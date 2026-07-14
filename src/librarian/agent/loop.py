import asyncio
import json
import logging
from functools import cache

from openai import AsyncOpenAI

from librarian.agent import tools
from librarian.agent.prompts import SYSTEM_PROMPT
from librarian.config import settings

logger = logging.getLogger('agent')

_MAX_TOOL_ROUNDS = 6
_LLM_TIMEOUT_SECONDS = 60

_TOOL_NAMES = {schema['function']['name'] for schema in tools.TOOL_SCHEMAS}


@cache
def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key, timeout=_LLM_TIMEOUT_SECONDS)


async def _execute(call) -> str:
    """Failures return as tool content for the model to recover from — a bad call never breaks the request."""
    name = call.function.name
    if name not in _TOOL_NAMES:
        return json.dumps({'error': f'unknown tool: {name}'})
    try:
        return json.dumps(await getattr(tools, name)(**json.loads(call.function.arguments)))
    except Exception as e:
        logger.warning(f'tool {name} failed: {type(e).__name__}: {e}')
        return json.dumps({'error': f'{type(e).__name__}: {e}'})


def _as_dict(message) -> dict:
    result = {'role': 'assistant', 'content': message.content}
    if message.tool_calls:
        result['tool_calls'] = [
            {'id': c.id, 'type': 'function', 'function': {'name': c.function.name, 'arguments': c.function.arguments}}
            for c in message.tool_calls
        ]
    return result


async def run(messages: list[dict]) -> list[dict]:
    """Advance the conversation by one agent turn; returns only the newly produced messages."""
    client = _client()
    conversation = [{'role': 'system', 'content': SYSTEM_PROMPT}, *messages]
    produced: list[dict] = []
    for _ in range(_MAX_TOOL_ROUNDS):
        response = await client.chat.completions.create(
            model=settings.chat_model, messages=conversation, tools=tools.TOOL_SCHEMAS
        )
        message = response.choices[0].message
        assistant = _as_dict(message)
        conversation.append(assistant)
        produced.append(assistant)
        if not message.tool_calls:
            return produced
        logger.info(f'tool calls: {[c.function.name for c in message.tool_calls]}')
        contents = await asyncio.gather(*(_execute(call) for call in message.tool_calls))
        for call, content in zip(message.tool_calls, contents):
            tool_message = {'role': 'tool', 'tool_call_id': call.id, 'content': content}
            conversation.append(tool_message)
            produced.append(tool_message)

    response = await client.chat.completions.create(model=settings.chat_model, messages=conversation)
    produced.append({'role': 'assistant', 'content': response.choices[0].message.content})
    return produced
