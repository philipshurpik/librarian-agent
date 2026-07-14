"""HTTP surface: a stateless chat endpoint driving the agent loop, plus a health probe."""

import logging

from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from librarian.agent import loop

logging.basicConfig(level=logging.INFO)
app = FastAPI(title='Ask-a-Librarian')


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

    @field_validator('history')
    @classmethod
    def reject_system_messages(cls, history: list[dict]) -> list[dict]:
        """The server owns the system prompt; a client-injected one would override it."""
        if any(m.get('role') == 'system' for m in history):
            raise ValueError('history must not contain system messages')
        return history


class ChatResponse(BaseModel):
    reply: str
    history: list[dict]


@app.post('/chat')
async def chat(request: ChatRequest) -> ChatResponse:
    """The client carries the history (raw OpenAI message dicts) and sends it back with each turn."""
    messages = [*request.history, {'role': 'user', 'content': request.message}]
    produced = await loop.run(messages)
    return ChatResponse(reply=produced[-1]['content'] or '', history=[*messages, *produced])


@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}
