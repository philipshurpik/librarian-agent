"""HTTP surface: a stateless chat endpoint driving the agent loop, plus a health probe."""

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from librarian.agent import loop

logging.basicConfig(level=logging.INFO)
app = FastAPI(title='Ask-a-Librarian')


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str
    history: list[dict]


@app.post('/chat')
async def chat(request: ChatRequest) -> ChatResponse:
    """The client carries the history (raw OpenAI message dicts) and sends it back with each turn."""
    messages = [*request.history, {'role': 'user', 'content': request.message}]
    produced = await loop.run(messages)
    return ChatResponse(reply=produced[-1]['content'], history=[*messages, *produced])


@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}
