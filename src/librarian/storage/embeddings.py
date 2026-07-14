from functools import cache

from openai import AsyncOpenAI

from librarian.config import settings

_BATCH_SIZE = 128


@cache
def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed texts preserving order; the SDK retries transient failures itself (max_retries=2)."""
    vectors = []
    for i in range(0, len(texts), _BATCH_SIZE):
        response = await _client().embeddings.create(model=settings.embedding_model, input=texts[i : i + _BATCH_SIZE])
        vectors.extend(d.embedding for d in sorted(response.data, key=lambda d: d.index))
    return vectors


async def embed_query(text: str) -> list[float]:
    return (await embed_batch([text]))[0]
