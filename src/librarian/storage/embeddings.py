from openai import OpenAI

from librarian.config import settings

_BATCH_SIZE = 128


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed texts preserving order; the SDK retries transient failures itself (max_retries=2)."""
    client = OpenAI(api_key=settings.openai_api_key)
    vectors = []
    for i in range(0, len(texts), _BATCH_SIZE):
        response = client.embeddings.create(model=settings.embedding_model, input=texts[i : i + _BATCH_SIZE])
        vectors.extend(d.embedding for d in sorted(response.data, key=lambda d: d.index))
    return vectors
