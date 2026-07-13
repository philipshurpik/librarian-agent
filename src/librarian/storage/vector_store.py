from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from librarian.config import settings
from librarian.models import Book


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient, dim: int) -> None:
    if not client.collection_exists(settings.qdrant_collection):
        vectors_config = VectorParams(size=dim, distance=Distance.COSINE)
        client.create_collection(settings.qdrant_collection, vectors_config=vectors_config)


def upsert_chunks(client: QdrantClient, chunks: list[tuple[Book, int, str]], vectors: list[list[float]]) -> None:
    """Point ids are uuid5(book_id:chunk_idx) — deterministic, so re-ingest overwrites instead of duplicating."""
    points = [
        PointStruct(
            id=str(uuid5(NAMESPACE_URL, f'{book.id}:{idx}')),
            vector=vector,
            payload={
                'book_id': book.id,
                'title': book.title,
                'chunk_idx': idx,
                'text': text,
                **book.attributes.model_dump(),
            },
        )
        for (book, idx, text), vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(settings.qdrant_collection, points=points)
