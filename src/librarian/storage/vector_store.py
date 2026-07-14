from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from librarian.config import settings
from librarian.models import Book


def _point_id(book_id: str, chunk_idx: int) -> str:
    """Deterministic ids for book + chunk index"""
    return str(uuid5(NAMESPACE_URL, f'{book_id}:{chunk_idx}'))


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient, dim: int) -> None:
    if not client.collection_exists(settings.qdrant_collection):
        vectors_config = VectorParams(size=dim, distance=Distance.COSINE)
        client.create_collection(settings.qdrant_collection, vectors_config=vectors_config)


def delete_points(client: QdrantClient, keys: list[tuple[str, int]]) -> None:
    """Remove chunks by (book_id, chunk_idx) — the orphaned tail of books whose chunk count shrank."""
    ids = [_point_id(book_id, idx) for book_id, idx in keys]
    client.delete(settings.qdrant_collection, points_selector=PointIdsList(points=ids))


def upsert_chunks(client: QdrantClient, chunks: list[tuple[Book, int, str]], vectors: list[list[float]]) -> None:
    points = [
        PointStruct(
            id=_point_id(book.id, idx),
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
