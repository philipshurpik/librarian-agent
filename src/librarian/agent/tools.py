from contextlib import closing
from functools import cache

from qdrant_client import QdrantClient

from librarian.config import settings
from librarian.storage import db, embeddings, vector_store

_SNIPPET_CHARS = 300
_WEAK_SCORE = 0.405  # midpoint of the F2-best threshold band measured by evals/retrieval.py: (0.381, 0.429]


@cache
def _qdrant() -> QdrantClient:
    return vector_store.get_client()


def _result(hit: dict) -> dict:
    keys = ('book_id', 'title', 'author', 'topic', 'level', 'year', 'score')
    return {k: hit[k] for k in keys} | {'snippet': hit['text'][:_SNIPPET_CHARS]}


def search_catalog(query: str, limit: int = 5) -> list[dict]:
    """Best-matching books for a free-text query."""
    hits = vector_store.search(_qdrant(), embeddings.embed_query(query), limit=limit)
    return [_result(h) for h in hits]


def check_availability(book_id: str) -> dict:
    with closing(db.connect(settings.sqlite_path)) as conn:
        row = db.get_book(conn, book_id)
    if row is None:
        return {'error': f'unknown book_id: {book_id}'}
    return {'book_id': book_id, 'title': row['title'], 'available': row['available']}


def reserve_book(book_id: str) -> dict:
    """Reserve one copy; the UPDATE is atomic, so concurrent requests cannot oversell."""
    with closing(db.connect(settings.sqlite_path)) as conn:
        if db.get_book(conn, book_id) is None:
            return {'error': f'unknown book_id: {book_id}'}
        reserved = db.reserve_book(conn, book_id)
        available = db.get_book(conn, book_id)['available']
    return {'book_id': book_id, 'reserved': reserved, 'available': available}


def recommend(interests: str, topic: str | None = None, level: str | None = None, limit: int = 3) -> dict:
    """Search + availability, with an honesty note when filtered matches are weak or absent."""
    filters = {k: v for k, v in (('topic', topic), ('level', level)) if v}
    hits = vector_store.search(_qdrant(), embeddings.embed_query(interests), limit=limit, filters=filters or None)
    with closing(db.connect(settings.sqlite_path)) as conn:
        results = [_result(h) | {'available': db.get_book(conn, h['book_id'])['available']} for h in hits]
    if not results:
        return {'results': [], 'note': 'no books matched these filters'}
    if results[0]['score'] < _WEAK_SCORE:
        return {'results': results, 'note': 'weak matches only — consider relaxing topic/level filters'}
    return {'results': results}
