from contextlib import closing
from functools import cache

from qdrant_client import AsyncQdrantClient

from librarian.config import settings
from librarian.storage import db, embeddings, vector_store

_SNIPPET_CHARS = 300
_WEAK_SCORE = 0.405  # midpoint of the F2-best threshold band measured by evals/retrieval.py: (0.381, 0.429]


@cache
def _qdrant() -> AsyncQdrantClient:
    return vector_store.get_client()


def _result(hit: dict) -> dict:
    keys = ('book_id', 'title', 'author', 'topic', 'level', 'year', 'score')
    return {k: hit[k] for k in keys} | {'snippet': hit['text'][:_SNIPPET_CHARS]}


_SEARCH_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'search_catalog',
        'description': 'Semantic search over the library catalog; returns best-matching books with scores.',
        'parameters': {
            'type': 'object',
            'properties': {'query': {'type': 'string', 'description': 'what the user wants to read about'}},
            'required': ['query'],
        },
    },
}


async def search_catalog(query: str, limit: int = 5) -> list[dict]:
    """Best-matching books for a free-text query; `limit` is deliberately not in the schema — callers (evals) set it."""
    hits = await vector_store.search(_qdrant(), await embeddings.embed_query(query), limit=limit)
    return [_result(h) for h in hits]


_AVAILABILITY_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'check_availability',
        'description': 'How many copies of a book are currently available to reserve.',
        'parameters': {
            'type': 'object',
            'properties': {'book_id': {'type': 'string', 'description': 'catalog id, e.g. bk-004'}},
            'required': ['book_id'],
        },
    },
}


async def check_availability(book_id: str) -> dict:
    with closing(db.connect(settings.sqlite_path)) as conn:
        row = db.get_book(conn, book_id)
    if row is None:
        return {'error': f'unknown book_id: {book_id}'}
    return {'book_id': book_id, 'title': row['title'], 'available': row['available']}


_RESERVE_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'reserve_book',
        'description': 'Reserve one copy of a book. Call only after the user explicitly asked to reserve it.',
        'parameters': {
            'type': 'object',
            'properties': {'book_id': {'type': 'string', 'description': 'catalog id, e.g. bk-004'}},
            'required': ['book_id'],
        },
    },
}


async def reserve_book(book_id: str) -> dict:
    """Reserve one copy; the UPDATE is atomic, so concurrent requests cannot oversell."""
    with closing(db.connect(settings.sqlite_path)) as conn:
        if db.get_book(conn, book_id) is None:
            return {'error': f'unknown book_id: {book_id}'}
        reserved = db.reserve_book(conn, book_id)
        available = db.get_book(conn, book_id)['available']
    return {'book_id': book_id, 'reserved': reserved, 'available': available}


def _topic_param() -> dict:
    """Enum of the catalog's actual topic vocabulary (read once at startup) — the model cannot invent values."""
    with closing(db.connect(settings.sqlite_path)) as conn:
        topics = db.list_topics(conn)
    return {'type': 'string', 'enum': topics} if topics else {'type': 'string'}


_RECOMMEND_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'recommend',
        'description': 'Recommend available books for the user interests, optionally filtered by topic/level. '
        'Omit topic and level unless the user explicitly asked to narrow. '
        'A "note" in the result means matches are weak — present them as closest alternatives.',
        'parameters': {
            'type': 'object',
            'properties': {
                'interests': {'type': 'string', 'description': 'what the user is looking for'},
                'topic': _topic_param(),
                'level': {'type': 'string', 'enum': ['beginner', 'intermediate', 'advanced']},
            },
            'required': ['interests'],
        },
    },
}


async def recommend(interests: str, topic: str | None = None, level: str | None = None, limit: int = 3) -> dict:
    """Search + availability, with an honesty note when filtered matches are weak or absent."""
    filters = {k: v for k, v in (('topic', topic), ('level', level)) if v}
    vector = await embeddings.embed_query(interests)
    hits = await vector_store.search(_qdrant(), vector, limit=limit, filters=filters or None)
    with closing(db.connect(settings.sqlite_path)) as conn:
        rows = [(h, db.get_book(conn, h['book_id'])) for h in hits]
    # a hit without a SQLite row is index drift (stale Qdrant point) — skip it rather than crash
    results = [_result(h) | {'available': row['available']} for h, row in rows if row]
    if not results:
        return {'results': [], 'note': 'no books matched these filters'}
    if results[0]['score'] < _WEAK_SCORE:
        return {'results': results, 'note': 'weak matches only — consider relaxing topic/level filters'}
    return {'results': results}


TOOL_SCHEMAS = [_SEARCH_SCHEMA, _AVAILABILITY_SCHEMA, _RESERVE_SCHEMA, _RECOMMEND_SCHEMA]
