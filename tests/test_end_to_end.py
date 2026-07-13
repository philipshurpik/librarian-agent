import sqlite3
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from librarian.config import settings
from librarian.ingest import run
from librarian.storage import embeddings, vector_store

CATALOG = Path(__file__).parents[1] / 'data' / 'catalog.json'


def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(t)), float(t.count(' ')), 1.0] for t in texts]


def test_run_end_to_end_is_idempotent(tmp_path, monkeypatch):
    client = QdrantClient(':memory:')
    monkeypatch.setattr(settings, 'catalog_path', str(CATALOG))
    monkeypatch.setattr(settings, 'sqlite_path', str(tmp_path / 'library.db'))
    monkeypatch.setattr(vector_store, 'get_client', lambda: client)
    monkeypatch.setattr(embeddings, 'embed_batch', _fake_embed)

    run()
    run()

    conn = sqlite3.connect(settings.sqlite_path)
    assert conn.execute('SELECT COUNT(*) FROM books').fetchone()[0] == 77
    assert client.count(settings.qdrant_collection).count == 82

    book_filter = Filter(must=[FieldCondition(key='book_id', match=MatchValue(value='bk-001'))])
    points, _ = client.scroll(settings.qdrant_collection, scroll_filter=book_filter, limit=10)
    payloads = sorted((p.payload for p in points), key=lambda p: p['chunk_idx'])
    assert [p['chunk_idx'] for p in payloads] == [0, 1, 2]
    assert payloads[0]['author'] == 'Martin Kleppmann'
    assert payloads[0]['text'].startswith('Designing Data-Intensive Applications by Martin Kleppmann.')
