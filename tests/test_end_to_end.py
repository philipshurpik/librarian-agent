"""Whole-pipeline tests: the real run() against in-memory Qdrant and stubbed embeddings."""

import json
import sqlite3
from pathlib import Path

import pytest
from conftest import RAW
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from librarian.config import settings
from librarian.ingest import run
from librarian.storage import embeddings, vector_store

CATALOG = Path(__file__).parents[1] / 'data' / 'catalog.json'


def setup_run(tmp_path, monkeypatch, catalog_path=CATALOG):
    """Patch settings + external clients; returns (qdrant client, list of embed batch sizes)."""
    client, calls = QdrantClient(':memory:'), []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(len(texts))
        return [[1.0, float(len(t))] for t in texts]

    monkeypatch.setattr(settings, 'catalog_path', str(catalog_path))
    monkeypatch.setattr(settings, 'sqlite_path', str(tmp_path / 'library.db'))
    monkeypatch.setattr(vector_store, 'get_client', lambda: client)
    monkeypatch.setattr(embeddings, 'embed_batch', fake_embed)
    return client, calls


def test_run_end_to_end_is_idempotent(tmp_path, monkeypatch):
    client, calls = setup_run(tmp_path, monkeypatch)

    run()
    run()
    assert calls == [82]  # second run: ledger says nothing changed -> zero embedding calls

    conn = sqlite3.connect(settings.sqlite_path)
    assert conn.execute('SELECT COUNT(*) FROM books').fetchone()[0] == 77
    assert client.count(settings.qdrant_collection).count == 82

    book_filter = Filter(must=[FieldCondition(key='book_id', match=MatchValue(value='bk-001'))])
    points, _ = client.scroll(settings.qdrant_collection, scroll_filter=book_filter, limit=10)
    payloads = sorted((p.payload for p in points), key=lambda p: p['chunk_idx'])
    assert [p['chunk_idx'] for p in payloads] == [0, 1, 2]
    assert payloads[0]['author'] == 'Martin Kleppmann'
    assert payloads[0]['text'].startswith('Designing Data-Intensive Applications by Martin Kleppmann.')


def test_run_reembeds_only_changed_and_deletes_orphaned_chunks(tmp_path, monkeypatch):
    catalog = tmp_path / 'catalog.json'
    client, calls = setup_run(tmp_path, monkeypatch, catalog_path=catalog)

    two_chunk_description = f'{"lorem ipsum " * 80}\n\n{"dolor sit amet " * 70}'
    books = [{**RAW, 'description': two_chunk_description}, {**RAW, 'id': 'bk-002', 'title': 'Another Title'}]
    catalog.write_text(json.dumps(books))
    run()
    catalog.write_text(json.dumps([{**RAW, 'description': 'A changed description.'}]))  # delta without bk-002
    run()

    assert calls == [3, 1]  # second run embeds only the changed book
    assert client.count(settings.qdrant_collection).count == 2  # bk-001 orphaned tail gone, absent bk-002 kept
    points, _ = client.scroll(settings.qdrant_collection, limit=10)
    texts = {p.payload['book_id']: p.payload['text'] for p in points}
    assert texts.keys() == {'bk-001', 'bk-002'}
    assert 'A changed description.' in texts['bk-001']

    conn = sqlite3.connect(settings.sqlite_path)
    assert [r[0] for r in conn.execute('SELECT id FROM books ORDER BY id')] == ['bk-001', 'bk-002']


def test_failed_embed_is_retried_on_next_run(tmp_path, monkeypatch):
    """Ledger is written after the index write: a run that dies mid-embed leaves its delta marked as pending."""
    catalog = tmp_path / 'catalog.json'
    client, calls = setup_run(tmp_path, monkeypatch, catalog_path=catalog)
    catalog.write_text(json.dumps([RAW]))
    run()  # collection exists so the fresh-index drift guard stays out of the picture

    catalog.write_text(json.dumps([{**RAW, 'id': 'bk-002', 'title': 'Another Title'}]))
    monkeypatch.setattr(embeddings, 'embed_batch', lambda texts: (_ for _ in ()).throw(RuntimeError('api down')))
    with pytest.raises(RuntimeError):
        run()

    monkeypatch.setattr(embeddings, 'embed_batch', lambda texts: [[1.0, 2.0]] * len(texts))
    run()
    assert client.count(settings.qdrant_collection).count == 2  # bk-002 retried: failed run never wrote its ledger
