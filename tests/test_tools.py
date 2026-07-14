from conftest import RAW, make_book
from qdrant_client import AsyncQdrantClient

from librarian.agent import tools
from librarian.config import settings
from librarian.storage import db, embeddings, vector_store

A_VEC, B_VEC = [1.0, 0.0], [0.0, 1.0]


async def setup_tools(tmp_path, monkeypatch):
    """bk-a (2 copies) matches the stub query vector; bk-b (topic=streaming, 0 copies) is orthogonal to it."""
    client = AsyncQdrantClient(':memory:')
    a = make_book(id='bk-a', title='Book A')
    b = make_book(id='bk-b', title='Book B', attributes={**RAW['attributes'], 'topic': 'streaming'}, available_units=0)
    await vector_store.ensure_collection(client, dim=2)
    await vector_store.upsert_chunks(client, [(a, 0, 'All about A. ' * 40), (b, 0, 'text b')], [A_VEC, B_VEC])

    conn = db.connect(tmp_path / 'library.db')
    db.upsert_books(conn, [a, b])
    conn.close()

    async def fake_embed_query(text):
        return A_VEC

    monkeypatch.setattr(settings, 'sqlite_path', str(tmp_path / 'library.db'))
    monkeypatch.setattr(tools, '_qdrant', lambda: client)
    monkeypatch.setattr(embeddings, 'embed_query', fake_embed_query)


async def test_search_catalog_returns_ranked_results(tmp_path, monkeypatch):
    await setup_tools(tmp_path, monkeypatch)

    results = (await tools.search_catalog('books like A'))['results']
    assert [r['book_id'] for r in results] == ['bk-a', 'bk-b']
    assert (results[0]['title'], results[0]['score']) == ('Book A', 1.0)
    assert len(results[0]['snippet']) == tools._SNIPPET_CHARS  # long chunk text is truncated


async def test_check_availability(tmp_path, monkeypatch):
    await setup_tools(tmp_path, monkeypatch)

    assert await tools.check_availability('bk-a') == {'book_id': 'bk-a', 'title': 'Book A', 'available': 2}
    assert 'error' in await tools.check_availability('bk-x')


async def test_reserve_book_success_exhaustion_and_unknown(tmp_path, monkeypatch):
    await setup_tools(tmp_path, monkeypatch)

    assert await tools.reserve_book('bk-a') == {'book_id': 'bk-a', 'reserved': True, 'available': 1}
    assert await tools.reserve_book('bk-b') == {'book_id': 'bk-b', 'reserved': False, 'available': 0}
    assert 'error' in await tools.reserve_book('bk-x')


async def test_recommend_filters_annotates_availability_and_flags_weak_matches(tmp_path, monkeypatch):
    await setup_tools(tmp_path, monkeypatch)

    good = await tools.recommend('something like A')
    assert good['results'][0]['book_id'] == 'bk-a'
    assert good['results'][0]['available'] == 2
    assert 'note' not in good

    weak = await tools.recommend('something like A', topic='streaming')  # filter leaves only the orthogonal book
    assert [r['book_id'] for r in weak['results']] == ['bk-b']
    assert weak['results'][0]['available'] == 0
    assert weak['note'].startswith('weak matches only')

    assert await tools.recommend('anything', level='expert') == {
        'results': [],
        'note': 'no books matched these filters',
    }


async def test_recommend_skips_hits_missing_from_sqlite(tmp_path, monkeypatch):
    await setup_tools(tmp_path, monkeypatch)
    conn = db.connect(settings.sqlite_path)
    conn.execute("DELETE FROM books WHERE id = 'bk-b'")
    conn.commit()
    conn.close()

    results = (await tools.recommend('something like A'))['results']
    assert [r['book_id'] for r in results] == ['bk-a']  # stale Qdrant point for bk-b is dropped, not a crash


async def test_topic_param_lists_catalog_vocabulary_or_stays_open(tmp_path, monkeypatch):
    await setup_tools(tmp_path, monkeypatch)
    assert tools._topic_param() == {'type': 'string', 'enum': ['databases', 'streaming']}

    monkeypatch.setattr(settings, 'sqlite_path', str(tmp_path / 'empty.db'))
    assert tools._topic_param() == {'type': 'string'}  # before first ingest: no enum, not an all-forbidding empty one
