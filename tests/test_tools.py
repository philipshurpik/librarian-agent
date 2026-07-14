from conftest import RAW, make_book
from qdrant_client import QdrantClient

from librarian.agent import tools
from librarian.config import settings
from librarian.storage import db, embeddings, vector_store

A_VEC, B_VEC = [1.0, 0.0], [0.0, 1.0]


def setup_tools(tmp_path, monkeypatch):
    """bk-a (2 copies) matches the stub query vector; bk-b (topic=streaming, 0 copies) is orthogonal to it."""
    client = QdrantClient(':memory:')
    vector_store.ensure_collection(client, dim=2)
    a = make_book(id='bk-a', title='Book A')
    b = make_book(id='bk-b', title='Book B', attributes={**RAW['attributes'], 'topic': 'streaming'}, available_units=0)
    vector_store.upsert_chunks(client, [(a, 0, 'All about A. ' * 40), (b, 0, 'text b')], [A_VEC, B_VEC])

    conn = db.connect(tmp_path / 'library.db')
    db.upsert_books(conn, [a, b])
    conn.close()

    monkeypatch.setattr(settings, 'sqlite_path', str(tmp_path / 'library.db'))
    monkeypatch.setattr(tools, '_qdrant', lambda: client)
    monkeypatch.setattr(embeddings, 'embed_query', lambda text: A_VEC)


def test_search_catalog_returns_ranked_results(tmp_path, monkeypatch):
    setup_tools(tmp_path, monkeypatch)

    results = tools.search_catalog('books like A')
    assert [r['book_id'] for r in results] == ['bk-a', 'bk-b']
    assert (results[0]['title'], results[0]['score']) == ('Book A', 1.0)
    assert len(results[0]['snippet']) == tools._SNIPPET_CHARS  # long chunk text is truncated


def test_check_availability(tmp_path, monkeypatch):
    setup_tools(tmp_path, monkeypatch)

    assert tools.check_availability('bk-a') == {'book_id': 'bk-a', 'title': 'Book A', 'available': 2}
    assert 'error' in tools.check_availability('bk-x')


def test_reserve_book_success_exhaustion_and_unknown(tmp_path, monkeypatch):
    setup_tools(tmp_path, monkeypatch)

    assert tools.reserve_book('bk-a') == {'book_id': 'bk-a', 'reserved': True, 'available': 1}
    assert tools.reserve_book('bk-b') == {'book_id': 'bk-b', 'reserved': False, 'available': 0}
    assert 'error' in tools.reserve_book('bk-x')


def test_recommend_filters_annotates_availability_and_flags_weak_matches(tmp_path, monkeypatch):
    setup_tools(tmp_path, monkeypatch)

    good = tools.recommend('something like A')
    assert good['results'][0]['book_id'] == 'bk-a'
    assert good['results'][0]['available'] == 2
    assert 'note' not in good

    weak = tools.recommend('something like A', topic='streaming')  # filter leaves only the orthogonal book
    assert [r['book_id'] for r in weak['results']] == ['bk-b']
    assert weak['results'][0]['available'] == 0
    assert weak['note'] == 'weak matches only — consider relaxing topic/level filters'

    assert tools.recommend('anything', level='expert') == {'results': [], 'note': 'no books matched these filters'}


def test_topic_param_lists_catalog_vocabulary_or_stays_open(tmp_path, monkeypatch):
    setup_tools(tmp_path, monkeypatch)
    assert tools._topic_param() == {'type': 'string', 'enum': ['databases', 'streaming']}

    monkeypatch.setattr(settings, 'sqlite_path', str(tmp_path / 'empty.db'))
    assert tools._topic_param() == {'type': 'string'}  # before first ingest: no enum, not an all-forbidding empty one
