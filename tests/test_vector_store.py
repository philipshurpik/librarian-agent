"""vector_store.search against in-memory Qdrant with hand-crafted 2d vectors."""

from conftest import RAW, make_book
from qdrant_client import QdrantClient

from librarian.storage import vector_store


def seeded_client():
    """Two books: bk-a with two near-identical chunks, bk-b (topic=streaming) pointing the other way."""
    client = QdrantClient(':memory:')
    vector_store.ensure_collection(client, dim=2)
    a = make_book(id='bk-a', title='Book A')
    b = make_book(id='bk-b', title='Book B', attributes={**RAW['attributes'], 'topic': 'streaming'})
    chunks = [(a, 0, 'a0'), (a, 1, 'a1'), (b, 0, 'b0')]
    vector_store.upsert_chunks(client, chunks, [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    return client


def test_search_returns_distinct_books_ranked_by_similarity():
    hits = vector_store.search(seeded_client(), [1.0, 0.0], limit=2)

    assert [h['book_id'] for h in hits] == ['bk-a', 'bk-b']  # bk-a once, despite two matching chunks
    assert hits[0]['score'] > hits[1]['score']
    assert (hits[0]['title'], hits[0]['author'], hits[0]['text']) == ('Book A', 'Jane Doe', 'a0')


def test_search_filters_by_payload():
    hits = vector_store.search(seeded_client(), [1.0, 0.0], limit=5, filters={'topic': 'streaming'})

    assert [h['book_id'] for h in hits] == ['bk-b']
