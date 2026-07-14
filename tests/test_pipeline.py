import json
from pathlib import Path

from conftest import RAW, make_book

from librarian.ingest import _chunk_text, build_chunks, dedupe_books, load_books

CATALOG = Path(__file__).parents[1] / 'data' / 'catalog.json'


def test_load_books_parses_and_cleans(tmp_path):
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps([{**RAW, 'title': '  Messy  Title '}]))
    assert [b.title for b in load_books(path)] == ['Messy Title']


def test_load_books_skips_invalid_records(tmp_path):
    no_author = {**RAW, 'id': 'bk-bad', 'attributes': {'topic': 'databases'}}
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps([RAW, no_author, {'id': 'bk-worse'}]))
    assert [b.id for b in load_books(path)] == ['bk-001']  # corrupt entries are skipped, the batch survives


def test_exact_id_duplicate_dropped():
    assert len(dedupe_books([make_book(), make_book()])) == 1


def test_id_duplicate_keeps_longer_description():
    books = [make_book(description='short'), make_book(description='a much longer description')]
    (kept,) = dedupe_books(books)
    assert kept.description == 'a much longer description'


def test_content_duplicate_dropped_deterministically():
    original = make_book(id='bk-013', description='a long, rich description of the book')
    messy = make_book(id='bk-079', title='some  title ', description='short blurb')
    assert dedupe_books([original, messy]) == dedupe_books([messy, original]) == [original]


def test_content_duplicate_tie_breaks_on_smaller_id():
    first, second = make_book(id='bk-014'), make_book(id='bk-078')
    assert dedupe_books([second, first]) == [first]


def test_same_title_different_author_is_not_a_duplicate():
    other = make_book(id='bk-999', attributes={**RAW['attributes'], 'author': 'John Smith'})
    assert len(dedupe_books([make_book(), other])) == 2


def test_chunk_text_packs_paragraphs_up_to_budget():
    paras = ['a' * 400, 'b' * 400, 'c' * 400]
    chunks = _chunk_text('\n\n'.join(paras), max_chars=900)
    assert chunks == ['\n\n'.join(paras[:2]), paras[2]]


def test_chunk_text_hard_splits_oversized_paragraph():
    chunks = _chunk_text('word ' * 400, max_chars=1500)  # one 2000-char paragraph, no \n\n
    assert len(chunks) == 2
    assert all(len(c) <= 1500 for c in chunks)
    assert ' '.join(chunks) == ('word ' * 400).strip()


def test_build_chunks_prefixes_title_and_author():
    (chunk,) = build_chunks(make_book(), max_chars=1500)
    assert chunk == 'Some Title by Jane Doe.\n\nA fine book.'


def test_build_chunks_fallback_without_description():
    (chunk,) = build_chunks(make_book(description=None), max_chars=1500)
    assert chunk == 'Some Title by Jane Doe. Topic: databases.'


def test_build_chunks_splits_long_description():
    book = make_book(description='\n\n'.join(['word ' * 100] * 4))
    chunks = build_chunks(book, max_chars=600)
    assert len(chunks) == 4
    assert all(c.startswith('Some Title by Jane Doe.') for c in chunks)


def test_real_catalog_dedupes_to_expected():
    books = dedupe_books(load_books(CATALOG))
    assert len(books) == 77  # 80 raw - repeated bk-030 - content dups bk-078 (of bk-014), bk-079 (of bk-013)
    ids = {b.id for b in books}
    assert {'bk-014', 'bk-013', 'bk-030'} <= ids
    assert {'bk-078', 'bk-079'}.isdisjoint(ids)
