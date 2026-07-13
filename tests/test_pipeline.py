import json
from pathlib import Path

from conftest import RAW, make_book

from librarian.ingest import dedupe_books, load_books

CATALOG = Path(__file__).parents[1] / 'data' / 'catalog.json'


def test_load_books_parses_and_cleans(tmp_path):
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps([{**RAW, 'title': '  Messy  Title '}]))
    assert [b.title for b in load_books(path)] == ['Messy Title']


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


def test_real_catalog_dedupes_to_expected():
    books = dedupe_books(load_books(CATALOG))
    assert len(books) == 77  # 80 raw - repeated bk-030 - content dups bk-078 (of bk-014), bk-079 (of bk-013)
    ids = {b.id for b in books}
    assert {'bk-014', 'bk-013', 'bk-030'} <= ids
    assert {'bk-078', 'bk-079'}.isdisjoint(ids)
