import pytest
from conftest import RAW, make_book

from librarian.models import Book


def test_title_and_author_whitespace_collapsed():
    book = make_book(
        title='  the pragmatic  programmer ',
        attributes={**RAW['attributes'], 'author': 'Andrew  Hunt,  David Thomas'},
    )
    assert book.title == 'the pragmatic programmer'
    assert book.attributes.author == 'Andrew Hunt, David Thomas'


@pytest.mark.parametrize('raw, expected', [(2017, 2017), ('c2013', 2013), ('2016', 2016), (None, None)])
def test_year_coercion(raw, expected):
    book = make_book(attributes={**RAW['attributes'], 'year': raw})
    assert book.attributes.year == expected


def test_year_missing():
    attrs = {k: v for k, v in RAW['attributes'].items() if k != 'year'}
    assert make_book(attributes=attrs).attributes.year is None


@pytest.mark.parametrize('raw, expected', [(-1, 0), (0, 0), (3, 3)])
def test_units_clamped(raw, expected):
    assert make_book(available_units=raw).available_units == expected


def test_units_missing_defaults_to_zero():
    raw = {k: v for k, v in RAW.items() if k != 'available_units'}
    assert Book(**raw).available_units == 0


@pytest.mark.parametrize('raw', [None, '', '  '])
def test_empty_description_becomes_none(raw):
    assert make_book(description=raw).description is None


def test_html_description_cleaned():
    raw = '<p>Dive into <b>velocity &amp; scaling</b>.</p><p>Covers pods &#8212; &quot;deeply&quot;.</p>'
    assert make_book(description=raw).description == 'Dive into velocity & scaling.\n\nCovers pods — "deeply".'


def test_paragraph_breaks_preserved():
    book = make_book(description='First  paragraph.\n\n  Second   paragraph.')
    assert book.description == 'First paragraph.\n\nSecond paragraph.'


def test_dedupe_key_matches_messy_duplicate():
    original = make_book(
        title='The Pragmatic Programmer', attributes={**RAW['attributes'], 'author': 'Andrew Hunt, David Thomas'}
    )
    messy = make_book(
        id='bk-079',
        title='the pragmatic  programmer ',
        attributes={**RAW['attributes'], 'author': 'Andrew  Hunt,  David Thomas'},
    )
    assert original.dedupe_key == messy.dedupe_key
