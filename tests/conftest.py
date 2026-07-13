from librarian.models import Book

RAW = {
    'id': 'bk-001',
    'title': 'Some Title',
    'attributes': {'author': 'Jane Doe', 'topic': 'databases', 'year': 2017, 'level': 'intermediate'},
    'description': 'A fine book.',
    'available_units': 2,
}


def make_book(**overrides) -> Book:
    return Book(**{**RAW, **overrides})
