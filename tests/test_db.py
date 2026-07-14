from conftest import make_book

from librarian.storage import db


def test_upsert_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / 'library.db')
    books = [make_book(id='bk-1'), make_book(id='bk-2', title='Other Book', available_units=5)]
    db.upsert_books(conn, books)
    db.upsert_books(conn, books)

    rows = conn.execute('SELECT * FROM books ORDER BY id').fetchall()
    assert len(rows) == 2
    assert (rows[0]['id'], rows[0]['title'], rows[0]['author']) == ('bk-1', 'Some Title', 'Jane Doe')
    assert (rows[1]['title'], rows[1]['available_units']) == ('Other Book', 5)


def test_upsert_updates_existing_row(tmp_path):
    conn = db.connect(tmp_path / 'library.db')
    db.upsert_books(conn, [make_book(available_units=2)])
    db.upsert_books(conn, [make_book(available_units=7, description=None)])

    row = conn.execute('SELECT available_units, description FROM books').fetchone()
    assert row['available_units'] == 7
    assert row['description'] is None


def test_reserve_book_stops_at_available_units(tmp_path):
    conn = db.connect(tmp_path / 'library.db')
    db.upsert_books(conn, [make_book(available_units=2)])

    assert db.reserve_book(conn, 'bk-001')
    assert db.reserve_book(conn, 'bk-001')
    assert not db.reserve_book(conn, 'bk-001')  # pool exhausted
    assert db.get_book(conn, 'bk-001')['available'] == 0
    assert not db.reserve_book(conn, 'missing')
    assert db.get_book(conn, 'missing') is None


def test_reingest_preserves_reservations(tmp_path):
    conn = db.connect(tmp_path / 'library.db')
    db.upsert_books(conn, [make_book(available_units=2)])
    db.reserve_book(conn, 'bk-001')
    db.upsert_books(conn, [make_book(available_units=5)])  # catalog update arrives

    row = db.get_book(conn, 'bk-001')
    assert (row['reserved_units'], row['available']) == (1, 4)
