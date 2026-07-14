import json
import sqlite3
from pathlib import Path

from librarian.models import Book

_SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    topic TEXT NOT NULL,
    year INTEGER,
    level TEXT,
    description TEXT,
    available_units INTEGER NOT NULL DEFAULT 0,
    reserved_units INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT,
    chunk_count INTEGER
);
"""

_UPSERT = """
INSERT INTO books (id, title, author, topic, year, level, description, available_units)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    title = excluded.title,
    author = excluded.author,
    topic = excluded.topic,
    year = excluded.year,
    level = excluded.level,
    description = excluded.description,
    available_units = excluded.available_units
"""


def connect(path: str | Path) -> sqlite3.Connection:
    if path != ':memory:':
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def upsert_books(conn: sqlite3.Connection, books: list[Book]) -> None:
    """Ledger columns (content_hash, chunk_count) are untouched — see update_ledger."""
    rows = [
        (
            b.id,
            b.title,
            b.attributes.author,
            b.attributes.topic,
            b.attributes.year,
            b.attributes.level,
            b.description,
            b.available_units,
        )
        for b in books
    ]
    conn.executemany(_UPSERT, rows)
    conn.commit()


def load_ledger(conn: sqlite3.Connection, ids: list[str]) -> dict[str, tuple[str | None, int]]:
    """What each book (in delta we are currently processing) looked like when it was last indexed."""
    query = 'SELECT id, content_hash, chunk_count FROM books WHERE id IN (SELECT value FROM json_each(?))'
    return {book_id: (h, count or 0) for book_id, h, count in conn.execute(query, [json.dumps(ids)])}


def get_book(conn: sqlite3.Connection, book_id: str) -> sqlite3.Row | None:
    query = 'SELECT *, available_units - reserved_units AS available FROM books WHERE id = ?'
    return conn.execute(query, (book_id,)).fetchone()


def reserve_book(conn: sqlite3.Connection, book_id: str) -> bool:
    """Atomic check-and-increment: cannot oversell under concurrent requests."""
    query = 'UPDATE books SET reserved_units = reserved_units + 1 WHERE id = ? AND reserved_units < available_units'
    ok = conn.execute(query, (book_id,)).rowcount == 1
    conn.commit()
    return ok


def update_ledger(conn: sqlite3.Connection, entries: dict[str, tuple[str, int]]) -> None:
    """Written only after the index write succeeds — the ledger records what Qdrant actually contains."""
    rows = [(h, count, book_id) for book_id, (h, count) in entries.items()]
    conn.executemany('UPDATE books SET content_hash = ?, chunk_count = ? WHERE id = ?', rows)
    conn.commit()
