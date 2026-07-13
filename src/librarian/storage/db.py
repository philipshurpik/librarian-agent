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
    available_units INTEGER NOT NULL DEFAULT 0
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
