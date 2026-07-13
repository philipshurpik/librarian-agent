import json
import logging
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from textwrap import wrap

from librarian.config import settings
from librarian.models import Book
from librarian.storage import db

logger = logging.getLogger('ingest')


def load_books(path: str | Path) -> list[Book]:
    """Parse raw catalog records; cleaning happens in the Book validators."""
    return [Book(**record) for record in json.loads(Path(path).read_text())]


def _rank(book: Book) -> tuple[int, str]:
    """Dedupe winner: longest description first, ties broken by smallest id — deterministic across runs."""
    return -len(book.description or ''), book.id


def _unique_by(books: list[Book], key: Callable[[Book], object]) -> list[Book]:
    best: dict[object, Book] = {}
    for book in books:
        k = key(book)
        if k not in best or _rank(book) < _rank(best[k]):
            best[k] = book
    return list(best.values())


def dedupe_books(books: list[Book]) -> list[Book]:
    """Two passes: duplicate ids, then same book re-entered under a different id (title+author match)."""
    by_id = _unique_by(books, lambda b: b.id)
    unique = _unique_by(by_id, lambda b: b.dedupe_key)
    if id_dups := len(books) - len(by_id):
        logger.info(f'dedupe: dropped {id_dups} records with duplicate ids')
    if content_dups := [b.id for b in by_id if b not in unique]:
        logger.info(f'dedupe: dropped content duplicates: {content_dups}')
    return unique


def _split_oversized(para: str, max_chars: int) -> list[str]:
    """Split oversized paragraphs by words up to max_chars, otherwise leave one parahraph"""
    return [para] if len(para) <= max_chars else wrap(para, max_chars)


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Pack paragraphs into chunks of at most max_chars; oversized paragraphs are word-wrapped."""
    paras = [p for para in text.split('\n\n') for p in _split_oversized(para, max_chars)]
    chunks, current = [], ''
    for para in paras:
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f'{current}\n\n{para}' if current else para
    return [*chunks, current] if current else chunks


def build_chunks(book: Book, max_chars: int) -> list[str]:
    """Texts to embed for one book, each prefixed with title/author for retrieval context.

    Books without a description get a metadata-only fallback so they stay discoverable.
    """
    prefix = f'{book.title} by {book.attributes.author}.'
    if not book.description:
        return [f'{prefix} Topic: {book.attributes.topic}.']
    return [f'{prefix}\n\n{chunk}' for chunk in _chunk_text(book.description, max_chars)]


def run() -> None:
    books = dedupe_books(load_books(settings.catalog_path))
    with closing(db.connect(settings.sqlite_path)) as conn:
        db.upsert_books(conn, books)
    logger.info(f'upserted {len(books)} books into {settings.sqlite_path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    run()
