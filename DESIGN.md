# Design

## Ingestion: architecture

```
data/catalog.json
      │
      ▼
 load + validate ──── cleaning lives in pydantic validators (parse boundary)
      │
      ▼
    dedupe ────────── by id, then by content key (title + author)
      │
      ▼
   SQLite ─────────── system of record: books + index ledger (content_hash, chunk_count)
      │
      ▼
  hash diff ───────── new / changed books
      │
      ▼
chunk → embed ─────── OpenAI, new/changed only
      │
      ▼
   Qdrant ─────────── serving index: one point per chunk, deterministic ids
```

SQLite is the system of record: content, hashes, availability, reservations
Qdrant: content, embeddings for semantic search

## Ingestion: design decisions

- **Cleaning at the model boundary:**
  - All normalization (whitespace, HTML stripping, year coercion, unit clamping) lives in `Book` validators.
- **Dedupe: drop, don't merge:**
  - Two passes: remove duplicate ids, then deduplicate normalized (title, author) under different ids. 
  - Assumption: a content duplicate is a double entry, not extra stock. 
  - The winner is deterministic (longest description, tie → smallest id) - achieve idempotency
- **Chunking: paragraphs packed to ~1500 chars:** 
  - Descriptions are split on paragraph boundaries and greedily packed; 
  - Each chunk is prefixed with `"{title} by {author}."` so a mid-book chunk stays attributable after retrieval. 
  - Oversized single paragraphs are word-wrapped as a fallback.
- **Idempotency at both layers:**
  - Using `ON CONFLICT DO UPDATE` in SQLite - preserves row identity for foreign keys).
  - Qdrant point ids are `uuid5(book_id:chunk_idx)` — re-ingest overwrites items, never duplicates them.
- **Incremental embedding via a ledger (`content_hash`, `chunk_count`):**
  - `content_hash = sha256(chunk_texts)` is stored in SQLite next to the data it describes.
    Only new/changed books are embedded (O(delta)); steady-state re-run makes zero embedding calls.
  - Upsert first, then delete chunk tail if description shrinks (`chunk_idx >= new chunk_count`)
  - The ledger is written only after the index write succeeds: crashed embeddings are self-healing.
- **Assumption: Catalog is a delta, not a snapshot:**
  - We only insert new items and change existing from it, never delete. Could work as delta batches.
  - Ledger reads are O(batch) too: scoped to the file's ids via a single `json_each` lookup.
  - Trade-off: if the Qdrant collection is lost, a re-run only rebuilds books present in the current file; 
    full recovery means re-embedding from SQLite (the system of record has everything needed).
- **Re-ingest cannot corrupt service state:**
  - The catalog owns stock (`available_units`); reservation state (planned `reserved_units` column) is never written 
    by ingest, and effective availability is computed at read time (`available - reserved`).
