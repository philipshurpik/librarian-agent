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

## Ingestion & embeddings: migration 

**Model migration / re-embedding cost:**
- Never mutate the live index. Implemented as one collection per model: the name is derived from 
  `embedding_model` (`books__openai-3-small`), so a model switch lands in a fresh collection with the 
  right dimensions - full embed there, old collection kept for existing consumers while they upgrade (and for rollback). 
  - Query model and index always match by construction - a service can never search vectors built by a different model.
- Production migration:
  - Backfill the new embeddings for existing books as supervised, re-runnable job, ingest dual-writes both collections. 
    The promotion gate is mechanical:
    - equal exact point counts in old and new collections - chunking is model-independent, so points match 1:1 
    - ledger's `sum(chunk_count)` arbitrates any mismatch
    - plus recall spot-checks on the new collection.
    Consumers then move one config change at a time, drop the old collection when the last consumer moves.
- Cost sanity:
  - 1M docs * ~300 tokens -> ~$6 with text-embedding-3-small;
  - 100M -> ~$600 - embedding model throughput limit and database are the real constraints, not dollars.

## Ingestion & embeddings: scaling

**100M documents.**
- Change detection moves to events: source-DB change events -> queue (Kafka/PubSub) -> idempotent embed consumers; 
  - Ingestion becomes an always-on pipeline fully decoupled from serving.
- Index size (assumption 1.5 chunks on avg for book library descriptions): 
  - ~150M points * 1536d * 4B -> ~920GB of raw vectors (by default)  + (HNSW on top - both on disk and RAM)
  - Possible levers:
    - Set datatype=Datatype.FLOAT16 (near-lossless, if embedding model run in half precision) -> 460GB
    - Using Matryoshka embeddings (1536d -> 768d) can reduce storage 2x even more 460 GB -> 230 GB (on disk)
    - Qdrant 1.18 ships [TurboQuant](https://qdrant.tech/articles/turboquant-quantization/). 
      Realistically we can use TQ 4-bit with minimal recall change and get ~290GB on disk with ~60 GB vectors on RAM for faster retrieval 
  - Separate evals for our data should be set up to evaluate which technique, or their combination works for our data and tasks

