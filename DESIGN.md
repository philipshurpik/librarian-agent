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
  - The winner is deterministic (longest description, tie -> smallest id) - achieve idempotency
- **Chunking: paragraphs packed to ~1500 chars:** 
  - Descriptions are split on paragraph boundaries and greedily packed; 
  - Each chunk is prefixed with `"{title} by {author}."` so a mid-book chunk stays attributable after retrieval. 
  - Oversized single paragraphs are word-wrapped as a fallback.
- **Idempotency at both layers:**
  - Using `ON CONFLICT DO UPDATE` in SQLite: only catalog columns are updated, so `reserved_units` and the ledger survive re-ingest.
  - Qdrant point ids are `uuid5(book_id:chunk_idx)` - re-ingest overwrites items, never duplicates them.
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
  - The catalog owns stock (`available_units`); reservation state (`reserved_units` column) is never written
    by ingest, and effective availability is computed at read time, clamped at zero
    (`max(available_units - reserved_units, 0)` - a catalog update may pull stock below existing reservations).

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

**New books arriving daily: batch vs streaming:**
- The pipeline is delta-shaped and idempotent, so daily arrivals are a scheduled re-run over the
  delta file - unchanged books cost zero embedding calls. Batch is okay as long it meets the freshness need.
- Streaming buys freshness: move to the event-driven pipeline only when a freshness SLO of minutes forces it.

**Parallelism and backfills:**
- Embedding is parallel: the binding constraint is provider rate limits (TPM/RPM).
- Large backfills (initial load, embedding-model migration) fit OpenAI's Batch API: ~50% cheaper with a 24h SLA.
- A backfill is resumable - a crashed backfill re-runs and continues from what was already indexed.

**100M documents:**
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

## Service: architecture

```
client (demo.py / curl)
      │  POST /chat {message, history}         GET /health
      ▼
   FastAPI ────────── stateless: the client carries history (raw OpenAI messages)
      │
      ▼
  agent loop ──────── OpenAI tool calling, ≤6 rounds, then a forced text answer
      │
      ├─ search_catalog ─────► Qdrant (best chunk per book, grouped)
      ├─ recommend ──────────► Qdrant + SQLite (topic/level filters, availability, weak-match note)
      ├─ check_availability ─► SQLite
      └─ reserve_book ───────► SQLite (atomic check-and-increment)
```

- **Stateless `/chat`:** 
  - The request carries the full conversation, the response returns it extended.
    No sessions, no server-side state - any replica can serve any turn.
- **Tools own their schemas:**
  - Each JSON schema sits directly above its implementation; the loop derives its dispatch whitelist from them. 
  - The `topic` filter enum is read from the catalog at startup, so the model picks from the real topics.
- **Retrieval:**
  - `recommend` flags weak matches (top score < 0.405 - the threshold based on `evals/retrieval.py`: midpoint of the F2-best band); 
  - the system prompt tells the model to present these as closest alternatives.
- **Reservations cannot oversell:** 
  - one atomic `UPDATE ... WHERE reserved_units < available_units`; effective availability computed at read time;

## Service: scaling to ~10k concurrent users

**What the current build already provides:**

- **Stateless service tier:**
  - `/chat` keeps no server-side state, so replicas behind a load balancer scale horizontally; any replica can serve any turn.
- **Bounded turns by construction:**
  - the hard 6-round cap plus an explicit 60s timeout per LLM call limit worst-case latency and LLM spend per request.
- **Reservations stay correct under concurrency:**
  - one atomic check-and-increment UPDATE, safe across replicas.
- **The load shape fits async serving:**
  - a request is 1–3 LLM round-trips (seconds each), so 10k concurrent users are non-blocking waiting, not computing.
  - The whole I/O path is async-native (`AsyncOpenAI`, `AsyncQdrantClient`), parallel tool calls from one model turn run concurrently,
    and the only sync calls left are local SQLite reads (would be swapped for production anyway).
    The hard ceiling is provider token throughput (TPM) - managed by the gateway.

**What we would add at scale (proposed, not in this build):**

- **Database:**
  - SQLite -> Postgres for production deploy. Add Qdrant read replicas.
- **Caching:**
  - Query-embedding cache for hot searches (maybe, depending on search load, if there are frequent ones);
  - For long conversations system prompt with tools could be optimized for KV cache
    - by default for gpt models prompts that are 1024 tokens or longer are cached
    - different settings are available for different model providers
    - self hosted models could provide KV cache reusage even for smaller prompts
- **Conversation history growth:**
  - Depending on model and setup could also benefit from KV cache re-usage
  - Possible to summarize older turns beyond a token budget

## Service: LLM gateway + Observability

- **LiteLLM proxy:**
  Proxy between the service and model providers enables us to centralize monitoring and model management:
  - Add tracking and token accounting per route/model
  - Set up retries with backoff, circuit breaking, model fallback (gpt-5.4-mini -> fallback model) and routing
  - Enables simpler model switching for self hosted open source models (replace vllm url in one place)
- **Langfuse:**
  - Enables observability - latency/token metrics and traces for every call the agent makes
  - LLM as judge evaluations

## Service: failure modes
- **Tool crashes** (vector store down, bad book id):
  - returned to the model as `{"error": ...}` tool content - it apologizes or works around
- **Hallucinated tool calls / arguments:**
  - unknown tool -> error content; 
  - unknown book ids -> error dict; 
  - the topic enum removes the invented-filter-value failure class at the schema level.
- **Runaway loops:**
  - hard 6-round cap, then a final call with tools disabled forces a text answer.
- **Prompt injection via catalog data:**
  - descriptions are untrusted input that reaches the model as tool content. 
    - tool results are structured JSON with 300-char snippets (small surface); 
  - System prompt requires an explicit user request before `reserve_book`;
    - tools are least-privilege (the only mutation is +1 reservation - no deletes). 
  - Possible further work:
    - Agentic eval suite, including focusing on prompt injection.
- **Prompt injection via client history:** 
  - `/chat` rejects `system` role messages in the history from client (422 error) - the server owns system prompt 
  - Forged `tool`/`assistant` content remains possible, could be fixed with server-side sessions

## What we cut 

- **Reranking:**
  - Cross-encoder reranking would fix homonym collisions that dense retrieval cannot separate (surfaced by eval), like 
    ("monolith buildings construction" -> *Monolith to Microservices* at 0.457)
- **Topic enum is a startup snapshot:**
  - compose orders ingest before the api, so first boot sees the full vocabulary; a topic added by a later
    re-ingest becomes filterable only after an api restart; production would refresh it on an ingest signal.
