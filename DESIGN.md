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
  - A record failing validation is skipped with a warning - one corrupt entry must not block the batch;
    production would route it to a dead-letter file for inspection.
- **Dedupe: drop, don't merge:**
  - Two passes: remove duplicate ids, then deduplicate normalized (title, author) under different ids.
  - Assumption: a content duplicate is a double entry, not extra stock.
  - The winner is deterministic (longest description, tie -> smallest id) - achieve idempotency
- **Chunking: paragraphs packed to ~1500 chars:**
  - Descriptions are split on paragraph boundaries and greedily packed;
  - Each chunk is prefixed with `"{title} by {author}."` so a mid-book chunk stays attributable after retrieval.
  - Oversized single paragraphs are word-wrapped as a fallback.
- **Idempotency at both layers:**
  - Using `ON CONFLICT DO UPDATE` in SQLite: only catalog columns are updated, so `reserved_units` and the
    ledger survive re-ingest.
  - Qdrant point ids are `uuid5(book_id:chunk_idx)` - re-ingest overwrites items, never duplicates them.
- **Incremental embedding via a ledger (`content_hash`, `chunk_count`):**
  - `content_hash = sha256(indexed metadata + chunk texts)` is stored in SQLite next to the data it describes.
    Metadata-only changes to topic, level, or year therefore refresh the Qdrant payload too.
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
- **Current implementation:**
  - The collection name is derived from `embedding_model` (`books__openai-3-small`), so different models cannot
    mix incompatible vectors or dimensions. A first switch to a new model creates and fully indexes a fresh collection.
  - The ledger tracks only the active collection. Once the model changes, the old collection stops receiving updates;
    switching back after catalog changes would serve stale content.
  - The current build provides model isolation, not safe rollback.
- **Production migration:**
  - Key the ledger by `(collection, book_id)` so each model has independent `content_hash` and `chunk_count` state.
  - Backfill the new collection as a supervised, re-runnable job while ingestion dual-writes both collections.
    The promotion gate is mechanical:
    - equal exact point counts in old and new collections - chunking is model-independent, so points match 1:1;
    - each collection ledger's `sum(chunk_count)` arbitrates any mismatch;
    - plus recall spot-checks on the new collection.
  - Promote the query embedding model and a Qdrant alias as one serving configuration change, keep dual-writing
    during a bounded rollback window, then stop writes and drop the old collection after the last consumer moves.
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
      Realistically we can use TQ 4-bit with minimal recall change and get ~290GB on disk with ~60 GB vectors
      on RAM for faster retrieval
  - Separate evals for our data should be set up to evaluate which technique, or their combination works for
    our data and tasks
- Query serving at this scale:
  - shard the collection across nodes (Qdrant distributed mode); replicas for QPS and failover;
  - payload indexes on the filter fields (topic, level) so filtered search does not scan;

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
    The agent orchestration keeps no session state, but SQLite still prevents multi-host horizontal scaling.
- **Tools own their schemas:**
  - Each JSON schema sits directly above its implementation; the loop derives its dispatch whitelist from them.
  - The `topic` filter enum is read from the catalog at startup, so the model picks from the real topics.
- **Retrieval:**
  - `search_catalog` and `recommend` flag weak matches (top score < 0.405 - the threshold based on
    `evals/retrieval.py`: midpoint of the F2-best band);
  - the system prompt tells the model to present these as closest alternatives.
- **Reservations cannot oversell:**
  - one atomic `UPDATE ... WHERE reserved_units < available_units`; effective availability computed at read time.
    The guarantee applies to concurrent requests sharing the same SQLite database.
  - Assumption: a reservation is an anonymous counter - no identity, no cancel, no expiry.
    The multi-user shape is in "Out of scope" below.

## Service: scaling to ~10k concurrent users

**What the current build already provides:**

- **Stateless agent orchestration:**
  - `/chat` keeps no server-side session state; the client carries bounded conversation history.
  - SQLite remains local state, so the current build is intentionally single-host.
- **Bounded turns by construction:**
  - the hard 6-round cap plus an explicit 60s timeout per LLM call bound individual calls and LLM spend.
- **Reservations stay correct under concurrency:**
  - one atomic check-and-increment UPDATE is safe for concurrent requests against the same SQLite database.
- **The load shape fits async serving:**
  - a request is 1–3 LLM round-trips (seconds each), so 10k concurrent users are non-blocking waiting, not computing.
  - The slow network path is async-native (`AsyncOpenAI`, `AsyncQdrantClient`), and parallel tool calls from one model
    turn run concurrently. SQLite reads and reservation writes remain synchronous; the Postgres move replaces them
    with an async driver before horizontal scale. The hard ceiling is provider token throughput (TPM) - managed by
    the gateway.

**What we would add at scale (proposed, not in this build):**

- **Database:**
  - SQLite -> Postgres before adding API replicas; the same atomic reservation update runs against one shared database.
  - API replicas behind a load balancer; Qdrant read replicas for retrieval QPS and failover.
- **Backpressure:**
  - Bound in-flight agent turns per replica with a semaphore; reject excess work with `429` and `Retry-After`
    instead of building an unbounded in-memory queue.
  - Apply one overall request deadline plus shorter tool deadlines, and cancel outstanding tool calls when it expires.
  - Feed gateway/provider `429` responses into admission control; retry only read-only calls with bounded
    jittered backoff.
- **Caching:**
  - Query-embedding cache for hot searches (maybe, depending on search load, if there are frequent ones);
  - For long conversations system prompt with tools could be optimized for KV cache
    - by default for gpt models prompts that are 1024 tokens or longer are cached
    - different settings are available for different model providers
    - self hosted models could provide KV cache reusage even for smaller prompts
- **Conversation history growth:**
  - Already bounded: `/chat` rejects oversized input (message length, history messages/chars) with 422 -
    the client carries the history, so the server must cap what it accepts and pays to send to the LLM.
  - Depending on model and setup could also benefit from KV cache re-usage
  - Possible to summarize older turns beyond a token budget

## Service: LLM gateway + Observability

- **LiteLLM proxy:**
  - Sits between the API and model providers; the application keeps the OpenAI protocol and changes only its base URL.
  - Enforces per-user/team rate limits, token quotas and spend budgets before requests reach a provider.
  - Tracks tokens, latency and cost per route/model; central policy can stop or downgrade expensive traffic.
  - Caches identical embedding requests and explicitly cacheable deterministic LLM calls; mutation-bearing agent
    turns are never cached.
  - Owns bounded retries with backoff, circuit breaking, model fallback (gpt-5.4-mini -> fallback model) and routing.
  - Enables simpler switching to self-hosted models by changing the vLLM endpoint in one place.
- **Langfuse:**
  - Enables observability - latency/token metrics and traces for every call the agent makes.
  - LLM-as-judge evaluations.

## Service: failure modes
- **Tool crashes** (vector store down, bad book id):
  - returned to the model as `{"error": ...}` tool content - it apologizes or works around
- **Hallucinated tool calls / arguments:**
  - unknown tool -> error content;
  - unknown book ids -> error dict;
  - the topic enum removes the invented-filter-value failure class at the schema level.
- **Runaway loops:**
  - hard 6-round cap, then a final call with tools disabled forces a text answer.
- **Provider outage / timeouts:**
  - LLM calls carry a 60s timeout; a provider failure surfaces as a clean 503, not a stack trace.
  - Retries with backoff and model fallback belong to the llm gateway;
  - Only `reserve_book` mutates state and is never auto-retried - the model reports failure and user can re-ask.
- **Prompt injection via catalog data:**
  - descriptions are untrusted input that reaches the model as tool content.
  - Mitigations reduce exposure, they do not prevent injection - the model still reads attacker text:
    - the system prompt explicitly marks catalog text as data, not instructions;
    - tool results are structured JSON with 300-char snippets (small surface);
    - system prompt requires an explicit user request before `reserve_book`;
    - tools are least-privilege (the only mutation is +1 reservation - no deletes).
  - Possible further work:
    - injection eval suite: adversarial descriptions -> assert no unauthorized tool call / rule break;
    - confirm-before-reserve turn in the UX, idempotency key on the mutation.
- **Prompt injection via client history:**
  - `/chat` rejects `system` role messages in the history from client (422 error) - the server owns system prompt
  - Forged `tool`/`assistant` content remains possible, could be fixed with server-side sessions

## Out of scope per the task: how each would land

- **Auth:**
  - A simple login (session cookie or JWT) in front of `/chat` - the app just needs to know who is asking.
  - Mainly so a reservation belongs to a person (below); also lets us cap per-user usage.
- **Multi-user session management:**
  - Move history server-side: a session store (Redis/Postgres) keyed by session id;
  - the client sends `session_id` + `message` instead of carrying the full history.
  - The service tier stays stateless - state moves to a shared tier, any replica still serves any turn.
  - Closes the forged `tool`/`assistant` history channel (failure modes above).
- **Reservations with identity:**
  - Today's anonymous counter becomes a `reservations` table (book_id, user_id, expires_at):
    reserve/cancel are inserts/deletes, expiry frees the copy - needs auth above.
- **Real vector-DB infrastructure:**
  - index sizing, replication and quantization levers are in "Ingestion & embeddings: scaling".
- **CI/CD:**
  - deliberately omitted; the gates exist as Make targets (`lint`, `test`, `eval` as a retrieval-quality gate).

## What we cut

- **Reranking:**
  - Cross-encoder reranking would fix homonym collisions that dense retrieval cannot separate (surfaced by eval), like
    ("monolith buildings construction" -> *Monolith to Microservices* at 0.457)
- **Topic enum is a startup snapshot:**
  - compose orders ingest before the api, so first boot sees the full vocabulary; a topic added by a later
    re-ingest becomes filterable only after an api restart; production would refresh it on an ingest signal.
