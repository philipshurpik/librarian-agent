# Librarian Agent

A chat service where an AI librarian helps you find tech books, get recommendations, check
availability, and reserve a copy. 
FastAPI + OpenAI tool-calling loop, Qdrant for semantic search, SQLite as the system of record. 
Architecture and reasoning live in [DESIGN.md](DESIGN.md).

## Quickstart (Docker)

```bash
cp .env.example .env        # put your OpenAI API key in
make up                     # build + start api and qdrant (docker compose up -d --build)
make docker-ingest          # populate the catalog (docker compose run --rm api python -m librarian.ingest)
make chat                   # chat from your terminal (needs uv locally — or use the curl examples below)
```

## Talking to the service

`make chat` is a tiny terminal client (`demo.py`) that keeps the conversation going and shows
which tools the agent called each turn. Raw HTTP works too:

```bash
curl -s localhost:8000/health

curl -s localhost:8000/chat -H 'Content-Type: application/json' -d '{"message": "Any good book about Rust for beginners?"}'
```

The response is `{"reply": ..., "history": [...]}`. 
The service is stateless — for a follow-up turn, send `history` back together with the new `message`:

```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message": "Reserve it for me, please", "history": [ ...history from the previous response... ]}'
```

## Local development

Requires [uv](https://docs.astral.sh/uv/) and a running Qdrant (`make up` provides one on `localhost:6333`).

```bash
make install    # uv sync
make ingest     # run the ingestion pipeline locally
make serve      # uvicorn with reload on localhost:8000
make test       # pytest
make lint       # ruff format + check
make eval       # retrieval eval: recall@3 + weak-match threshold calibration
```

## The dataset

`data/catalog.json` — 80 hand-curated records of real tech books, generated with LLM assistance
and seeded with deliberate real-world grime the pipeline must handle: 5 records with missing
descriptions, a duplicate id, two books duplicated under different ids, HTML fragments, and
description lengths from 0 to ~3k chars. After cleaning and dedupe, 77 books are indexed.

## Project layout

```
src/librarian/
  ingest.py         # one-command pipeline: load → clean → dedupe → chunk → embed → index
  api.py            # FastAPI: POST /chat, GET /health
  agent/            # tool-calling loop, tool implementations + schemas, system prompt
  storage/          # SQLite (books table: catalog, reservation counter, index ledger), Qdrant, OpenAI embeddings
evals/retrieval.py  # retrieval quality: golden queries + off-topic threshold calibration
tests/              # tests: pipeline, storage, tools, agent loop, api
```
