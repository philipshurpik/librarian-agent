.PHONY: install lint test ingest eval serve chat up down docker-ingest

install:
	uv sync

lint:
	uv run ruff format . && uv run ruff check . --fix

test:
	uv run python -m pytest tests/ -v

ingest:
	uv run python -m librarian.ingest

eval:
	uv run python evals/retrieval.py

serve:
	uv run uvicorn librarian.api:app --reload

chat:
	uv run python demo.py

up:
	docker compose up -d --build

down:
	docker compose down

docker-ingest:
	docker compose run --rm api python -m librarian.ingest
