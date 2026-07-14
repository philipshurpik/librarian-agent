.PHONY: install lint test ingest eval serve chat

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
