.PHONY: install lint test ingest serve

install:
	uv sync

lint:
	uv run ruff format . && uv run ruff check . --fix

test:
	uv run python -m pytest tests/ -v

ingest:
	uv run python -m librarian.ingest
