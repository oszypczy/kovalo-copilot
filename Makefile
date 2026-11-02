.PHONY: up down logs test test-integration test-model test-live lint first-call first-call-local db-reset ingest ingest-local eval eval-local mcp-inspector

S ?=
STAGE ?= e1
ARGS ?=

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f $(S)

test:
	uv run pytest

lint:
	uv run ruff check --fix
	uv run ruff format
	uv run mypy

first-call:
	docker compose run --rm api kovalo first-call

first-call-local:
	uv run --env-file .env kovalo first-call

test-integration:
	uv run pytest -m integration

test-model:
	uv run pytest -m model

# Testy z prawdziwym LLM-em (darmowy tier): wybierz jeden plik, np. make test-live T=tests/live/test_e1_live.py
T ?= tests/live
test-live:
	uv run --env-file .env pytest -m live $(T)

# usuwa wolumeny: pgvector (dane i checkpointy) oraz cache embeddingów (przeliczenie 47 chunków to sekundy)
db-reset:
	docker compose down -v
	docker compose up -d postgres

ingest:
	docker compose run --rm api kovalo ingest

ingest-local:
	uv run --env-file .env kovalo ingest

# STAGE=e1|e2|...; ARGS przekazuje opcje, np. make eval STAGE=e1 ARGS="--backend pgvector"
eval:
	docker compose run --rm api kovalo eval $(STAGE) $(ARGS)

# Lokalnie (bez Dockera), np. make eval-local STAGE=e2 ARGS="--with-llm" (wymaga GOOGLE_API_KEY w .env)
eval-local:
	uv run --env-file .env kovalo eval $(STAGE) $(ARGS)

mcp-inspector:
	npx @modelcontextprotocol/inspector uv run kovalo-mcp
