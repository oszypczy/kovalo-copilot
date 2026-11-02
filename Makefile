.PHONY: up down logs test lint first-call first-call-local db-reset ingest eval mcp-inspector

S ?=
STAGE ?= e2

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f $(S)

test:
	uv run pytest -m "not live and not integration"

lint:
	uv run ruff check --fix
	uv run ruff format
	uv run mypy

first-call:
	docker compose run --rm api kovalo first-call

first-call-local:
	uv run --env-file .env kovalo first-call

# usuwa wolumen pgvector (dane i checkpointy)
db-reset:
	docker compose down -v
	docker compose up -d postgres

# F1: kovalo ingest powstaje w etapie F1
ingest:
	docker compose run --rm api kovalo ingest

# F1+: kovalo eval powstaje w kolejnych etapach (STAGE=e1|e2|...)
eval:
	docker compose run --rm api kovalo eval $(STAGE)

mcp-inspector:
	npx @modelcontextprotocol/inspector uv run kovalo-mcp
