.PHONY: up down logs test test-integration test-model test-live lint first-call first-call-local db-reset ingest ingest-local eval eval-local mcp-inspector mcp-inspector-http mcp-tools

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

# usuwa wolumeny: pgvector (dane, tabele checkpointów LangGraph F3 i pauzy zatwierdzeń: ten sam pgdata) oraz cache embeddingów (przeliczenie 47 chunków to sekundy)
db-reset:
	docker compose down -v
	docker compose up -d postgres

ingest:
	docker compose run --rm api kovalo ingest

ingest-local:
	uv run --env-file .env kovalo ingest

# STAGE=e1|e2|e3...; ARGS przekazuje opcje, np. make eval STAGE=e1 ARGS="--backend pgvector"
# F3: make eval STAGE=e3 (odtworzenie D1-D8, zero wywołań LLM-a)
# F4: make eval STAGE=e4 (przeliczenie i odtworzenie przebiegów kursu, zero wywołań LLM-a)
# F5: make eval STAGE=e5 ARGS="--transport http --backend pgvector" (serwer MCP z compose, zero wywołań LLM-a bez --live)
eval:
	docker compose run --rm api kovalo eval $(STAGE) $(ARGS)

# Lokalnie (bez Dockera), np. make eval-local STAGE=e2 ARGS="--with-llm" albo
# make eval-local STAGE=e3 ARGS="--live" (wymaga GOOGLE_API_KEY w .env)
# F5: make eval-local STAGE=e5 ARGS="--live" (agent przez MCP stdio; wymaga GOOGLE_API_KEY)
# F4: make eval-local STAGE=e4 ARGS="--live --ids P1,T1"; pojedyncze pytanie do agenta: uv run kovalo ask "..."
eval-local:
	uv run --env-file .env kovalo eval $(STAGE) $(ARGS)

# MCP Inspector na serwerze stdio (wyszukiwanie w pamięci, bez Postgresa)
mcp-inspector:
	npx @modelcontextprotocol/inspector uv run kovalo-mcp --search memory

# Inspector dla serwera z compose (make up): uruchom `npx @modelcontextprotocol/inspector`,
# wybierz transport "Streamable HTTP" i URL http://localhost:8765/mcp
mcp-inspector-http:
	@echo 'Uruchom: npx @modelcontextprotocol/inspector'
	@echo 'Transport: Streamable HTTP, URL: http://localhost:8765/mcp (wymaga make up)'

# Lista narzędzi serwera (wynik do opisu PR-a)
mcp-tools:
	npx @modelcontextprotocol/inspector --cli uv run kovalo-mcp --search memory --method tools/list
