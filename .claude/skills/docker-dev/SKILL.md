---
name: docker-dev
description: Praca z kontenerami Kovalo Decision Copilot - polecenia docker compose i Makefile, make ingest/eval, logi, reset bazy pgvector, ponowne indeksowanie, MCP Inspector i typowe problemy (porty, healthchecki, brak klucza, limity Gemini).
---

# Docker w projekcie

| Cel | Polecenie |
|---|---|
| start całości | `make up` (= `docker compose up -d --build`) |
| pierwsze indeksowanie | `make ingest` (= `docker compose run --rm api kovalo ingest`) |
| logi | `make logs S=api` |
| ewaluacja | `make eval STAGE=e2` (= `docker compose run --rm api kovalo eval e2`) |
| reset bazy | `make db-reset` (usuwa wolumen pgvector i checkpointy: pauzy akceptacji przepadną) |
| MCP Inspector | `make mcp-inspector` (stdio przez `uv run kovalo-mcp`) albo podłącz `http://localhost:8765/mcp` |
| testy w kontenerze | `docker compose run --rm api uv run pytest -m "not live"` |

Usługi i porty: ui 8000, api 8080, mcp-server 8765, postgres 5432.

## Typowe problemy
- `api` nie startuje: sprawdź `docker compose ps`. Postgres musi być `healthy`, a `mcp-server` musi odpowiadać.
- Puste wyniki wyszukiwania: nie uruchomiono `make ingest` albo zmieniono model embeddingów bez ponownego indeksowania (inna kolekcja/wymiar).
- 401/403 od Gemini: brak `GOOGLE_API_KEY` w `.env`. 429: limit darmowego tieru, fallback przełącza z Flash-Lite na Flash (20 zapytań dziennie); jeśli oba limity dzienne wyczerpane, poczekaj do resetu.
- Zmiana `pyproject.toml`: `uv lock` lokalnie, potem `make up` (przebudowa warstwy zależności).
