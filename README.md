# Kovalo Decision Copilot

Asystent AI dla kadry zarządzającej fikcyjnej firmy **Kovalo Distribution**. Odpowiada na podstawie dokumentów firmy i podaje źródła, liczy na danych (OTIF, faktury, kary umowne), stosuje politykę rabatową i prowadzi workflow akceptacji z pauzami na decyzję człowieka. Narzędzia udostępnia przez MCP.

To projekt zaliczeniowy z przedmiotu **Technologie Generatywne (TEG)**. Łączy pięć miniprojektów z [wodecki/TEG_PL_Project_2026](https://github.com/wodecki/TEG_PL_Project_2026) w jedną aplikację.

**Stos:** LangChain 1.x · LangGraph 1.x · MCP (FastMCP) · Gemini Flash-Lite (darmowy tier) · fastembed · Postgres + pgvector · FastAPI · Chainlit · Docker.

![Architektura](docs/architecture/kovalo-architecture-1-containers.png)

## Status

Etap **F0: planowanie i fundament** ([roadmapa](docs/07-roadmap.md)). Kod aplikacji jeszcze nie powstał.

## Dokumentacja

Zacznij od [docs/README.md](docs/README.md): przegląd, wymagania, architektura, stos, domena, ewaluacja, roadmapa i sposób pracy z Claude Code (subagenci, skille, MCP).

## Struktura repozytorium

| Ścieżka | Zawartość |
|---|---|
| `docs/` | dokumentacja projektu, ADR-y, diagramy draw.io, materiały kursu |
| `data/` | korpus 18 dokumentów, dane firmy, zestawy ewaluacyjne (z kursu, tylko do odczytu) |
| `reference/course/` | kod referencyjny kursu: szkielety kroków, testy, helpers (tylko do odczytu) |
| `.claude/` | subagenci i skille Claude Code |
| `.mcp.json` | serwery MCP: draw.io, dokumentacja LangChain, Context7 (budowa) oraz `kovalo` (nasz serwer, stdio) |
| `src/kovalo/` | *(od F0)* kod aplikacji |

## Uruchomienie (docelowo)

```bash
cp .env.example .env      # wpisz GOOGLE_API_KEY z https://aistudio.google.com/apikey
make up                   # docker compose up -d --build
make ingest               # pierwsze indeksowanie korpusu
open http://localhost:8000
```

## Serwer MCP w innym kliencie

Serwer `kovalo` udostępnia 7 narzędzi tylko do odczytu, zasoby i prompt `supplier_review`. Dwa transporty:

- **Claude Code:** wpis `kovalo` w `.mcp.json` (stdio, `uv run --quiet kovalo-mcp --transport stdio --search memory`, wyszukiwanie w pamięci, bez Postgresa i klucza). Sprawdzenie: `claude mcp list`. Uruchamiaj Claude Code z katalogu projektu; wpis nie używa `--directory`.
- **Claude Desktop:** w `claude_desktop_config.json` ta sama komenda z bezwzględną ścieżką, np. `{"command": "uv", "args": ["run", "--quiet", "--directory", "/ścieżka/do/teg", "kovalo-mcp", "--transport", "stdio", "--search", "memory"]}`.
- **MCP Inspector, stdio:** `make mcp-inspector`; lista narzędzi z CLI: `make mcp-tools`.
- **MCP Inspector, HTTP:** po `make up` uruchom `npx @modelcontextprotocol/inspector`, transport „Streamable HTTP”, URL `http://localhost:8765/mcp` (`make mcp-inspector-http` wypisuje instrukcję). Port jest związany z `127.0.0.1`.

---

*Kovalo Distribution oraz wszystkie dokumenty, osoby i liczby w tym projekcie są fikcyjne (materiały kursu TEG).*
