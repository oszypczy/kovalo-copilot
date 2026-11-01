# Kovalo Decision Copilot: instrukcje dla Claude Code

Projekt na przedmiot TEG: pięć miniprojektów kursu połączonych w jedną aplikację dockerową (LangChain + LangGraph + MCP, darmowy LLM Gemini).

## Gdzie czego szukać
- Wizja, mapowanie miniprojektów → moduły: `docs/01-overview.md`
- Wymagania FR/NFR z kryteriami: `docs/02-requirements.md`
- Architektura i zasady zależności modułów: `docs/03-architecture.md`
- Domena (polityka course-v1, kary OTIF, narzędzia): `docs/05-domain.md`
- Ewaluacja i cele: `docs/06-evaluation.md`; roadmapa: `docs/07-roadmap.md`
- Decyzje: `docs/adr/`. Zmiana technologii, kontraktu, wartości PINNED albo promptu kursu wymaga nowego ADR-u.
- Specyfikacja kursu: `docs/course/` (README/CONCEPTS) i `reference/course/` (szkielety kroków, testy, helpers). Tylko do odczytu.

## Subagenci (`.claude/agents/`)
architect (kontrakty, ADR, diagramy, review architektury) → test-engineer (testy najpierw, ewaluacje) → ai-engineer (implementacja LLM/LangGraph/MCP) → devops-engineer (Docker, CI). Szczegóły: `docs/08-dev-workflow.md`.

## Zasady
- Python 3.12 + uv. Kod w `src/kovalo/`, testy w `tests/`. Lint: `uv run ruff check --fix && uv run ruff format`.
- Testy domyślnie bez LLM-a (`pytest -m "not live"`); prawdziwe wywołania tylko z markerem `live`.
- Modele LLM tylko przez `kovalo.llm`; żadnych kluczy ani nazw modeli na sztywno.
- Kwoty w całkowitych groszach / Decimal. Reguły biznesowe w kodzie deterministycznym, nie w promptach.
- API LangChain/LangGraph sprawdzaj w MCP `langchain-docs`, inne biblioteki w `context7`.
- Dokumentacja i UI po polsku; identyfikatory, klucze, statusy po angielsku.
- `data/` i `reference/` są tylko do odczytu.
