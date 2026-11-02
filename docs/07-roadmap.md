# 07. Roadmapa

Etapy idą w kolejności miniprojektów, bo każda warstwa korzysta z poprzedniej. Każdy etap kończy się: zielonymi testami w CI, raportem ewaluacji (od E1) i aktualizacją dokumentacji. Kolumna „Agent” wskazuje głównego wykonawcę ([08-dev-workflow.md](08-dev-workflow.md)).

## F0: fundament projektu

| Zadanie | Agent |
|---|---|
| repo, `pyproject.toml` (uv), `ruff`, `mypy`, `pre-commit`, struktura `src/kovalo/` | devops-engineer |
| `config.yaml` (PINNED z kursu + nasze sekcje `llm`, `embeddings`, `db`), `config.py` (pydantic-settings) | architect → ai-engineer |
| `docker-compose.yml` z `postgres` (pgvector), szkielety `api`, `ui`, `mcp-server`; `Makefile` | devops-engineer |
| `llm.py`: fabryka Gemini, rate limiter, fallback Flash → Flash-Lite; test „pierwszego wywołania” | ai-engineer |
| GitHub Actions: lint + testy unit | devops-engineer |

**Gotowe, gdy:** `docker compose up` startuje Postgresa i puste usługi z `/health`; `make first-call` zwraca odpowiedź Gemini.

**Status:** zrealizowane (szczegóły i przegląd: [contracts/f0.md](contracts/f0.md) §1 i §9). Otwarte przed F2: czas odpowiedzi modelu głównego (`llm.timeout_s`, `llm.max_retries`, §9.4) i poprawka ładowania `.env` (A4).

## F1: wiedza (MP1)

loader z metadanymi, tokeny i kalkulator kosztów, chunking 800/120, fastembed + cache, PGVector, wyszukiwanie + własny cosine, polecenie `kovalo ingest` (`make ingest`), ewaluacja E1 (hit@k, porównanie kilku modeli embeddingów → decyzja w ADR-0003).

**Gotowe, gdy:** 18 dokumentów / 8762 tokeny / 47 chunków; raport E1; wybrany model embeddingów.

## F2: odpowiedzi ze źródłami (MP2)

chunki z kontekstem (nagłówki → rozmiar), odpowiedź w schemacie z cytowaniami, rekord przebiegu, ocena 4-poziomowa bez LLM-a, sędzia LLM, hybryda BM25 (FR-2.5), ewaluacja E2.

**Gotowe, gdy:** raport E2 z porównaniem chunki MP1 / z kontekstem / hybryda.

## F3: workflow decyzyjny (MP3)

silnik polityki `course-v1` (TDD z tabelą przypadków), parser polskich odpowiedzi człowieka, `understand` (structured output), graf z `interrupt()` i wersjonowaniem wniosku, `PostgresSaver`, API `/messages` + `/resume`, **pierwsza wersja UI Chainlit** (czat + przyciski akceptacji), ewaluacja E3 (odtworzenie + na żywo).

**Gotowe, gdy:** dialogi D1–D8 przechodzą w odtworzeniu 100%; scenariusz D2 da się przeklikać w UI.

## F4: agent z narzędziami (MP4)

repozytorium danych firmy, kalkulator kary (Decimal/grosze), 6 narzędzi + `check_termination_condition`, agent `create_agent` z limitami, trace w Postgresie, ocena trace (5 testów), widok trace w UI (`cl.Step`), opcjonalnie supervisor, ewaluacja E4.

**Gotowe, gdy:** P1–P3 dają dokładnie 729 600 / 362 400 / 0 groszy; raport E4; T1 poprawne.

## F5: MCP i końcowy asystent (MP5)

serwer FastMCP (stdio + streamable HTTP) jako kontener, klient MCP z limitami i 4 klasami błędów, podmiana węzła `answer_with_tools`, prompt `assistant_understand_system`, ewaluacja E5, test w MCP Inspector i Claude Desktop/Claude Code.

**Gotowe, gdy:** raport E5; `docker compose up` daje pełnego asystenta w przeglądarce; serwer `kovalo` dodany do `.mcp.json` działa w Claude Code.

## F6: dopracowanie

licznik kosztów w UI, dashboard ewaluacji, README z demo (GIF), porządki w dokumentacji, odświeżony diagram architektury, prezentacja projektu.

## Kamienie milowe a testy cząstkowe kursu

Quizy na Edux sprawdzają teorię z `CONCEPTS.md` i zastosowanie na Kovalo. Przed każdym quizem warto mieć skończony odpowiadający mu etap (F1 → quiz 1 … F5 → quiz 5) i przeczytany [course/mpN_CONCEPTS.md](course/).
