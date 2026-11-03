# 07. Roadmapa

Etapy idą w kolejności miniprojektów, bo każda warstwa korzysta z poprzedniej. Każdy etap kończy się: zielonymi testami w CI, raportem ewaluacji (od E1) i aktualizacją dokumentacji. Kolumna „Agent” wskazuje głównego wykonawcę ([08-dev-workflow.md](08-dev-workflow.md)).

## F0: fundament projektu

| Zadanie | Agent |
|---|---|
| repo, `pyproject.toml` (uv), `ruff`, `mypy`, `pre-commit`, struktura `src/kovalo/` | devops-engineer |
| `config.yaml` (PINNED z kursu + nasze sekcje `llm`, `embeddings`, `db`), `config.py` (pydantic-settings) | architect → ai-engineer |
| `docker-compose.yml` z `postgres` (pgvector), szkielety `api`, `ui`, `mcp-server`; `Makefile` | devops-engineer |
| `llm.py`: fabryka Gemini, rate limiter, fallback między modelami Gemini (kolejność: ADR-0006); test „pierwszego wywołania” | ai-engineer |
| GitHub Actions: lint + testy unit | devops-engineer |

**Gotowe, gdy:** `docker compose up` startuje Postgresa i puste usługi z `/health`; `make first-call` zwraca odpowiedź Gemini.

**Status:** zrealizowane (szczegóły i przegląd: [contracts/f0.md](contracts/f0.md) §1 i §9). Otwarte przed F2: czas odpowiedzi modelu głównego (`llm.timeout_s`, `llm.max_retries`, §9.4) i poprawka ładowania `.env` (A4).

## F1: wiedza (MP1)

loader z metadanymi, tokeny i kalkulator kosztów, chunking 800/120, fastembed + cache, PGVector, wyszukiwanie + własny cosine, polecenie `kovalo ingest` (`make ingest`), ewaluacja E1 (hit@k, porównanie kilku modeli embeddingów → decyzja w ADR-0003).

**Gotowe, gdy:** 18 dokumentów / 8762 tokeny / 47 chunków; raport E1; wybrany model embeddingów.

**Kontrakt i zadania:** [contracts/f1.md](contracts/f1.md); decyzje: [ADR-0007](adr/0007-knowledge-index-f1.md).

**Status (2026-11-02):** kod, testy jednostkowe i architektury, `-m model` i `-m live` zielone; E1 zrobione ([raport](../reports/mp1/2026-11-02-e1.md)), wybrany `intfloat/multilingual-e5-large` ([ADR-0003](adr/0003-local-embeddings.md) przyjęty; MP1 0,9 / 1,0, MP2 0,9 / 1,0, trafia też Q10/M03). Otwarte: weryfikacja w compose (D3: `make ingest` dwa razy, `--backend pgvector`, `pytest -m integration`), bo Postgres w Dockerze nie był dostępny, oraz drobne poprawki z przeglądu ([contracts/f1.md §12](contracts/f1.md)).

## F2: odpowiedzi ze źródłami (MP2)

chunki z kontekstem (nagłówki → rozmiar), odpowiedź w schemacie z cytowaniami, rekord przebiegu, ocena 4-poziomowa bez LLM-a, sędzia LLM, hybryda BM25 (FR-2.5), ewaluacja E2.

**Gotowe, gdy:** raport E2 z porównaniem chunki MP1 / z kontekstem / hybryda.

**Kontrakt i zadania:** [contracts/f2.md](contracts/f2.md); decyzje: [ADR-0008](adr/0008-rag-hybrid-e2.md) (przyjęty), zmiana modelu głównego na `gemini-3.5-flash-lite` w [ADR-0006](adr/0006-config-and-llm-factory.md). Budżet: `kovalo eval e2` bez LLM-a, `--with-llm` do 48 wywołań przy domyślnych wariantach, powtórka z cache 0.

**Status (2026-11-02):** kod, testy jednostkowe i architektury, `-m model`, `-m integration` (compose) i `-m live` zielone; E2 zrobione ([raport](../reports/mp2/2026-11-02-e2.md)): poziomy 1–2 referencji odtwarzają `check.py` kursu co do znaku, poziomy 3–4 spełniają cel „≥ referencja − 1” dla `ctx` i `hybrid`, przeliczenie z cache daje 0 wywołań. `retrieval.mode: vector` (ADR-0008). Komentarze z przeglądu ręcznego w `reports/mp2/review.yaml`, raport przeliczony z cache (zadanie P1 w [contracts/f2.md §14.3](contracts/f2.md)); F2 zmergowane. Znane ograniczenia (pytania PL do dokumentów EN, samoocena sędziego) są w ADR-0008 jako kandydaci na F3–F6.

## F3: workflow decyzyjny (MP3)

silnik polityki `course-v1` (TDD z tabelą przypadków), parser polskich odpowiedzi człowieka, `understand` (structured output), graf z `interrupt()` i wersjonowaniem wniosku, `PostgresSaver`, API `/messages` + `/resume`, **pierwsza wersja UI Chainlit** (czat + przyciski akceptacji), ewaluacja E3 (odtworzenie + na żywo).

**Gotowe, gdy:** dialogi D1–D8 przechodzą w odtworzeniu 100%; scenariusz D2 da się przeklikać w UI.

**Kontrakt i zadania:** [contracts/f3.md](contracts/f3.md) (zadania D1, T1, A0–A9, T2, D2, R1; w tym uwagi N1–N3 z F2); decyzje: [ADR-0009](adr/0009-workflow-f3.md) (workflow, `PostgresSaver`, `CopilotService`, E3, tryb offline modeli) i [ADR-0010](adr/0010-api-ui-f3.md) (API JSON, przyciski Chainlit), oba przyjęte. Budżet: odtworzenie E3 i testy bez LLM-a; `eval e3 --live` ≤ 14 wywołań, powtórka z cache 0, test na żywo 1, przeklikanie D2 w UI 2 na przejście (zamknięcie F3 ≈ 19).

**Status (2026-11-03):** kod, testy jednostkowe i architektury (641), `-m integration` (17/17, compose) i `-m live` (1 wywołanie) zielone; przegląd R1 bez problemów blokujących ([contracts/f3.md §18](contracts/f3.md)). E3 ([raport](../reports/mp3/2026-11-03-e3.md)): odtworzenie 100% na każdym etapie bez LLM-a; na żywo (`gemini-3.5-flash-lite`, 14 wywołań) kierowanie 13/13, pola 10/11, wniosek 8/8, zatwierdzający 11/11, pauzy 13/13, status 13/13, 0 błędów; przeliczenie z cache bez klucza: 0 wywołań. Pauza przetrwała `docker compose restart api`. Otwarte przed merge: ręczne przeklikanie D2 w UI (do potwierdzenia przez użytkownika, 2 wywołania). Znane ograniczenie: przy „marża bez zmian” Flash-Lite nie przenosi marży do pola `Turn` (D2-t2, bez skutku dla wniosku; zmiana promptu PINNED wymaga ADR-u). Zużycie Gemini w F3: 16 wywołań (+2 na przeklikanie D2).

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
