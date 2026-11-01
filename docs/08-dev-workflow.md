# 08. Jak budujemy projekt z Claude Code

Projekt budujemy z pomocą Claude Code. Trzy mechanizmy:
- **subagenci** (`.claude/agents/*.md`): specjaliści z własnym promptem, narzędziami i zakresem plików,
- **skille** (`.claude/skills/*/SKILL.md`): wiedza i procedury, które agent ładuje, gdy zadanie ich dotyczy,
- **serwery MCP** (`.mcp.json`): dostęp do zewnętrznych narzędzi i dokumentacji.

## Subagenci

| Agent | Kiedy go używamy | Czego NIE robi | Model |
|---|---|---|---|
| **architect** | projekt modułów i kontraktów, ADR-y, diagramy draw.io, przegląd zgodności kodu z architekturą (zasady zależności z [03](03-architecture.md)), rozbicie etapu na zadania | nie pisze kodu produkcyjnego | opus |
| **ai-engineer** | implementacja wszystkiego, co dotyka LLM-a: `llm.py`, RAG, `understand`, graf LangGraph, agent, narzędzia, serwer i klient MCP, prompty | nie zmienia kontraktów bez ADR-u; nie pisze Dockerfile | opus |
| **test-engineer** | testy przed implementacją (TDD) na podstawie kontraktów i testów kursu, atrapy modeli, fixtures, programy ewaluacji, analiza raportów i rozbieżności | nie „naprawia” testów pod kod; nie wywołuje płatnych API | sonnet |
| **devops-engineer** | Dockerfile (multi-stage, uv), docker-compose i profile, Makefile, GitHub Actions, `.env.example`, healthchecki | nie dotyka logiki w `src/kovalo/` poza `api/` (wiring) | sonnet |

Typowy cykl jednej funkcji (np. FR-3.1, silnik polityki):

1. **architect**: kontrakt (sygnatury, typy wyników), miejsce w module, ewentualny ADR.
2. **test-engineer**: testy z tabeli przypadków (czerwone).
3. **ai-engineer**: implementacja aż do zielonych testów.
4. **test-engineer**: ewaluacja na żywo (jeśli dotyczy) i raport.
5. **architect**: przegląd zależności i aktualizacja docs/diagramu.

Wbudowane polecenia `/code-review` i `/simplify` uzupełniają ten cykl przed commitem.

## Skille

| Skill | Zawartość | Kto korzysta |
|---|---|---|
| `kovalo-domain` | polityka `course-v1` z przypadkami, narzędzia i statusy, dane firmy, zapis liczb po polsku, wskazania do korpusu i słownika | wszyscy |
| `langgraph-patterns` | konwencje projektu: stan i reducery, `interrupt`/`Command(resume)`, checkpointer, structured output, `create_agent`, wstrzykiwanie narzędzi, testy z atrapami modeli | ai-engineer, test-engineer |
| `course-step` | jak przenieść krok kursu do aplikacji: przeczytać docstring kroku w `reference/course`, testy kursu, README; zachować wartości PINNED; dopisać wymaganie FR | wszyscy |
| `run-eval` | jak uruchomić ewaluację E1–E5, oszczędzać limit darmowego tieru, czytać raport i porównać z referencją | test-engineer |
| `docker-dev` | polecenia compose, profile, logi, reset bazy, typowe problemy | devops-engineer |
| `drawio` | oficjalny skill draw.io (Apache-2.0): generowanie `.drawio` i eksport PNG/SVG przez draw.io Desktop CLI | architect |

## Serwery MCP (`.mcp.json`)

| Serwer | Po co | Uruchomienie |
|---|---|---|
| `drawio` | otwiera diagramy (XML / CSV / Mermaid) w edytorze draw.io w przeglądarce | `npx -y @drawio/mcp` (oficjalny, jgraph) |
| `langchain-docs` | aktualna dokumentacja LangChain / LangGraph / LangSmith (API 1.x zmienia się szybko) | HTTP: `https://docs.langchain.com/mcp` |
| `context7` | dokumentacja innych bibliotek w konkretnej wersji (FastAPI, Chainlit, pgvector, fastembed, MCP SDK) | `npx -y @upstash/context7-mcp` |
| `kovalo` *(od F5)* | nasz własny serwer: testowanie narzędzi z Claude Code jak z innego klienta MCP | `uv run kovalo-mcp --transport stdio` |

Plik `.drawio` w repozytorium jest źródłem prawdy. Eksporty PNG robimy przez draw.io Desktop CLI (`/Applications/draw.io.app/Contents/MacOS/draw.io -x -f png -e ...`), więc PNG zawiera osadzony XML i da się go dalej edytować.

## Zasady pracy

- Branch na etap/funkcję (`f3-policy-engine`), PR z checklistą: testy, raport ewaluacji (jeśli dotyczy), docs.
- Wartości `PINNED` i prompty kursu zmieniamy tylko przez ADR.
- Testy domyślnie bez LLM-a; testy `@pytest.mark.live` uruchamiamy ręcznie.
- Każda zmiana promptu = nowa wersja + ewaluacja przed i po.
