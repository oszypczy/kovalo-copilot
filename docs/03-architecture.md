# 03. Architektura

Diagramy: [`architecture/kovalo-architecture.drawio`](architecture/kovalo-architecture.drawio). Plik ma trzy strony: kontenery, graf copilota, indeksowanie i wyszukiwanie. Eksporty PNG (z osadzonym XML-em, edytowalne w draw.io) leżą obok pliku.

![Kontenery](architecture/kovalo-architecture-1-containers.png)

## 1. Kontenery (docker-compose)

| Usługa | Obraz / technologia | Port | Odpowiedzialność |
|---|---|---|---|
| `ui` | Chainlit | 8000 | czat, kroki agenta, przyciski akceptacji, źródła, licznik kosztów (linia tury i suma rozmowy z gotowych kwot `usage.cost`; UI nie zna cen); rozmawia tylko z `api` |
| `api` | FastAPI + LangGraph | 8080 (na hoście `API_PORT`) | graf copilota (`understand`, routing, workflow akceptacji, agent), checkpointy, koszty, trace |
| `mcp-server` | FastMCP (`mcp` SDK 1.30), streamable HTTP (`/mcp`) | 8765 | 7 narzędzi tylko do odczytu (6 kursu + `check_termination_condition`), szablon zasobu `kovalo://documents/{doc_id}`, zasób `kovalo://policy/course-v1`, prompt `supplier_review`; `pgvector` (`kovalo_ctx`) i `./data` tylko do odczytu; bez klucza LLM-a ([ADR-0013](adr/0013-mcp-server-client-f5.md)) |
| `postgres` | `pgvector/pgvector:pg16` | 5432 | indeks wektorowy (`langchain-postgres`, `PGVectorStore`: tabela na kolekcję), checkpointy LangGraph (`langgraph-checkpoint-postgres`), w nich trace agenta (koperta `agent_trace` w stanie wątku, [ADR-0011](adr/0011-agent-tools-f4.md)) |

Indeksowanie, ewaluacja i dashboard to **nie osobne usługi**, tylko polecenia CLI (`kovalo ingest`, `kovalo eval`) uruchamiane jednorazowo w obrazie `api`: `make ingest` = `docker compose run --rm api kovalo ingest` (korpus → chunki MP1 i chunki z kontekstem → embeddingi → podmiana tabel `kovalo_mp1` i `kovalo_ctx`), `make eval STAGE=e2` = `docker compose run --rm api kovalo eval e2` (raport w `reports/`, montowanym jako wolumen). Dashboard (`make dashboard` = `uv run kovalo eval dashboard`, w obrazie `make eval STAGE=dashboard`) czyta raporty i `reports/dashboard.yaml` i pisze `reports/dashboard.md` i `.json`, bez LLM-a i bez sieci.

Zewnętrznie: **Gemini API** (domyślny LLM, darmowy klucz z Google AI Studio), **LangSmith** (tracing każdej tury: węzły grafu, wywołania LLM-a, narzędzia; darmowy plan dla jednej osoby).

Embeddingi liczymy **lokalnie** (fastembed, model ONNX wbudowany w obraz), więc ani indeksowanie, ani `mcp-server` nie potrzebują klucza.

```
Chainlit (ui) ──HTTP (JSON; SSE: …/stream)──> FastAPI (api) ──LangGraph──> Gemini API
                                 │   └──MCP streamable HTTP──> mcp-server ──> pgvector + data/company + data/corpus
Claude Code / Inspector / Claude Desktop ──stdio (własny proces kovalo-mcp --search memory)
                                 └──checkpointy / trace──> postgres
make ingest ──> data/corpus ──> chunki ──> fastembed ──> pgvector
```

## 2. Moduły Pythona (`src/kovalo/`)

```
src/kovalo/
  config.py          ustawienia: config.yaml (PINNED z kursu) + .env (pydantic-settings), get_settings()
  llm.py             fabryka modeli: init_chat_model(provider), rate limiter, fallback Flash-Lite → Flash (ADR-0006); fabryka embeddingów (F1); tryb offline z cache (F3)
  text_pl.py         liczby po polsku i odpowiedzi człowieka w pauzach (port helpers/text_pl.py kursu)
  health.py          wspólny format /health i sprawdzenia (Postgres, HTTP, klucz LLM-a)
  cli.py             polecenie `kovalo`: first-call, ingest, eval
  knowledge/         MP1/MP2: loader, chunking (zwykły i z kontekstem), embeddingi, vector store, BM25 (rank-bm25), Retriever (wektorowy | hybryda RRF)
  cost/              MP1: tokenizer o200k_base, scenariusz kosztów, licznik kosztów per tura
  rag/               MP2: schematy Answer/Verdict, rekord przebiegu, normalizacja odwołań, answer_question (model przez parametr)
  policy/            MP3: DiscountRequest, Decision, required_approvers (course-v1, Decimal; czysty kod, zero LLM-a)
  quantities.py      liczby, jednostki i frazy w odpowiedziach (port helpers/quantities.py; ocena E4)
  company/           MP4: dane firmy, rekordy wyników, odczyty, kalkulator kary (Decimal, grosze), warunek rozwiązania umowy
  tools/             MP4: wyszukiwanie fragmentów, specyfikacje 7 narzędzi (wspólne dla LangChaina i FastMCP), adapter StructuredTool
  agents/            MP4: trace, budżet i run_question (port kursu), create_agent z fallbackiem i licznikiem wywołań, ścieżka dowodowa
  workflow/          MP3/MP5: Turn, understand_turn, update_request, stan, węzły, graf, checkpointer, CopilotService, fabryka
  mcp_server/        MP5: FastMCP „kovalo” z TOOL_SPECS (nakładki async w wątku), zasoby, prompt, /health; stdio + streamable-http
  mcp_client/        MP5: kontrakt rekordów (check_record), McpConnection (stdio / HTTP / w pamięci, limity, 4 warstwy błędów), McpToolSource (jedno połączenie, ponowne łączenie)
  eval/              MP1–MP5: retrieval (E1), rag (E2: 4 poziomy + sędzia), cache LLM-a na dysku, workflow (E3: odtworzenie + na żywo), trace (E4: przeliczenie kursu, odtworzenie, na żywo; 5 testów; agent przez MCP z `--transport`), assistant (E5: zgodność MCP, odtworzenie 25 tur przez MCP, na żywo)
  api/               FastAPI: router /threads nad CopilotService, schematy wejścia i błędów, dostawca serwisu (lifespan)
  ui/                Chainlit (app.py) zamontowany w FastAPI (server.py, /health); klient HTTP do api, renderowanie odpowiedzi, pauz i źródeł
```

Punkty wejścia: `kovalo` (CLI), `kovalo-api`, `kovalo-ui`, `kovalo-mcp`. Kontrakty modułów fundamentu (`config`, `llm`, `health`, `cli`) opisuje [contracts/f0.md](contracts/f0.md), modułów wiedzy i kosztów (`knowledge`, `cost`, `eval` E1) [contracts/f1.md](contracts/f1.md), odpowiedzi ze źródłami i E2 (`rag`, `knowledge` z hybrydą, `eval/rag`) [contracts/f2.md](contracts/f2.md), workflow, API, UI i E3 (`policy`, `workflow`, `api`, `ui`, `eval/workflow`) [contracts/f3.md](contracts/f3.md), agent, narzędzia, SSE i E4 (`quantities`, `company`, `tools`, `agents`, `eval/trace`) [contracts/f4.md](contracts/f4.md), a MCP i E5 (`mcp_server`, `mcp_client`, `eval/assistant`) [contracts/f5.md](contracts/f5.md).

Zasady zależności (sprawdzane w review):
- `policy/`, `company/`, `cost/` nie importują LangChaina ani niczego z LLM-em: to deterministyczny rdzeń.
- `tools/` to cienkie nakładki na `knowledge/`, `company/`, `policy/`. `mcp_server/` to cienkie nakładki na `tools/`.
- `workflow/` nie zna transportu narzędzi: dostaje listę narzędzi (bezpośrednich albo z MCP) przez parametr.
- `api/` i `ui/` nie zawierają logiki biznesowej. `ui/` nie importuje `llm`, `workflow`, `agents`, `tools`, `knowledge`, `rag`: rozmawia z `api` przez HTTP.
- Tylko `llm.py` importuje integracje dostawców (`langchain_google_genai`, `langchain_openai`) i `init_chat_model`. `config.py` i `health.py` nie importują LangChaina.
- `fastembed` importuje tylko `knowledge/embeddings.py`, `langchain_postgres` tylko `knowledge/pgvector.py`, `tiktoken` tylko `cost/tokens.py`. `knowledge/` i `cost/` nie importują `llm`: model embeddingów dostają przez parametr z `llm.get_embeddings()` ([contracts/f1.md §2.3](contracts/f1.md)).
- `rank_bm25` importuje tylko `knowledge/lexical.py`, `langchain_text_splitters` tylko `knowledge/chunking.py`; `langchain_community` nigdzie ([ADR-0008](adr/0008-rag-hybrid-e2.md)).
- Generacja RAG żyje w `rag/`: `answer_question` dostaje `Retriever` i model przez parametr, a `kovalo.llm` importuje tylko `rag/factory.py`. `knowledge/` nie importuje `rag/`; `rag/` nie importuje `eval/`, `workflow/`, `api/`, `ui/`. `llm.py` dostaje cache LLM-a parametrem i nie importuje `eval/` ([contracts/f2.md §2.3](contracts/f2.md)).
- `policy/` importuje tylko bibliotekę standardową, `pydantic`, `kovalo.config`, `kovalo.text_pl`. `workflow/` dostaje model `understand` i `DocumentAnswerer` przez `CopilotDeps`; `kovalo.llm`, `kovalo.rag.factory` i `kovalo.knowledge` importuje tylko `workflow/factory.py`, a `langgraph.checkpoint.postgres` i `psycopg_pool` tylko `workflow/checkpoint.py`. `interrupt()` wywołują tylko węzły w `workflow/graph.py` ([contracts/f3.md §2.3](contracts/f3.md)).
- `company/` i `quantities.py` bez LangChaina; `tools/specs.py` i `tools/search.py` też (F5 rejestruje te funkcje w FastMCP), `langchain_core` tylko w `tools/langchain.py`. `langchain.agents` (`create_agent`, middleware) importuje tylko `agents/`; `agents/` dostaje model, narzędzia i wyjątki fallbacku przez parametr (bez `kovalo.llm`, `kovalo.tools`). `workflow/` importuje `kovalo.agents`, a `kovalo.tools` i `kovalo.company` tylko w `workflow/factory.py`; `get_stream_writer` tylko w `workflow/graph.py` ([contracts/f4.md §2.3](contracts/f4.md)).
- Pakiet `mcp` importują tylko `mcp_server/` i `mcp_client/`; `langchain_mcp_adapters` i `langchain.mcp` nigdzie. `mcp_server/` to cienkie nakładki na `tools/` (z `kovalo.llm` tylko `get_embeddings`, bez modelu czatu), `mcp_client/` zna z naszych modułów tylko `config` i `tools.specs` (klasy rekordów). `kovalo.mcp_client` importuje w `workflow/` tylko `factory.py`; graf, węzły i serwis nie znają transportu narzędzi. `LazyRetriever` żyje w `knowledge/retrievers.py` i dostaje funkcję otwierającą ([contracts/f5.md §2.3](contracts/f5.md)).
- `api/` woła wyłącznie `workflow.CopilotService` (bez `langgraph`, `kovalo.llm`, `kovalo.policy`); `ui/` ma własne modele widoku i nie importuje `kovalo.api` ([ADR-0010](adr/0010-api-ui-f3.md)).
- Nazwy modeli (czatu i embeddingów) są tylko w `config.yaml` (model embeddingów także w `ARG` Dockerfile, pilnowanym testem). Część zasad sprawdza test `tests/architecture/test_dependencies.py`.

## 3. Graf copilota (LangGraph)

![Graf copilota](architecture/kovalo-architecture-2-graph.png)

```
START → understand ─┬─ document_question ─> answer_documents   (E3: RAG z F2; aplikacja: agent z 7 narzędziami, od F5 przez MCP)
                    ├─ policy_question   ─> answer_policy      (silnik polityki, tylko informacja, bez pauzy)
                    ├─ other / error     ─> reply_other
                    └─ approval_request  ─> update_request ─┬─(ValueError: niejasne new|update, brak rabatu)─> END (error)
                                                            ├─(brak marży, nie wyprzedaż)─> clarify ⟲ (interrupt)
                                                            └─> decide ─┬─(już zatwierdzone)─> END
                                                                        └─> approve ⟲ (interrupt, rola po roli) ─> END
```

- **Stan** (`CopilotState`, TypedDict, 1:1 z kursu): `messages` (reducer `add_messages`), `turn` (`Turn`: kind, request_action, standalone, pola rabatu), `request` (`DiscountRequest` z `revision`), `decision` (`Decision`), `approvals` (rola → `approved`/`rejected` dla bieżącej wersji), `status`, `record` (`RunRecord` jako słownik), `error`; NASZE: `calls` (wywołania LLM-a w turze: model, tokeny, czas).
- **Checkpointer:** `PostgresSaver` na `psycopg_pool.ConnectionPool` z serializerem znającym `Turn`, `DiscountRequest`, `Decision`; w testach i E3 `InMemorySaver` z tym samym serializerem. `thread_id` = UUID rozmowy z `POST /threads`. Pauza przetrwa restart kontenera ([ADR-0009](adr/0009-workflow-f3.md)).
- **Pauzy:** ładunki 1:1 z kursu: `interrupt({"type": "clarification", "field": "margin_after_pct", "question": ...})` i `interrupt({"type": "approval", "role": ..., "revision": ..., "approvers": [...], "request": {...}})`. Jedno `interrupt()` na wywołanie węzła; zła odpowiedź wraca krawędzią do tego samego węzła. Wznowienie: `Command(resume=liczba | "approved" | "rejected" | "cancel")`; polskie słowa i „7,5” zamienia `text_pl.human_answer` w serwisie.
- **Serwis** `workflow.CopilotService` (`send_message`, `resume`, `get_thread`) to jedyne wejście do grafu dla API i E3: strażnik pauzy (nowa wiadomość w pauzie → `ThreadPausedError`, w API 409), jedna tura naraz na wątek, `TurnResult` (odpowiedzi tury, pauza, wniosek, źródła, tokeny).
- **Węzeł `answer_documents`** dostaje przez `CopilotDeps` dokładnie jedno: `document_answerer` (RAG z F2, `rag.build_document_answerer`, [contracts/f2.md §7.4](contracts/f2.md); E3) albo `agent` (od F4 w aplikacji; w F5 ta sama ścieżka z narzędziami MCP). Obie wersje używają tej samej fabryki grafu.
- **Serwis od F6:** `TurnResult.usage` (z kanału `calls` tury) ma podział na faktyczne modele (`by_model`) i koszt (`cost`: `ours_usd` po cenach `cost.our_prices_per_1m_tokens`, `reference_usd` po cenach `cost.reference_model`), liczony przez `cost/usage.py` przy składaniu wyniku; koszt nie trafia do stanu grafu. Przerwana tura (ostatni checkpoint z `next != ()` bez pauzy, np. po zabiciu `api` w trakcie węzła) jest zamykana leniwie przy następnej wiadomości, pod blokadą wątku: `update_state(..., as_node="reply_other")` ze statusem `error` (`turn_interrupted`) i komunikatem o niedokończonej odpowiedzi, potem nowa wiadomość biegnie normalnie; przerwanej tury nie dokańczamy ([ADR-0015](adr/0015-polish-f6.md), [contracts/f6.md §7](contracts/f6.md)). Warunek: jeden proces `api`.
- **Serwis od F4:** `start_message` / `start_resume` zwracają `TurnStream` (tura w wątku roboczym, zdarzenia z `stream_mode=["custom", "updates"]`), `send_message` / `resume` czytają go do `final`; `get_trace` czyta ostatnią kopertę `agent_trace` z historii stanu. Błędy połączenia z Postgresem → `CheckpointerUnavailableError` (API: 503).


**Agent w węźle `answer_documents` (F4).** W aplikacji (`workflow.answerer: agent`, `understand_prompt: assistant_understand_system`) węzeł uruchamia `AgentRunner`: `create_agent` z promptem `agent_system`, 7 narzędziami tylko do odczytu, budżetem kursu (24 kroki, 180 s), fallbackiem tylko przy błędach dostępności, `checkpointer=False`, w pustym `contextvars.Context`. Wywołania narzędzi idą do strumienia przez `get_stream_writer`; wynik to odpowiedź agenta (albo przeprosiny), status i trace w `record`. `check_termination_condition` liczy warunek CON-101 w kodzie, a jej odczyt OTIF trafia do trace jako zdarzenie składowe `get_supplier_otif`. E3 dalej buduje graf z RAG i `understand_system` ([ADR-0011](adr/0011-agent-tools-f4.md)).


**Agent przez MCP i końcowy asystent (F5).** Z `tools.source: mcp` `LazyAgentRunner` przy pierwszym pytaniu otwiera `McpToolSource` (`tools.mcp_transport: http` → `services.mcp_url`; `stdio` → podproces `kovalo-mcp`) i buduje tego samego agenta na narzędziach z `list_tools()`. Graf, węzły, serwis i koperta trace się nie zmieniają (ADR-0011 pkt 6). Każde wywołanie przechodzi przez `check_record` (surowy `structuredContent` wobec rekordu z `TOOL_SPECS`, zanim cokolwiek uzupełni wartości domyślne); błąd ma warstwę `tool / protocol / transport / validation`, a połączenie po błędzie transportu nie jest używane ponownie (następne wywołanie otwiera nowe). Model widzi przez MCP dokładnie ten sam rekord co przy narzędziu bezpośrednim; pochodzenie wywołania (`transport`, `invocation_id`) idzie w artefakcie `ToolMessage`, którego `StableUsageMiddleware` nie wysyła do modelu, więc klucze cache są te same co w F4. Agent dostaje callbacki węzła (run agenta pod turą grafu w LangSmith). Pytania o politykę i akceptacje nie używają MCP. Inni klienci (Claude Code przez `.mcp.json`, Inspector, Claude Desktop) uruchamiają własny proces `kovalo-mcp --search memory` przez stdio i widzą tylko narzędzia, zasoby i prompt, nie workflow akceptacji (FR-5.5). E5: [contracts/f5.md §10](contracts/f5.md), [ADR-0014](adr/0014-assistant-eval-e5.md).


## 4. Kontrakty

### Wynik narzędzia (MP4/MP5)

Każde narzędzie zwraca słownik z polem `status` (`ok / no_data / unknown_supplier / not_applicable / out_of_scope / invalid_request`) i polami zależnymi od narzędzia. Przez MCP wynik idzie jako `structuredContent` (ten sam słownik) + kopia JSON w tekście. Klient sprawdza surowy wynik z rekordem (`mcp_client.check_record`: komplet pól bez nadmiarowych, także w rekordach zagnieżdżonych, ścisłe typy, reguły statusów, waluta PLN), zanim użyje wartości domyślnych; zgodność z wywołaniem bezpośrednim mierzy E5 (`compare_call`, 20 wywołań).

### API

Kontrakt F3: [contracts/f3.md §8](contracts/f3.md), decyzje: [ADR-0010](adr/0010-api-ui-f3.md). Każdy błąd ma ciało `ApiError {error, detail, pending}` (także walidacja: 422 `invalid_request`; nieoczekiwany: 500 `internal_error`). Ścieżki JSON zwracają `TurnResult`; od F4 obok nich są ścieżki strumieniowe `…/stream` (`text/event-stream`, zdarzenia `tool_start`, `tool_end`, `final` z `TurnResult`, `error`; bez tokenów). Błędy wykrywalne przed turą wracają jako JSON, a tura biegnie w wątku roboczym i dobiega do końca także po rozłączeniu klienta ([ADR-0012](adr/0012-sse-agent-steps-f4.md), [contracts/f4.md §9](contracts/f4.md)).

| Metoda | Ścieżka | Ciało / wynik | Od |
|---|---|---|---|
| POST | `/threads` | → 201 `{thread_id}` (UUID) | F3 |
| POST | `/threads/{id}/messages` | `{text}` → `TurnResult`; 409 `thread_paused` (z ładunkiem pauzy), `thread_busy`, `thread_stuck` (od F6 tylko awaryjnie, gdy zamknięcie przerwanej tury się nie uda); 422 `invalid_request`; 503 `copilot_unavailable` | F3 (SSE: `/messages/stream`, F4) |
| POST | `/threads/{id}/resume` | `{value}` (liczba / `approved` / `rejected` / `cancel`) albo `{text}` („7,5”, „zatwierdź”), opcjonalnie `pause_ref` → `TurnResult`; 409 `not_paused` (także po zamknięciu przerwanej tury), `stale_pause` | F3 |
| GET | `/threads/{id}` | `ThreadState`: status, aktywna pauza, wniosek, decyzja, akceptacje, wiadomości | F3 |
| POST | `/threads/{id}/resume/stream` | jak `/resume`, odpowiedź SSE | F4 |
| GET | `/threads/{id}/trace` | `TraceView`: ostatni trace agenta w wątku (z historii checkpointów) i ścieżka dowodowa; wątek bez tury agenta (także nieznany) → 200 z `trace: null` | F4 |
| GET | `/health` | `HealthReport`: `ok` / `degraded` (200) / `error` (503); sprawdzenia Postgresa (krytyczne), MCP i klucza LLM-a bez wywołania modelu ([contracts/f0.md §6](contracts/f0.md)) | F0 |

## 5. Przepływ indeksowania

![Indeksowanie](architecture/kovalo-architecture-3-ingest.png)

`catalog.csv` + `corpus/*.md` → `Document` z metadanymi → `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter(800/120)` → nagłówek `<tytuł> > <sekcja>` (87 chunków, id `<doc_id>#ctxNNN`) → `CacheBackedEmbeddings(fastembed)` (osobny cache dokumentów i pytań) → `PGVectorStore`, tabela `kovalo_ctx`. Obok, do porównań E1/E2, chunki MP1 (47) w `kovalo_mp1`. `kovalo ingest` buduje każdą tabelę obok i podmienia ją w jednej transakcji, a model, wymiar i liczbę chunków zapisuje w `kovalo_index_meta` ([ADR-0007](adr/0007-knowledge-index-f1.md)).

Wyszukiwanie (`knowledge.Retriever`): tryb `retrieval.mode` = `vector` (cosine, top 3) albo `hybrid`: top 10 z wektorów i top 10 z BM25 (`rank-bm25`, indeks w pamięci budowany przy starcie z korpusu, zgodność sprawdzana z `kovalo_index_meta.n_chunks`) → ważona Reciprocal Rank Fusion → top 3 ([ADR-0008](adr/0008-rag-hybrid-e2.md)). Domyślnie `vector`: w E2 z e5 hybryda nie dała przewagi (hit@1 9/10 wobec 10/10, reszta remis), a `hybrid` zostaje do włączenia w configu.

## 5a. Odpowiedź ze źródłami i E2


`rag.answer_question`: pytanie → `Retriever` (top 3) → `build_prompt(answer_system)` → `get_structured_model(Answer)` (Gemini z fallbackiem, `include_raw=True`) → `RunRecord` (kontekst, prompt, `Answer` albo błąd, tokeny, model). Ewaluacja E2 ocenia rekord bez LLM-a (`score_record`: dowody, fakty, odwołania, odmowy) i sędzią (`judge_record`, prompt `judge_system`), a odpowiedzi modeli trzyma w cache na dysku (`paths.llm_cache_dir`), więc powtórny przebieg nie zużywa limitu ([contracts/f2.md §8](contracts/f2.md)).

## 6. Obserwowalność i koszty


- Licznik tokenów i kosztów per tura z `usage_metadata` (kanał `calls` stanu tury → `group_by_model` → `price_usage` → `TurnResult.usage.cost`). Ceny z sekcji `cost` w `config.yaml`: `our_prices_per_1m_tokens` dla faktycznego modelu (płatny tier Gemini, NASZE) i `prices_per_1m_tokens[reference_model]` (cennik kursu, PINNED); ta sama funkcja `tokens_cost` co w raportach E2–E5. Nieznany model daje „brak ceny”, nie zgadywanie; free tier = 0 USD. UI pokazuje linię kosztu na końcu ostatniej wiadomości tury i sumę rozmowy z sesji Chainlita ([ADR-0015](adr/0015-polish-f6.md)).
- Dashboard ewaluacji: `reports/dashboard.md` i `.json` z JSON-ów raportów E1–E5 (gdy są), poprzedniego `dashboard.json` i ręcznych danych w `reports/dashboard.yaml` (liczby oryginalnych przebiegów na żywo, zużycie limitu na etap, znane ograniczenia); wynik deterministyczny, `--check` w CI ([contracts/f6.md §8](contracts/f6.md)).
- Trace agenta w checkpointach Postgresa (kanał `record`, koperta `agent_trace` v1 z MP5; bez osobnej tabeli `agent_traces`: decyzja ostateczna, ADR-0011 pkt 7 i ADR-0015; dashboard czyta raporty ewaluacji, nie tury produkcyjne), widoczny w UI (kroki `cl.Step` na żywo, „Ścieżka dowodowa”) i przez `GET /threads/{id}/trace`.
- LangSmith (`LANGSMITH_TRACING=true`, klucz w `.env`, projekt `kovalo-copilot`): każda tura to drzewo runów; od F5 run agenta jest zagnieżdżony pod turą grafu (callbacki węzła, R1-2 z F4).
- Wywołania MCP: `invocation_id` w trace (koperta w checkpoincie) i w logu `mcp-server` (stderr, z czasem i statusem rekordu).
