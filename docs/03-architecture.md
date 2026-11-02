# 03. Architektura

Diagramy: [`architecture/kovalo-architecture.drawio`](architecture/kovalo-architecture.drawio). Plik ma trzy strony: kontenery, graf copilota i przepływ danych w indeksowaniu. Eksporty PNG leżą obok pliku.

![Kontenery](architecture/kovalo-architecture-1-containers.png)

## 1. Kontenery (docker-compose)

| Usługa | Obraz / technologia | Port | Odpowiedzialność |
|---|---|---|---|
| `ui` | Chainlit | 8000 | czat, kroki agenta, przyciski akceptacji, źródła; rozmawia tylko z `api` |
| `api` | FastAPI + LangGraph | 8080 | graf copilota (`understand`, routing, workflow akceptacji, agent), checkpointy, koszty, trace |
| `mcp-server` | FastMCP (`mcp` SDK), streamable HTTP (`/mcp`) | 8765 | 6 narzędzi tylko do odczytu, zasoby (dokumenty), prompt; dostęp do indeksu i danych firmy |
| `postgres` | `pgvector/pgvector:pg16` | 5432 | indeks wektorowy (`langchain-postgres`, `PGVectorStore`: tabela na kolekcję), checkpointy LangGraph (`langgraph-checkpoint-postgres`), zapisy przebiegów |

Indeksowanie i ewaluacja to **nie osobne usługi**, tylko polecenia CLI (`kovalo ingest`, `kovalo eval`) uruchamiane jednorazowo w obrazie `api`: `make ingest` = `docker compose run --rm api kovalo ingest` (korpus → chunki z kontekstem → embeddingi → upsert do pgvector), `make eval STAGE=e2` = `docker compose run --rm api kovalo eval e2` (raport w `reports/`, montowanym jako wolumen).

Zewnętrznie: **Gemini API** (domyślny LLM, darmowy klucz z Google AI Studio), opcjonalnie **LangSmith** (tracing, darmowy plan dla jednej osoby).

Embeddingi liczymy **lokalnie** (fastembed, model ONNX wbudowany w obraz), więc ani indeksowanie, ani `mcp-server` nie potrzebują klucza.

```
Chainlit (ui) ──HTTP/SSE──> FastAPI (api) ──LangGraph──> Gemini API
                                 │   └──MCP streamable HTTP──> mcp-server ──> pgvector + data/company
                                 └──checkpointy / trace──> postgres
make ingest ──> data/corpus ──> chunki ──> fastembed ──> pgvector
```

## 2. Moduły Pythona (`src/kovalo/`)

```
src/kovalo/
  config.py          ustawienia: config.yaml (PINNED z kursu) + .env (pydantic-settings), get_settings()
  llm.py             fabryka modeli: init_chat_model(provider), rate limiter, fallback Flash → Flash-Lite; fabryka embeddingów (F1)
  health.py          wspólny format /health i sprawdzenia (Postgres, HTTP, klucz LLM-a)
  cli.py             polecenie `kovalo`: first-call, ingest, eval
  knowledge/         MP1/MP2: loader, tokeny, chunking (zwykły i z kontekstem), embeddingi, vector store, retriever (+ hybryda BM25)
  cost/              MP1: tokenizer o200k_base, scenariusz kosztów, licznik kosztów per tura
  rag/               MP2: odpowiedź ze źródłami (structured output), rekord przebiegu
  policy/            MP3: silnik polityki course-v1 (czysty kod, zero LLM-a)
  company/           MP4: odczyt danych firmy, kalkulator kary (Decimal, grosze)
  tools/             MP4: narzędzia LangChain (@tool) z kontraktami wyników i polem status
  agents/            MP4: agent ReAct (create_agent), supervisor (opcjonalnie), trace
  workflow/          MP3/MP5: stan, węzły, graf copilota, parser odpowiedzi człowieka
  mcp_server/        MP5: FastMCP, transport stdio + streamable-http
  mcp_client/        MP5: połączenie, limity czasu, klasy błędów, walidacja structuredContent
  eval/              MP1–MP5: retrieval, rag (4 poziomy + sędzia), dialogues, trace (5 testów), assistant
  api/               FastAPI: routery, schematy, zależności (graf, checkpointer)
  ui/                Chainlit (app.py) zamontowany w FastAPI (server.py, /health); renderowanie kroków, pauz i źródeł
```

Punkty wejścia: `kovalo` (CLI), `kovalo-api`, `kovalo-ui`, `kovalo-mcp`. Kontrakty modułów fundamentu (`config`, `llm`, `health`, `cli`) opisuje [contracts/f0.md](contracts/f0.md), a modułów wiedzy i kosztów (`knowledge`, `cost`, `eval` E1) [contracts/f1.md](contracts/f1.md).

Zasady zależności (sprawdzane w review):
- `policy/`, `company/`, `cost/` nie importują LangChaina ani niczego z LLM-em: to deterministyczny rdzeń.
- `tools/` to cienkie nakładki na `knowledge/`, `company/`, `policy/`. `mcp_server/` to cienkie nakładki na `tools/`.
- `workflow/` nie zna transportu narzędzi: dostaje listę narzędzi (bezpośrednich albo z MCP) przez parametr.
- `api/` i `ui/` nie zawierają logiki biznesowej. `ui/` nie importuje `llm`, `workflow`, `agents`, `tools`, `knowledge`, `rag`: rozmawia z `api` przez HTTP.
- Tylko `llm.py` importuje integracje dostawców (`langchain_google_genai`, `langchain_openai`) i `init_chat_model`. `config.py` i `health.py` nie importują LangChaina.
- `fastembed` importuje tylko `knowledge/embeddings.py`, `langchain_postgres` tylko `knowledge/pgvector.py`, `tiktoken` tylko `cost/tokens.py`. `knowledge/` i `cost/` nie importują `llm`: model embeddingów dostają przez parametr z `llm.get_embeddings()` ([contracts/f1.md §2.3](contracts/f1.md)).
- Nazwy modeli (czatu i embeddingów) są tylko w `config.yaml` (model embeddingów także w `ARG` Dockerfile, pilnowanym testem). Część zasad sprawdza test `tests/architecture/test_dependencies.py`.

## 3. Graf copilota (LangGraph)

![Graf copilota](architecture/kovalo-architecture-2-graph.png)

```
START → understand ─┬─ document_question ─> answer_with_tools  (agent ReAct, narzędzia przez MCP)
                    ├─ policy_question   ─> answer_policy       (silnik polityki, bez narzędzi)
                    ├─ other / error     ─> reply_other
                    └─ approval_request  ─> update_request ─┬─(brak marży / niejasne new|update)─> clarify ⟲ (interrupt)
                                                            └─> decide ─> approve ⟲ (interrupt, rola po roli) ─> END
```

- **Stan** (`CopilotState`, TypedDict): `messages` (z reducerem `add_messages`), `turn` (kind, request_action, standalone, pola rabatu), `request` (aktualny wniosek: warunki, wersja, wymagane role, decyzje), `status`, `trace`, `usage`.
- **Checkpointer:** `PostgresSaver`; `thread_id` = identyfikator rozmowy w UI. Pauza przetrwa restart kontenera.
- **Pauzy:** `interrupt({"kind": "clarify_margin" | "approval", "role": ..., "request": ...})`; API zwraca ładunek pauzy, UI pokazuje przyciski, odpowiedź wraca jako `Command(resume=...)`.
- **Węzeł `answer_with_tools`** w MP3 to RAG z FR-2; w MP5 podmieniamy go na agenta z narzędziami MCP. Implementujemy to jako fabrykę grafu z parametrem `document_answerer`, więc obie wersje istnieją i są testowane.

## 4. Kontrakty

### Wynik narzędzia (MP4/MP5)

Każde narzędzie zwraca słownik z polem `status` (`ok / no_data / unknown_supplier / not_applicable / out_of_scope / invalid_request`) i polami zależnymi od narzędzia. Przez MCP wynik idzie jako `structuredContent` (ten sam słownik) + tekst. Klient waliduje go modelem Pydantic, zanim użyje wartości domyślnych.

### API (szkic)

| Metoda | Ścieżka | Ciało / wynik |
|---|---|---|
| POST | `/threads` | → `{thread_id}` |
| POST | `/threads/{id}/messages` | `{text}` → strumień SSE: `token`, `tool_start`, `tool_end`, `interrupt`, `final` |
| POST | `/threads/{id}/resume` | `{value}` (liczba / `approved` / `rejected` / `cancel`) → jak wyżej |
| GET | `/threads/{id}` | stan: status, aktywna pauza, wniosek |
| GET | `/threads/{id}/trace` | ścieżka dowodowa ostatniej tury |
| GET | `/health` | `HealthReport`: `ok` / `degraded` (200) / `error` (503); sprawdzenia Postgresa (krytyczne), MCP i klucza LLM-a bez wywołania modelu ([contracts/f0.md §6](contracts/f0.md)) |

## 5. Przepływ indeksowania

![Indeksowanie](architecture/kovalo-architecture-3-ingest.png)

`catalog.csv` + `corpus/*.md` → `Document` z metadanymi → `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter(800/120)` → nagłówek `<tytuł> > <sekcja>` → `CacheBackedEmbeddings(fastembed)` (osobny cache dokumentów i pytań) → `PGVectorStore` (tabela `kovalo_ctx` od F2; w F1 chunki MP1 w `kovalo_mp1`) + indeks BM25 (w pamięci, budowany przy starcie z tych samych chunków). `kovalo ingest` buduje tabelę obok i podmienia ją w jednej transakcji, a model i wymiar zapisuje w `kovalo_index_meta` ([ADR-0007](adr/0007-knowledge-index-f1.md)).

## 6. Obserwowalność i koszty

- Licznik tokenów i kosztów per tura z `usage_metadata` (przeliczany po cenach referencyjnych z configu i po faktycznych cenach dostawcy; free tier = 0 USD).
- Trace agenta w Postgresie (`agent_traces`), widoczny w UI i przez `GET /threads/{id}/trace`.
- Opcjonalnie LangSmith (`LANGSMITH_TRACING=true`).
