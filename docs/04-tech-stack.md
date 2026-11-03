# 04. Stos technologiczny

| Obszar | Wybór | Uwagi |
|---|---|---|
| Język, pakiety | Python 3.12, `uv` | jak w kursie |
| LLM: framework | **LangChain 1.x** (`init_chat_model`, `create_agent`, structured output, `@tool`) | wymóg |
| Workflow i agent | **LangGraph 1.x** (`StateGraph`, `interrupt`, `Command`, checkpointer) | wymóg |
| LLM: domyślny | **Gemini Flash-Lite** (`gemini-3.5-flash-lite`) przez `langchain-google-genai` (darmowy klucz z Google AI Studio); do F1 był to Gemini Flash, ale ma w darmowym tierze tylko 20 zapytań dziennie | [ADR-0002](adr/0002-free-llm-gemini.md), [ADR-0006](adr/0006-config-and-llm-factory.md) (pomiar F2 / E2) |
| LLM: zapasowe | Gemini Flash `gemini-3.6-flash` (ten sam klucz, osobny limit 20 zapytań/dzień), OpenAI (tryb zgodny z kursem, jeśli ktoś ma klucz) | wybór w `config.yaml` → `llm.provider` |
| Embeddingi | **fastembed** (ONNX, CPU) przez własny adapter + `CacheBackedEmbeddings` (`langchain-classic`); model **`intfloat/multilingual-e5-large`** (1024, prefiksy `query: `/`passage: `), wybrany w E1 spośród 4 kandydatów (jina-v3, mpnet, MiniLM; wyniki w ADR-0003); wektory nieznormalizowane, więc tylko odległość kosinusowa | [ADR-0003](adr/0003-local-embeddings.md), [ADR-0007](adr/0007-knowledge-index-f1.md) |
| Vector store | **Postgres 16 + pgvector**, `langchain-postgres` (`PGVectorStore`; `PGVector` jest przestarzały) | [ADR-0004](adr/0004-postgres-pgvector-checkpointer.md), [ADR-0007](adr/0007-knowledge-index-f1.md) |
| Checkpointy | `langgraph-checkpoint-postgres` 3.2 (`PostgresSaver` na `psycopg_pool.ConnectionPool`, serializer z zarejestrowanymi typami workflow); w testach `InMemorySaver` | ten sam Postgres; graf synchroniczny ([ADR-0009](adr/0009-workflow-f3.md)) |
| Wyszukiwanie leksykalne | `rank-bm25` (`BM25Okapi`) bezpośrednio + własna ważona fuzja RRF (bez `BM25Retriever` z `langchain-community` i bez `EnsembleRetriever`) | rozszerzenie FR-2.5, [ADR-0008](adr/0008-rag-hybrid-e2.md); tryb `retrieval.mode`: `vector` albo `hybrid` |
| Tokenizer do kosztów | `tiktoken` `o200k_base` | PINNED z kursu; dla Gemini liczymy też faktyczne tokeny z `usage_metadata`; ceny Gemini w `cost.our_prices_per_1m_tokens` |
| MCP | `mcp` SDK 1.x (FastMCP, `<2` jak w kursie), transport `stdio` + `streamable-http`; klient bezpośrednio na SDK, `langchain-mcp-adapters` tylko do porównania | kurs wyjaśnia, dlaczego nie `load_mcp_tools` (gubi `structuredContent`) |
| API | FastAPI, Uvicorn (jeden proces) | F3: odpowiedzi JSON; F4: SSE przez `fastapi.sse.EventSourceResponse` ([ADR-0010](adr/0010-api-ui-f3.md)) |
| UI | **Chainlit** | `cl.Action` + `@cl.action_callback` do akceptacji, `cl.Text` do źródeł, `cl.Step` do kroków agenta (F4); z `api` tylko przez HTTP (`httpx`) ([ADR-0010](adr/0010-api-ui-f3.md)) |
| Testy | pytest, `GenericFakeChatModel`, skryptowane modele z narzędziami, `httpx` do API | testy bez LLM-a: `-m "not live"` |
| Ewaluacja | własne programy (port z `reference/course/helpers`); cache odpowiedzi LLM-a na dysku (`JsonFileLlmCache`, `langchain_core.caches.BaseCache`); DeepEval pomijamy | [ADR-0008](adr/0008-rag-hybrid-e2.md) |
| Jakość | `ruff` (lint + format), `mypy`, `pre-commit` | |
| Kontenery | Docker, docker-compose (4 usługi: ui, api, mcp-server, postgres; indeksowanie i ewaluacja jako `make ingest` / `make eval`) | |
| CI | GitHub Actions: lint, testy unit, build obrazów | testy `live` tylko ręcznie (`workflow_dispatch`) z sekretem |

## Konfiguracja LLM-a (`config.yaml`)

Pełny kontrakt: [contracts/f0.md §4](contracts/f0.md), uzasadnienie: [ADR-0006](adr/0006-config-and-llm-factory.md).

```yaml
llm:
  provider: google_genai          # google_genai | openai
  model: gemini-3.5-flash-lite    # stabilna wersja; Flash (3.5/3.6) ma w darmowym tierze tylko 20 zapytań/dzień (ADR-0006)
  temperature: null               # domyślna dostawcy (Gemini 3: 1.0, zalecenie Google)
  max_retries: 1
  timeout_s: 60
  params: {thinking_level: low}   # obsługiwane przez 3.5 Flash-Lite i 3.6 Flash
  rate_limit: {requests_per_second: 0.25, check_every_n_seconds: 0.1, max_bucket_size: 1}
  fallbacks:
    - {provider: google_genai, model: gemini-3.6-flash,
       rate_limit: {requests_per_second: 0.15, check_every_n_seconds: 0.1, max_bucket_size: 1}}
embeddings:
  model: text-embedding-3-small   # PINNED (model referencyjny kursu)
  dimensions: 1536                # PINNED
  batch_size: 64
  provider: fastembed
  local: {model: intfloat/multilingual-e5-large, dimensions: 1024, query_prefix: "query: ", passage_prefix: "passage: ",
         threads: null, local_files_only: false, models_dir: .cache/fastembed, cache_dir: .cache/embeddings}
  candidates: [...]               # E1, pełna lista: contracts/f1.md §4.1
```

`.env`:

```
GOOGLE_API_KEY=        # https://aistudio.google.com/apikey (darmowy)
LANGSMITH_API_KEY=     # opcjonalny tracing
OPENAI_API_KEY=        # opcjonalny tryb zgodny z kursem
```

## Ryzyka darmowego tieru

| Ryzyko | Co robimy |
|---|---|
| Niskie limity RPM/RPD (Google nie publikuje dokładnych liczb, są w AI Studio per projekt; 2026-11-02: Flash 20 zapytań/dzień) | Flash-Lite jako model główny, rate limiter, ewaluacje z cache odpowiedzi i możliwość wznowienia, fallback na Flash |
| Model z free tieru zmieni nazwę albo zniknie | konkretna wersja w configu, fallback przy 404, jeden punkt zmiany w `config.yaml` |
| Dane wysyłane na free tier Google mogą służyć do ulepszania usług | dane Kovalo są fikcyjne; nie wysyłamy niczego prawdziwego |
| Inny tokenizer niż `o200k_base` | szacunek kosztów zgodny z kursem + faktyczne tokeny dostawcy obok siebie |
| Structured output / tool calling słabsze niż w modelu referencyjnym | testy kontraktów schematów, ponowienie przy błędzie parsowania, status `error` zamiast zgadywania |

## Ograniczenia zależności

| Ograniczenie | Gdzie | Powód | Kiedy usunąć |
|---|---|---|---|
| `cryptography>=46,<47` (zależność pośrednia: `google-auth`, `pyjwt[crypto]` przez `chainlit` i `mcp`) | `pyproject.toml`, `[tool.uv] constraint-dependencies` | `cryptography` 47.x kończy się `SIGILL` przy imporcie w kontenerze na Apple Silicon (Docker Desktop, jądro linuxkit z SVE/SME); sprawdzone na czystym `python:3.12-slim-bookworm`, 46.x działa | gdy nowe wydanie `cryptography` importuje się w tym samym obrazie na Apple Silicon: usunąć wpis, `uv lock --upgrade-package cryptography`, `make up` i `make first-call` |
| `onnxruntime>=1.20,<1.23` (zależność pośrednia przez `fastembed`) | `pyproject.toml`, `[tool.uv] constraint-dependencies` | `onnxruntime` 1.23+ (kernele SME/KleidiAI) kończy się `SIGILL` (kod 132) przy liczeniu embeddingów w kontenerze na Apple M4/M5 (Docker Desktop ogłasza SME); 1.22.1 i 1.20.1 działają, na hoście macOS problem nie występuje | gdy nowe wydanie przechodzi `make ingest` w kontenerze na Apple Silicon: usunąć wpis, `uv lock --upgrade-package onnxruntime`, `make up`, `make ingest` |

Ograniczenia pośrednich zależności zapisujemy tutaj; ADR jest potrzebny dopiero wtedy, gdy ograniczenie zmienia wybór technologii albo kontrakt.
