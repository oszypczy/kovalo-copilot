# 04. Stos technologiczny

| Obszar | Wybór | Uwagi |
|---|---|---|
| Język, pakiety | Python 3.12, `uv` | jak w kursie |
| LLM: framework | **LangChain 1.x** (`init_chat_model`, `create_agent`, structured output, `@tool`) | wymóg |
| Workflow i agent | **LangGraph 1.x** (`StateGraph`, `interrupt`, `Command`, checkpointer) | wymóg |
| LLM: domyślny | **Gemini Flash** przez `langchain-google-genai` (darmowy klucz z Google AI Studio) | [ADR-0002](adr/0002-free-llm-gemini.md) |
| LLM: zapasowe | Gemini Flash-Lite (ten sam klucz, osobny limit), OpenAI (tryb zgodny z kursem, jeśli ktoś ma klucz) | wybór w `config.yaml` → `llm.provider` |
| Embeddingi | **fastembed** (ONNX, CPU), model wielojęzyczny, np. `intfloat/multilingual-e5-large` albo `BAAI/bge-m3`; wybór po pomiarze hit@k | [ADR-0003](adr/0003-local-embeddings.md) |
| Vector store | **Postgres 16 + pgvector**, `langchain-postgres` (`PGVector`) | [ADR-0004](adr/0004-postgres-pgvector-checkpointer.md) |
| Checkpointy | `langgraph-checkpoint-postgres` (`PostgresSaver`) | ten sam Postgres |
| Wyszukiwanie leksykalne | `BM25Retriever` (`rank-bm25`) + `EnsembleRetriever` | rozszerzenie FR-2.5 |
| Tokenizer do kosztów | `tiktoken` `o200k_base` | PINNED z kursu; dla Gemini liczymy też faktyczne tokeny z `usage_metadata` |
| MCP | `mcp` SDK 1.x (FastMCP, `<2` jak w kursie), transport `stdio` + `streamable-http`; klient bezpośrednio na SDK, `langchain-mcp-adapters` tylko do porównania | kurs wyjaśnia, dlaczego nie `load_mcp_tools` (gubi `structuredContent`) |
| API | FastAPI + `sse-starlette`, Uvicorn | |
| UI | **Chainlit** | `cl.Step` do kroków agenta, `cl.AskActionMessage` do akceptacji |
| Testy | pytest, `GenericFakeChatModel`, skryptowane modele z narzędziami, `httpx` do API | testy bez LLM-a: `-m "not live"` |
| Ewaluacja | własne programy (port z `reference/course/helpers`), opcjonalnie DeepEval | |
| Jakość | `ruff` (lint + format), `mypy`, `pre-commit` | |
| Kontenery | Docker, docker-compose (4 usługi: ui, api, mcp-server, postgres; indeksowanie i ewaluacja jako `make ingest` / `make eval`) | |
| CI | GitHub Actions: lint, testy unit, build obrazów | testy `live` tylko ręcznie (`workflow_dispatch`) z sekretem |

## Konfiguracja LLM-a (`config.yaml`)

Pełny kontrakt: [contracts/f0.md §4](contracts/f0.md), uzasadnienie: [ADR-0006](adr/0006-config-and-llm-factory.md).

```yaml
llm:
  provider: google_genai          # google_genai | openai
  model: gemini-3.8-flash         # stabilna wersja, nie alias -latest
  temperature: null               # domyślna dostawcy (Gemini 3: 1.0, zalecenie Google)
  max_retries: 2
  timeout_s: 60
  params: {thinking_level: low}
  rate_limit: {requests_per_second: 0.15, check_every_n_seconds: 0.1, max_bucket_size: 1}
  fallbacks:
    - {provider: google_genai, model: gemini-3.5-flash-lite,
       rate_limit: {requests_per_second: 0.25, check_every_n_seconds: 0.1, max_bucket_size: 1}}
embeddings:
  model: text-embedding-3-small   # PINNED (model referencyjny kursu)
  dimensions: 1536                # PINNED
  batch_size: 64
  provider: fastembed
  local: {model: intfloat/multilingual-e5-large, dimensions: 1024, models_dir: .cache/fastembed, cache_dir: .cache/embeddings}
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
| Niskie limity RPM/RPD (Google nie publikuje dokładnych liczb, są w AI Studio per projekt) | rate limiter, ewaluacje z cache odpowiedzi i możliwość wznowienia, fallback na Flash-Lite |
| Model z free tieru zmieni nazwę albo zniknie | konkretna wersja w configu, fallback przy 404, jeden punkt zmiany w `config.yaml` |
| Dane wysyłane na free tier Google mogą służyć do ulepszania usług | dane Kovalo są fikcyjne; nie wysyłamy niczego prawdziwego |
| Inny tokenizer niż `o200k_base` | szacunek kosztów zgodny z kursem + faktyczne tokeny dostawcy obok siebie |
| Structured output / tool calling słabsze niż w modelu referencyjnym | testy kontraktów schematów, ponowienie przy błędzie parsowania, status `error` zamiast zgadywania |

## Ograniczenia zależności

| Ograniczenie | Gdzie | Powód | Kiedy usunąć |
|---|---|---|---|
| `cryptography>=46,<47` (zależność pośrednia: `google-auth`, `pyjwt[crypto]` przez `chainlit` i `mcp`) | `pyproject.toml`, `[tool.uv] constraint-dependencies` | `cryptography` 47.x kończy się `SIGILL` przy imporcie w kontenerze na Apple Silicon (Docker Desktop, jądro linuxkit z SVE/SME); sprawdzone na czystym `python:3.12-slim-bookworm`, 46.x działa | gdy nowe wydanie `cryptography` importuje się w tym samym obrazie na Apple Silicon: usunąć wpis, `uv lock --upgrade-package cryptography`, `make up` i `make first-call` |

Ograniczenia pośrednich zależności zapisujemy tutaj; ADR jest potrzebny dopiero wtedy, gdy ograniczenie zmienia wybór technologii albo kontrakt.
