# 04. Stos technologiczny

| Obszar | Wybór | Uwagi |
|---|---|---|
| Język, pakiety | Python 3.12, `uv` | jak w kursie |
| LLM: framework | **LangChain 1.x** (`init_chat_model`, `create_agent`, structured output, `@tool`) | wymóg |
| Workflow i agent | **LangGraph 1.x** (`StateGraph`, `interrupt`, `Command`, checkpointer) | wymóg |
| LLM: domyślny | **Gemini Flash** przez `langchain-google-genai` (darmowy klucz z Google AI Studio) | [ADR-0002](adr/0002-free-llm-gemini.md) |
| LLM: zapasowe | Groq (`langchain-groq`, darmowy tier), Ollama (`langchain-ollama`, lokalnie), OpenAI (tryb zgodny z kursem, jeśli ktoś ma klucz) | wybór w `config.yaml` → `llm.provider` |
| Embeddingi | **fastembed** (ONNX, CPU), model wielojęzyczny, np. `intfloat/multilingual-e5-large` albo `BAAI/bge-m3`; wybór po pomiarze hit@k | [ADR-0003](adr/0003-local-embeddings.md) |
| Vector store | **Postgres 16 + pgvector**, `langchain-postgres` (`PGVector`) | [ADR-0004](adr/0004-postgres-pgvector-checkpointer.md) |
| Checkpointy | `langgraph-checkpoint-postgres` (`PostgresSaver`) | ten sam Postgres |
| Wyszukiwanie leksykalne | `BM25Retriever` (`rank-bm25`) + `EnsembleRetriever` | rozszerzenie FR-2.5 |
| Tokenizer do kosztów | `tiktoken` `o200k_base` | PINNED z kursu; dla Gemini liczymy też faktyczne tokeny z `usage_metadata` |
| MCP | `mcp` SDK (FastMCP), transport `stdio` + `streamable-http`; klient bezpośrednio na SDK, `langchain-mcp-adapters` tylko do porównania | kurs wyjaśnia, dlaczego nie `load_mcp_tools` (gubi `structuredContent`) |
| API | FastAPI + `sse-starlette`, Uvicorn | |
| UI | **Chainlit** | `cl.Step` do kroków agenta, `cl.AskActionMessage` do akceptacji |
| Testy | pytest, `GenericFakeChatModel`, skryptowane modele z narzędziami, `httpx` do API | testy bez LLM-a: `-m "not live"` |
| Ewaluacja | własne programy (port z `reference/course/helpers`), opcjonalnie DeepEval | |
| Jakość | `ruff` (lint + format), `mypy`, `pre-commit` | |
| Kontenery | Docker, docker-compose (profile: domyślny, `eval`, `local-llm`) | |
| CI | GitHub Actions: lint, testy unit, build obrazów | testy `live` tylko ręcznie (`workflow_dispatch`) z sekretem |

## Konfiguracja LLM-a (szkic `config.yaml`)

```yaml
llm:
  provider: google_genai          # google_genai | groq | ollama | openai
  model: gemini-flash-latest      # alias Google; konkretną wersję sprawdź w AI Studio
  temperature: 0
  rate_limit_rps: 0.15            # ok. 9 zapytań/min, poniżej limitu darmowego tieru
  fallbacks:
    - {provider: groq, model: llama-3.3-70b-versatile}
embeddings:
  provider: fastembed
  model: intfloat/multilingual-e5-large
```

`.env`:

```
GOOGLE_API_KEY=        # https://aistudio.google.com/apikey (darmowy)
GROQ_API_KEY=          # opcjonalny fallback, https://console.groq.com
LANGSMITH_API_KEY=     # opcjonalny tracing
OPENAI_API_KEY=        # opcjonalny tryb zgodny z kursem
```

## Ryzyka darmowego tieru

| Ryzyko | Co robimy |
|---|---|
| Niskie limity RPM/RPD (Google nie publikuje dokładnych liczb, są w AI Studio per projekt) | rate limiter, ewaluacje z cache odpowiedzi i możliwość wznowienia, fallback na Groq |
| Model z free tieru zmieni nazwę albo zniknie | alias w configu, jeden punkt zmiany w `llm.py` |
| Dane wysyłane na free tier Google mogą służyć do ulepszania usług | dane Kovalo są fikcyjne; nie wysyłamy niczego prawdziwego |
| Inny tokenizer niż `o200k_base` | szacunek kosztów zgodny z kursem + faktyczne tokeny dostawcy obok siebie |
| Structured output / tool calling słabsze niż w modelu referencyjnym | testy kontraktów schematów, ponowienie przy błędzie parsowania, status `error` zamiast zgadywania |
