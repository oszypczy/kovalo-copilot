# Zapisy decyzji architektonicznych (ADR)

| ADR | Decyzja | Status |
|---|---|---|
| [0001](0001-single-integrated-app.md) | jedna zintegrowana aplikacja zamiast 5 miniprojektów | przyjęty |
| [0002](0002-free-llm-gemini.md) | Gemini z darmowego tieru, fallback w obrębie Gemini (kolejność modeli: ADR-0006) | przyjęty |
| [0003](0003-local-embeddings.md) | lokalne embeddingi fastembed: `intfloat/multilingual-e5-large` (wybór po E1) | przyjęty |
| [0004](0004-postgres-pgvector-checkpointer.md) | Postgres + pgvector jako vector store i checkpointer | przyjęty |
| [0005](0005-mcp-streamable-http.md) | MCP przez streamable HTTP w kontenerze + stdio | przyjęty |
| [0006](0006-config-and-llm-factory.md) | kontrakt `config.yaml` i fabryki modeli: od F2 `gemini-3.5-flash-lite` → `gemini-3.6-flash` (Flash: 20 zapytań/dzień), temperatura dostawcy, fallback tylko przy błędach dostępności | przyjęty |
| [0007](0007-knowledge-index-f1.md) | indeks wiedzy F1: `PGVectorStore`, własny adapter fastembed z osobnym cache pytań, kandydaci E1, cennik Gemini w `cost.our_*` | przyjęty |
| [0008](0008-rag-hybrid-e2.md) | RAG F2: chunki z kontekstem w `kovalo_ctx`, hybryda `rank-bm25` + własna fuzja RRF, `get_structured_model` z fallbackiem, prompt sędziego `judge_system`, cache LLM-a na dysku, E2 domyślnie bez LLM-a; `retrieval.mode: vector` po E2 | przyjęty |
| [0009](0009-workflow-f3.md) | workflow F3: port MP3 (schematy 1:1, polityka w `Decimal`), graf z pauzami kursu, `PostgresSaver` na puli, `CopilotService` jako jedyny punkt wejścia, E3 (odtworzenie + na żywo), tryb `offline` fabryk modeli; E3: odtworzenie 100%, na żywo kierowanie 13/13 | przyjęty |
| [0010](0010-api-ui-f3.md) | API i UI w F3: odpowiedzi JSON (SSE w F4), wątki bez rejestru, `/resume` z `value`/`text`/`pause_ref`, przyciski Chainlit z `action_callback` | przyjęty |

Nowy ADR: skopiuj [0000-template.md](0000-template.md).
