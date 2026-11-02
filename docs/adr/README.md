# Zapisy decyzji architektonicznych (ADR)

| ADR | Decyzja | Status |
|---|---|---|
| [0001](0001-single-integrated-app.md) | jedna zintegrowana aplikacja zamiast 5 miniprojektów | przyjęty |
| [0002](0002-free-llm-gemini.md) | Gemini Flash (free tier), fallback Flash-Lite | przyjęty |
| [0003](0003-local-embeddings.md) | lokalne embeddingi fastembed: `intfloat/multilingual-e5-large` (wybór po E1) | przyjęty |
| [0004](0004-postgres-pgvector-checkpointer.md) | Postgres + pgvector jako vector store i checkpointer | przyjęty |
| [0005](0005-mcp-streamable-http.md) | MCP przez streamable HTTP w kontenerze + stdio | przyjęty |
| [0006](0006-config-and-llm-factory.md) | kontrakt `config.yaml` i fabryki modeli: `gemini-3.6-flash` → `gemini-3.5-flash-lite`, temperatura dostawcy, fallback tylko przy błędach dostępności | przyjęty |
| [0007](0007-knowledge-index-f1.md) | indeks wiedzy F1: `PGVectorStore`, własny adapter fastembed z osobnym cache pytań, kandydaci E1, cennik Gemini w `cost.our_*` | przyjęty |

Nowy ADR: skopiuj [0000-template.md](0000-template.md).
