# ADR-0004: Postgres + pgvector jako vector store i checkpointer

- Status: przyjęty
- Data: 2026-11-01

## Kontekst
Kurs używa `InMemoryVectorStore` i `InMemorySaver`. W Dockerze indeks i pauzy akceptacji muszą przetrwać restart, a `api` i `mcp-server` muszą korzystać z tego samego indeksu.

## Decyzja
Jeden kontener `pgvector/pgvector:pg16`: indeks przez `langchain-postgres` (`PGVector`), checkpointy przez `langgraph-checkpoint-postgres` (`PostgresSaver`), zapisy przebiegów agenta w tabeli `agent_traces`.

## Konsekwencje
- Jedna baza do backupu i resetu (`make db-reset`).
- Testy jednostkowe korzystają z `InMemoryVectorStore` / `InMemorySaver` przez te same interfejsy; testy integracyjne z Postgresem w compose.

## Rozważane alternatywy
- Chroma + SQLite checkpointer: dwie technologie zamiast jednej.
- FAISS w pliku: brak współdzielenia między kontenerami bez wolumenu i blokad.

## Aktualizacja (F6)
Tabela `agent_traces` nie powstała. Trace agenta zostaje w checkpointach LangGraph (koperta `agent_trace` w stanie wątku, [ADR-0011](0011-agent-tools-f4.md) pkt 7), a zbiorczy widok ewaluacji czyta raporty, nie tury produkcyjne ([ADR-0015](0015-polish-f6.md)). Indeks to `PGVectorStore` (`PGVector` jest przestarzały, [ADR-0007](0007-knowledge-index-f1.md)).
