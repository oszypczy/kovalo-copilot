# Dokumentacja: Kovalo Decision Copilot

Projekt łączy pięć miniprojektów kursu TEG w jedną aplikację: asystenta dla menedżerów fikcyjnej firmy Kovalo Distribution. Asystent działa w Dockerze. Zbudowaliśmy go na LangChain i LangGraph, a narzędzia udostępnia przez MCP.

| Dokument | Co zawiera |
|---|---|
| [01-overview.md](01-overview.md) | cel projektu i to, jak pięć miniprojektów składa się w jeden system |
| [02-requirements.md](02-requirements.md) | wymagania funkcjonalne i niefunkcjonalne z kryteriami akceptacji |
| [03-architecture.md](03-architecture.md) | architektura: kontenery, moduły, przepływ rozmowy, kontrakty |
| [04-tech-stack.md](04-tech-stack.md) | technologie, darmowy LLM, embeddingi, wersje |
| [05-domain.md](05-domain.md) | domena Kovalo: korpus, dane firmy, polityka rabatowa `course-v1`, narzędzia |
| [06-evaluation.md](06-evaluation.md) | ewaluacja: retrieval, RAG na 4 poziomach, dialogi, trace agenta, cały asystent |
| [07-roadmap.md](07-roadmap.md) | etapy realizacji i definicja ukończenia każdego z nich |
| [08-dev-workflow.md](08-dev-workflow.md) | jak pracujemy z Claude Code: subagenci, skille, serwery MCP |
| [contracts/](contracts/) | kontrakty etapów (moduły, sygnatury, testy) przed implementacją; F0: [contracts/f0.md](contracts/f0.md), F1: [contracts/f1.md](contracts/f1.md), F2: [contracts/f2.md](contracts/f2.md) |
| [adr/](adr/) | zapisy decyzji architektonicznych (ADR) |
| [architecture/](architecture/) | diagramy draw.io (`.drawio` i eksporty PNG) |
| [course/](course/) | oryginalne materiały kursu: README i CONCEPTS miniprojektów, scenariusz, słownik |

Kod referencyjny kursu (szkielety kroków, testy, helpers) leży w [`/reference/course`](../reference/course/README.md) i służy tylko do odczytu.
