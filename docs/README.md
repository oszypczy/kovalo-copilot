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
| [contracts/](contracts/) | kontrakty etapów (moduły, sygnatury, testy, zadania) spisane przed implementacją: [F0](contracts/f0.md) fundament, [F1](contracts/f1.md) indeks wiedzy i E1, [F2](contracts/f2.md) RAG i E2, [F3](contracts/f3.md) workflow, API, UI i E3, [F4](contracts/f4.md) agent z narzędziami i E4, [F5](contracts/f5.md) MCP i E5, [F6](contracts/f6.md) dopracowanie (koszty, dashboard, demo) |
| [adr/](adr/) | zapisy decyzji architektonicznych (ADR-0001 … ADR-0015); lista ze statusami: [adr/README.md](adr/README.md) |
| [demo/](demo/) | scenariusz pokazu aplikacji na żywo: [demo/scenario.md](demo/scenario.md) |
| [architecture/](architecture/) | diagramy draw.io (`.drawio` i eksporty PNG) |
| [course/](course/) | oryginalne materiały kursu: README i CONCEPTS miniprojektów, scenariusz, słownik |

Wyniki ewaluacji E1–E5 w jednym miejscu: [reports/dashboard.md](../reports/dashboard.md) (generowany poleceniem `make dashboard`), raporty etapów w [`/reports`](../reports/).

Kod referencyjny kursu (szkielety kroków, testy, helpers) leży w [`/reference/course`](../reference/course/README.md) i służy tylko do odczytu.
