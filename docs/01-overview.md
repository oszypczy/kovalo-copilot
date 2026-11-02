# 01. Przegląd projektu

## Cel

Kovalo Decision Copilot to asystent dla kadry zarządzającej Kovalo Distribution. Zastępuje 20 minut szukania po dyskach sieciowych. Menedżer pyta normalnym językiem, a asystent:

1. odpowiada na podstawie **wewnętrznych dokumentów** (18 plików w `data/corpus/`) i pokazuje źródła (`[POL-001]`),
2. liczy na **danych firmy** (OTIF dostawców, faktury, kary umowne) i pokazuje ścieżkę dowodową,
3. stosuje **politykę rabatową jako kod**: mówi, kto musi zatwierdzić rabat,
4. prowadzi **workflow akceptacji**: dopytuje o brakujące dane, wstrzymuje się i czeka na decyzję każdej wymaganej roli,
5. udostępnia narzędzia przez **MCP**, więc te same narzędzia działają w naszym asystencie, w MCP Inspector i w Claude Desktop,
6. ma **ewaluację** na każdym poziomie i **szacunek kosztów**.

Sponsor (dyrektor operacyjny): *„Chcę asystenta, którego zapytam normalnym językiem, który odpowie na podstawie naszych dokumentów, pokaże, skąd ma odpowiedź, i pomoże mi podjąć decyzję. A zanim dam go czterdziestu menedżerom, muszę wiedzieć, ile będzie kosztował”.*

## Jeden projekt zamiast pięciu

Prowadzący dopuszcza jeden większy projekt zamiast pięciu osobnych. Kurs i tak buduje jeden system warstwami, a każdy miniprojekt korzysta z poprzednich. My budujemy ten system od razu jako aplikację:

| Miniprojekt kursu | Warstwa naszej aplikacji | Moduł (`src/kovalo/`) |
|---|---|---|
| **MP1** fundament wiedzy: wczytywanie, tokeny i koszt, chunki, embeddingi, wyszukiwanie | indeksowanie i wyszukiwanie, kalkulator kosztów | `knowledge/`, `cost/` |
| **MP2** odpowiedzi ze źródłami: chunki z kontekstem, odpowiedź w schemacie, 4 poziomy ewaluacji, sędzia LLM | RAG z odwołaniami do źródeł, ewaluacja RAG | `rag/`, `eval/rag/` |
| **MP3** workflow decyzyjny: reguły polityki, `understand`, graf z pauzami | graf copilota w LangGraph z `interrupt()` i akceptacjami | `policy/`, `workflow/` |
| **MP4** agent z narzędziami: dane firmy, 6 narzędzi, trace, ocena, supervisor | agent ReAct z narzędziami i zapisem przebiegu | `company/`, `tools/`, `agents/`, `eval/trace/` |
| **MP5** MCP i finalny asystent: serwer FastMCP, klient, podmiana węzła, pełna ewaluacja | serwer MCP jako osobny kontener, asystent korzystający z narzędzi przez MCP | `mcp_server/`, `mcp_client/`, `eval/assistant/` |
| *(nasze rozszerzenie)* | API, UI czatu, Docker, CI | `api/`, `ui/`, `docker/` |

## Czym różnimy się od wersji kursowej

| Kurs | Nasz projekt | Dlaczego |
|---|---|---|
| OpenAI `gpt-5.4-mini` (płatny) | **Gemini Flash-Lite** z darmowego tieru (zapasowo Flash; do F1 odwrotnie, zmiana w [ADR-0006](adr/0006-config-and-llm-factory.md)) | wymóg: darmowy klucz API ([ADR-0002](adr/0002-free-llm-gemini.md)) |
| embeddingi OpenAI z cache kursu | **lokalny wielojęzyczny model embeddingów** (fastembed, ONNX) | bez klucza i bez limitów; cache OpenAI wymaga płatnych embeddingów zapytań ([ADR-0003](adr/0003-local-embeddings.md)) |
| `InMemoryVectorStore` | **Postgres + pgvector** (ten sam Postgres trzyma checkpointy LangGraph) | trwały indeks i trwałe pauzy akceptacji w Dockerze |
| skrypty uruchamiane z terminala | **FastAPI** + **Chainlit** w docker-compose | aplikacja zamiast zbioru skryptów |
| MCP przez stdio (podproces) | MCP przez **streamable HTTP** w osobnym kontenerze; stdio dalej działa dla Inspector i Claude Desktop | w Dockerze serwer to usługa |
| `check.py` po każdym kroku | pytest (unit bez LLM-a) + osobne programy ewaluacji na żywo | ten sam podział: testy deterministyczne i ewaluacja na żywo |

Wartości kursu oznaczone `PINNED` (chunk 800/120, `top_k=3`, tokenizer `o200k_base` do szacowania kosztów, polityka `course-v1`, limity agenta i MCP) zostawiamy bez zmian, żeby wyniki dało się porównać z przebiegami referencyjnymi. Zmiana LLM-a i embeddingów zmieni liczby. Każdy raport ewaluacji pokazuje nasz wynik obok referencyjnego.

## Poza zakresem

- prawdziwe akceptacje i tożsamość osób zatwierdzających (akceptacje są symulowane w UI),
- zapis do CRM-u i obietnice wobec klienta,
- aktualizacja danych firmy na bieżąco (dane to stan na 2026-10-01),
- uwierzytelnianie wielu użytkowników (jedna instancja, lokalnie).
