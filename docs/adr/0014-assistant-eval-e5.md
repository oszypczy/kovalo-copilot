# ADR-0014: Ewaluacja E5 (zgodność MCP, odtworzenie rozmów, przebieg na żywo), E4 przez MCP, nieudane próby w limicie wywołań

- Status: przyjęty (2026-11-04; decyzje U7–U10 i U12 w [contracts/f5.md §14.3](../contracts/f5.md); wynik E5: §16.6)
- Data: 2026-11-04

## Kontekst
MP5 ocenia całego asystenta tura po turze według oczekiwanej ścieżki (`helpers/assistant_eval.py`): 5 rozmów MP5 (12 tur) i 8 dialogów MP3 (13 tur) z nakładką dla D7, razem 25 tur i 12 wymiarów (ścieżka, status, pola, pauzy, zatwierdzający, wniosek, trace tylko tam, gdzie oczekiwany, oraz pięć sprawdzeń MP4 w czterech turach o dane). Kurs uruchamia to tylko na żywo (`run_assistant_eval.py`, około 4,4 centa), a zgodność MCP z wywołaniem bezpośrednim sprawdza testami kroku 2 (`compare_call`, 17 wywołań, 5 mutantów). Drugim kryterium ukończenia MP5 jest E4 przez MCP (`run_agent_mcp.py`). W E3 i E4 rozdzieliliśmy błędy portu (odtworzenie bez LLM-a) od błędów modelu (przebieg na żywo z cache, `--cache-only`, `--max-calls`). Przegląd F4 zostawił R1-6: `CallMeter` nie liczy nieudanych prób do limitu. Darmowy tier: `gemini-3.5-flash-lite` z wysokim limitem dziennym, `gemini-3.6-flash` około 20 zapytań dziennie.

## Decyzja
1. **`kovalo eval e5`** w `eval/assistant/` składa się z trzech części: (a) **zgodność**: 20 wywołań (17 z macierzy kursu + 3 dla `check_termination_condition`) bezpośrednio i przez MCP, porównanych `compare_call` (port kroku 2); (b) **odtworzenie**: 13 rozmów z idealnym `understand` ze wzorca i agentem ze skryptu, przez prawdziwy serwer MCP, przez `CopilotService` i `run_turn`, ocenione bez wzorca agenta; (c) **na żywo** (`--live`): `understand` z promptem `assistant_understand_system` i agent przez MCP, ocena portem `assistant_eval` 1:1 ze `score_trace` z E4. Części (a) i (b) nie wywołują LLM-a i muszą dać 100% (inaczej kod 4).
2. **Transport E5:** domyślnie stdio (podproces `kovalo-mcp --search memory`, jak w kursie), `http` do kontenera w compose, `memory` w testach. Raport podaje transport i backend wyszukiwania.
3. **Budżet:** cache `JsonFileLlmCache` od pierwszego przebiegu, `--cache-only` bez klucza, `--plan` (25 `understand` + około 9 tur agenta = 34; górna granica 73), `--max-calls` (domyślnie 40) jako **jeden** licznik dla `understand` i agenta, liczący także nieudane próby (`CallMeter(count_errors=True, counter=...)`). Wyczerpany limit: bieżąca i następne tury „pominięte”, kod 7, dokończenie z cache.
4. **E4 przez MCP:** `kovalo eval e4 --live --transport memory|stdio|http` buduje agenta na narzędziach MCP (odtworzenie kursu zostaje bezpośrednie). Najpierw `--cache-only`: jeśli narzędzia MCP wyglądają dla modelu jak bezpośrednie (ten sam opis, schemat argumentów i treść wyniku, ADR-0013 pkt 4), przebieg F4 odtwarza się bez wywołań; raport E4 dostaje wtedy kolumnę kursu „przez MCP” (9/9, 3/3, 8/9, 0 (1), 0/9).
5. **R1-6:** `CallMeter(count_errors=...)`: w E5 zawsze włączone, w E4 flaga `--count-errors` (domyślnie wyłączona, żeby liczby F4 zostały porównywalne).
6. **Raport** `reports/mp5/<data>-e5.md` + JSON + `assistant_run.{replay,live}.json` w kształcie kursu; komentarze tylko z `reports/mp5/review.yaml` (etykiety `ours-replay`, `ours-live`, id tur `C1-t1` … `D8-t1`). Cele: ścieżka ≥ 24/25, pozostałe wymiary ≥ kurs − 1, 0 nieudanych wywołań MCP.
7. **`agent_system` bez zmian w F5** (U10 = A, rekomendacja): w asystencie prośba o zatwierdzenie idzie do workflow akceptacji, nie do agenta, więc słabość B1 z E4 nie dotyczy E5. Ewentualne `agent_system_v2` wymaga osobnego ADR-u z E4 i E5 przed i po.

## Konsekwencje
- E5 rozróżnia trzy przyczyny błędu: nakładka/transport (zgodność), port grafu i oceny (odtworzenie), model (na żywo).
- Przebieg na żywo to około 34 wywołań jednego dnia na Flash-Lite; fallback Flash tylko przy błędach dostępności.
- Cache lokalnego przebiegu (`memory`) i przebiegu w compose (`pgvector`) się nie mieszają, bo wyszukiwanie może zwrócić inne fragmenty.
- Odtworzenie z agentem ze skryptu nie mierzy jakości odpowiedzi agenta; tę mierzy E4 i część na żywo E5.
- Wspólny licznik prób może zatrzymać przebieg wcześniej niż w E3/E4 (liczy błędy); to celowe przy darmowym tierze. Wyjątek (R1-1 w [contracts/f5.md §16.2](../contracts/f5.md)): po stronie `understand` licznik owija cały łańcuch z fallbackiem, więc nieudana próba modelu głównego, po której odpowiedział fallback, nie jest liczona; poprawka w F6.
- Wynik (2026-11-04): zgodność 20/20, odtworzenie 100%, na żywo 100% we wszystkich wymiarach (30 wywołań sieciowo, 0 błędów tur), powtórka `--cache-only` 25/25; E4 przez MCP z cache F4 za 0 wywołań, wynik identyczny ([reports/mp5/2026-11-04-e5.md](../../reports/mp5/2026-11-04-e5.md)).

## Rozważane alternatywy
- Tylko przebieg na żywo jak w kursie: błąd nakładki albo grafu wyglądałby jak błąd modelu, a każda poprawka kosztowałaby nowy przebieg.
- Zgodność tylko w testach (bez sekcji w raporcie): test wystarcza do CI, ale raport E5 ma pokazać, że transport nie zmienia wyników.
- Osobne limity dla `understand` i agenta: dwa parametry bez zysku, skoro oba zużywają ten sam dzienny limit modelu.
- Pełny przebieg E4 przez MCP zawsze (≈ 24 wywołania): zbędny, jeśli model widzi przez MCP dokładnie to samo co bezpośrednio, bo wtedy wynik jest z definicji ten sam.
- `agent_system_v2` od razu: zmienia klucze cache wszystkich pytań E4 i tur agenta E5 (około 35 wywołań na pomiar przed i po), a problem nie występuje w ścieżce asystenta.
