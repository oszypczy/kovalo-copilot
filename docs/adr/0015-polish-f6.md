# ADR-0015: Dopracowanie F6: koszt tury, zamknięcie przerwanej tury, dashboard z raportów

- Status: przyjęty (2026-11-07, po przeglądzie R1)
- Data: 2026-11-07

## Kontekst

F6 domyka aplikację przed oddaniem: licznik kosztów w UI (FR-6.5), zbiorczy widok ewaluacji, README z demo; roadmapa wymieniała też prezentację projektu. Zostały też zaległości z F3–F5: wątek `thread_stuck` po przerwanej turze (R1-8 z F5, U13 = B), tokeny w SSE (ADR-0012: „kandydat na F6”), tabela `agent_traces` (ADR-0011 pkt 7: „rozważymy przy dashboardzie”), lista rozmów, poprawki liczenia wywołań i tokenów w E5 (R1-1–R1-3 z F5). Kontrakt i pytania: [contracts/f6.md](../contracts/f6.md) §13.2 (U1–U11).

Ograniczenia: darmowy tier Gemini (Flash-Lite główny, Flash około 20 zapytań dziennie), więc licznik i dashboard nie mogą wywoływać LLM-a; JSON-y raportów są poza gitem (w gicie tylko Markdown i `review.yaml`); `api` to jeden proces Uvicorna; wątki nie mają rejestru (ADR-0010).

## Decyzja

Decyzje użytkownika z 2026-11-07 (contracts/f6.md §13.3): rekomendacje U1–U7, U10 (A) i U11 przyjęte, U9 = C; po D2 zmiana U8 i U10: bez nagrania demo.

1. **Koszt tury w `TurnResult`** (U4, U5). `TurnUsage` dostaje pola `by_model: list[ModelUsage]` (podział na faktyczne modele, fallback widoczny) i `cost: TurnCost | None` (`ours_usd` po cenach `cost.our_prices_per_1m_tokens` dla faktycznego modelu, `reference_usd` po cenach `cost.reference_model`, `unpriced_models`). Liczy `CopilotService` nowym modułem `cost/usage.py` tą samą funkcją `tokens_cost` co raporty E2–E5; serwis bez cennika (E3, testy) daje `cost None`. Koszt nie trafia do stanu grafu. UI pokazuje jedną linię na końcu ostatniej wiadomości tury i sumę rozmowy z sesji; nie zna cen. Zmiana addytywna, ścieżki API bez zmian.
2. **Zamknięcie przerwanej tury** (U7). Wątek z `next != ()` i bez pauzy jest zamykany leniwie przy następnej wiadomości, pod blokadą wątku: `graph.update_state(..., as_node="reply_other")` ze statusem `error`, błędem `turn_interrupted` i komunikatem dla menedżera; potem nowa wiadomość biegnie normalnie (odpowiedź zawiera komunikat i nową odpowiedź). Przerwanej tury nie dokańczamy. `start_resume` po zamknięciu zgłasza `not_paused`. 409 `thread_stuck` zostaje jako błąd awaryjny, gdy zamknięcie się nie uda. Warunek: jeden proces `api`.
3. **Dashboard z raportów** (U2, U3). `kovalo eval dashboard` generuje `reports/dashboard.md` i `reports/dashboard.json` z JSON-ów raportów E1–E5 (gdy są) i z poprzedniego `dashboard.json` (gdy ich nie ma); ręczne dane (liczby oryginalnych przebiegów na żywo nadpisanych powtórką z cache, zużycie limitu na etap, znane ograniczenia) są w `reports/dashboard.yaml`. Wszystkie trzy pliki w gicie; wynik deterministyczny, `--check` w CI. Nowy klucz NASZ `paths.reports_dir`. Zero wywołań LLM-a.
4. **Czego nie robimy** (U1, U6). Bez tokenów w SSE: FR-6.2 „strumieniowanie odpowiedzi” oznacza strumień kroków agenta i wyniku tury (ADR-0012). Bez listy rozmów (historia Chainlita wymaga warstwy danych i uwierzytelniania). Bez tabeli `agent_traces`: trace zostaje w checkpoincie (ADR-0011 pkt 7), a dashboard czyta raporty ewaluacji, nie tury produkcyjne. R1-1–R1-3 z F5 zostają znanymi ograniczeniami opisanymi w dashboardzie i 06-evaluation. Bez prezentacji ze slajdami (U9 = C) i bez nagrania demo (GIF/MP4; zmiana U8 i U10 z 2026-11-07, „nie będę nagrywać demo, będę to pokazywać”): projekt pokazujemy w działającej aplikacji według scenariusza [demo/scenario.md](../demo/scenario.md), a plan B przy limicie albo braku sieci to automatyczny fallback, jedna ponowna próba, rozmowa z porannej próby i dashboard; bez trybu demo z cache LLM-a w `api`.

Wartości PINNED i prompty kursu bez zmian.

## Konsekwencje

- FR-6.5 spełnione w UI (per tura i per rozmowa) i w raportach (dashboard). FR-6.2 i 02-requirements doprecyzowane.
- `workflow/service.py` importuje `kovalo.cost.usage` (jedyny nowy import w `workflow/`); `cost/__init__.py` musi zostać bez importów, żeby `workflow/` nie wciągał `tiktoken`. `ui/` nie importuje `kovalo.cost`.
- Kwoty w UI to szacunek „gdyby płacić” (darmowy tier = 0 USD), aktualny na `cost.our_prices_checked_on`; nieznany model daje „brak ceny”, a nie zgadywanie.
- Menedżer nie widzi już 409 po restarcie `api` w trakcie tury; przerwana odpowiedź przepada (trzeba zadać pytanie ponownie). Gdyby `api` miało kiedyś kilka procesów, zamknięcie wymaga blokady między procesami (np. blokady doradczej w Postgresie).
- Dashboard trzeba przegenerować po każdym nowym raporcie; `--check` w CI pilnuje zgodności `dashboard.md` z `dashboard.json` i `dashboard.yaml`, ale nie z JSON-ami poza gitem (to sprawdza `make dashboard` lokalnie).
- ADR-0004 (tabela `agent_traces`) i ADR-0012 (tokeny) dostają dopisek odsyłający tutaj.

## Rozważane alternatywy

- Koszt liczony w UI z cennika z nowego endpointu: reguła cenowa w UI i drugi kontrakt; koszt w stanie grafu: zmiana stanu 1:1 z kursu dla wartości pochodnej.
- Dokończenie przerwanej tury (`invoke(None)`): zużywa limit na odpowiedź, na którą nikt nie czeka; skan wątków przy starcie: wymaga SQL-a na tabelach LangGraph i rejestru, którego nie ma.
- Dashboard w Chainlicie albo jako endpoint API: nowa usługa lub strona dla danych, które są statyczne; parsowanie raportów Markdown: kruche; commitowanie JSON-ów raportów: duże pliki z surowymi rekordami.
- Prawdziwe tokeny w SSE (`stream_mode="messages"`): częściowy tekst z modelu, który zawiódł przed fallbackiem, brak tokenów przy trafieniu w cache, nowe zdarzenie w kontrakcie; zysk mały, bo odpowiedź agenta jest krótka i przychodzi po krokach.
- Slajdy (Marp w repozytorium albo Keynote/PowerPoint): odrzucone przez użytkownika, pokaz to sama aplikacja. Nagranie GIF do README (ffmpeg, `make demo-gif`): zaplanowane (U8 = A), potem odrzucone przez użytkownika. Tryb demo `api` z cache LLM-a: nowy kontrakt dla jednego pokazu.
- Lista rozmów przez warstwę danych Chainlita (SQLAlchemy na tym samym Postgresie + logowanie hasłem + `on_chat_resume` z odtworzeniem pauz): realna, ale to osobny etap pracy bez wpływu na wymagania kursu.

## Wynik F6

Przegląd R1 z 2026-11-07 ([contracts/f6.md §15](../contracts/f6.md)):
- **Licznik kosztów:** w UI potwierdzony przez użytkownika: P1 → 7296,00 PLN z linią „Tura: 4 wywołania (gemini-3.6-flash, gemini-3.5-flash-lite), 8311 tokenów wejścia, 650 wyjścia. Koszt: 0 USD w darmowym tierze; po cenach płatnych 0,44 centa (Gemini) / 0,92 centa (cennik kursu). Rozmowa: 1 tura, 0,44 / 0,92 centa.”; fallback widać w nazwach modeli; wznowienie pauzy bez linii. Przez API: `usage.cost.ours_usd` = 0,00047 USD dla 501 / 129 tokenów, zgodnie z cennikiem.
- **Przerwana tura:** po `docker compose kill api` w trakcie P1 (po 3 krokach narzędzi) i restarcie: wątek bez pauzy, następna wiadomość 200 z dwiema odpowiedziami (komunikat o przerwanej turze + odpowiedź), WARNING w logu. Na `PostgresSaver` checkpoint zamknięcia ma `source == "update"` i `metadata["writes"] == None`, więc test integracyjny sprawdza status, błąd i `next`, a nie nazwę węzła.
- **Dashboard:** `reports/dashboard.{yaml,md,json}` w gicie, `--check` w CI, liczby zgodne z 06-evaluation (test spójności). Odstępstwa addytywne od kontraktu (pola `RunCalls.run_id` i `origin`, parametr `warn=`, ręczne pola zawsze z `dashboard.yaml`, koszt E1/E2 z cen w configu) przyjęte w R1.
- Zależności modułów zgodne z §2.2 kontraktu (testy architektury zielone); 1442 testy bez LLM-a, `-m integration` 25/25.
- Zużycie Gemini w F6: około 9 wywołań (plan zakładał około 40 razem z nagraniem).

### Aktualizacja po klikaniu UI (2026-11-07)

Zmiany w UI po przeglądzie R1, na prośbę użytkownika (bez zmian w API i kontrakcie odpowiedzi):
- **Koszt tury jako panel boczny** zamiast linii na końcu wiadomości: element `cl.Text(display="side")` „Koszt tury” (numerowany w rozmowie: „Koszt tury 2”…) obok „Ścieżki dowodowej”, link w linii „Szczegóły: …” ostatniej wiadomości tury. Treść to lista: tura (modele), tokeny, koszt tury (0 USD w darmowym tierze, ceny płatne Gemini / cennik kursu), suma rozmowy.
- **Panel boczny tylko po kliknięciu:** Chainlit 2.12 otwiera panel przy każdej zmianie elementów bocznych i nie ma na to opcji w configu; `public/kovalo.js` zamyka panel, gdy otworzył się bez kliknięcia linku `a.element-link` w ostatnich 1,5 s.
- **Wygląd:** `public/kovalo.css` zastępuje biały kursor ładowania trzema kropkami w kolorze motywu; kroki narzędzi mają ikony lucide (`metadata.icon`: `file-search`, `database`, `calculator`, domyślnie `wrench`) zamiast pustych szarych awatarów.
- **Tylko tekst:** wysyłanie plików wyłączone (`features.spontaneous_file_upload`); wiadomość bez tekstu dostaje podpowiedź `NO_TEXT` zamiast angielskiego błędu 422 z API.
- **MCP Inspector 2.x:** `make mcp-inspector` i `make mcp-inspector-http` z `--protocol-era auto` (w domyślnym trybie „legacy” zapytania wisiały jako PENDING).
