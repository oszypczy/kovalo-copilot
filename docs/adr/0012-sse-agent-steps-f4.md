# ADR-0012: Strumień SSE tury, kroki agenta w UI, tura niezależna od połączenia, 503 przy utracie Postgresa

- Status: przyjęty (2026-11-03; decyzje użytkownika U9 i U12 w [contracts/f4.md §16.3](../contracts/f4.md); zweryfikowany w compose i UI, §18.6)
- Data: 2026-11-03

## Kontekst
ADR-0010 odłożył strumień na F4: `POST /threads/{id}/messages` i `/resume` jako SSE, `TurnResult` jako ładunek zdarzenia `final`, `GET /threads/{id}/trace`, kroki agenta w Chainlit (`cl.Step`). W F4 tura z agentem trwa kilka–kilkadziesiąt sekund (3–5 wywołań modelu przy limiterze 0,25 zapytania/s), więc użytkownik powinien widzieć wywołania narzędzi na bieżąco. Stan (2026-11-03): FastAPI 0.162 ma `fastapi.sse.EventSourceResponse` i `ServerSentEvent` (działa z `POST`); Chainlit 2.12 ma `cl.Step(name, type="tool", parent_id, show_input, language)` z `send()` i `update()`; `httpx-sse` 0.4.3 jest już w `uv.lock` (zależność `mcp`). Przegląd F3 zostawił dwie uwagi o serwisie: R1-2 (błąd Postgresa w trakcie żądania daje 500 zamiast 503) i R1-3 (słownik blokad wątków rośnie bez końca).

Graf jest synchroniczny (ADR-0009 pkt 6). Jeśli tura biegnie w generatorze odpowiedzi HTTP, rozłączenie klienta zamyka generator w środku węzła, a checkpoint zostaje z `next` bez pauzy, czyli wątek na stałe `thread_stuck` (P6 z F3).

## Decyzja
1. **Osobne ścieżki strumieniowe** obok niezmienionych JSON-owych: `POST /threads/{id}/messages/stream` i `POST /threads/{id}/resume/stream` (te same ciała co wersje JSON), odpowiedź `text/event-stream`. Błędy wykrywalne przed startem tury (409, 422, 503) wracają jako zwykły JSON `ApiError` z kodem HTTP, zanim zacznie się strumień.
2. **Zdarzenia:** `tool_start` (wywołanie narzędzia: `call_id`, `parent_call_id`, `agent`, `tool`, `arguments`, `origin`), `tool_end` (`call_id`, `status`, `brief`, `result`, `error`), `final` (`TurnResult` bez zmian pól poza nowymi z F4) i `error` (nieoczekiwany wyjątek po starcie strumienia: `error`, `detail`). Pauza jest częścią `final.pending` (osobne zdarzenie `interrupt` niepotrzebne). Bez strumieniowania tokenów w F4.
3. **Tura niezależna od połączenia:** serwis uruchamia `graph.stream(..., stream_mode=["custom", "updates"], version="v2", durability="sync")` w wątku roboczym, który zawsze dobiega do końca; zdarzenia idą przez kolejkę do generatora odpowiedzi. Rozłączenie klienta przestaje tylko czytać kolejkę; blokada tury jest zwalniana na końcu pracy wątku. Wersje JSON (`send_message`, `resume`) czytają ten sam strumień do `final`, więc E3 i API dalej idą jedną ścieżką.
4. **Zdarzenia agenta** wysyła węzeł `answer_documents` przez `get_stream_writer()` (`stream_mode="custom"`), z wywołania zwrotnego `run_question(..., on_event=...)`; `agents/` nie zna LangGraph-owego strumienia zewnętrznego grafu.
5. **`GET /threads/{id}/trace`** zwraca ostatni trace agenta w wątku (z historii checkpointów) i jego ścieżkę dowodową w formacie `show_trace` kursu; wątek bez tury agenta → `trace: null` (200).
6. **UI:** klient strumienia w `ui/client.py` (`httpx` + `httpx-sse`, jawna zależność). Każde `tool_start` to `cl.Step(type="tool")` pod krokiem `on_message` tury (`cl.context.current_run.id`; zdarzenie składowe pod krokiem rodzica), `tool_end` uzupełnia wynik; po `final` jedna wiadomość z odpowiedzią, przyciskami pauzy i źródłami jak w F3, a ścieżka dowodowa jako element boczny `cl.Text(display="side")` podlinkowany linią „Szczegóły: Ścieżka dowodowa”. (Doprecyzowanie z implementacji, Chainlit 2.12: dzieci wiadomości renderują się pod jej treścią, a krok ma stały nagłówek „{used} {name}”, więc ścieżka dowodowa nie jest krokiem.) Interfejs po polsku: `.chainlit/translations/pl-PL.json`, `language = "pl-PL"`. Bez logiki biznesowej w UI: `brief` i ścieżka dowodowa przychodzą z API.
7. **R1-2:** `workflow/checkpoint.py` udostępnia krotkę błędów połączenia; serwis zamienia je w trakcie tury i odczytu stanu na `CheckpointerUnavailableError`, a API na 503 `copilot_unavailable` (bez DSN). **R1-3:** wpis blokady wątku znika, gdy nikt go nie trzyma (licznik użytkowników pod blokadą słownika).

## Konsekwencje
- FR-6.1 (SSE, `/trace`) i FR-6.2 (kroki agenta jako `Step`) spełnione; „strumieniowanie odpowiedzi” oznacza w F4 strumień kroków i wyniku tury, nie tokenów (tokeny: kandydat na F6).
- Odpowiedź tury, której klient nie odebrał (rozłączenie), jest w checkpoincie: `GET /threads/{id}` ją pokazuje, a następna wiadomość działa normalnie.
- Jeden proces `api` (blokady w pamięci) zostaje; wątek roboczy na turę mieści się w puli wątków serwera.
- Testy API strumienia czytają SSE z `TestClient` (`client.stream("POST", ...)`); testy UI używają `httpx.MockTransport` z ciałem `text/event-stream`.

## Rozważane alternatywy
- Negocjacja przez nagłówek `Accept` na tych samych ścieżkach: jedna ścieżka, ale dwa typy odpowiedzi w jednym endpoincie i niejasny OpenAPI; osobne ścieżki są prostsze w testach.
- Tura w generatorze odpowiedzi (bez wątku roboczego): mniej kodu, ale rozłączenie zostawia wątek `thread_stuck`.
- Strumieniowanie tokenów odpowiedzi agenta (`stream_mode="messages"` z podgrafu): odpowiedź końcowa przychodzi po wszystkich narzędziach i jest krótka, a tokeny z podgrafu wywołanego w węźle wymagają przekazania konfiguracji i filtrowania tur z wywołaniami narzędzi.
- Asynchroniczny graf i `AsyncPostgresSaver`: nadal bez korzyści przy synchronicznych modelach i narzędziach (ADR-0009).
- Własny parser SSE w UI zamiast `httpx-sse`: kilkadziesiąt linii do utrzymania; biblioteka już jest w zależnościach pośrednich.

## Aktualizacja (F6)
Tokeny w SSE („kandydat na F6”) rozstrzygnięte w [ADR-0015](0015-polish-f6.md): nie strumieniujemy tokenów odpowiedzi. FR-6.2 „strumieniowanie odpowiedzi” oznacza strumień kroków agenta i wyniku tury. Zdarzenie `final` niesie od F6 koszt tury (`TurnResult.usage.by_model`, `usage.cost`), zmiana addytywna bez nowych zdarzeń. Przerwana tura nie kończy się już 409 `thread_stuck` przy następnej wiadomości: serwis ją zamyka (ADR-0015).
