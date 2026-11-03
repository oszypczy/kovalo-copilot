# ADR-0010: API i UI w F3: odpowiedzi JSON, wątki bez rejestru, przyciski Chainlit z callbackami

- Status: przyjęty (2026-11-03, przegląd R1: [contracts/f3.md §18](../contracts/f3.md); odpowiedzi na U1–U3, U9 w §16.1; ręczne przeklikanie D2 w UI do potwierdzenia przez użytkownika)
- Data: 2026-11-03

## Kontekst
F3 ma udostępnić workflow przez API (`/threads/{id}/messages`, `/threads/{id}/resume`) i dać pierwszą wersję UI Chainlit, w której scenariusz D2 da się przeklikać. Szkic w 03-architecture §4 i FR-6.1 zakładał strumień SSE (`token`, `tool_start`, `tool_end`, `interrupt`, `final`), a FR-6.2 przyciski przez `cl.AskActionMessage`. Stan (2026-11-03): FastAPI 0.162 ma wbudowane `fastapi.sse.EventSourceResponse`; Chainlit 2.12: `cl.AskActionMessage(...).send()` czeka na kliknięcie najwyżej `timeout` sekund (domyślnie 90) i blokuje obsługę wiadomości, a `cl.Action(name, payload, label)` z `@cl.action_callback(name)` obsługuje kliknięcie bez blokowania.

W F3 żadna tura nie ma czego strumieniować token po tokenie: `understand` i odpowiedź RAG to structured output (cały obiekt naraz), a kroków agenta jeszcze nie ma. Pauza akceptacji może trwać dowolnie długo, bo stan jest w Postgresie.

## Decyzja
1. **API w F3 zwraca JSON**, jedna odpowiedź na żądanie: `TurnResult` (status, odpowiedzi tej tury, aktywna pauza, wniosek, decyzja, akceptacje, źródła, tokeny). SSE dokładamy w F4 (kroki agenta), z `TurnResult` jako ładunkiem zdarzenia `final`, więc kontrakt się nie łamie.
2. **Wątki bez rejestru:** `POST /threads` zwraca nowe UUID; checkpoint powstaje przy pierwszej wiadomości. `GET /threads/{id}` dla wątku bez checkpointu zwraca pusty stan (`status: null`), nie 404. Jedna instancja, jeden użytkownik (01-overview, „Poza zakresem”).
3. **Wznowienie** przyjmuje wartość z przycisku (`value`) albo tekst wpisany przez człowieka (`text`, zamieniany po stronie API przez `text_pl.human_answer`), dokładnie jedno z nich. Opcjonalne `pause_ref` (`type`, `role`/`field`, `revision`) chroni przed nieaktualnym przyciskiem: niezgodność z bieżącą pauzą daje 409 `stale_pause`.
4. **Błędy jako 409** z kodem: `thread_paused` (nowa wiadomość w pauzie, FR-3.6, z ładunkiem pauzy), `not_paused`, `thread_busy` (druga tura w tym samym wątku w trakcie), `stale_pause`, `thread_stuck`. `503 copilot_unavailable`, gdy nie da się zbudować grafu (brak klucza LLM-a, Postgres niedostępny); `/health` działa dalej.
5. **UI:** przyciski akceptacji to `cl.Action` w wiadomości z pauzą i `@cl.action_callback("resume")` (bez `AskActionMessage`). Gdy wątek czeka na pauzę, tekst wpisany w czacie idzie do `/resume` jako `text` (jak `chat.py` kursu: marża „7,5”, „zatwierdź”), a nie do `/messages`. UI nie formatuje logiki biznesowej: opis wniosku (`summary`) i treść pytań przychodzą z API. Źródła z odpowiedzi RAG jako `cl.Text` z treścią chunku.
6. **UI rozmawia z API tylko przez HTTP** (`httpx.AsyncClient`, `services.api_url`, limit `services.ui_api_timeout_s`) i ma własne, minimalne modele odpowiedzi; zgodność z modelami API pilnuje test kontraktu.

## Konsekwencje
- FR-6.1 (SSE) i FR-6.2 (strumieniowanie, `cl.Step`) są w F3 spełnione częściowo; reszta w F4. FR-6.2 zmienia mechanizm przycisków z `AskActionMessage` na akcje z callbackami.
- Odświeżenie strony w Chainlit zaczyna nową sesję i nowy wątek; poprzedni wątek w pauzie zostaje w Postgresie bez widoku w UI (lista rozmów to kandydat na F6).
- Brak strumieniowania: użytkownik czeka na całą turę (zwykle 1–3 s; z fallbackiem i limiterem do ok. 60 s), więc UI pokazuje wskaźnik pracy i ma limit 180 s.
- Port hosta `api` dalej z `API_PORT` (u nas 8081); wewnątrz sieci compose UI łączy się z `http://api:8080`.

## Rozważane alternatywy
- SSE od razu (zgodnie ze szkicem): więcej kodu i testów asynchronicznych bez korzyści w F3; jedyne zdarzenia byłyby `interrupt` i `final`.
- `cl.AskActionMessage` (jak w FR-6.2): prostszy kod jednej tury, ale blokuje obsługę wiadomości do kliknięcia, kończy się po `timeout`, a po przekroczeniu UI traci przyciski, choć pauza trwa w Postgresie.
- `cl.AskUserMessage` do marży: ten sam problem z blokadą i limitem czasu; wpis w zwykłym polu czatu robi to samo bez blokady.
- Rejestr wątków w tabeli: potrzebny dopiero przy liście rozmów i wielu użytkownikach.
