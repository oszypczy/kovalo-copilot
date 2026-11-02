# ADR-0006: Kontrakt config.yaml i fabryki modeli (F0)

- Status: przyjęty
- Data: 2026-11-01

## Kontekst
ADR-0002 wybrał Gemini Flash z fallbackiem na Flash-Lite, ale nazwy modeli i ustawienia zostawił do sprawdzenia w F0. Szkic w [04-tech-stack.md](../04-tech-stack.md) używał aliasów `gemini-flash-latest` i `temperature: 0`. `config.yaml` kursu ma sekcję `llm` dla OpenAI/OpenRouter/Ollama i nie ma miejsca na embeddingi lokalne, bazę ani adresy usług.

Stan na 2026-11-01 (ai.google.dev, strony models, pricing, rate-limits, gemini-3):
- stabilne modele z darmowym tierem: `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`; Google poleca 3.8 Flash i 3.5 Flash-Lite do nowych projektów; serie 2.0 są wyłączone, a 2.5 dostępne tylko dla dotychczasowych użytkowników;
- aliasy `-latest` mogą wskazać nowy model bez ostrzeżenia;
- limity RPM/RPD darmowego tieru są widoczne tylko w AI Studio, per projekt;
- dla Gemini 3 Google zdecydowanie zaleca domyślną temperaturę 1.0; niższa może powodować zapętlenia; `thinking_level` domyślnie `high`;
- `langchain-google-genai` 4.4 mapuje błędy HTTP na klasy `langchain_core.exceptions.Model*Error`.

## Decyzja
1. **Modele:** główny `gemini-3.6-flash`, fallback `gemini-3.5-flash-lite` (zmiana z pierwotnego `gemini-3.8-flash`, zob. „Pomiar 2026-11-02”). Konkretne wersje zamiast aliasów `-latest`, żeby ewaluacje dało się powtórzyć. Zmiana modelu to zmiana w `config.yaml` i nowy przebieg ewaluacji.
2. **Parametry:** `temperature: null` (domyślna dostawcy), `thinking_level: low` (mniej tokenów myślenia w limicie darmowego tieru i krótszy czas odpowiedzi), `max_retries: 1` (bez ponowień w SDK: po przekroczeniu limitu od razu fallback), `timeout_s: 60`.
3. **Rate limiter:** `InMemoryRateLimiter`, jeden na model w procesie: 0,15 req/s dla Flash, 0,25 req/s dla Flash-Lite, `max_bucket_size: 1`.
4. **Fallback:** tylko przy błędach dostępności (`ModelRateLimitError`, `ModelAPIError`, `ModelNotFoundError`, `ModelConnectionError`, `ModelTimeoutError`, timeout i błąd transportu `httpx`). Bez fallbacku przy 400/401/403, przepełnieniu kontekstu i błędach parsowania.
5. **Układ `config.yaml`:** sekcje PINNED i `corpus` kopiujemy 1:1 z kursu (pilnuje tego test). Sekcję `llm` kursu zastępujemy naszą (w kursie to „TWÓJ WYBÓR”, nie PINNED). Do `embeddings` dopisujemy `provider` i `local`, nie zmieniając kluczy PINNED. Dodajemy sekcje `db` i `services`. Katalogi wyników `paths.mini_project_N_output_dir` wskazują `reports/mpN`.
6. **Konfiguracja w kodzie:** `pydantic-settings` ze źródłem YAML; nadpisania z env z prefiksem `KOVALO_` i separatorem `__`. Sekrety pod nazwami z `.env.example`. Brak klucza Gemini nie blokuje startu usług; błąd zgłasza dopiero fabryka modeli.

Szczegóły kontraktu: [contracts/f0.md](../contracts/f0.md).

## Konsekwencje
- Temperatura 1.0 sprawia, że odpowiedzi nie są deterministyczne. Testy bez LLM-a tego nie dotyczą, a ewaluacje na żywo raportują wyniki z kilku przebiegów albo z cache.
- `thinking_level: low` może obniżyć jakość trudnego rozumowania. Jeśli `understand` albo agent wypadną słabo w E3/E4, podnosimy go per cel (`llm.purposes`), bez zmian w kodzie.
- `max_retries: 1`: pierwszy pomiar (F0) pokazał, że `gemini-3.8-flash` w darmowym tierze potrafi nie odpowiedzieć w 60 s; z `max_retries: 2` SDK ponawiał wywołanie i odpowiedź z fallbacku przychodziła po 131 s. Bez ponowień najgorszy przypadek to ok. 60 s + czas Flash-Lite. Przejściowe 429/503 nie są ponawiane na tym samym modelu, tylko od razu idą do fallbacku. Czas odpowiedzi Flasha mierzymy ponownie w F1.
- Wycofanie modelu przez Google (404) przełącza na fallback, a naprawa to jedna linia w configu.
- Limiter działa w obrębie procesu, więc `api` i jednorazowy `kovalo eval` mają osobne budżety. Ewaluacji nie uruchamiamy równolegle z intensywnym używaniem UI.
- `create_agent` wymaga `BaseChatModel`, więc agent (F4) bierze fallback przez `ModelFallbackMiddleware`, a nie przez `with_fallbacks`.

## Rozważane alternatywy
- Aliasy `gemini-flash-latest` / `gemini-flash-lite-latest`: zero utrzymania, ale model może się zmienić między dwoma ewaluacjami bez naszej wiedzy.
- `gemini-3.8-flash` jako główny (pierwotny wybór, polecany przez Google): w darmowym tierze nie odpowiada, zob. pomiar niżej.
- `gemini-3.5-flash` jako główny: odpowiada równie szybko jak 3.6, ale jest starszy; zostaje kandydatem na drugi fallback.
- `temperature: 0` jak w kursie: lepsza powtarzalność, ale wbrew wyraźnemu zaleceniu Google dla Gemini 3 (ryzyko zapętleń przy rozumowaniu).
- Fallback przy każdym `Exception` (domyślne w `with_fallbacks`): maskowałby błędy schematów i promptów, które fallback powtórzyłby tak samo.

## Pomiar 2026-11-02

Jedno wywołanie na model (prompt jednozdaniowy, `thinking_level: low`, `timeout: 60`, `max_retries: 1`), klucz z darmowego tieru:

| Model | Wynik | Czas |
|---|---|---|
| `gemini-3.8-flash` | `504 DEADLINE_EXCEEDED` (także przy limicie 180 s) | 60–186 s |
| `gemini-3.7-flash` | `504 DEADLINE_EXCEEDED` | 59 s |
| `gemini-3.6-flash` | odpowiedź poprawna (4 z 4 prób) | 2,3–2,9 s |
| `gemini-3.5-flash` | odpowiedź poprawna | 2,3 s |
| `gemini-2.5-flash` | `404 NOT_FOUND` (niedostępny dla nowych użytkowników) | – |
| `gemini-3.5-flash-lite` (fallback) | odpowiedź poprawna | ok. 10 s |

Wniosek: główny model to `gemini-3.6-flash`. Do `gemini-3.8-flash` wracamy, gdy zacznie odpowiadać w darmowym tierze (sprawdzamy przy każdym etapie z ewaluacją na żywo).
