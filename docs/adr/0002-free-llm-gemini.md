# ADR-0002: Darmowy LLM: Gemini Flash

- Status: przyjęty; kolejność modeli zmieniona przez [ADR-0006](0006-config-and-llm-factory.md) (od F2 model główny to Flash-Lite, a Flash jest fallbackiem: darmowy tier Flash ma 20 zapytań dziennie)
- Data: 2026-11-01

## Kontekst
Model referencyjny kursu (OpenAI `gpt-5.4-mini`) jest płatny. Wymóg projektu: darmowy klucz API. Potrzebujemy structured output (`understand`, odpowiedź ze schematem, sędzia), tool calling (agent) i dobrego polskiego.

## Decyzja
Domyślnie **Gemini Flash** przez `langchain-google-genai` (klucz z Google AI Studio, darmowy tier). Wybór modelu przez `init_chat_model` i `config.yaml`. Jedyny dostawca to Google: fallback `with_fallbacks` na **Gemini Flash-Lite** (ten sam klucz, osobny limit na model). Bez innych dostawców, żeby nie mnożyć kluczy, kontenerów i różnic w zachowaniu modeli. Rate limiter LangChaina poniżej limitu RPM.

## Konsekwencje
- Wyniki ewaluacji będą się różnić od referencyjnych; raporty pokazują oba.
- Limity dzienne darmowego tieru wymagają cache odpowiedzi w ewaluacjach i oszczędnego uruchamiania.
- Google może wykorzystywać dane z darmowego tieru; dane Kovalo są fikcyjne.
- Dokładne limity i nazwy modeli sprawdzamy w AI Studio przy F0. Zewnętrzne źródła z 2026 r. podają je niespójnie. Wynik: [ADR-0006](0006-config-and-llm-factory.md).

## Rozważane alternatywy
- Groq jako domyślny: szybki, ale słabszy polski i structured output, niski limit tokenów na minutę.
- Ollama jako domyślny: w pełni offline, ale małe modele często psują structured output (kurs to potwierdza dla `gemma3:4b`).
- OpenRouter `:free`: niestabilna dostępność modeli.
