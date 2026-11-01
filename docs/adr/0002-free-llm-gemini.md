# ADR-0002: Darmowy LLM: Gemini Flash z fallbackiem na Groq i Ollamę

- Status: przyjęty
- Data: 2026-11-01

## Kontekst
Model referencyjny kursu (OpenAI `gpt-5.4-mini`) jest płatny. Wymóg projektu: darmowy klucz API. Potrzebujemy structured output (`understand`, odpowiedź ze schematem, sędzia), tool calling (agent) i dobrego polskiego.

## Decyzja
Domyślnie **Gemini Flash** przez `langchain-google-genai` (klucz z Google AI Studio, darmowy tier). Wybór modelu przez `init_chat_model` i `config.yaml`. Fallback `with_fallbacks` na **Groq** (`llama-3.3-70b-versatile` albo inny model z tool calling dostępny w darmowym tierze), awaryjnie **Ollama** lokalnie (profil `local-llm`). Rate limiter LangChaina poniżej limitu RPM.

## Konsekwencje
- Wyniki ewaluacji będą się różnić od referencyjnych; raporty pokazują oba.
- Limity dzienne darmowego tieru wymagają cache odpowiedzi w ewaluacjach i oszczędnego uruchamiania.
- Google może wykorzystywać dane z darmowego tieru; dane Kovalo są fikcyjne.
- Dokładne limity i nazwy modeli sprawdzamy w AI Studio przy F0. Zewnętrzne źródła z 2026 r. podają je niespójnie.

## Rozważane alternatywy
- Groq jako domyślny: szybki, ale słabszy polski i structured output, niski limit tokenów na minutę.
- Ollama jako domyślny: w pełni offline, ale małe modele często psują structured output (kurs to potwierdza dla `gemma3:4b`).
- OpenRouter `:free`: niestabilna dostępność modeli.
