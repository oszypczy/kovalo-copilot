---
name: run-eval
description: Uruchamianie i interpretacja ewaluacji Kovalo Decision Copilot (E1 retrieval i koszty, E2 RAG na 4 poziomach z sędzią LLM, E3 dialogi workflow, E4 trace agenta, E5 cały asystent) z oszczędzaniem limitów darmowego tieru Gemini i porównaniem z przebiegami referencyjnymi kursu.
---

# Ewaluacja

Metryki, cele i zasady: `docs/06-evaluation.md`.

## Uruchamianie
- Lokalnie: `uv run python -m kovalo.eval <e1|e2|e3|e4|e5> [--limit N] [--ids M01,M03] [--no-cache]`
- W Dockerze: `make eval STAGE=e2` (profil `eval`, raport w `reports/`).
- Najpierw tryb deterministyczny tam, gdzie istnieje (E3: odtworzenie z idealnym understand; E1: bez LLM-a).

## Oszczędzanie darmowego tieru
- Cache LLM-a (`SQLiteCache` w `.cache/llm.sqlite`) jest domyślnie włączony; `--no-cache` tylko do pomiaru zmienności.
- `--ids` do powtórki pojedynczych przypadków; całość dopiero po poprawce.
- Przy błędzie 429: rate limiter i backoff działają same; jeśli wyczerpano dzienny limit, przełącz `llm.provider` na fallback i zapisz to w raporcie.

## Czytanie raportu
1. Najpierw przejdź wyniki **przypadek po przypadku**, dopiero potem patrz na sumy.
2. Porównaj z kolumną „referencja kursu”. Różnica ≥ 2 na którymkolwiek poziomie wymaga wyjaśnienia.
3. Sprawdź warstwę: retrieval (brak dokumentu) → dowody (brak frazy) → odpowiedź → odwołania. Odmowa z `phrases_missing` oznacza problem z wyszukiwaniem, nie z modelem.
4. Rozbieżności ewaluatora (np. dowód sformułowany inaczej niż fraza) zapisz w sekcji „rozbieżności przejrzane ręcznie”. Nie poprawiaj wzorców w `data/eval/` bez uzasadnienia.
5. Wnioski: 3–5 punktów, każdy z identyfikatorem przypadku i proponowaną akcją.
