---
name: run-eval
description: Uruchamianie i interpretacja ewaluacji Kovalo Decision Copilot (E1 retrieval i koszty, E2 RAG na 4 poziomach z sędzią LLM, E3 dialogi workflow, E4 trace agenta, E5 cały asystent) z oszczędzaniem limitów darmowego tieru Gemini i porównaniem z przebiegami referencyjnymi kursu.
---

# Ewaluacja

Metryki, cele i zasady: `docs/06-evaluation.md`.

## Uruchamianie
- Lokalnie: `make eval-local STAGE=e1 ARGS="..."` (= `uv run --env-file .env kovalo eval e1 ...`).
- W Dockerze: `make eval STAGE=e1 ARGS="--backend pgvector"` (= `docker compose run --rm api kovalo eval e1 ...`, raport w `reports/`).
- E1 (F1): `kovalo eval e1 [--candidates] [--model NAME]... [--backend memory|pgvector] [--with-llm] [--out DIR]`; opis opcji i kodów wyjścia: `docs/contracts/f1.md` §7.2. Porównanie kandydatów (`--candidates`) tylko lokalnie (pobiera ok. 5,8 GB modeli). `--with-llm` = jedno wywołanie Gemini.
- E2 (F2): `kovalo eval e2 [--with-llm] [--variants mp1,ctx,hybrid] [--llm-side ours|reference] [--ids M03,M10] [--no-judge] [--no-cache | --cache-only] [--backend memory|pgvector] [--plan]`; opis i kody wyjścia: `docs/contracts/f2.md` §8.7. Bez `--with-llm` zero wywołań LLM-a; `--plan` wypisuje budżet i kończy.
- Najpierw tryb deterministyczny tam, gdzie istnieje (E3: odtworzenie z idealnym understand; E1: bez `--with-llm`).
- Testy na żywo: `uv run --env-file .env pytest -m live tests/live/test_e1_live.py` (jedno wywołanie); samo `-m live` uruchamia wszystkie testy na żywo (w F1 dwa wywołania). Szczegóły: `docs/06-evaluation.md`, „Testy z prawdziwymi usługami”.

## Oszczędzanie darmowego tieru
- Cache LLM-a (`JsonFileLlmCache` w `paths.llm_cache_dir`, domyślnie `.cache/llm/`, ADR-0008) jest domyślnie włączony; `--no-cache` tylko do pomiaru zmienności, `--cache-only` gwarantuje zero wywołań sieciowych (brak w cache = rekord z błędem).
- `--ids` do powtórki pojedynczych przypadków; całość dopiero po poprawce.
- Przy błędzie 429: rate limiter i backoff działają same; jeśli wyczerpano dzienny limit, dokończ następnego dnia (ewaluacje wznawiają się z cache); model główny to Flash-Lite, fallback Flash ma tylko 20 zapytań dziennie (ADR-0006). Raport zapisuje faktyczny model odpowiedzi i sędziego.

## Czytanie raportu
1. Najpierw przejdź wyniki **przypadek po przypadku**, dopiero potem patrz na sumy.
2. Porównaj z kolumną „referencja kursu”. Różnica ≥ 2 na którymkolwiek poziomie wymaga wyjaśnienia.
3. Sprawdź warstwę: retrieval (brak dokumentu) → dowody (brak frazy) → odpowiedź → odwołania. Odmowa z `phrases_missing` oznacza problem z wyszukiwaniem, nie z modelem.
4. Rozbieżności ewaluatora (np. dowód sformułowany inaczej niż fraza) zapisz w sekcji „rozbieżności przejrzane ręcznie”. Nie poprawiaj wzorców w `data/eval/` bez uzasadnienia.
5. Wnioski: 3–5 punktów, każdy z identyfikatorem przypadku i proponowaną akcją.
