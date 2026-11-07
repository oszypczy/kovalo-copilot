# Dashboard ewaluacji

Plik generowany poleceniem `uv run kovalo eval dashboard` (sprawdzenie: `uv run kovalo eval dashboard --check`) z `reports/dashboard.yaml`, JSON-ów raportów E1–E5 i poprzedniego `reports/dashboard.json`. Nie poprawiaj go ręcznie: zmiany wpisuj w `dashboard.yaml` albo w raportach etapów. Metryki i cele: [docs/06-evaluation.md](../docs/06-evaluation.md) (06-evaluation).

Stan na: 2026-11-04 (data najnowszego raportu)

## Podsumowanie

| Etap | Data | Model | Najważniejszy wynik (kurs → nasz) | Cel spełniony | Raport |
|---|---|---|---|---|---|
| E1 | 2026-11-02 | gemini-3.6-flash | E1: hit@1 / hit@3 na Q01–Q10 0,9 / 0,9 → 0,9 / 1,0 (intfloat/multilingual-e5-large); M03 trafione; koszt kursu $17,78 / $69,86 odtworzony | tak | [mp1/2026-11-02-e1.md](mp1/2026-11-02-e1.md) |
| E2 | 2026-11-02 | gemini-3.5-flash-lite | E2: odpowiedzi 8/10 → 9/10 / 9/10, sędzia 8/8 → 9/9 / 9/9 (ctx / hybrid); poziomy 1–2 jak kurs | tak | [mp2/2026-11-02-e2.md](mp2/2026-11-02-e2.md) |
| E3 | 2026-11-03 | gemini-3.5-flash-lite | E3: na żywo kierowanie 13/13 → 13/13, pola 11/11 → 10/11; odtworzenie 100%; 0,56 centa | tak | [mp3/2026-11-03-e3.md](mp3/2026-11-03-e3.md) |
| E4 | 2026-11-03 | gemini-3.5-flash-lite | E4: na żywo narzędzia 8/9 → 8/9, fakty 8/9 → 8/9 (cel 9/9 niespełniony: B1); T1 67% / 0% → 100% / 100%; 2,49 centa | częściowo | [mp4/2026-11-03-e4.md](mp4/2026-11-03-e4.md) |
| E5 | 2026-11-04 | gemini-3.5-flash-lite | E5: na żywo 25/25 ścieżki przez MCP, jak kurs; 30 wywołań, 2,03 centa | tak | [mp5/2026-11-04-e5.md](mp5/2026-11-04-e5.md) |

## E1: wyszukiwanie i koszty (MP1)

Data: 2026-11-02 · Model: gemini-3.6-flash · Embeddingi: intfloat/multilingual-e5-large

E1: hit@1 / hit@3 na Q01–Q10 0,9 / 0,9 → 0,9 / 1,0 (intfloat/multilingual-e5-large); M03 trafione; koszt kursu $17,78 / $69,86 odtworzony.

| Metryka | Kurs | Bez LLM-a / odtworzenie | Na żywo | Cel | Spełniony |
|---|---|---|---|---|---|
| hit@1 (MP1, Q01–Q10) | 0,9 | 0,9 | – | ≥ 0,8 | tak |
| hit@3 (MP1, Q01–Q10) | 0,9 | 1,0 | – | ≥ 0,9 | tak |
| hit@1 (MP2, M01–M10) | 0,9 | 0,9 | – | ≥ 0,8 | tak |
| hit@3 (MP2, M01–M10) | 0,9 | 1,0 | – | ≥ 0,9 | tak |
| M03 (SUP-114) w top-3 | nietrafione | trafione | – | trafione | tak |
| Koszt miesięczny: retrieval / cały korpus (ceny referencyjne kursu) | $17,78 / $69,86 | $17,78 / $69,86 | – | równy kursowi ($17,78 / $69,86) | tak |
| Koszt miesięczny po cenach gemini-3.6-flash (płatny tier Gemini) | – | $15,58 / $67,65 | – | – | – |

- Wywołania LLM-a: plan 1, sieciowo 1, z cache 0, nieudane 0, 12,3 s (przebieg 20261102-104911; źródło: raport)
- Tokeny i koszt: tokeny 804 / 959; 0,42 centa po cenach płatnego tieru Gemini; 0,49 centa po cenach kursu
- Raport: [mp1/2026-11-02-e1.md](mp1/2026-11-02-e1.md)
- Uwaga: Koszt miesięczny liczony dla tokenów przebiegu kursu (wiersz „referencja kursu” raportu E1) po cenach gemini-3.6-flash, modelu z pomiaru E1. Od F2 model odpowiedzi i cost.our_model to gemini-3.5-flash-lite: $9,17 / $30,00 (FR-1.2, ADR-0006); model embeddingów z configu (intfloat/multilingual-e5-large) daje po cenach gemini-3.6-flash $15,66 / $67,65, i tę kwotę podaje 06-evaluation.
- Uwaga: Koszt E1 w scenariuszu to dolna granica: faktyczne wywołanie z myśleniem zużyło 959 tokenów wyjścia (w tym 714 myślenia) wobec 350 w scenariuszu kursu (raport E1, przebieg 20261102-104911). ADR-0006 podaje 677 tokenów: to starszy pomiar z F1 (06-evaluation podaje oba); raport E1 jest nowszy i to jego liczbę bierze dashboard.

## E2: odpowiedzi ze źródłami (MP2)

Data: 2026-11-02 · Model: gemini-3.5-flash-lite · Embeddingi: intfloat/multilingual-e5-large

E2: odpowiedzi 8/10 → 9/10 / 9/10, sędzia 8/8 → 9/9 / 9/9 (ctx / hybrid); poziomy 1–2 jak kurs.

| Metryka | Kurs | Bez LLM-a / odtworzenie | Na żywo | Cel | Spełniony |
|---|---|---|---|---|---|
| Poziomy 1–2, mp1: hit@1, hit@3, komplet dowodów, tokeny promptu | 9/10, 9/10, 9/10, 850,3 | 9/10, 10/10, 9/10, 874,6 | – | równe kursowi (check.py) | tak |
| Poziomy 1–2, ctx: hit@1, hit@3, komplet dowodów, tokeny promptu | 9/10, 9/10, 8/10, 610,4 | 10/10, 10/10, 9/10, 649,3 | – | równe kursowi (check.py) | tak |
| Poziomy 1–2, hybrid: hit@1, hit@3, komplet dowodów, tokeny promptu | – | 9/10, 10/10, 9/10, 638,5 | – | – | – |
| Poziom 1: znaleziono trafny dokument (ctx / hybrid) | 9/10 | – | 10/10 / 10/10 | ≥ referencja − 1 | tak |
| Poziom 2: komplet fraz dowodowych w kontekście (ctx / hybrid) | 8/10 | – | 9/10 / 9/10 | ≥ referencja − 1 | tak |
| Poziom 3: udzielone odpowiedzi (ctx / hybrid) | 8/10 | – | 9/10 / 9/10 | ≥ referencja − 1 | tak |
| Poziom 3: wszystkie kluczowe fakty (ctx / hybrid) | 8/8 | – | 9/9 / 9/9 | ≥ referencja − 1 | tak |
| Poziom 3: sędzia, odpowiedź poprawna (ctx / hybrid) | 8/8 | – | 9/9 / 9/9 | ≥ referencja − 1 | tak |
| Poziom 3: sędzia, odpowiedź oparta na kontekście (ctx / hybrid) | 8/8 | – | 9/9 / 9/9 | ≥ referencja − 1 | tak |
| Poziom 4: odwołania prawidłowe (ctx / hybrid) | 8/8 | – | 9/9 / 9/9 | ≥ referencja − 1 | tak |
| Poziom 4: odwołania do trafnego dokumentu (ctx / hybrid) | 8/8 | – | 9/9 / 9/9 | ≥ referencja − 1 | tak |
| Odmowy trafne (pytania spoza korpusu) (ctx / hybrid) | 2/2 | – | 2/2 / 2/2 | ≥ referencja − 1 | tak |

- Wywołania LLM-a: plan 48, sieciowo 0, z cache 42, nieudane 0 (przebieg 20261102-155513-ccd9ddd2; źródło: raport)
- Tokeny i koszt: tokeny 15 132 / 1776; 0,9 centa po cenach płatnego tieru Gemini; 1,93 centa po cenach kursu
- Raport: [mp2/2026-11-02-e2.md](mp2/2026-11-02-e2.md)
- Uwaga: JSON E2 to przeliczenie z cache (--with-llm --cache-only): 0 wywołań sieciowych, 42 z cache. Pierwszy przebieg na gemini-3.5-flash-lite (20261102-153627-87b0e6c8): 11 wywołań, 31 z cache (przegląd w raporcie E2).
- Uwaga: Wiersz hybrid (poziomy 1–2) bez kolumny „Kurs”: kurs nie ma wariantu hybrid. Z embeddingami referencyjnymi kursu hybrid daje 9/10, 10/10, 10/10, 631 tokenów (raport E2).
- Uwaga: Tokeny i koszt E2 to szacunek: średnie tokeny odpowiedzi z raportu razy liczba odpowiedzi (ctx i hybrid, bez sędziego).
- Uwaga: Sędzia to ten sam model co odpowiedzi: „sędzia 9/9” to górna granica (M09, ADR-0008); M04 to realne pudło wyszukiwania (pytanie PL, dokument EN).

## E3: workflow decyzyjny (MP3)

Data: 2026-11-03 · Model: gemini-3.5-flash-lite · Embeddingi: intfloat/multilingual-e5-large

E3: na żywo kierowanie 13/13 → 13/13, pola 11/11 → 10/11; odtworzenie 100%; 0,56 centa.

| Metryka | Kurs | Bez LLM-a / odtworzenie | Na żywo | Cel | Spełniony |
|---|---|---|---|---|---|
| Kierowanie (rodzaj wiadomości) | 13/13 | 13/13 | 13/13 | odtworzenie 100%; na żywo ≥ 12/13 | tak |
| Pola odczytane przez LLM | 11/11 | 11/11 | 10/11 | odtworzenie 100% | tak |
| Stan wniosku po aktualizacji | 8/8 | 8/8 | 8/8 | odtworzenie 100% | tak |
| Osoby zatwierdzające (polityka) | 11/11 | 11/11 | 11/11 | odtworzenie 100% | tak |
| Pauzy (typ, rola, kolejność) | 13/13 | 13/13 | 13/13 | odtworzenie 100% | tak |
| Status końcowy | 13/13 | 13/13 | 13/13 | odtworzenie 100% | tak |
| Błędy (nieudane tury) | 0 | 0 | 0 | 0 | tak |

- Wywołania LLM-a: plan 14, sieciowo 0, z cache 14, nieudane 0, 0,1 s (przebieg 20261103T081441-e6c75d13; źródło: raport)
- Tokeny i koszt: tokeny 6626 / 1439; 0,56 centa po cenach płatnego tieru Gemini; 1,14 centa po cenach kursu; kurs: 1,65 centa
- Raport: [mp3/2026-11-03-e3.md](mp3/2026-11-03-e3.md)
- Uwaga: JSON E3 to powtórka --live --cache-only (0 wywołań sieciowych, 14 z cache). Oryginalny przebieg na żywo: 14 wywołań (13 understand + 1 odpowiedź RAG), 64 s, bez fallbacku (contracts/f3.md, A9); identyfikatora tamtego przebiegu nie zapisano, więc bez first_live.
- Uwaga: D2-t2: na żywo pola 10/11, model nie przejął marży 10% z rozmowy; skutek zerowy (reports/mp3/review.yaml). Cel E3 dla pól to odtworzenie 100%.

## E4: agent z narzędziami (MP4)

Data: 2026-11-03 · Model: gemini-3.5-flash-lite · Embeddingi: intfloat/multilingual-e5-large

E4: na żywo narzędzia 8/9 → 8/9, fakty 8/9 → 8/9 (cel 9/9 niespełniony: B1); T1 67% / 0% → 100% / 100%; 2,49 centa.

| Metryka | Kurs | Bez LLM-a / odtworzenie | Na żywo | Cel | Spełniony |
|---|---|---|---|---|---|
| Test 1: pokrycie narzędzi (każde potrzebne wywołanie) | 8/9 | 8/9 | 8/9 | 9/9 | nie |
| Test 2: poprawne argumenty (gdzie liczono karę) | 3/3 | 3/3 | 3/3 | 3/3 | tak |
| Test 3: fakty (każdy kluczowy fakt w odpowiedzi) | 8/9 | 8/9 | 8/9 | 9/9 | nie |
| Test 4: liczby bez źródła (podane przez użytkownika) | 0 (0) | 0 (0) | 0 (0) | 0 | tak |
| Test 5: twierdzenia o zatwierdzeniu w odpowiedziach | 0/9 | 0/9 | 0/9 | 0/9 | tak |
| Zespół kursu (supervisor + 3 specjalistów): testy 1–5 | 8/9, 3/3, 8/9, 3 (2), 0/9 | – | – | – | – |
| T1: warunek rozwiązania umowy (narzędzia / fakty) | 67% / 0% | 67% / 0% | 100% / 100% | 100% / 100% | tak |
| Wywołania narzędzi (nieudane) | 21 (0) | 21 (0) | 21 (0) | – | – |

- Wywołania LLM-a: plan 22, sieciowo 0, z cache 24, nieudane 0, 0,1 s (przebieg 20261103T211153-c91b6a43; źródło: raport)
- Tokeny i koszt: tokeny 62 092 / 2509; 2,49 centa po cenach płatnego tieru Gemini; 5,79 centa po cenach kursu; kurs: 3,8 centa
- Raport: [mp4/2026-11-03-e4.md](mp4/2026-11-03-e4.md); powiązane: [mp4/2026-11-04-e4.md](mp4/2026-11-04-e4.md)
- Uwaga: B1: agent poprawnie odmówił zatwierdzenia, ale nie wywołał check_discount_policy (narzędzia 0%, fakty 50%), więc testy 1 i 3 są poniżej celu 9/9; agent_system bez zmian (U10 z F5).
- Uwaga: JSON E4 to powtórka --live --cache-only (0 wywołań sieciowych, 24 z cache). Oryginalny przebieg A10 szedł w dwóch dniach (--ids, --max-calls 20): 24 tury modelu, 0 błędów, bez fallbacku (contracts/f4.md, reports/mp4/review.yaml); liczby sieciowe obu części nie są zapisane w jednym przebiegu, więc bez first_live.

## E5: końcowy asystent przez MCP (MP5)

Data: 2026-11-04 · Model: gemini-3.5-flash-lite · Embeddingi: intfloat/multilingual-e5-large

E5: na żywo 25/25 ścieżki przez MCP, jak kurs; 30 wywołań, 2,03 centa.

| Metryka | Kurs | Bez LLM-a / odtworzenie | Na żywo | Cel | Spełniony |
|---|---|---|---|---|---|
| Zgodność MCP z wywołaniem bezpośrednim | – | 20/20 | – | 100% | tak |
| Ścieżka (oczekiwany rodzaj wiadomości) | 25/25 | 25/25 | 25/25 | odtworzenie 100%; na żywo ≥ 24/25 | tak |
| Status końcowy | 25/25 | 25/25 | 25/25 | odtworzenie 100%; na żywo ≥ kurs − 1 | tak |
| Pola wyodrębnione przez LLM | 11/11 | 11/11 | 11/11 | odtworzenie 100%; na żywo ≥ kurs − 1 | tak |
| Pauzy (rodzaj, rola, kolejność) | 22/22 | 22/22 | 22/22 | odtworzenie 100%; na żywo ≥ kurs − 1 | tak |
| Osoby zatwierdzające (polityka) | 19/19 | 19/19 | 19/19 | odtworzenie 100%; na żywo ≥ kurs − 1 | tak |
| Stan wniosku | 15/15 | 15/15 | 15/15 | odtworzenie 100%; na żywo ≥ kurs − 1 | tak |
| Trace agenta | 25/25 | 25/25 | 25/25 | odtworzenie 100%; na żywo ≥ kurs − 1 | tak |
| Pytania o dane: pokrycie narzędzi | 4/4 | – | 4/4 | ≥ kurs − 1 | tak |
| Pytania o dane: poprawne argumenty | 1/1 | – | 1/1 | ≥ kurs − 1 | tak |
| Pytania o dane: fakty w odpowiedzi | 4/4 | – | 4/4 | ≥ kurs − 1 | tak |
| Pytania o dane: liczby z pokryciem źródłami | 4/4 | – | 4/4 | ≥ kurs − 1 | tak |
| Pytania o dane: bez twierdzenia o akceptacji | 4/4 | – | 4/4 | ≥ kurs − 1 | tak |
| Wywołania MCP w turach (nieudane) | 7 (0) | – | 7 (0) | 0 nieudanych | tak |

- Wywołania LLM-a: plan 34, sieciowo 30, z cache 4, nieudane 1, 132 s (przebieg 20261104T092904-0f4dd55b; źródło: dashboard.yaml, reports/mp5/review.yaml)
- Tokeny i koszt: tokeny 36 347 / 3751; 2,03 centa po cenach płatnego tieru Gemini; 4,41 centa po cenach kursu; kurs: 4,4 centa (liczą też tury z cache)
- Raport: [mp5/2026-11-04-e5.md](mp5/2026-11-04-e5.md)
- Uwaga: Tokeny i koszt E5 liczą też tury z cache (około +20%, R1-3 z F5).
- Uwaga: R1-1 z F5: nieudana próba modelu głównego w D7-t2 (odebrana przez fallback gemini-3.6-flash) nie jest liczona ani jako błąd tury, ani do --max-calls; w first_live jest jako failed_attempts.
- Uwaga: R1-2 z F5: JSON E5 to powtórka --live --cache-only (0 wywołań sieciowych, 34 z cache), liczby wywołań oryginalnego przebiegu pochodzą z reports/mp5/review.yaml.

## Koszty

| Etap | Wywołania sieciowe | Z cache | Tokeny wej. / wyj. | Nasz koszt (centy, cennik płatny Gemini) | Po cenach kursu (centy) | Koszt kursu (README, centy) |
|---|---|---|---|---|---|---|
| E1 | 1 | 0 | 804 / 959 | 0,42 | 0,49 | – |
| E2 | 0 | 42 | 15 132 / 1776 | 0,9 | 1,93 | – |
| E3 | 0 | 14 | 6626 / 1439 | 0,56 | 1,14 | 1,65 |
| E4 | 0 | 24 | 62 092 / 2509 | 2,49 | 5,79 | 3,8 |
| E5 | 30 | 4 | 36 347 / 3751 | 2,03 | 4,41 | 4,4 |

Tokeny i koszt E5 liczą też tury z cache.

Scenariusz miesięczny kursu (E1: 400 pytań dziennie, 21 dni roboczych, 350 tokenów odpowiedzi), retrieval / cały korpus: po cenach referencyjnych kursu $17,78 / $69,86, po cenach płatnego tieru gemini-3.6-flash $15,58 / $67,65.

Faktyczny koszt projektu w darmowym tierze: 0 USD (kwoty wyżej to przeliczenie tych samych tokenów po cennikach płatnych).

## Zużycie darmowego limitu Gemini

| Etap | Wywołania | Uwagi |
|---|---|---|
| F0 | brak pomiaru | first-call i test_first_call_live; odpowiedź dopiero z fallbacku po 131 s (contracts/f0.md, 9.4); pomiaru modeli w ADR-0006 nie zliczono |
| F1 | brak pomiaru | E1 --with-llm: 1 wywołanie (raport E1) i testy na żywo: 2 (contracts/f1.md); pomiaru modeli w ADR-0006 nie zliczono |
| F2 | brak pomiaru | E2 na Flash-Lite: 11 wywołań (31 z cache, raport E2) i test na żywo: 2; wcześniejszy przebieg na gemini-3.6-flash wyczerpał limit 20 zapytań, a pomiaru 8 modeli (ADR-0006) nie zliczono |
| F3 | 18 | 16 + 2 na przeklikanie D2 w UI (07-roadmap) |
| F4 | 47 | około (07-roadmap) |
| F5 | 55 | około (07-roadmap) |
| F6 | 9 | około: weryfikacja w compose (przerwana tura ≈ 4) i przeklikanie UI (≈ 5, jedno przez fallback); pokaz na żywo osobno, bez nagrania (07-roadmap) |

Razem w etapach z pomiarem: 129 wywołań.

## Znane ograniczenia

- E1: Koszt miesięczny liczony dla tokenów przebiegu kursu (wiersz „referencja kursu” raportu E1) po cenach gemini-3.6-flash, modelu z pomiaru E1. Od F2 model odpowiedzi i cost.our_model to gemini-3.5-flash-lite: $9,17 / $30,00 (FR-1.2, ADR-0006); model embeddingów z configu (intfloat/multilingual-e5-large) daje po cenach gemini-3.6-flash $15,66 / $67,65, i tę kwotę podaje 06-evaluation.
- E1: Koszt E1 w scenariuszu to dolna granica: faktyczne wywołanie z myśleniem zużyło 959 tokenów wyjścia (w tym 714 myślenia) wobec 350 w scenariuszu kursu (raport E1, przebieg 20261102-104911). ADR-0006 podaje 677 tokenów: to starszy pomiar z F1 (06-evaluation podaje oba); raport E1 jest nowszy i to jego liczbę bierze dashboard.
- E2: JSON E2 to przeliczenie z cache (--with-llm --cache-only): 0 wywołań sieciowych, 42 z cache. Pierwszy przebieg na gemini-3.5-flash-lite (20261102-153627-87b0e6c8): 11 wywołań, 31 z cache (przegląd w raporcie E2).
- E2: Wiersz hybrid (poziomy 1–2) bez kolumny „Kurs”: kurs nie ma wariantu hybrid. Z embeddingami referencyjnymi kursu hybrid daje 9/10, 10/10, 10/10, 631 tokenów (raport E2).
- E2: Tokeny i koszt E2 to szacunek: średnie tokeny odpowiedzi z raportu razy liczba odpowiedzi (ctx i hybrid, bez sędziego).
- E2: Sędzia to ten sam model co odpowiedzi: „sędzia 9/9” to górna granica (M09, ADR-0008); M04 to realne pudło wyszukiwania (pytanie PL, dokument EN).
- E3: JSON E3 to powtórka --live --cache-only (0 wywołań sieciowych, 14 z cache). Oryginalny przebieg na żywo: 14 wywołań (13 understand + 1 odpowiedź RAG), 64 s, bez fallbacku (contracts/f3.md, A9); identyfikatora tamtego przebiegu nie zapisano, więc bez first_live.
- E3: D2-t2: na żywo pola 10/11, model nie przejął marży 10% z rozmowy; skutek zerowy (reports/mp3/review.yaml). Cel E3 dla pól to odtworzenie 100%.
- E4: B1: agent poprawnie odmówił zatwierdzenia, ale nie wywołał check_discount_policy (narzędzia 0%, fakty 50%), więc testy 1 i 3 są poniżej celu 9/9; agent_system bez zmian (U10 z F5).
- E4: JSON E4 to powtórka --live --cache-only (0 wywołań sieciowych, 24 z cache). Oryginalny przebieg A10 szedł w dwóch dniach (--ids, --max-calls 20): 24 tury modelu, 0 błędów, bez fallbacku (contracts/f4.md, reports/mp4/review.yaml); liczby sieciowe obu części nie są zapisane w jednym przebiegu, więc bez first_live.
- E5: Tokeny i koszt E5 liczą też tury z cache (około +20%, R1-3 z F5).
- E5: R1-1 z F5: nieudana próba modelu głównego w D7-t2 (odebrana przez fallback gemini-3.6-flash) nie jest liczona ani jako błąd tury, ani do --max-calls; w first_live jest jako failed_attempts.
- E5: R1-2 z F5: JSON E5 to powtórka --live --cache-only (0 wywołań sieciowych, 34 z cache), liczby wywołań oryginalnego przebiegu pochodzą z reports/mp5/review.yaml.
