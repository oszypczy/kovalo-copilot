# 06. Ewaluacja

Kurs mierzy system na każdym poziomie osobno, bo jedna liczba „trafności” ukrywa, gdzie coś poszło źle. Przenosimy ten układ 1:1 i każdy raport pokazuje nasz wynik obok referencyjnego (OpenAI `gpt-5.4-mini`).

Dwie warstwy:
- **testy deterministyczne** (`uv run pytest`; markery `live`, `integration`, `model` są wyłączone w `addopts`): bez LLM-a, atrapy modeli, odtwarzanie dialogów z idealnym `understand`; uruchamiane w CI,
- **ewaluacja na żywo** (`make eval STAGE=eN`, polecenie `kovalo eval` w obrazie `api`): prawdziwy LLM, raport w `reports/mpN/<data>-eN.md` (w repozytorium) + JSON z pełnymi rekordami (poza repozytorium). E1 i E2 bez `--with-llm` oraz E3 bez `--live` nie wywołują LLM-a ([contracts/f1.md §7](contracts/f1.md), [contracts/f2.md §8](contracts/f2.md), [contracts/f3.md §10](contracts/f3.md)).

## Zestawy danych (`data/eval/`)

| Plik | Etap | Zawartość |
|---|---|---|
| `mini_project_1_queries.yaml` | retrieval | 10 pytań z trafnym dokumentem |
| `mini_project_2_questions.yaml` | RAG | 12 pytań (10 z odpowiedzią w korpusie, 2 spoza korpusu): frazy dowodowe, kluczowe fakty, wzorzec |
| `mini_project_3_dialogues.yaml` | workflow | 8 dialogów (13 tur) z oczekiwaną ścieżką, polami, pauzami, decyzjami |
| `mini_project_4_questions.yaml` | agent | 9 pytań: `needs`, `calculations`, `key_facts`, frazy twierdzenia o akceptacji |
| `mini_project_5_conversations.yaml` | asystent | 5 rozmów, które przechodzą przez wszystkie ścieżki |

## Metryki i cele

| Etap | Metryka | Referencja kursu | Nasz cel (free LLM + lokalne embeddingi) |
|---|---|---|---|
| E1 retrieval | hit@1 / hit@3 (chunki MP1; Q01–Q10 i M01–M10) | 0,9 / 0,9 (oba zestawy) | ≥ 0,8 / ≥ 0,9 na Q01–Q10; M03 trafione (osiągnięte już bez hybrydy, zob. „Wyniki”) |
| E1 koszty | miesięczny koszt retrieval / cały korpus | $17,78 / $69,86 | identycznie po cenach referencyjnych; obok szacunek po cenach płatnego tieru Gemini (free tier = $0) |
| E2 RAG, poziomy 1–2 (bez LLM-a) | hit@1 / hit@3 / komplet dowodów / tokeny promptu: chunki MP1, z kontekstem, hybryda | MP1 9/10, 9/10, 9/10, 850,3; z kontekstem 9/10, 9/10, 8/10, 610,4 | referencja odtworzona co do znaku; M03 trafione hybrydą (FR-2.5) |
| E2 RAG, poziomy 3–4 (z LLM-em) | retrieval, dowody, odpowiedź (fakty, sędzia), odwołania, odmowy | 9/10, 8/10, 8/10 (8/8, 8/8), 8/8, 2/2 | ≥ referencja − 1 na każdym poziomie |
| E3 dialogi | kierowanie, pola, stan wniosku, osoby zatwierdzające, pauzy, status (odtworzenie z idealnym `understand` i przebieg na żywo: 13 wywołań `understand` + 1 odpowiedź RAG) | 13/13, 11/11, 8/8, 11/11, 13/13, 13/13, 0 błędów | odtworzenie 100%; na żywo ≥ 12/13 kierowania. Nasz wynik (2026-11-03, gemini-3.5-flash-lite, 14 wywołań, 0,56 centa): odtworzenie 100%; na żywo 13/13, 10/11, 8/8, 11/11, 13/13, 13/13, 0 błędów (jedyna rozbieżność: D2-t2, model nie przejął marży 10% z rozmowy; skutek zerowy, przejrzane w `reports/mp3/review.yaml`) |
| E4 agent | pokrycie narzędzi, argumenty, fakty, liczby bez źródła, twierdzenia o akceptacji | 8/9, 3/3, 8/9, 0, 0/9 | 9/9, 3/3, 9/9 (T1 dzięki FR-4.7), 0, 0/9 |
| E5 asystent | zgodność MCP (20 wywołań), ścieżka, status, pola, pauzy, zatwierdzający, stan, trace, 5 sprawdzeń pytań o dane, nieudane wywołania MCP | 25/25, 25/25, 11/11, 22/22, 19/19, 15/15, 25/25; 4/4, 1/1, 4/4, 4/4, 4/4; MCP 7 (0) | zgodność i odtworzenie 100%; na żywo ścieżka ≥ 24/25, reszta ≥ kurs − 1, 0 nieudanych MCP. Nasz wynik (2026-11-04): 100% we wszystkich, zob. „Wyniki” |

## Testy z prawdziwymi usługami

Każdy marker uruchamiamy ręcznie i osobno. `-m live` zbiera wszystkie testy na żywo (w F1: `test_first_call_live` i `test_e1_with_llm_live`, czyli 2 wywołania Gemini), więc przy oszczędzaniu limitu wskazujemy plik albo test:

```bash
uv run --env-file .env pytest -m live tests/live/test_e1_live.py            # jedno wywołanie (E1 --with-llm)
uv run --env-file .env pytest -m live -k first_call                         # jedno wywołanie (F0)
uv run --env-file .env pytest -m live tests/live/test_e2_live.py            # dwa wywołania (odpowiedź + sędzia, F2)
uv run --env-file .env pytest -m live tests/live/test_e3_live.py            # jedno wywołanie (understand z historią, F3)
uv run --env-file .env pytest -m live tests/live/test_e4_live.py            # ≤ 4 wywołania (P1 przez agenta, F4)
uv run --env-file .env pytest -m live tests/live/test_e5_live.py            # ≤ 6 wywołań (rozmowa C5 przez MCP, F5)
uv run pytest -m model                                                      # model fastembed z configu (bez sieci po pobraniu)
uv run pytest -m integration                                                # wymaga make up (Postgres z compose)
```

Bez klucza w środowisku testy `live` są pomijane (`skip`), a nie zielone: sprawdzaj podsumowanie pytesta.

## Wyniki

| Etap | Data | Raport | Wynik |
|---|---|---|---|
| E1 | 2026-11-02 | [reports/mp1/2026-11-02-e1.md](../reports/mp1/2026-11-02-e1.md) | referencja odtworzona co do znaku (0,9 / 0,9 w MP1 i MP2; $17,78 / $69,86, x3,9). Model z configu `intfloat/multilingual-e5-large`: MP1 0,9 / 1,0, MP2 0,9 / 1,0, jako jedyny trafia Q10 i M03 (`SUP-114`). Cel spełniony; wybór i porównanie 4 modeli w [ADR-0003](adr/0003-local-embeddings.md). Koszt po cenach `gemini-3.6-flash`: $15,66 / $67,65 (dolna granica: faktyczne wywołanie dało 677 tokenów wyjścia z myśleniem wobec 350 w scenariuszu, zob. [ADR-0006](adr/0006-config-and-llm-factory.md)). Nie zweryfikowano jeszcze `--backend pgvector` (Postgres w compose). Od F2 model odpowiedzi i `cost.our_model` to `gemini-3.5-flash-lite` (ADR-0006, trzeci pomiar); liczby E1 to stan z pomiaru |
| E2 | 2026-11-02 | [reports/mp2/2026-11-02-e2.md](../reports/mp2/2026-11-02-e2.md) | `gemini-3.5-flash-lite`, embeddingi `intfloat/multilingual-e5-large`; poziomy 1–2 zgodne z referencją. Poziomy 3–4 (ctx / hybrid; kurs gpt-5.4-mini, ctx 8/10, 8/8, 8/8, 2/2): odpowiedzi 9/10 / 9/10, fakty 9/9 / 9/9, sędzia 9/9 / 9/9, odmowy 2/2 / 2/2; cel ≥ referencja − 1 spełniony. Przeliczenie z cache (`--with-llm --cache-only`): 0 wywołań, te same wyniki. Rekomendacja reguły i decyzja: `retrieval.mode: vector` (hybryda gorsza w hit@1, reszta remis; [ADR-0008](adr/0008-rag-hybrid-e2.md)). Przegląd ręczny (T2): M04 to realne pudło wyszukiwania (pytanie PL, chunk PL-601 po angielsku), odmowa uczciwa; M07 hybrid był błędem ewaluatora („500 000,00 PLN”, poprawione); sędzia (ten sam model) przepuścił przesadę w M09 (ceny jako „główna” przyczyna odejść, wg danych druga po dostawach), więc „sędzia 9/9” to górna granica. Oba przypadki są w ADR-0008 jako znane ograniczenia (tłumaczenie zapytania, sędzia z innego modelu). Test na żywo `test_e2_live.py` zielony |
| E3 | 2026-11-03 | [reports/mp3/2026-11-03-e3.md](../reports/mp3/2026-11-03-e3.md) | odtworzenie z idealnym `understand` (0 wywołań LLM-a): 13/13, 11/11, 8/8, 11/11, 13/13, 13/13, 0 błędów, czyli 100% jak w kursie. Na żywo `gemini-3.5-flash-lite` (14 wywołań: 13 `understand` + 1 odpowiedź RAG, bez fallbacku, 64 s, 0,56 centa po cenach Flash-Lite; kurs gpt-5.4-mini 1,65 centa): kierowanie 13/13, pola 10/11, wniosek 8/8, zatwierdzający 11/11, pauzy 13/13, status 13/13, 0 błędów; cel „kierowanie ≥ 12/13” spełniony. Przeliczenie `--live --cache-only` bez klucza: 0 wywołań, te same liczby. Przegląd ręczny (T2, `reports/mp3/review.yaml`): jedyna rozbieżność D2-t2, model przy „marża bez zmian” zostawił `margin_after_pct` puste zamiast przenieść 10; ocena zgodna z kursem, skutek zerowy (wniosek w wersji 2 ma marżę 10% z wersji 1). Prompt `understand_system` jest PINNED, więc poprawka tylko przez nowy ADR z E3 przed i po ([contracts/f3.md §18.4](contracts/f3.md)). Odpowiedzi tekstowe i cytowanie D7 (MTG-402) przejrzane ręcznie. Test na żywo `test_e3_live.py` zielony (1 wywołanie) |
| E4 | 2026-11-03 | [reports/mp4/2026-11-03-e4.md](../reports/mp4/2026-11-03-e4.md) | przeliczenie zapisanych przebiegów kursu naszym `score_trace` (0 wywołań): co do pytania jak README (jeden agent 8/9, 3/3, 8/9, 0 (0), 0/9; zespół 8/9, 3/3, 8/9, 3 (2), 0/9); odtworzenie wywołań kursu na naszych narzędziach: 21 wywołań, argumenty 3/3, P1–P3 = 729 600 / 362 400 / 0 groszy. Na żywo `gemini-3.5-flash-lite`, 7 narzędzi (z `check_termination_condition`), `agent_system` bez zmian: narzędzia 8/9, argumenty 3/3, fakty 8/9, liczby bez źródła 0, twierdzenia o zatwierdzeniu 0/9; 9 × `answered`, 21 wywołań narzędzi bez błędów, 24 tury modelu (62 092 / 2 509 tokenów; 2,49 centa po cenach Flash-Lite, 5,79 po referencyjnych; kurs 3,8 centa), 0 błędów, bez fallbacku. **T1 poprawne** (100% narzędzi i faktów; kurs 67% / 0%). Cel 9/9 nie jest spełniony tylko przez B1: agent poprawnie odmówił zatwierdzenia, ale nie wywołał `check_discount_policy` (0% narzędzi, 50% faktów), jedyne pytanie gorsze od kursu; kandydat na `agent_system_v2` w F5 (decyzja użytkownika). Przeliczenie `--live --cache-only`: 0 wywołań, te same liczby. Przegląd ręczny w `reports/mp4/review.yaml` (T1, B1, odpowiedzi na 4 pytania kursu o zespół). Test na żywo `test_e4_live.py` zielony |
| E5 | 2026-11-04 | [reports/mp5/2026-11-04-e5.md](../reports/mp5/2026-11-04-e5.md) | zgodność MCP z wywołaniem bezpośrednim 20/20 (17 wywołań kursu + 3 `check_termination_condition`); odtworzenie 13 rozmów (25 tur) z idealnym `understand` przez serwer MCP: 100% w każdym wymiarze, 0 wywołań LLM-a. Na żywo `gemini-3.5-flash-lite`, agent przez MCP (stdio, wyszukiwanie w pamięci), prompty `assistant_understand_system` i `agent_system` bez zmian: ścieżka 25/25, status 25/25, pola 11/11, pauzy 22/22, zatwierdzający 19/19, wniosek 15/15, trace 25/25, pytania o dane 4/4, 1/1, 4/4, 4/4, 4/4, wywołania MCP 7 (0), czyli jak kurs. 30 wywołań sieciowych (24 `understand` + 6 agenta), 4 z cache (3 tury agenta C1-t1 z E4, D3-t1 w obrębie przebiegu), 0 błędów tur; jedna tura `understand` przez fallback `gemini-3.6-flash` (D7-t2). „Prześlij to.” i „Spróbuj jeszcze raz…” rozpoznane jako `update`. Powtórka `--live --cache-only` bez klucza: 25/25, 0 wywołań (po poprawce fallbacku w trybie offline, ADR-0008). E4 przez MCP (`--transport stdio --cache-only`, [raport](../reports/mp4/2026-11-04-e4.md)): 24 trafienia w cache F4, 0 wywołań, wynik identyczny z F4. Przegląd ręczny w `reports/mp5/review.yaml` (D7-t2, C1-t1, D3-t1, C5-t1). Tokeny i koszt w raporcie liczą też tury z cache (około +20%). Test na żywo `test_e5_live.py` zielony |

### E3: budżet i polecenia

`kovalo eval e3` (odtworzenie, 0 wywołań), `kovalo eval e3 --live` (≤ 14 wywołań: 13 `understand` + 1 odpowiedź RAG; fallback w najgorszym razie drugie tyle z limitu Flash), powtórka `--live --cache-only` (0 wywołań, bez klucza). Pełna tabela: [contracts/f3.md §11](contracts/f3.md). Komentarze przeglądu ręcznego: `reports/mp3/review.yaml` (etykiety `ours-replay`, `ours-live`, id `D2-t2`).

### E5: budżet i polecenia

`kovalo eval e5` (zgodność MCP + odtworzenie, 0 wywołań; domyślnie serwer przez stdio z `--search memory`, więc potrzebny lokalny model fastembed), `kovalo eval e5 --live` (plan 34: 25 `understand` + około 9 tur agenta; górna granica 73; `--max-calls 40` wspólny dla obu), powtórka `--live --cache-only` (0 wywołań, bez klucza). W compose: `make eval STAGE=e5 ARGS="--transport http --backend pgvector"`. E4 z agentem przez MCP: `kovalo eval e4 --live --transport stdio`. Pełna tabela: [contracts/f5.md §5](contracts/f5.md). Komentarze przeglądu ręcznego: `reports/mp5/review.yaml` (etykiety `ours-replay`, `ours-live`, id tur `C1-t1` … `D8-t1`).

## Zasady (z CONCEPTS kursu)

- Błąd (brak czytelnej odpowiedzi) **nigdy** nie liczy się jako trafna odmowa.
- Hit@k liczy dokumenty, nie dowody: dowody mierzymy osobno frazami.
- **Ewaluatora też trzeba sprawdzić:** dopasowanie fraz może przeoczyć dowód sformułowany inaczej. Każdą rozbieżność przeglądamy ręcznie i zapisujemy w raporcie.
- Liczby w odpowiedzi agenta: jednostki są jawne (grosze ≠ PLN, % ≠ pp), polski zapis liczb, liczby niejednoznaczne („7,296”) zawsze niezweryfikowane.
- **Niezależny wzorzec (test oracle):** osoby zatwierdzające w `mini_project_3_dialogues.yaml` wyliczono ręcznie; E3 nigdy nie porównuje wyniku z wynikiem naszego silnika polityki.
- Sędzia LLM to kolejne wywołanie ze schematem. Jego uzasadnienie porównujemy z odpowiedzią, źródłem i kryteriami.
- Na darmowym tierze uruchamiamy ewaluacje z cache odpowiedzi LLM-a na dysku (`JsonFileLlmCache` w `paths.llm_cache_dir`, [ADR-0008](adr/0008-rag-hybrid-e2.md); nie `SQLiteCache`, który wymaga `langchain-community`), żeby ponowne przeliczenie metryk nie zużywało limitu. `--cache-only` gwarantuje zero wywołań sieciowych. Każdy raport podaje liczbę wywołań (plan, wykonane, z cache).

## Raport

Każdy raport zawiera: dostawcę i model, identyfikator przebiegu, tabelę metryk z kolumną „referencja”, wyniki tura po turze, tokeny i koszt (faktyczny oraz przeliczony po cenach referencyjnych) oraz sekcję „rozbieżności przejrzane ręcznie”. Komentarze z przeglądu ręcznego trzymamy w `reports/<mp>/review.yaml` (commitowany, klucz: wariant + id pytania), a generator wstawia je do sekcji; ręczne poprawki w wygenerowanym Markdownie są zabronione, bo raport musi dać się przeliczyć z cache i zgadzać z JSON-em ([contracts/f2.md §14.3](contracts/f2.md)).
