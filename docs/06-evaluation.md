# 06. Ewaluacja

Kurs mierzy system na każdym poziomie osobno, bo jedna liczba „trafności” ukrywa, gdzie coś poszło źle. Przenosimy ten układ 1:1 i każdy raport pokazuje nasz wynik obok referencyjnego (OpenAI `gpt-5.4-mini`).

Dwie warstwy:
- **testy deterministyczne** (`uv run pytest`; markery `live`, `integration`, `model` są wyłączone w `addopts`): bez LLM-a, atrapy modeli, odtwarzanie dialogów z idealnym `understand`; uruchamiane w CI,
- **ewaluacja na żywo** (`make eval STAGE=eN`, polecenie `kovalo eval` w obrazie `api`): prawdziwy LLM, raport w `reports/mpN/<data>-eN.md` (w repozytorium) + JSON z pełnymi rekordami (poza repozytorium). E1 i E2 bez `--with-llm` nie wywołują LLM-a ([contracts/f1.md §7](contracts/f1.md), [contracts/f2.md §8](contracts/f2.md)).

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
| E3 dialogi | kierowanie, pola, stan wniosku, osoby zatwierdzające, pauzy, status | 13/13, 11/11, 8/8, 11/11, 13/13, 13/13 | odtworzenie 100%; na żywo ≥ 12/13 kierowania |
| E4 agent | pokrycie narzędzi, argumenty, fakty, liczby bez źródła, twierdzenia o akceptacji | 8/9, 3/3, 8/9, 0, 0/9 | 9/9, 3/3, 9/9 (T1 dzięki FR-4.7), 0, 0/9 |
| E5 asystent | ścieżka, status, pola, pauzy, zatwierdzający, stan, trace, dane | 25/25 we wszystkich | ≥ 24/25 |

## Testy z prawdziwymi usługami

Każdy marker uruchamiamy ręcznie i osobno. `-m live` zbiera wszystkie testy na żywo (w F1: `test_first_call_live` i `test_e1_with_llm_live`, czyli 2 wywołania Gemini), więc przy oszczędzaniu limitu wskazujemy plik albo test:

```bash
uv run --env-file .env pytest -m live tests/live/test_e1_live.py            # jedno wywołanie (E1 --with-llm)
uv run --env-file .env pytest -m live -k first_call                         # jedno wywołanie (F0)
uv run --env-file .env pytest -m live tests/live/test_e2_live.py            # dwa wywołania (odpowiedź + sędzia, F2)
uv run pytest -m model                                                      # model fastembed z configu (bez sieci po pobraniu)
uv run pytest -m integration                                                # wymaga make up (Postgres z compose)
```

Bez klucza w środowisku testy `live` są pomijane (`skip`), a nie zielone: sprawdzaj podsumowanie pytesta.

## Wyniki

| Etap | Data | Raport | Wynik |
|---|---|---|---|
| E1 | 2026-11-02 | [reports/mp1/2026-11-02-e1.md](../reports/mp1/2026-11-02-e1.md) | referencja odtworzona co do znaku (0,9 / 0,9 w MP1 i MP2; $17,78 / $69,86, x3,9). Model z configu `intfloat/multilingual-e5-large`: MP1 0,9 / 1,0, MP2 0,9 / 1,0, jako jedyny trafia Q10 i M03 (`SUP-114`). Cel spełniony; wybór i porównanie 4 modeli w [ADR-0003](adr/0003-local-embeddings.md). Koszt po cenach `gemini-3.6-flash`: $15,66 / $67,65 (dolna granica: faktyczne wywołanie dało 677 tokenów wyjścia z myśleniem wobec 350 w scenariuszu, zob. [ADR-0006](adr/0006-config-and-llm-factory.md)). Nie zweryfikowano jeszcze `--backend pgvector` (Postgres w compose). Od F2 model odpowiedzi i `cost.our_model` to `gemini-3.5-flash-lite` (ADR-0006, trzeci pomiar); liczby E1 to stan z pomiaru |
| E2 | 2026-11-02 | [reports/mp2/2026-11-02-e2.md](../reports/mp2/2026-11-02-e2.md) | `gemini-3.5-flash-lite`, embeddingi `intfloat/multilingual-e5-large`; poziomy 1–2 zgodne z referencją. Poziomy 3–4 (ctx / hybrid; kurs gpt-5.4-mini, ctx 8/10, 8/8, 8/8, 2/2): odpowiedzi 9/10 / 9/10, fakty 9/9 / 9/9, sędzia 9/9 / 9/9, odmowy 2/2 / 2/2; cel ≥ referencja − 1 spełniony. Przeliczenie z cache (`--with-llm --cache-only`): 0 wywołań, te same wyniki. Rekomendacja reguły i decyzja: `retrieval.mode: vector` (hybryda gorsza w hit@1, reszta remis; [ADR-0008](adr/0008-rag-hybrid-e2.md)). Przegląd ręczny (T2): M04 to realne pudło wyszukiwania (pytanie PL, chunk PL-601 po angielsku), odmowa uczciwa; M07 hybrid był błędem ewaluatora („500 000,00 PLN”, poprawione); sędzia (ten sam model) przepuścił przesadę w M09 (ceny jako „główna” przyczyna odejść, wg danych druga po dostawach), więc „sędzia 9/9” to górna granica. Oba przypadki są w ADR-0008 jako znane ograniczenia (tłumaczenie zapytania, sędzia z innego modelu). Test na żywo `test_e2_live.py` zielony |

## Zasady (z CONCEPTS kursu)

- Błąd (brak czytelnej odpowiedzi) **nigdy** nie liczy się jako trafna odmowa.
- Hit@k liczy dokumenty, nie dowody: dowody mierzymy osobno frazami.
- **Ewaluatora też trzeba sprawdzić:** dopasowanie fraz może przeoczyć dowód sformułowany inaczej. Każdą rozbieżność przeglądamy ręcznie i zapisujemy w raporcie.
- Liczby w odpowiedzi agenta: jednostki są jawne (grosze ≠ PLN, % ≠ pp), polski zapis liczb, liczby niejednoznaczne („7,296”) zawsze niezweryfikowane.
- Sędzia LLM to kolejne wywołanie ze schematem. Jego uzasadnienie porównujemy z odpowiedzią, źródłem i kryteriami.
- Na darmowym tierze uruchamiamy ewaluacje z cache odpowiedzi LLM-a na dysku (`JsonFileLlmCache` w `paths.llm_cache_dir`, [ADR-0008](adr/0008-rag-hybrid-e2.md); nie `SQLiteCache`, który wymaga `langchain-community`), żeby ponowne przeliczenie metryk nie zużywało limitu. `--cache-only` gwarantuje zero wywołań sieciowych. Każdy raport podaje liczbę wywołań (plan, wykonane, z cache).

## Raport

Każdy raport zawiera: dostawcę i model, identyfikator przebiegu, tabelę metryk z kolumną „referencja”, wyniki tura po turze, tokeny i koszt (faktyczny oraz przeliczony po cenach referencyjnych) oraz sekcję „rozbieżności przejrzane ręcznie”. Komentarze z przeglądu ręcznego trzymamy w `reports/<mp>/review.yaml` (commitowany, klucz: wariant + id pytania), a generator wstawia je do sekcji; ręczne poprawki w wygenerowanym Markdownie są zabronione, bo raport musi dać się przeliczyć z cache i zgadzać z JSON-em ([contracts/f2.md §14.3](contracts/f2.md)).
