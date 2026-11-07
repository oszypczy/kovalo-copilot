# ADR-0008: RAG ze źródłami, hybryda BM25 i ewaluacja E2 (F2)

- Status: przyjęty (2026-11-02, przegląd R1: [contracts/f2.md §14](../contracts/f2.md); odpowiedzi na U1–U7 w §13.1, decyzja o `retrieval.mode` niżej)
- Data: 2026-11-02

## Kontekst
F2 przenosi MP2 do aplikacji: chunki z kontekstem, odpowiedź w schemacie z odwołaniami, rekord przebiegu, ocena bez LLM-a, sędzia LLM i nasze rozszerzenie FR-2.5 (hybryda BM25 + wektory). Przy projektowaniu kontraktu sprawdziliśmy biblioteki i zrobiliśmy sondę offline (2026-11-02):

- `BM25Retriever` jest tylko w `langchain-community` (`langchain_classic.retrievers.bm25` to przekierowanie do `langchain_community`). ADR-0007 wykluczył `langchain-community` z zależności. `EnsembleRetriever` (`langchain_classic.retrievers`) zwraca dokumenty bez wyników i bez informacji, z której listy pochodzą.
- `rank_bm25.BM25Okapi` z argumentem `tokenizer` tokenizuje korpus przez `multiprocessing.Pool`; z korpusem już ztokenizowanym działa w jednym procesie.
- `RunnableWithFallbacks` (wynik `get_resilient_chat_model`) nie ma `with_structured_output`; schemat trzeba związać z każdym modelem osobno. `with_structured_output(..., include_raw=True)` zwraca surowy `AIMessage` z `usage_metadata` (tokeny, w tym myślenia) i błąd parsowania w polu `parsing_error` zamiast wyjątku.
- `SQLiteCache` jest w `langchain-community`. `langchain_core.caches.BaseCache` działa w `BaseChatModel._generate_with_cache` przed rate limiterem, więc trafienie w cache nie zużywa limitu, a zwraca oryginalne `usage_metadata`.
- Krok 4 kursu zostawia rubrykę sędziego studentowi; w `config.yaml` kursu nie ma promptu sędziego.
- Kurs nie dołącza zapisanych odpowiedzi MP2, więc poziomów 3–4 referencji nie da się przeliczyć offline; README podaje tylko podsumowanie przebiegu `20261006T212509-3ab56303`.
- Sonda offline (port kroków 1 i 3 kursu, cache kursu i `intfloat/multilingual-e5-large`): chunki z kontekstem odtwarzają cache kursu (87 chunków, 0 brakujących), a `check.py` kursu co do znaku. Hybryda z embeddingami referencyjnymi poprawia hit@3 z 9/10 do 10/10 i komplet dowodów z 8/10 do 10/10 (M03, M10: dokładne identyfikatory). Z e5 sam wariant z kontekstem ma hit@1 10/10, a hybryda 9/10 (M04: polskie pytanie, angielski dokument, BM25 bez wspólnych słów). Tabela: [contracts/f2.md §9.3](../contracts/f2.md).

## Decyzja
1. **Chunki z kontekstem** zgodnie z krokiem 1 kursu (`MarkdownHeaderTextSplitter` `#`/`##` → `RecursiveCharacterTextSplitter` 800/120 → nagłówek `<tytuł> > <sekcja>`, id `<doc_id>#ctxNNN`), w tabeli `kovalo_ctx` (`db.vector_collection`). `kovalo ingest` buduje domyślnie obie tabele (`kovalo_mp1` i `kovalo_ctx`), bo E2 porównuje oba podziały także na Postgresie.
2. **Hybryda:** BM25 przez `rank-bm25` bezpośrednio (`knowledge/lexical.py`, tokenizacja `\w+(?:-\w+)*` po `casefold`, więc `SUP-114` to jeden token; bez stemmingu) i własna ważona Reciprocal Rank Fusion (`knowledge/retrievers.py`, ok. 20 linii, ten sam wzór co `EnsembleRetriever`: `Σ w / (c + pozycja)`, `c = 60`). Bez `langchain-community` i bez `EnsembleRetriever`. Indeks BM25 powstaje w pamięci z korpusu przy starcie procesu; zgodność z indeksem wektorowym sprawdza `kovalo_index_meta.n_chunks`.
3. **Konfiguracja wyszukiwania:** nowe klucze NASZE obok PINNED `retrieval.top_k`: `retrieval.mode` (`vector | hybrid`, domyślnie `vector`, zob. „Decyzja o `retrieval.mode`”) i `retrieval.hybrid` (`fetch_k: 10`, `rrf_c: 60`, wagi 0,5/0,5, BM25 `k1 1.5, b 0.75, epsilon 0.25`, domyślne `rank_bm25`). Wag nie stroimy na 10 pytaniach ewaluacyjnych.
4. **Generacja w `rag/`:** `answer_question` dostaje `Retriever` i model przez parametr; `kovalo.llm` importuje tylko `rag/factory.py`. Wspólny interfejs `Retriever.retrieve(question, k)` obsługuje wyszukiwanie wektorowe i hybrydowe w pamięci i w Postgresie.
5. **Structured output z fallbackiem:** `llm.get_structured_model(schema, purpose)` wiąże schemat z modelem głównym i każdym fallbackiem (`include_raw=True`), a potem `with_fallbacks` z `FALLBACK_EXCEPTIONS`. Błąd parsowania nie przełącza modelu (zgodnie z ADR-0006 pkt 4), tylko kończy się rekordem ze statusem błędu. Schematy `Answer` i `Verdict` kopiujemy z kursu 1:1 razem z opisami pól (są częścią tego, co widzi model).
6. **Cache LLM-a w ewaluacjach:** własny `JsonFileLlmCache(BaseCache)` w `paths.llm_cache_dir` (plik JSON na wpis, zapis atomowy, tryby `read_write`, `read_only`, `strict`) przekazywany do modeli parametrem `cache` w `kovalo.llm`. Zastępuje zapowiadany w 06-evaluation `SQLiteCache` (który wymaga `langchain-community`). Wyjątki dostawcy nie trafiają do cache. Ten sam mechanizm obsłuży wywołania z narzędziami w E4/E5. W aplikacji (api) cache nie jest włączony.
7. **Sędzia:** nowy prompt NASZ `prompts.judge_system` (wersja 1, treść w [contracts/f2.md §4.1](../contracts/f2.md)): kryteria `correct` i `grounded` z opisów pól `Verdict` kursu plus przypadki z README MP2 (dodatki, negacja, zapis liczb, nazwy stanowisk). Prompty PINNED bez zmian. Kolejne wersje: `judge_system_v2` z ewaluacją przed i po.
8. **E2:** `kovalo eval e2` domyślnie liczy tylko poziomy 1–2 (retrieval i dowody) dla trzech wariantów (MP1, z kontekstem, hybryda) i dwóch stron embeddingów (cache kursu, model z configu): zero wywołań LLM-a. Poziomy 3–4 wymagają `--with-llm`; domyślne warianty i budżet: [contracts/f2.md §8.6](../contracts/f2.md) (pytanie U1). Kolumna referencji poziomów 3–4 to stałe z README kursu; poziomy 1–2 referencji są przeliczane i porównywane z kursem.
9. **Parametry per cel:** `llm.purposes.<cel>.params` nakładane na `llm.params` (np. `thinking_level` dla `answer` albo `judge`), czego ADR-0006 zakładał w konsekwencjach, a config nie obsługiwał.
10. **Rekord przebiegu:** `RunRecord` kursu rozszerzony o pola z wartościami domyślnymi (`chunks`, `retrieval`, `prompt`, `answered_by`, `usage`, `seconds`; `ContextItem.score`), więc pliki czyta też `load_run` kursu.

## Konsekwencje
- Nowa zależność `rank-bm25` (czysty Python, ostatnie wydanie 0.2.2 z 2022 r.; stabilny algorytm, ale bez aktywnego utrzymania). Gdyby zniknęła albo przestała się instalować, zastąpimy ją ok. 40 liniami własnego BM25 Okapi za tym samym interfejsem `BM25Index`.
- Wyniki hybrydy nie są podobieństwem kosinusowym (to wynik RRF), więc `ContextItem.score` ma znaczenie zależne od `RunRecord.retrieval`; raporty to opisują.
- BM25 bez stemmingu nie łączy odmian polskich słów ani języków (PL-601 po angielsku). Hybryda pomaga przy dokładnych identyfikatorach, a może szkodzić przy pytaniach międzyjęzycznych. Dlatego domyślny tryb wybiera właściciel projektu po E2 według reguły z [contracts/f2.md §8.5](../contracts/f2.md).
- `api` i `mcp-server` budują indeks BM25 z korpusu przy starcie (milisekundy dla 87 chunków); zmiana korpusu bez `make ingest` kończy się błędem zgodności zamiast cichej rozbieżności.
- Cache LLM-a sprawia, że powtórne ewaluacje są darmowe i deterministyczne mimo temperatury 1,0, ale zmiana promptu, schematu albo modelu unieważnia wpisy (inny klucz). Katalog cache nie trafia do gita (zawiera odpowiedzi modelu).
- Sędzia z tym samym modelem co odpowiedzi może zawyżać oceny własnych odpowiedzi (pytanie U3); raport to zaznacza, a T2 przegląda uzasadnienia ręcznie.
- Pełny przebieg E2 z LLM-em to do 48 wywołań przy domyślnych wariantach (do 72 dla trzech); przy limiterze 0,25 zapytania/s (Flash-Lite, model główny od F2, [ADR-0006](0006-config-and-llm-factory.md)) trwa co najmniej ok. 3,2 min, a przy Flash (0,15 zapytania/s) ok. 5,5 min.

## Rozważane alternatywy
- `langchain-community` (`BM25Retriever`) + `EnsembleRetriever`: mniej własnego kodu, ale duża zależność dla jednej klasy (ADR-0007), wyniki bez punktacji i bez kontroli remisów.
- BM25 w Postgresie (`tsvector`/`ts_rank` albo rozszerzenie `pg_textsearch`): jedno źródło danych, ale `ts_rank` to nie BM25, konfiguracja polskiego słownika wymaga dodatkowych plików w obrazie, a testy jednostkowe potrzebowałyby Postgresa.
- `bm25s`: szybszy, ale przy 87 chunkach szybkość nie ma znaczenia, a API jest większe.
- Generacja w `knowledge/` albo bezpośrednio w `workflow/`: łamie zasadę, że `knowledge/` nie zna LLM-a, albo wiąże RAG z grafem F3 i utrudnia E2.
- `SQLiteCache` z `langchain-community` albo `InMemoryCache`: pierwszy łamie ADR-0007, drugi nie przetrwa między uruchomieniami.
- Cache na poziomie rekordu (pytanie → `RunRecord`): prostszy dla E2, ale nie zadziała dla agenta z narzędziami w E4/E5.
- Stały `retrieval.mode: hybrid` od razu (bo FR-2.5): sonda pokazuje, że z e5 hybryda bywa gorsza na pierwszym miejscu; decyzja należy do E2.

## Decyzja o `retrieval.mode` (2026-11-02, pytanie U2)

**`retrieval.mode: vector`** (bez zmian w `config.yaml`). Hybryda zostaje dostępna przez config (`KOVALO_RETRIEVAL__MODE=hybrid`) i w E2 jako wariant `hybrid`.

Podstawa: reguła z [contracts/f2.md §8.5](../contracts/f2.md) na raporcie [reports/mp2/2026-11-02-e2.md](../../reports/mp2/2026-11-02-e2.md) (strona `ours`, `intfloat/multilingual-e5-large`, odpowiedzi i sędzia `gemini-3.5-flash-lite`), po poprawce `normalize_text` (pkt 11) i przeliczeniu z cache (`--cache-only`, 0 wywołań):

| Miara | ctx (vector) | hybrid |
|---|---|---|
| hit@1 | 10/10 | 9/10 (gorsza: M04) |
| hit@3 / komplet dowodów | 10/10 / 9/10 | 10/10 / 9/10 |
| udzielone / kluczowe fakty / sędzia: poprawna | 9/10 / 9/9 / 9/9 | 9/10 / 9/9 / 9/9 |

Hybryda jest gorsza w hit@1 i nie jest lepsza w żadnej mierze, więc reguła daje `vector`; właściciel projektu przyjął wcześniej rekomendację „`vector`, chyba że E2 pokaże przewagę hybrydy”. FR-2.5 jest spełnione bez hybrydy, bo e5 z chunkami z kontekstem znajduje CON-101 dla M03 na 1. miejscu; hybryda wygrywa tylko przy embeddingach referencyjnych kursu (hit@3 10/10, dowody 10/10), których aplikacja nie używa. Wracamy do pytania, gdy zmieni się model embeddingów, korpus (więcej identyfikatorów) albo zestaw pytań (E3–E5 pokażą pudła na identyfikatorach).

## Uzupełnienia z przeglądu R1 (2026-11-02)

11. **Zerowe ułamki w ocenie bez LLM-a:** `eval.rag.scoring.normalize_text` usuwa „,00”/„.00” po cyfrze (`(?<=\d)[.,]00(?!\d)`), zanim porówna frazy. Prompt kursu każe pisać kwoty z groszami („7296,00 PLN”), a warianty faktów w `data/eval` ich nie mają, więc „500 000,00 PLN” dawało fałszywe pokrycie 0 (M07 hybrid w pierwszym przebiegu). Reguła jest symetryczna (tekst i wariant) i nie łamie wymagań kroku 3 kursu („9%” nadal nie pasuje do „9,5%”, „500 000” nie pasuje do „1 500 000,00”); testy kursu i odtworzenie `check.py` zostają zielone. To rozszerzenie naszego portu, nie zmiana wartości PINNED ani promptu.
12. **Rozszerzenia kontraktu przyjęte w R1** (pola i funkcje pomocnicze, bez nowych zależności): lista w [contracts/f2.md §14.2](../contracts/f2.md).
13. **Komentarze z przeglądu ręcznego poza wygenerowanym raportem:** plik `reports/<mp>/review.yaml` (commitowany) wczytywany przez generator raportu, klucz: wariant + id pytania. Raport da się wtedy przeliczyć z cache bez utraty komentarzy. Kontrakt: [contracts/f2.md §14.3](../contracts/f2.md).

## Adaptacja F5: brak w cache a fallback w trybie `--cache-only` (2026-11-04)

Powtórka E5 `--live --cache-only` po przebiegu A7 dała 24/25: w turze D7-t2 `understand` na żywo odpowiedział fallback (`gemini-3.6-flash`), bo model główny zawiódł błędem dostępności, więc w cache był tylko wpis fallbacku. `LlmCacheMiss` nie należy do `FALLBACK_EXCEPTIONS`, więc łańcuch nie sprawdzał wpisu fallbacku. Zmiana: `kovalo.llm.CacheMissError` (bazowa klasa `LlmCacheMiss`; `kovalo.llm` nie importuje `kovalo.eval`) i `fallback_exceptions(offline)`. Tylko w trybie offline (`offline=True`: `get_structured_model` i `AvailabilityFallbackMiddleware` agenta z `build_agent_runner`) brak trafienia u modelu głównego przełącza na następny model; żaden nie trafi → `LlmCacheMiss` jak dotąd. Na żywo bez zmian: brak w cache nie jest błędem dostępności. Testy: `tests/unit/test_llm.py::test_strict_cache_falls_back_to_the_fallback_entry`, `tests/unit/workflow/test_factory.py::test_agent_replays_a_fallback_answer_from_strict_cache`.

## Znane ograniczenia (kandydaci na późniejsze etapy)

- **Pytania po polsku do dokumentów po angielsku (M04, PL-601).** e5 trafia dokument, ale nie chunk z dowodem (`PL-601#ctx002`), a BM25 bez wspólnych tokenów podbija niezwiązany `CON-103#ctx000`. Model uczciwie odmawia (`phrases_missing`). Kandydaci: tłumaczenie albo rozszerzenie zapytania przed wyszukiwaniem (węzeł grafu w F3 albo opcja `Retriever`), dwujęzyczny nagłówek kontekstu w chunkach dokumentów EN (zmiana `build_contextual_chunks` = zmiana kroku 1 kursu, więc nowy ADR i E1/E2 przed i po). Bez strojenia na 10 pytaniach.
- **Samoocena sędziego (M09).** Sędzia to ten sam model co odpowiedzi (U3: A). W M09 przepuścił przesadę („klienci odchodzą głównie z powodu cen”, a według REP-302 ceny są drugą przyczyną po dostawach), a pokrycie fraz też jej nie wykrywa. „Sędzia 9/9” to górna granica jakości. Kandydaci: sędzia z innego modelu przez `llm.purposes.judge.model` (np. `gemini-3.6-flash` przy jednorazowym przebiegu, limit 20/dzień), rubryka `judge_system_v2` wymagająca cytatu liczby lub zdania ze źródła (ewaluacja przed i po), ręczna próbka uzasadnień w każdym T2.

