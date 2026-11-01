# 06. Ewaluacja

Kurs mierzy system na każdym poziomie osobno, bo jedna liczba „trafności” ukrywa, gdzie coś poszło źle. Przenosimy ten układ 1:1 i każdy raport pokazuje nasz wynik obok referencyjnego (OpenAI `gpt-5.4-mini`).

Dwie warstwy:
- **testy deterministyczne** (`pytest -m "not live"`): bez LLM-a, atrapy modeli, odtwarzanie dialogów z idealnym `understand`; uruchamiane w CI,
- **ewaluacja na żywo** (`make eval STAGE=eN`, polecenie `kovalo eval` w obrazie `api`): prawdziwy LLM, raport w `reports/<data>-<etap>.md` + JSON z pełnymi rekordami.

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
| E1 retrieval | hit@1 / hit@3 (chunki MP1) | 0,9 / 0,9 | ≥ 0,8 / ≥ 0,9; z hybrydą: M03 trafione |
| E1 koszty | miesięczny koszt retrieval / cały korpus | $17,78 / $69,86 | identycznie po cenach referencyjnych; obok szacunek po cenach płatnego tieru Gemini (free tier = $0) |
| E2 RAG | retrieval, dowody, odpowiedź (fakty, sędzia), odwołania, odmowy | 9/10, 8/10, 8/10 (8/8, 8/8), 8/8, 2/2 | ≥ referencja − 1 na każdym poziomie |
| E3 dialogi | kierowanie, pola, stan wniosku, osoby zatwierdzające, pauzy, status | 13/13, 11/11, 8/8, 11/11, 13/13, 13/13 | odtworzenie 100%; na żywo ≥ 12/13 kierowania |
| E4 agent | pokrycie narzędzi, argumenty, fakty, liczby bez źródła, twierdzenia o akceptacji | 8/9, 3/3, 8/9, 0, 0/9 | 9/9, 3/3, 9/9 (T1 dzięki FR-4.7), 0, 0/9 |
| E5 asystent | ścieżka, status, pola, pauzy, zatwierdzający, stan, trace, dane | 25/25 we wszystkich | ≥ 24/25 |

## Zasady (z CONCEPTS kursu)

- Błąd (brak czytelnej odpowiedzi) **nigdy** nie liczy się jako trafna odmowa.
- Hit@k liczy dokumenty, nie dowody: dowody mierzymy osobno frazami.
- **Ewaluatora też trzeba sprawdzić:** dopasowanie fraz może przeoczyć dowód sformułowany inaczej. Każdą rozbieżność przeglądamy ręcznie i zapisujemy w raporcie.
- Liczby w odpowiedzi agenta: jednostki są jawne (grosze ≠ PLN, % ≠ pp), polski zapis liczb, liczby niejednoznaczne („7,296”) zawsze niezweryfikowane.
- Sędzia LLM to kolejne wywołanie ze schematem. Jego uzasadnienie porównujemy z odpowiedzią, źródłem i kryteriami.
- Na darmowym tierze uruchamiamy ewaluacje z cache odpowiedzi LLM-a (`SQLiteCache` z LangChaina), żeby ponowne przeliczenie metryk nie zużywało limitu.

## Raport

Każdy raport zawiera: dostawcę i model, identyfikator przebiegu, tabelę metryk z kolumną „referencja”, wyniki tura po turze, tokeny i koszt (faktyczny oraz przeliczony po cenach referencyjnych) oraz sekcję „rozbieżności przejrzane ręcznie”.
