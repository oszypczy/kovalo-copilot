# ADR-0003: Lokalne wielojęzyczne embeddingi (fastembed)

- Status: przyjęty (model: `intfloat/multilingual-e5-large`)
- Data: 2026-11-01 (propozycja), 2026-11-02 (przyjęcie po E1)

## Kontekst
Kurs dostarcza cache embeddingów OpenAI `text-embedding-3-small` dla chunków i pytań testowych, ale każde nowe pytanie użytkownika wymaga płatnego wywołania OpenAI. Gemini ma darmowe embeddingi, ale z limitami tego samego tieru co czat.

## Decyzja
Embeddingi liczymy lokalnie przez **fastembed** (ONNX, CPU, bez PyTorcha, mały obraz), opakowane w `CacheBackedEmbeddings`. Model pobieramy w czasie budowania obrazu.

**Wybrany model: `intfloat/multilingual-e5-large`** (1024 wymiary, 2,24 GB, prefiksy `query: ` / `passage: `). Jest już ustawiony w `embeddings.local` w `config.yaml` i w `ARG FASTEMBED_MODEL` w `docker/Dockerfile`, więc wybór nie zmienia konfiguracji ani wymiaru tabeli `kovalo_mp1`.

Kandydaci i reguła wyboru: [ADR-0007](0007-knowledge-index-f1.md) pkt 4 i [contracts/f1.md §7.3](../contracts/f1.md). Pierwotny kandydat `BAAI/bge-m3` odpadł, bo `fastembed` 0.9 (`TextEmbedding`) go nie obsługuje. Do hit@k MP2 wchodzą tylko pytania z odpowiedzią w korpusie (M01–M10).

## Wynik E1

Przebieg `20261102-093631`, raport: [reports/mp1/2026-11-02-e1.md](../../reports/mp1/2026-11-02-e1.md). Chunki identyczne z podziałem kursu (0 spoza cache kursu), `top_k = 3`, CPU lokalnie (`uv run`).

| Model | Rozmiar | MP1 hit@1 / hit@3 | MP2 hit@1 / hit@3 | Czas 47 chunków | Pytanie | Cel |
|---|---|---|---|---|---|---|
| referencja kursu (`text-embedding-3-small`, cache) | – | 0,9 / 0,9 | 0,9 / 0,9 | – | – | tak |
| **`intfloat/multilingual-e5-large`** | 2,24 GB | **0,9 / 1,0** | **0,9 / 1,0** | 18,8 s | 71 ms | tak |
| `jinaai/jina-embeddings-v3` | 2,29 GB | 0,8 / 0,9 | 0,9 / 0,9 | 18,1 s | 485 ms | tak |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 1,00 GB | 0,9 / 0,9 | 0,6 / 0,9 | 4,8 s | 13 ms | tak |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 0,22 GB | 0,6 / 0,9 | 0,6 / 0,8 | 1,3 s | 5 ms | nie |

Cel: MP1 hit@1 ≥ 0,8 i hit@3 ≥ 0,9.

Uzasadnienie z reguły §7.3:
1. Cel spełniają e5-large, jina-v3 i mpnet; MiniLM odpada (hit@1 0,6).
2. Najlepszy po sumie trafień MP1+MP2: e5-large (hit@3 20/20, hit@1 18/20). jina-v3 i mpnet mają po 18/20 w hit@3.
3. Tolerancja „co najwyżej 1 trafienie mniej, wtedy mniejszy model” nie zadziałała: jina-v3 i mpnet tracą po 2 trafienia hit@3, a mpnet także 3 trafienia hit@1. Zostaje e5-large.

Dodatkowe obserwacje:
- e5-large jako jedyny model (także jedyny wobec referencji kursu) trafia Q10 i M03: pytania o dokładny identyfikator `SUP-114` (CON-101 na 2. miejscu). Cel z [06-evaluation.md](../06-evaluation.md) „z hybrydą (F2): M03 trafione” jest więc osiągnięty już bez BM25; hybryda w F2 zostaje dla zestawu MP2 i chunków z kontekstem.
- jina-v3 liczy pytanie ok. 7 razy wolniej niż e5-large (485 ms wobec 71 ms) przy większym rozmiarze; licencja CC BY-NC 4.0 była zaakceptowana przez właściciela projektu tylko na potrzeby porównania.
- Czas indeksowania 47 chunków (18,8 s bez cache) dotyczy tylko pierwszego `kovalo ingest`; kolejne biorą wektory z cache.

## Konsekwencje
- Zero kosztów i limitów przy indeksowaniu i wyszukiwaniu; działa offline.
- Wyniki retrieval różnią się od referencyjnych (tu: lepsze o 1 trafienie hit@3 w MP1 i MP2). Raporty zawsze pokazują wiersz referencji obok.
- Opcjonalny tryb zgodny z kursem (`embeddings.provider: openai_cache`) dla osób z kluczem OpenAI; bez klucza obsługuje tylko pytania testowe z cache kursu.
- Obraz `app` jest większy o ok. 2,2 GB, a model zajmuje ok. 2,5 GB RAM w każdym procesie, który liczy pytania (`api` od F1, `mcp-server` od F5). Gdyby RAM okazał się problemem, zapasowym wyborem jest mpnet (1,0 GB) z jawną stratą w MP2 hit@1 (0,6).
- **Ograniczenie:** wektory e5-large zwracane przez fastembed nie są znormalizowane do długości 1. Dla podobieństwa kosinusowego (pamięć i `COSINE_DISTANCE` w pgvector) to bez znaczenia, ale **nie wolno** zastępować go iloczynem skalarnym (`INNER_PRODUCT`, `<#>`) ani indeksem HNSW z operatorem iloczynu bez wcześniejszej normalizacji.
- Pomiar czasu dotyczy jednego laptopa (CPU); w kontenerze z ograniczonymi wątkami (`embeddings.local.threads`) czasy będą inne.

## Rozważane alternatywy
- Gemini `text-embedding` (darmowy tier): limity współdzielone z czatem.
- `sentence-transformers` z PyTorchem: obraz większy o ok. 1–2 GB.
- `BAAI/bge-m3`: nieobsługiwany przez `fastembed` 0.9 (ADR-0007).
- `jinaai/jina-embeddings-v3`, mpnet, MiniLM: zmierzone w E1, wyniki wyżej.
