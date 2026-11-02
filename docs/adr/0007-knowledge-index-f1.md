# ADR-0007: Indeks wiedzy w F1: PGVectorStore, własny adapter fastembed, cennik Gemini

- Status: przyjęty
- Data: 2026-11-02

## Kontekst
F1 przenosi MP1 do aplikacji. Kurs używa `InMemoryVectorStore`, embeddingów OpenAI z dołączonego cache i cen `gpt-5.4-mini`. ADR-0003 wybrał lokalne embeddingi fastembed, a ADR-0004 Postgres + pgvector przez `langchain-postgres`. Przy projektowaniu kontraktu F1 sprawdziliśmy biblioteki (PyPI i kod pakietów, 2026-11-02):

- `langchain-postgres` 0.0.19: klasa `PGVector` jest od 0.0.14 oznaczona jako przestarzała, zalecana jest `PGVectorStore` (osobna tabela na kolekcję, `PGEngine`, upsert po id, filtr po kolumnach metadanych). Domyślna kolumna id ma typ `UUID`, a nasze id to `POL-001#000`.
- `PGVectorStore.similarity_search_with_score` zwraca **odległość** kosinusową (`<=>`, mniejsza = lepsza), a `InMemoryVectorStore` **podobieństwo** (większe = lepsze). Kurs wymaga podobieństwa i zgodności z własnym cosine.
- `fastembed` 0.9.1: `intfloat/multilingual-e5-large` (1024, 2,24 GB), `jinaai/jina-embeddings-v3` (1024, 2,29 GB), `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768, 1,0 GB), `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384, 0,22 GB). `BAAI/bge-m3` (kandydat z ADR-0003) **nie jest** obsługiwany przez `TextEmbedding`.
- `fastembed` nie dodaje prefiksów `query: ` / `passage: ` wymaganych przez modele E5; `query_embed`/`passage_embed` różnią się tylko dla modeli wielozadaniowych (jina v3: `task_id`).
- `CacheBackedEmbeddings.from_bytes_store(..., query_embedding_cache=True)` zapisuje pytania i dokumenty w **tym samym** magazynie i z tym samym kluczem. Dla modeli asymetrycznych (E5, jina v3) ten sam tekst jako pytanie i jako dokument ma inne wektory.
- `LocalFileStore` dopuszcza w kluczach tylko `[a-zA-Z0-9_.\-/]`.
- Gemini API, płatny tier Standard (ai.google.dev/gemini-api/docs/pricing, sprawdzone 2026-11-02): `gemini-3.6-flash` 0,75 USD wejście / 3,75 USD wyjście (z tokenami myślenia) za 1M tokenów do 2026-12-31, od 2027-01-01 1,50 / 7,50; `gemini-3.5-flash-lite` 0,30 / 2,50.

## Decyzja
1. **Indeks:** `langchain_postgres.PGVectorStore` (nie `PGVector`). Jedna tabela na kolekcję (`db.vector_collection_mp1: kovalo_mp1` w F1, `db.vector_collection: kovalo_ctx` od F2), kolumna id typu `TEXT` z `chunk_id`, kolumny metadanych `doc_id`, `doc_type`, `chunk_index`, pełne metadane w JSON. Odległość kosinusowa, bez indeksu HNSW (47–90 wierszy, pełne przeszukanie jest dokładne). `kovalo ingest` buduje tabelę obok i podmienia ją w jednej transakcji, a w tabeli `kovalo_index_meta` zapisuje model i wymiar. Technologia z ADR-0004 się nie zmienia.
2. **Wynik wyszukiwania:** `kovalo.knowledge.search` zawsze zwraca podobieństwo kosinusowe (`InMemoryVectorStore`: bez zmian, `PGVectorStore`: `1 - odległość`). Tolerancja zgodności z własnym cosine: `1e-6` w pamięci, `1e-5` w Postgresie (pgvector przechowuje `float4`).
3. **Embeddingi:** własny adapter `FastEmbedEmbeddings(Embeddings)` na `fastembed.TextEmbedding` zamiast `langchain_community.embeddings.FastEmbedEmbeddings`. Prefiksy pytań i dokumentów są w `config.yaml` przy modelu. Cache: `CacheBackedEmbeddings` z **osobnymi** magazynami dla dokumentów i pytań (`<cache_dir>/<model>/passage`, `<cache_dir>/<model>/query`). Bez `langchain-community` w zależnościach; dokumenty wczytujemy przez `pathlib` (wymagania kroku 1 kursu dotyczą wyniku, nie narzędzia).
4. **Kandydaci E1:** `intfloat/multilingual-e5-large`, `jinaai/jina-embeddings-v3`, `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (lista w `embeddings.candidates`). `BAAI/bge-m3` odpada. Reguła wyboru: [contracts/f1.md §7.3](../contracts/f1.md). Wynik zapisujemy w ADR-0003.
5. **Tryb `openai_cache`:** odtwarza przebieg referencyjny kursu offline z `data/embeddings/cache` (tylko odczyt; brak tekstu w cache to błąd, nigdy wywołanie OpenAI). Służy jako wiersz „referencja” w E1 i w testach.
6. **Koszty:** nowe klucze `cost.our_model`, `cost.our_prices_per_1m_tokens`, `cost.our_prices_checked_on` obok kluczy PINNED. Szacunek zawsze liczy tokeny `o200k_base` (PINNED), więc obie kolumny różnią się tylko ceną. Wpisujemy cenę obowiązującą w dniu sprawdzenia (0,75 / 3,75), a zmianę od 2027 r. opisujemy w komentarzu i w raporcie.
7. **Testy z prawdziwym modelem ONNX:** nowy marker `model` (pobranie 0,2–2,3 GB), domyślnie wyłączony jak `live` i `integration`.

## Konsekwencje
- Klucze PINNED i ich wartości bez zmian; `config.yaml` dostaje nowe klucze w sekcjach `embeddings`, `cost` i `db` (kontrakt w [contracts/f1.md §4](../contracts/f1.md)).
- Zmiana modelu embeddingów zmienia wymiar wektora, więc wymaga `kovalo ingest` (tabela jest odtwarzana, nie migrowana). Odczyt sprawdza `kovalo_index_meta` i przy niezgodności zgłasza błąd z poleceniem `make ingest`.
- Obraz `app` zawiera jeden model (wybrany), obraz `ui` żadnego. Porównanie kandydatów w E1 robimy lokalnie (`uv run`), bo wymaga pobrania ok. 5,8 GB.
- Własny adapter to ok. 60 linii do utrzymania, ale daje kontrolę nad prefiksami, `local_files_only` i liczeniem wywołań (raport `kovalo ingest`).
- `jinaai/jina-embeddings-v3` ma licencję CC BY-NC 4.0: dopuszczalny w projekcie akademickim, ale przy wyborze go to zaznaczamy.
- Szacunek po cenach Gemini nie uwzględnia tokenów myślenia ani innego tokenizera Gemini; raport podaje to wprost, a jedno wywołanie `--with-llm` pokazuje faktyczne liczby dostawcy.

## Rozważane alternatywy
- `PGVector` (wspólna tabela `langchain_pg_embedding`): prostsze API, ale przestarzałe, a wspólna kolumna wektora nie pozwala trzymać kolekcji o różnych wymiarach z kontrolą długości.
- `langchain_community.embeddings.FastEmbedEmbeddings`: gotowe, ale bez prefiksów E5, a `langchain-community` dokłada dużo zależności dla jednej klasy.
- Jeden magazyn cache dla pytań i dokumentów (jak w kursie): poprawne tylko dla modeli symetrycznych; dla E5 dałoby błędne wektory pytań, gdy pytanie jest identyczne z tekstem chunku.
- Ceny Gemini w `cost.prices_per_1m_tokens`: klucz jest PINNED i test porównuje go z kursem.
- `BAAI/bge-m3` przez `TextEmbedding.add_custom_model`: wymaga własnego eksportu ONNX i sprawdzenia poolingu; nie warto przy trzech gotowych kandydatach.
