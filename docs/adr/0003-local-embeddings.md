# ADR-0003: Lokalne wielojęzyczne embeddingi (fastembed)

- Status: proponowany (model ostateczny po pomiarze w F1)
- Data: 2026-11-01

## Kontekst
Kurs dostarcza cache embeddingów OpenAI `text-embedding-3-small` dla chunków i pytań testowych, ale każde nowe pytanie użytkownika wymaga płatnego wywołania OpenAI. Gemini ma darmowe embeddingi, ale z limitami tego samego tieru co czat.

## Decyzja
Embeddingi liczymy lokalnie przez **fastembed** (ONNX, CPU, bez PyTorcha, mały obraz), opakowane w `CacheBackedEmbeddings`. Kandydaci: `intfloat/multilingual-e5-large`, `BAAI/bge-m3`, `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`. Wybieramy w F1 po hit@1/hit@3 na `mini_project_1_queries.yaml` i M01–M12 z MP2. Model pobieramy w czasie budowania obrazu.

## Konsekwencje
- Zero kosztów i limitów przy indeksowaniu i wyszukiwaniu; działa offline.
- Wyniki retrieval różnią się od referencyjnych.
- Opcjonalny tryb zgodny z kursem (`embeddings.provider: openai_cache`) dla osób z kluczem OpenAI.

## Rozważane alternatywy
- Gemini `text-embedding` (darmowy tier): limity współdzielone z czatem.
- `sentence-transformers` z PyTorchem: obraz większy o ok. 1–2 GB.
