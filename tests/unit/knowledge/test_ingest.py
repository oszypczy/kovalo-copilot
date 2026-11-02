"""Ingest MP1 (FR-6.3; kontrakt F1, 5.7), bez Postgresa: writer jest atrapą."""

from __future__ import annotations

from langchain_core.documents import Document


def _stack(settings, cache_dir, fake):
    from kovalo.knowledge.embeddings import CountingEmbeddings, build_cached_embeddings

    counting = CountingEmbeddings(fake)
    return build_cached_embeddings(counting, cache_dir=cache_dir, model="fake/model"), counting


def test_ingest_counts_embeddings_and_cache(settings, tmp_path, fake_embeddings):
    from kovalo.knowledge.ingest import ingest_mp1

    written: list[tuple[list[Document], object]] = []

    def writer(chunks, meta):
        written.append((list(chunks), meta))
        return 47, (0 if len(written) == 1 else 47)

    emb, _ = _stack(settings, tmp_path / "cache", fake_embeddings)
    first = ingest_mp1(settings, embeddings=emb, writer=writer)
    assert (first.embedded, first.from_cache) == (47, 0)
    assert (first.rows_after, first.rows_before) == (47, 0)
    assert first.table == settings.db.vector_collection_mp1
    assert first.corpus.n_documents == 18 and first.corpus.total_tokens == 8762
    assert first.chunks.n_chunks == 47 and first.chunks.max_chars == 798
    assert first.seconds >= 0

    # drugi przebieg: nowy stos (nowy licznik) na tym samym cache na dysku
    emb2, _ = _stack(settings, tmp_path / "cache", fake_embeddings)
    second = ingest_mp1(settings, embeddings=emb2, writer=writer)
    assert (second.embedded, second.from_cache) == (0, 47)
    assert (second.rows_after, second.rows_before) == (47, 47)

    chunks, meta = written[0]
    assert len(chunks) == 47
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(set(ids)) == 47
    assert meta.table == settings.db.vector_collection_mp1
    assert meta.n_chunks == 47
    assert meta.embedding_provider == settings.embeddings.provider
    assert meta.embedding_model == settings.embeddings.local.model


def test_ingest_without_counting_wrapper_reports_zero_embedded(settings, fake_embeddings):
    from kovalo.knowledge.ingest import ingest_mp1

    stats = ingest_mp1(settings, embeddings=fake_embeddings, writer=lambda c, m: (47, 0))
    assert (stats.embedded, stats.from_cache) == (0, 47)
