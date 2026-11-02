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


# ---------------------------------------------------------------------------
# F2: dwie tabele (kontrakt F2, 5.4)
# ---------------------------------------------------------------------------


def test_ingest_ctx_counts_and_table(settings, tmp_path, fake_embeddings):
    from kovalo.knowledge.ingest import ingest_chunks

    written: list[tuple[list[Document], object]] = []

    def writer(chunks, meta):
        written.append((list(chunks), meta))
        return 87, (0 if len(written) == 1 else 87)

    emb, _ = _stack(settings, tmp_path / "cache", fake_embeddings)
    first = ingest_chunks(settings, variant="ctx", embeddings=emb, writer=writer)
    assert first.variant == "ctx"
    assert first.table == settings.db.vector_collection == "kovalo_ctx"
    assert first.chunks.n_chunks == 87 and first.chunks.max_chars == 775
    assert (first.embedded, first.from_cache) == (87, 0)
    chunks, meta = written[0]
    assert len(chunks) == 87 and len({c.metadata["chunk_id"] for c in chunks}) == 87
    assert meta.table == "kovalo_ctx" and meta.n_chunks == 87

    emb2, _ = _stack(settings, tmp_path / "cache", fake_embeddings)
    second = ingest_chunks(settings, variant="ctx", embeddings=emb2, writer=writer)
    assert (second.embedded, second.from_cache) == (0, 87)


def test_ingest_chunks_mp1_variant_matches_ingest_mp1(settings, fake_embeddings):
    from kovalo.knowledge.ingest import ingest_chunks, ingest_mp1

    seen = []
    via_chunks = ingest_chunks(
        settings,
        variant="mp1",
        embeddings=fake_embeddings,
        writer=lambda c, m: (seen.append(m.table), (47, 0))[1],
    )
    legacy = ingest_mp1(settings, embeddings=fake_embeddings, writer=lambda c, m: (47, 0))
    assert via_chunks.variant == legacy.variant == "mp1"
    assert via_chunks.table == legacy.table == "kovalo_mp1"
    assert via_chunks.chunks == legacy.chunks
    assert seen == ["kovalo_mp1"]
