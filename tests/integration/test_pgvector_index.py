"""Indeks pgvector (FR-1.4, FR-6.3; kontrakt F1, 5.6). Wymaga compose (marker integration)."""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest
import yaml

pytestmark = pytest.mark.integration

DIM = 16


def _meta(settings, table, n=47, model="fake/model", dimensions=DIM):
    from kovalo.knowledge.pgvector import IndexMeta

    return IndexMeta(
        table=table,
        embedding_provider="fastembed",
        embedding_model=model,
        dimensions=dimensions,
        n_chunks=n,
        updated_at=datetime.now(UTC),
    )


def _fill(settings, table, chunks, fake):
    from kovalo.knowledge.pgvector import pg_engine, replace_table

    engine = pg_engine(settings)
    return engine, replace_table(
        settings, engine, fake, chunks, _meta(settings, table, len(chunks)), table=table
    )


def _count(settings, table):
    with psycopg.connect(settings.postgres_dsn()) as conn:
        return conn.execute(
            f'SELECT count(*), count(DISTINCT langchain_id) FROM "{table}"'
        ).fetchone()


def test_replace_table_idempotent(pg_settings, pg_table, chunks, fake_embeddings):
    _, first = _fill(pg_settings, pg_table, chunks, fake_embeddings)
    assert first == (47, 0)
    _, second = _fill(pg_settings, pg_table, chunks, fake_embeddings)
    assert second == (47, 47)
    assert _count(pg_settings, pg_table) == (47, 47)


def test_pg_search_matches_memory(pg_settings, pg_table, chunks, fake_embeddings):
    from kovalo.knowledge.pgvector import open_store
    from kovalo.knowledge.retrieval import build_vector_store, search

    engine, _ = _fill(pg_settings, pg_table, chunks, fake_embeddings)
    pg_store = open_store(engine, fake_embeddings, table=pg_table)
    mem_store = build_vector_store(chunks, fake_embeddings)
    queries = yaml.safe_load(pg_settings.paths.eval_queries.read_text(encoding="utf-8"))["queries"]
    for q in queries:
        pg_ids = [c.metadata["chunk_id"] for c, _ in search(pg_store, q["question"], k=3)]
        mem_ids = [c.metadata["chunk_id"] for c, _ in search(mem_store, q["question"], k=3)]
        assert pg_ids == mem_ids, q["query_id"]


def test_pg_score_is_cosine(pg_settings, pg_table, chunks, fake_embeddings):
    from kovalo.knowledge.pgvector import open_store
    from kovalo.knowledge.retrieval import cosine_similarity, search

    engine, _ = _fill(pg_settings, pg_table, chunks, fake_embeddings)
    store = open_store(engine, fake_embeddings, table=pg_table)
    question = "Kto musi zatwierdzić rabat 15% dla dużej sieci handlowej?"
    for chunk, score in search(store, question, k=3):
        mine = cosine_similarity(
            fake_embeddings.embed_query(question),
            fake_embeddings.embed_documents([chunk.page_content])[0],
        )
        assert score == pytest.approx(mine, abs=1e-5)


def test_pg_doc_type_filter(pg_settings, pg_table, chunks, fake_embeddings):
    from kovalo.knowledge.pgvector import open_store
    from kovalo.knowledge.retrieval import search

    engine, _ = _fill(pg_settings, pg_table, chunks, fake_embeddings)
    store = open_store(engine, fake_embeddings, table=pg_table)
    results = search(store, "Rabaty i zatwierdzanie", k=5, doc_type="policy")
    assert results and {c.metadata["doc_type"] for c, _ in results} == {"policy"}


def test_index_meta_and_mismatch(pg_settings, pg_table, chunks, fake_embeddings):
    from kovalo.knowledge.pgvector import IndexMismatchError, check_index, read_index_meta

    with pytest.raises(IndexMismatchError, match="make ingest"):
        check_index(pg_settings, pg_table, provider="fastembed", model="fake/model", dimensions=DIM)
    assert read_index_meta(pg_settings, pg_table) is None

    _fill(pg_settings, pg_table, chunks, fake_embeddings)
    meta = check_index(
        pg_settings, pg_table, provider="fastembed", model="fake/model", dimensions=DIM
    )
    assert meta.n_chunks == 47 and meta.embedding_model == "fake/model"
    with pytest.raises(IndexMismatchError, match=r"Uruchom make ingest\."):
        check_index(pg_settings, pg_table, provider="fastembed", model="inny/model", dimensions=DIM)
    with pytest.raises(IndexMismatchError):
        check_index(
            pg_settings, pg_table, provider="fastembed", model="fake/model", dimensions=1024
        )


# ---------------------------------------------------------------------------
# F2: tabela kovalo_ctx i open_retriever (kontrakt F2, 5.3-5.4)
# ---------------------------------------------------------------------------


def test_replace_ctx_table(pg_settings, pg_ctx_table, ctx_chunks, fake_embeddings):
    _, first = _fill(pg_settings, pg_ctx_table, ctx_chunks, fake_embeddings)
    assert first == (87, 0)
    assert _count(pg_settings, pg_ctx_table) == (87, 87)
    with psycopg.connect(pg_settings.postgres_dsn()) as conn:
        row = conn.execute(
            f"SELECT count(*) FILTER (WHERE doc_id IS NOT NULL AND doc_type IS NOT NULL "
            f'AND chunk_index IS NOT NULL), count(DISTINCT doc_id) FROM "{pg_ctx_table}"'
        ).fetchone()
    assert row == (87, 18)
    _, second = _fill(pg_settings, pg_ctx_table, ctx_chunks, fake_embeddings)
    assert second == (87, 87)


@pytest.fixture
def ctx_pg_setup(monkeypatch, pg_ctx_table, ctx_chunks, fake_embeddings):
    """Ustawienia wskazujące na tabelę testową, z wymiarem atrapy (16)."""
    from kovalo.config import load_settings
    from kovalo.knowledge.pgvector import pg_engine, replace_table

    monkeypatch.setenv("KOVALO_DB__VECTOR_COLLECTION", pg_ctx_table)
    monkeypatch.setenv("KOVALO_EMBEDDINGS__LOCAL__DIMENSIONS", str(DIM))
    monkeypatch.setenv("KOVALO_EMBEDDINGS__CANDIDATES", "[]")  # e5 (1024) kłóci się z atrapą
    settings = load_settings(google_api_key="test-key")
    model = settings.embeddings.local.model
    meta = _meta(settings, pg_ctx_table, len(ctx_chunks), model=model)
    replace_table(
        settings, pg_engine(settings), fake_embeddings, ctx_chunks, meta, table=pg_ctx_table
    )
    return settings


def test_open_retriever_hybrid_matches_memory(ctx_pg_setup, ctx_chunks, fake_embeddings):
    from kovalo.knowledge.retrieval import build_vector_store
    from kovalo.knowledge.retrievers import build_retriever, open_retriever

    settings = ctx_pg_setup
    pg = open_retriever(settings, fake_embeddings, variant="ctx", mode="hybrid")
    assert pg.name == "hybrid"
    memory = build_retriever(
        ctx_chunks,
        build_vector_store(ctx_chunks, fake_embeddings),
        mode="hybrid",
        hybrid=settings.retrieval.hybrid,
    )
    questions = yaml.safe_load(settings.paths.mini_project_2_questions.read_text(encoding="utf-8"))
    for q in questions["questions"][:10]:
        got = [c.metadata["chunk_id"] for c, _ in pg.retrieve(q["question"], 3)]
        want = [c.metadata["chunk_id"] for c, _ in memory.retrieve(q["question"], 3)]
        assert got == want, q["id"]


def test_open_retriever_default_mode_from_config(ctx_pg_setup, fake_embeddings):
    from kovalo.knowledge.retrievers import open_retriever

    assert open_retriever(ctx_pg_setup, fake_embeddings).name == "vector"  # config: retrieval.mode


def test_open_retriever_detects_corpus_mismatch(
    monkeypatch, pg_ctx_table, ctx_chunks, fake_embeddings
):
    from kovalo.config import load_settings
    from kovalo.knowledge.pgvector import IndexMismatchError, pg_engine, replace_table
    from kovalo.knowledge.retrievers import open_retriever

    monkeypatch.setenv("KOVALO_DB__VECTOR_COLLECTION", pg_ctx_table)
    monkeypatch.setenv("KOVALO_EMBEDDINGS__LOCAL__DIMENSIONS", str(DIM))
    monkeypatch.setenv("KOVALO_EMBEDDINGS__CANDIDATES", "[]")  # e5 (1024) kłóci się z atrapą
    settings = load_settings(google_api_key="test-key")
    meta = _meta(settings, pg_ctx_table, 86, model=settings.embeddings.local.model)
    replace_table(
        settings, pg_engine(settings), fake_embeddings, ctx_chunks, meta, table=pg_ctx_table
    )
    # replace_table zapisuje faktyczną liczbę wierszy (87), więc rozjazd korpusu symulujemy w meta
    with psycopg.connect(settings.postgres_dsn()) as conn:
        conn.execute(
            f'UPDATE "{settings.db.index_meta_table}" SET n_chunks = 86 WHERE table_name = %s',
            (pg_ctx_table,),
        )
    with pytest.raises(IndexMismatchError, match="make ingest"):
        open_retriever(settings, fake_embeddings, variant="ctx")
