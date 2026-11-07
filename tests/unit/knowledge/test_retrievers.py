"""Retrievery: wektorowy, hybrydowy, RRF (FR-1.5, FR-2.5; kontrakt F2, 5.3 i 9.3)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document


def doc(chunk_id: str, text: str = "x") -> Document:
    return Document(page_content=text, metadata={"chunk_id": chunk_id, "doc_id": chunk_id[:7]})


def ranking(*ids: str) -> list[tuple[Document, float]]:
    # wyniki wejściowe (cosine/BM25) są ignorowane: liczą się pozycje
    return [(doc(i), 99.0 - n) for n, i in enumerate(ids)]


def ids(results) -> list[str]:
    return [c.metadata["chunk_id"] for c, _ in results]


def test_rrf_fuse_table():
    from kovalo.knowledge.retrievers import rrf_fuse

    # dokument w obu listach wygrywa
    fused = rrf_fuse([ranking("A#000", "B#000"), ranking("C#000", "B#000")], weights=[1, 1], c=60)
    assert ids(fused)[0] == "B#000"
    assert len(fused) == 3  # chunk z kilku list tylko raz
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 62)
    assert dict((c.metadata["chunk_id"], s) for c, s in fused)["A#000"] == pytest.approx(1 / 61)

    # wagi
    fused = rrf_fuse([ranking("A#000"), ranking("C#000")], weights=[0.3, 0.7], c=60)
    assert ids(fused) == ["C#000", "A#000"]
    assert fused[0][1] == pytest.approx(0.7 / 61)

    # remis po chunk_id rosnąco
    fused = rrf_fuse([ranking("B#000"), ranking("A#000")], weights=[1, 1], c=60)
    assert ids(fused) == ["A#000", "B#000"]

    # c = 0 i pozycje od 1
    fused = rrf_fuse([ranking("A#000", "B#000")], weights=[2.0], c=0)
    assert [s for _, s in fused] == pytest.approx([2.0, 1.0])

    assert rrf_fuse([[], []], weights=[1, 1], c=60) == []


def test_rrf_fuse_keeps_document_from_first_list():
    from kovalo.knowledge.retrievers import rrf_fuse

    first = [(doc("A#000", "z pierwszej"), 0.1)]
    second = [(doc("A#000", "z drugiej"), 0.9)]
    fused = rrf_fuse([first, second], weights=[1, 1], c=60)
    assert fused[0][0].page_content == "z pierwszej"


@pytest.mark.parametrize(
    ("weights", "n_lists"),
    [([1], 2), ([1, 1, 1], 2), ([-1, 1], 2), ([0, 0], 2)],
    ids=["too_few", "too_many", "negative", "all_zero"],
)
def test_rrf_fuse_bad_weights(weights, n_lists):
    from kovalo.knowledge.retrievers import rrf_fuse

    with pytest.raises(ValueError):
        rrf_fuse([ranking("A#000")] * n_lists, weights=weights, c=60)


def test_vector_retriever_equals_search(toy_store):
    from kovalo.knowledge.retrieval import search
    from kovalo.knowledge.retrievers import VectorRetriever

    retriever = VectorRetriever(toy_store)
    assert retriever.name == "vector"
    question = "Kto zatwierdza wyjazdy zagraniczne?"
    got = retriever.retrieve(question, 2)
    want = search(toy_store, question, 2)
    assert ids(got) == ids(want)
    assert [s for _, s in got] == pytest.approx([s for _, s in want])


def test_hybrid_k_above_fetch_k(toy_store, settings):
    from kovalo.knowledge.lexical import BM25Index
    from kovalo.knowledge.retrievers import HybridRetriever
    from tests.fakes import TOY_CHUNKS

    params = settings.retrieval.hybrid
    retriever = HybridRetriever(toy_store, BM25Index(TOY_CHUNKS, params.bm25), params)
    assert retriever.name == "hybrid"
    with pytest.raises(ValueError):
        retriever.retrieve("pytanie", params.fetch_k + 1)


def test_hybrid_is_rrf_of_vector_and_bm25(toy_store, settings):
    from kovalo.knowledge.lexical import BM25Index
    from kovalo.knowledge.retrieval import search
    from kovalo.knowledge.retrievers import HybridRetriever, rrf_fuse
    from tests.fakes import TOY_CHUNKS

    params = settings.retrieval.hybrid
    lexical = BM25Index(TOY_CHUNKS, params.bm25)
    question = "Kto zatwierdza wyjazdy zagraniczne CFO?"
    got = HybridRetriever(toy_store, lexical, params).retrieve(question, 2)
    want = rrf_fuse(
        [search(toy_store, question, params.fetch_k), lexical.search(question, params.fetch_k)],
        weights=[params.weights.vector, params.weights.bm25],
        c=params.rrf_c,
    )[:2]
    assert ids(got) == ids(want)
    assert [s for _, s in got] == pytest.approx([s for _, s in want])


def test_build_retriever_modes(toy_store, settings):
    from kovalo.knowledge.retrievers import HybridRetriever, VectorRetriever, build_retriever
    from tests.fakes import TOY_CHUNKS

    params = settings.retrieval.hybrid
    vec = build_retriever(TOY_CHUNKS, toy_store, mode="vector", hybrid=params)
    hyb = build_retriever(TOY_CHUNKS, toy_store, mode="hybrid", hybrid=params)
    assert isinstance(vec, VectorRetriever) and isinstance(hyb, HybridRetriever)


@pytest.fixture(scope="module")
def reference_hybrid(ctx_chunks, session_settings):
    """Strona reference: wektory z cache kursu (bez modelu i sieci), chunki z kontekstem."""
    from kovalo.knowledge.embeddings import build_course_cache_embeddings
    from kovalo.knowledge.retrieval import build_vector_store
    from kovalo.knowledge.retrievers import build_retriever
    from tests.conftest import ForbiddenEmbeddings

    emb = build_course_cache_embeddings(
        session_settings.paths.embedding_cache,
        model=session_settings.embeddings.model,
        underlying=ForbiddenEmbeddings(),
    )
    store = build_vector_store(ctx_chunks, emb)
    return {
        mode: build_retriever(
            ctx_chunks, store, mode=mode, hybrid=session_settings.retrieval.hybrid
        )
        for mode in ("vector", "hybrid")
    }


def test_reference_hybrid_finds_sup_114(reference_hybrid):
    # FR-2.5: M03 "Co SUP-114 mówi o karach?"
    top = reference_hybrid["hybrid"].retrieve("Co SUP-114 mówi o karach?", 3)
    assert ids(top) == ["CON-101#ctx002", "CON-101#ctx003", "CON-101#ctx000"]
    vector_only = reference_hybrid["vector"].retrieve("Co SUP-114 mówi o karach?", 3)
    assert ids(vector_only) == ["CON-102#ctx004", "CON-102#ctx002", "CON-102#ctx001"]


# ---------------------------------------------------------------------------
# F3 (kontrakt F3, 7.1): build_memory_retriever, wspólne dla E2 i E3
# ---------------------------------------------------------------------------


def test_build_memory_retriever_variants(settings, fake_embeddings):
    from kovalo.knowledge.retrievers import HybridRetriever, VectorRetriever, build_memory_retriever

    ctx = build_memory_retriever(settings, fake_embeddings, variant="ctx")
    assert isinstance(ctx, VectorRetriever)  # retrieval.mode: vector
    assert len(ctx.store.store) == 87, "wariant ctx: 87 chunków w magazynie"

    mp1 = build_memory_retriever(settings, fake_embeddings, variant="mp1")
    assert len(mp1.store.store) == 47

    hybrid = build_memory_retriever(settings, fake_embeddings, variant="ctx", mode="hybrid")
    assert isinstance(hybrid, HybridRetriever)
    hits = hybrid.retrieve("Kto zatwierdza rabat powyżej 20%?", 3)
    assert len(hits) == 3


# ---------------------------------------------------------------------------
# F5: LazyRetriever przeniesiony do knowledge (kontrakt F5, 2 i 13.2)
# ---------------------------------------------------------------------------


def test_lazy_retriever_opens_once_and_retries_after_error(toy_retriever):
    """`LazyRetriever(opener)`: jedno otwarcie przy wielu wątkach; wyjątek otwarcia leci
    z `retrieve`, a następne `retrieve` próbuje ponownie; po sukcesie nie otwiera już więcej."""
    import threading
    import time

    from kovalo.knowledge.pgvector import IndexUnavailableError
    from kovalo.knowledge.retrievers import LazyRetriever

    attempts: list[int] = []

    def opener():
        attempts.append(1)
        time.sleep(0.05)  # okno na wyścig wątków
        if len(attempts) == 1:
            raise IndexUnavailableError("Postgres niedostępny")
        return toy_retriever

    lazy = LazyRetriever(opener)
    with pytest.raises(IndexUnavailableError):
        lazy.retrieve("kto zatwierdza", 2)
    assert len(attempts) == 1

    results: list = []
    errors: list = []

    def worker():
        try:
            results.append(lazy.retrieve("kto zatwierdza", 2))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert errors == [] and len(results) == 8
    assert len(attempts) == 2, "drugie otwarcie (po błędzie) raz, mimo ośmiu wątków"
    assert results[0] == toy_retriever.retrieve("kto zatwierdza", 2)
    assert lazy.name == "lazy"


def test_lazy_retriever_has_no_llm_import():
    """`knowledge/` nie importuje kovalo.llm: opener przychodzi z zewnątrz (kontrakt F5, 2.3)."""
    import inspect

    import kovalo.knowledge.retrievers as module

    assert "kovalo.llm" not in inspect.getsource(module)
