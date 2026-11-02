"""BM25 (FR-2.5; kontrakt F2, 5.2)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document


def doc(chunk_id: str, text: str) -> Document:
    return Document(page_content=text, metadata={"chunk_id": chunk_id, "doc_id": chunk_id[:7]})


@pytest.fixture
def params(settings):
    return settings.retrieval.hybrid.bm25


TOY = [
    doc("CON-101#ctx000", "Umowa Mleczna Dolina (SUP-114): kara umowna za OTIF"),
    doc("POL-001#ctx000", "Rabaty powyżej 12% zatwierdza dyrektor handlowy"),
    doc("POL-002#ctx000", "Nowi klienci płacą w ciągu 14 dni"),
]


def test_tokenize_keeps_ids():
    from kovalo.knowledge.lexical import tokenize

    assert tokenize("Co SUP-114 mówi o karach?") == ["co", "sup-114", "mówi", "o", "karach"]
    assert tokenize("(SUP-114) CON-101#ctx003") == ["sup-114", "con-101", "ctx003"]
    assert tokenize("") == []
    assert tokenize("?!  ...") == []
    assert tokenize("ŻÓŁĆ Zażółć") == ["żółć", "zażółć"]


def test_bm25_finds_exact_id(params):
    from kovalo.knowledge.lexical import BM25Index

    index = BM25Index(TOY, params)
    assert len(index) == 3
    hits = index.search("SUP-114", 3)
    assert [c.metadata["chunk_id"] for c, _ in hits] == ["CON-101#ctx000"]
    assert hits[0][1] > 0
    assert index.search("zupełnie obce słowa", 3) == []
    assert index.search("?!", 3) == []


def test_bm25_results_best_first_and_limited(params):
    from kovalo.knowledge.lexical import BM25Index

    index = BM25Index(TOY, params)
    hits = index.search("klienci płacą rabaty dyrektor", 1)
    assert len(hits) == 1
    all_hits = index.search("klienci płacą rabaty dyrektor", 3)
    scores = [s for _, s in all_hits]
    assert scores == sorted(scores, reverse=True) and all(s > 0 for s in scores)


def test_bm25_tie_break_and_validation(params):
    from kovalo.knowledge.lexical import BM25Index

    twins = [doc("POL-002#ctx000", "alfa beta"), doc("POL-001#ctx000", "alfa beta"), TOY[0]]
    hits = BM25Index(twins, params).search("alfa", 5)
    assert [c.metadata["chunk_id"] for c, _ in hits] == ["POL-001#ctx000", "POL-002#ctx000"]
    index = BM25Index(TOY, params)
    with pytest.raises(ValueError):
        index.search("SUP-114", 0)
    with pytest.raises(ValueError):
        BM25Index([], params)
    with pytest.raises(ValueError):
        BM25Index([TOY[0], TOY[0]], params)


def test_bm25_on_real_corpus_finds_supplier_code(ctx_chunks, params):
    from kovalo.knowledge.lexical import BM25Index

    index = BM25Index(ctx_chunks, params)
    assert len(index) == 87
    top = index.search("Co SUP-114 mówi o karach?", 3)
    assert top and top[0][0].metadata["doc_id"] == "CON-101"
