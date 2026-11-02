"""Wyszukiwanie (FR-1.5; port 7 testów kroku 5 kursu + kontrakt F1, 5.5 i 8.1)."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


def chunk(chunk_id: str, text: str, doc_type: str = "policy") -> Document:
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "doc_id": chunk_id[:7], "doc_type": doc_type},
    )


CHUNKS = [
    chunk("POL-001#000", "Rabaty powyżej 12% zatwierdza dyrektor handlowy."),
    chunk("POL-002#000", "Nowi klienci płacą w ciągu 14 dni."),
    chunk("REP-301#000", "OTIF do klientów spadł w Q2 do 94,1%.", doc_type="report"),
    chunk("MTG-402#000", "Od września ceny wody rosną o 4%.", doc_type="meeting_notes"),
]


@pytest.fixture
def store(fake_embeddings):
    from kovalo.knowledge.retrieval import build_vector_store

    return build_vector_store(CHUNKS, fake_embeddings)


def test_store_uses_chunk_ids(store):
    ids = [c.metadata["chunk_id"] for c in CHUNKS]
    assert sorted(d.id for d in store.get_by_ids(ids)) == sorted(ids)


def test_duplicate_chunk_id_raises(fake_embeddings):
    from kovalo.knowledge.retrieval import build_vector_store

    with pytest.raises(ValueError):
        build_vector_store([CHUNKS[0], CHUNKS[0]], fake_embeddings)


def test_search_returns_k_pairs_best_first(store):
    from kovalo.knowledge.retrieval import search

    results = search(store, "Nowi klienci płacą w ciągu 14 dni.", k=3)
    assert len(results) == 3
    top_chunk, top_score = results[0]
    assert top_chunk.metadata["chunk_id"] == "POL-002#000"
    assert top_score == pytest.approx(1.0)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_filters_by_doc_type(store):
    from kovalo.knowledge.retrieval import search

    results = search(store, "OTIF do klientów spadł w Q2 do 94,1%.", k=3, doc_type="policy")
    assert [c.metadata["doc_type"] for c, _ in results] == ["policy", "policy"]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1, 0], [1, 0], 1.0),
        ([1, 0], [0, 1], 0.0),
        ([1, 0], [-1, 0], -1.0),
        ([1, 0], [1, 1], 1 / np.sqrt(2)),
        ([3, 4], [6, 8], 1.0),
    ],
)
def test_cosine_similarity(a, b, expected):
    from kovalo.knowledge.retrieval import cosine_similarity

    got = cosine_similarity(a, b)
    assert isinstance(got, float)
    assert got == pytest.approx(expected)


def test_cosine_similarity_invalid_raises():
    from kovalo.knowledge.retrieval import cosine_similarity

    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 1])
    with pytest.raises(ValueError):
        cosine_similarity([1, 0], [1, 0, 0])


def test_store_score_is_your_cosine_similarity(store, fake_embeddings):
    from kovalo.knowledge.retrieval import cosine_similarity, search

    question = "Kto zatwierdza rabat 15%?"
    top_chunk, score = search(store, question, k=1)[0]
    mine = cosine_similarity(
        fake_embeddings.embed_query(question), fake_embeddings.embed_query(top_chunk.page_content)
    )
    assert score == pytest.approx(mine)


def test_similarity_from_score_in_memory_is_identity(store):
    from kovalo.knowledge.retrieval import similarity_from_score

    assert similarity_from_score(store, 0.3) == 0.3


def test_search_k_validation_and_unknown_store(store):
    from kovalo.knowledge.retrieval import search

    with pytest.raises(ValueError):
        search(store, "pytanie", k=0)

    class OtherStore(VectorStore):
        @classmethod
        def from_texts(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def similarity_search(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return []

    with pytest.raises(TypeError):
        search(OtherStore(), "pytanie", k=3)


REFERENCE_TOP3 = {
    "Q01": ["POL-001#002", "POL-001#001", "POL-001#000"],
    "Q02": ["POL-003#000", "POL-003#001", "PRO-201#002"],
    "Q03": ["CON-101#001", "POL-005#001", "POL-005#000"],
    "Q04": ["POL-002#000", "POL-002#001", "PRO-201#000"],
    "Q05": ["FAQ-501#000", "FAQ-501#001", "REP-301#001"],
    "Q06": ["PRO-203#001", "PRO-203#002", "PRO-203#000"],
    "Q07": ["PL-601#000", "PRO-203#001", "PRO-203#002"],
    "Q08": ["REP-302#000", "REP-302#002", "FAQ-501#002"],
    "Q09": ["MTG-402#000", "CON-102#000", "CON-101#000"],
    "Q10": ["PRO-203#002", "PRO-202#001", "CON-102#000"],
}


def test_reference_retrieval_top_ids(chunks, settings, forbidden_embeddings):
    from kovalo.cost.tokens import count_tokens
    from kovalo.knowledge.embeddings import build_course_cache_embeddings
    from kovalo.knowledge.prompts import build_prompt
    from kovalo.knowledge.retrieval import build_vector_store, search

    emb = build_course_cache_embeddings(
        settings.paths.embedding_cache,
        model=settings.embeddings.model,
        underlying=forbidden_embeddings,
    )
    store = build_vector_store(chunks, emb)
    queries = yaml.safe_load(settings.paths.eval_queries.read_text(encoding="utf-8"))["queries"]
    got = {}
    for q in queries:
        results = search(store, q["question"], k=settings.retrieval.top_k)
        got[q["query_id"]] = [c.metadata["chunk_id"] for c, _ in results]
    assert got == REFERENCE_TOP3

    q01 = queries[0]
    results = search(store, q01["question"], k=3)
    assert [round(s, 4) for _, s in results] == pytest.approx([0.5826, 0.5779, 0.5361], abs=1e-4)
    prompt = build_prompt(settings.prompts.system, [c for c, _ in results], q01["question"])
    assert count_tokens(prompt, settings.tokenizer.encoding) == 808
