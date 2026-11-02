from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Chunki leżą na okręgu w rosnącej odległości kątowej od wektora pytań [1, 0].
VECTORS = {
    "c1": [1.0, 0.1],
    "c2": [1.0, 0.5],
    "c3": [1.0, 1.0],
    "c4": [0.0, 1.0],
}


class ControlledEmbeddings(Embeddings):
    def embed_documents(self, texts):  # type: ignore[no-untyped-def]
        return [VECTORS[t] for t in texts]

    def embed_query(self, text):  # type: ignore[no-untyped-def]
        return [1.0, 0.0]


def test_hit_and_mrr():
    from kovalo.eval.datasets import EvalQuery
    from kovalo.eval.retrieval.metrics import evaluate_retrieval
    from kovalo.knowledge.retrieval import build_vector_store

    chunks = [
        Document(
            page_content=text,
            metadata={"chunk_id": f"D{i}#000", "doc_id": f"D{i}", "doc_type": "policy"},
        )
        for i, text in enumerate(["c1", "c2", "c3", "c4"], start=1)
    ]
    store = build_vector_store(chunks, ControlledEmbeddings())
    queries = [
        EvalQuery("A", "pytanie A", ("D1",)),  # 1. miejsce
        EvalQuery("B", "pytanie B", ("D3", "D9")),  # 3. miejsce
        EvalQuery("C", "pytanie C", ("D4",)),  # 4. miejsce: poza k=3
    ]
    res = evaluate_retrieval(store, queries, k=3, set_name="mp1")
    assert res.set_name == "mp1" and res.k == 3
    a, b, c = res.results
    assert [r.first_relevant_rank for r in res.results] == [1, 3, None]
    assert (a.hit_at_1, a.hit_at_3) == (True, True)
    assert (b.hit_at_1, b.hit_at_3) == (False, True)
    assert (c.hit_at_1, c.hit_at_3) == (False, False)
    assert a.top_ids == ["D1#000", "D2#000", "D3#000"]
    assert a.top_doc_ids == ["D1", "D2", "D3"]
    assert len(a.top_scores) == 3 and a.top_scores == [round(s, 4) for s in a.top_scores]
    assert a.top_scores == sorted(a.top_scores, reverse=True)
    assert (res.hits_at_1, res.hits_at_3) == (1, 2)
    assert res.hit_rate_at_1 == 0.33 and res.hit_rate_at_3 == 0.67
    assert res.mrr == pytest.approx(0.444)
    assert a.relevant_docs == ["D1"] and a.question == "pytanie A"
