"""Metryki wyszukiwania: hit@1, hit@k, MRR (kontrakt F1, 7.3; jak run_pipeline.py kursu)."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean
from typing import Literal

from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel

from kovalo.eval.datasets import EvalQuery
from kovalo.knowledge.retrieval import search

__all__ = ["QueryResult", "RetrievalResult", "evaluate_retrieval"]

SetName = Literal["mp1", "mp2"]


class QueryResult(BaseModel):
    query_id: str
    question: str
    relevant_docs: list[str]
    top_ids: list[str]  # chunk_id
    top_doc_ids: list[str]
    top_scores: list[float]  # round(…, 4)
    hit_at_1: bool
    hit_at_3: bool  # dowolny chunk istotnego dokumentu w top-k (k = retrieval.top_k)
    first_relevant_rank: int | None  # 1..k


class RetrievalResult(BaseModel):
    set_name: SetName
    k: int
    results: list[QueryResult]
    hits_at_1: int
    hits_at_3: int
    hit_rate_at_1: float  # round(mean, 2), jak w kursie
    hit_rate_at_3: float
    mrr: float  # round(…, 3)


def evaluate_retrieval(
    store: VectorStore, queries: Sequence[EvalQuery], *, k: int, set_name: SetName
) -> RetrievalResult:
    """Wyszukiwanie top-k dla każdego pytania i metryki trafień."""
    if not queries:
        raise ValueError("Brak pytań do ewaluacji.")
    results: list[QueryResult] = []
    for query in queries:
        hits = search(store, query.question, k)
        doc_ids = [str(chunk.metadata["doc_id"]) for chunk, _ in hits]
        relevant = set(query.relevant_docs)
        rank = next((i for i, d in enumerate(doc_ids, start=1) if d in relevant), None)
        results.append(
            QueryResult(
                query_id=query.query_id,
                question=query.question,
                relevant_docs=list(query.relevant_docs),
                top_ids=[str(chunk.metadata["chunk_id"]) for chunk, _ in hits],
                top_doc_ids=doc_ids,
                top_scores=[round(float(score), 4) for _, score in hits],
                hit_at_1=rank == 1,
                hit_at_3=rank is not None,
                first_relevant_rank=rank,
            )
        )
    return RetrievalResult(
        set_name=set_name,
        k=k,
        results=results,
        hits_at_1=sum(r.hit_at_1 for r in results),
        hits_at_3=sum(r.hit_at_3 for r in results),
        hit_rate_at_1=round(float(mean(r.hit_at_1 for r in results)), 2),
        hit_rate_at_3=round(float(mean(r.hit_at_3 for r in results)), 2),
        mrr=round(
            float(
                mean(1 / r.first_relevant_rank if r.first_relevant_rank else 0.0 for r in results)
            ),
            3,
        ),
    )
