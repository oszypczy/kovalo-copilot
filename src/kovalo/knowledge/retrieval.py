"""Wyszukiwanie po znaczeniu (FR-1.5, krok 5 kursu; kontrakt F1, 5.5, ADR-0007 pkt 2).

`search` zawsze zwraca podobieństwo kosinusowe (większe = lepsze), także dla PGVectorStore,
który zwraca odległość. Moduł nie importuje langchain_postgres: PGVectorStore rozpoznaje
leniwie przez kovalo.knowledge.pgvector.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

__all__ = [
    "ScoredChunk",
    "build_vector_store",
    "cosine_similarity",
    "search",
    "similarity_from_score",
]

ScoredChunk = tuple[Document, float]


def build_vector_store(chunks: Sequence[Document], embeddings: Embeddings) -> InMemoryVectorStore:
    """InMemoryVectorStore z chunkami; identyfikator = metadata['chunk_id'] (unikalny)."""
    ids = [c.metadata["chunk_id"] for c in chunks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Powtórzone chunk_id: {', '.join(duplicates)}")
    store = InMemoryVectorStore(embedding=embeddings)
    if chunks:
        store.add_documents(list(chunks), ids=ids)
    return store


def similarity_from_score(store: VectorStore, score: float) -> float:
    """Wynik store'u → podobieństwo kosinusowe.

    InMemoryVectorStore: bez zmian; PGVectorStore z odległością kosinusową: 1 - odległość;
    inna strategia albo typ store'u → TypeError.
    """
    if isinstance(store, InMemoryVectorStore):
        return float(score)
    from kovalo.knowledge.pgvector import is_cosine_distance, is_pg_store

    if is_pg_store(store):
        if not is_cosine_distance(store):  # type: ignore[arg-type]
            raise TypeError("PGVectorStore musi używać odległości kosinusowej (COSINE_DISTANCE).")
        return 1.0 - float(score)
    raise TypeError(f"Nieobsługiwany vector store: {type(store).__name__}")


def search(
    store: VectorStore, question: str, k: int, doc_type: str | None = None
) -> list[ScoredChunk]:
    """k par (chunk, podobieństwo kosinusowe), od najlepszej; opcjonalny filtr doc_type."""
    if k < 1:
        raise ValueError(f"k musi być ≥ 1, jest {k}")
    if isinstance(store, InMemoryVectorStore):
        if doc_type is None:
            return store.similarity_search_with_score(question, k=k)
        wanted = doc_type
        return store.similarity_search_with_score(
            question, k=k, filter=lambda d: d.metadata.get("doc_type") == wanted
        )
    from kovalo.knowledge.pgvector import is_pg_store

    if is_pg_store(store):
        pg_filter = {"doc_type": doc_type} if doc_type is not None else None
        results = store.similarity_search_with_score(question, k=k, filter=pg_filter)
        return [(doc, similarity_from_score(store, score)) for doc, score in results]
    raise TypeError(f"Nieobsługiwany vector store: {type(store).__name__}")


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """dot(a, b) / (|a|·|b|) jako float Pythona. Wektor zerowy albo różne długości → ValueError."""
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if va.shape != vb.shape or va.ndim != 1:
        raise ValueError(f"Wektory mają różne wymiary: {va.shape} i {vb.shape}")
    norm = float(np.linalg.norm(va)) * float(np.linalg.norm(vb))
    if norm == 0:
        raise ValueError("Podobieństwo kosinusowe nie jest określone dla wektora zerowego.")
    return float(np.dot(va, vb) / norm)
