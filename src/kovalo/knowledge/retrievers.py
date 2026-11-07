"""Wspólny interfejs wyszukiwania: wektorowe, hybrydowe (BM25 + wektory, RRF) i Postgres.

FR-1.5, FR-2.5; kontrakt F2, sekcja 5.3; ADR-0008 pkt 2 (własna fuzja zamiast
EnsembleRetriever, bez langchain-community). Moduł nie importuje langchain_postgres:
open_retriever sięga po kovalo.knowledge.pgvector leniwie.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from kovalo.config import HybridConfig, RetrievalMode, Settings
from kovalo.knowledge.chunking import ChunkVariant, build_chunks
from kovalo.knowledge.corpus import load_corpus
from kovalo.knowledge.lexical import BM25Index
from kovalo.knowledge.retrieval import ScoredChunk, build_vector_store, search

__all__ = [
    "HybridRetriever",
    "LazyRetriever",
    "Retriever",
    "VectorRetriever",
    "build_memory_retriever",
    "build_retriever",
    "open_retriever",
    "rrf_fuse",
]


class Retriever(Protocol):
    """Wyszukiwanie top-k: lista par (chunk, wynik), od najlepszej."""

    @property
    def name(self) -> str:  # "vector" | "hybrid"
        ...

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]: ...


@dataclass(frozen=True)
class VectorRetriever:
    """Wyszukiwanie po znaczeniu; wynik = podobieństwo kosinusowe (F1, 5.5)."""

    store: VectorStore
    name: str = "vector"

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        return search(self.store, question, k)


def rrf_fuse(
    rankings: Sequence[Sequence[ScoredChunk]], *, weights: Sequence[float], c: int
) -> list[ScoredChunk]:
    """Ważona Reciprocal Rank Fusion po metadata['chunk_id']: Σ w_i / (c + pozycja_i), od 1.

    Chunk z kilku list występuje raz (Document z pierwszej listy, w której wystąpił).
    Malejąco po wyniku, remis → rosnąco po chunk_id. Wyniki wejściowe są ignorowane.
    """
    if len(rankings) != len(weights):
        raise ValueError(f"Liczba list ({len(rankings)}) różni się od liczby wag ({len(weights)}).")
    if any(w < 0 for w in weights):
        raise ValueError(f"Wagi RRF nie mogą być ujemne: {list(weights)}")
    if not any(w > 0 for w in weights):
        raise ValueError("Co najmniej jedna waga RRF musi być większa od 0.")
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for position, (doc, _) in enumerate(ranking, start=1):
            chunk_id = str(doc.metadata["chunk_id"])
            docs.setdefault(chunk_id, doc)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (c + position)
    order = sorted(scores, key=lambda i: (-scores[i], i))
    return [(docs[i], scores[i]) for i in order]


@dataclass(frozen=True)
class HybridRetriever:
    """BM25 + wektory, fuzja RRF; wynik = wynik RRF (nie cosine)."""

    store: VectorStore
    lexical: BM25Index
    params: HybridConfig
    name: str = "hybrid"

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        fetch_k = self.params.fetch_k
        if k > fetch_k:
            raise ValueError(f"k ({k}) nie może być większe niż hybrid.fetch_k ({fetch_k})")
        vector = search(self.store, question, fetch_k)
        lexical = self.lexical.search(question, fetch_k)
        weights = self.params.weights
        fused = rrf_fuse(
            [vector, lexical], weights=[weights.vector, weights.bm25], c=self.params.rrf_c
        )
        return fused[:k]


def build_retriever(
    chunks: Sequence[Document],
    store: VectorStore,
    *,
    mode: RetrievalMode,
    hybrid: HybridConfig,
) -> Retriever:
    """'vector' → VectorRetriever; 'hybrid' → HybridRetriever z BM25 nad tymi samymi chunkami."""
    if mode == "vector":
        return VectorRetriever(store)
    if mode == "hybrid":
        return HybridRetriever(store, BM25Index(chunks, hybrid.bm25), hybrid)
    raise ValueError(f"Nieznany tryb wyszukiwania: {mode!r} (dozwolone: vector, hybrid)")


def build_memory_retriever(
    settings: Settings,
    embeddings: Embeddings,
    *,
    variant: ChunkVariant = "ctx",
    mode: RetrievalMode | None = None,
    documents: Sequence[Document] | None = None,
) -> Retriever:
    """Korpus (albo documents) → build_chunks(variant) → InMemoryVectorStore → build_retriever.

    Wyszukiwanie w pamięci, wspólne dla E2 i E3 (kontrakt F3, 7.1); mode None → retrieval.mode.
    """
    docs = list(documents) if documents is not None else load_corpus(settings)
    chunks = build_chunks(docs, settings.chunking, variant)
    return build_retriever(
        chunks,
        build_vector_store(chunks, embeddings),
        mode=mode or settings.retrieval.mode,
        hybrid=settings.retrieval.hybrid,
    )


def open_retriever(
    settings: Settings,
    embeddings: Embeddings,
    *,
    variant: ChunkVariant = "ctx",
    mode: RetrievalMode | None = None,
) -> Retriever:
    """Wyszukiwanie produkcyjne w Postgresie (tabela wariantu) + BM25 z korpusu.

    Niezgodny model/wymiar albo liczba chunków korpusu ≠ meta.n_chunks → IndexMismatchError.
    """
    from kovalo.knowledge.pgvector import (
        IndexMismatchError,
        check_index,
        configured_embedding_model,
        index_table,
        open_store,
        pg_engine,
    )

    table = index_table(settings, variant)
    model, dimensions = configured_embedding_model(settings)
    meta = check_index(
        settings,
        table,
        provider=settings.embeddings.provider,
        model=model,
        dimensions=dimensions,
    )
    chunks = build_chunks(load_corpus(settings), settings.chunking, variant)
    if len(chunks) != meta.n_chunks:
        raise IndexMismatchError(
            f"Indeks {table} ma {meta.n_chunks} chunków, a korpus daje {len(chunks)} "
            f"(wariant {variant}): korpus i indeks się różnią. Uruchom make ingest."
        )
    store = open_store(pg_engine(settings), embeddings, table=table)
    return build_retriever(
        chunks,
        store,
        mode=mode or settings.retrieval.mode,
        hybrid=settings.retrieval.hybrid,
    )


class LazyRetriever:
    """Retriever otwierany przy pierwszym retrieve() funkcją `opener` (F5: przeniesiony
    z workflow/factory.py; opener przychodzi z zewnątrz, więc knowledge/ nie zna modeli).

    Błąd otwarcia (IndexUnavailableError, IndexMismatchError, ConfigError) idzie wyżej
    z retrieve() (narzędzie → błąd w trace albo isError w MCP); kolejne retrieve() próbuje
    ponownie. Wątkowo bezpieczny: jedno otwarcie przy wielu wątkach.
    """

    name = "lazy"

    def __init__(self, opener: Callable[[], Retriever]) -> None:
        self._opener = opener
        self._retriever: Retriever | None = None
        self._lock = threading.Lock()

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        with self._lock:
            if self._retriever is None:
                self._retriever = self._opener()
            retriever = self._retriever
        return retriever.retrieve(question, k)
