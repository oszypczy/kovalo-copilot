"""Ingest MP1: korpus → chunki → embeddingi (cache) → zapis indeksu (FR-6.3; kontrakt F1, 5.7)."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from kovalo.config import Settings
from kovalo.knowledge.chunking import ChunkStats, chunk_documents, chunk_stats
from kovalo.knowledge.corpus import CorpusStats, corpus_stats, load_corpus
from kovalo.knowledge.embeddings import CountingEmbeddings
from kovalo.knowledge.pgvector import IndexMeta

__all__ = ["IndexWriter", "IngestStats", "configured_embedding_model", "ingest_mp1"]

IndexWriter = Callable[[Sequence[Document], IndexMeta], tuple[int, int]]


class IngestStats(BaseModel):
    corpus: CorpusStats
    chunks: ChunkStats
    table: str
    embedding_provider: str
    embedding_model: str
    dimensions: int
    embedded: int  # teksty policzone przez model w tym przebiegu (nie z cache)
    from_cache: int  # n_chunks - embedded
    rows_after: int
    rows_before: int
    seconds: float


def configured_embedding_model(settings: Settings) -> tuple[str, int]:
    """(model, wymiar) indeksu według embeddings.provider."""
    emb = settings.embeddings
    if emb.provider == "openai_cache":
        return emb.model, emb.dimensions
    return emb.local.model, emb.local.dimensions


def _counter(embeddings: Embeddings) -> CountingEmbeddings | None:
    if isinstance(embeddings, CountingEmbeddings):
        return embeddings
    if isinstance(embeddings, CacheBackedEmbeddings) and isinstance(
        embeddings.underlying_embeddings, CountingEmbeddings
    ):
        return embeddings.underlying_embeddings
    return None


def ingest_mp1(settings: Settings, *, embeddings: Embeddings, writer: IndexWriter) -> IngestStats:
    """load_corpus → chunk_documents → embed_documents (przez cache) → writer(chunks, meta).

    `embedded` to przyrost licznika CountingEmbeddings w tym przebiegu (0, gdy model nie jest
    opakowany licznikiem). Wektory liczymy jawnie przed zapisem, więc writer bierze je z cache.
    """
    start = time.perf_counter()
    docs = load_corpus(settings)
    chunks = chunk_documents(docs, settings.chunking)
    counter = _counter(embeddings)
    before = counter.documents_embedded if counter is not None else 0
    embeddings.embed_documents([c.page_content for c in chunks])
    embedded = (counter.documents_embedded - before) if counter is not None else 0

    model, dimensions = configured_embedding_model(settings)
    table = settings.db.vector_collection_mp1
    meta = IndexMeta(
        table=table,
        embedding_provider=settings.embeddings.provider,
        embedding_model=model,
        dimensions=dimensions,
        n_chunks=len(chunks),
        updated_at=datetime.now(UTC),
    )
    rows_after, rows_before = writer(chunks, meta)
    return IngestStats(
        corpus=corpus_stats(docs, encoding=settings.tokenizer.encoding),
        chunks=chunk_stats(chunks),
        table=table,
        embedding_provider=settings.embeddings.provider,
        embedding_model=model,
        dimensions=dimensions,
        embedded=embedded,
        from_cache=len(chunks) - embedded,
        rows_after=rows_after,
        rows_before=rows_before,
        seconds=time.perf_counter() - start,
    )
