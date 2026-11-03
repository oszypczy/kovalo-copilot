"""Pakiet `knowledge` (MP1, MP2): korpus, chunki, embeddingi, wyszukiwanie, indeks.

Wyszukiwanie: wektorowe, BM25 i hybryda (RRF).

Import pakietu nie ładuje fastembed ani langchain_postgres (oba importowane leniwie).
Kontrakty: docs/contracts/f1.md, sekcja 5; docs/contracts/f2.md, sekcja 5.
"""

from kovalo.knowledge.chunking import (
    ChunkStats,
    ChunkVariant,
    build_chunks,
    build_contextual_chunks,
    chunk_documents,
    chunk_stats,
    make_chunk_id,
    make_ctx_chunk_id,
)
from kovalo.knowledge.corpus import (
    CorpusError,
    CorpusStats,
    corpus_stats,
    load_corpus,
    load_documents,
)
from kovalo.knowledge.embeddings import (
    CountingEmbeddings,
    CourseCacheMissError,
    CourseCacheOnlyEmbeddings,
    FastEmbedEmbeddings,
    build_cached_embeddings,
    build_course_cache_embeddings,
    cache_namespace,
    missing_from_course_cache,
    model_size_gb,
)
from kovalo.knowledge.ingest import IngestStats, ingest_chunks, ingest_mp1
from kovalo.knowledge.lexical import BM25Index, tokenize
from kovalo.knowledge.pgvector import (
    IndexMeta,
    IndexMismatchError,
    IndexUnavailableError,
    check_index,
    open_store,
    pg_engine,
    read_index_meta,
    replace_table,
)
from kovalo.knowledge.prompts import build_prompt
from kovalo.knowledge.retrieval import (
    ScoredChunk,
    build_vector_store,
    cosine_similarity,
    search,
    similarity_from_score,
)
from kovalo.knowledge.retrievers import (
    HybridRetriever,
    Retriever,
    VectorRetriever,
    build_memory_retriever,
    build_retriever,
    open_retriever,
    rrf_fuse,
)

__all__ = [
    "BM25Index",
    "ChunkStats",
    "ChunkVariant",
    "CorpusError",
    "CorpusStats",
    "CountingEmbeddings",
    "CourseCacheMissError",
    "CourseCacheOnlyEmbeddings",
    "FastEmbedEmbeddings",
    "HybridRetriever",
    "IndexMeta",
    "IndexMismatchError",
    "IndexUnavailableError",
    "IngestStats",
    "Retriever",
    "ScoredChunk",
    "VectorRetriever",
    "build_cached_embeddings",
    "build_chunks",
    "build_contextual_chunks",
    "build_course_cache_embeddings",
    "build_memory_retriever",
    "build_prompt",
    "build_retriever",
    "build_vector_store",
    "cache_namespace",
    "check_index",
    "chunk_documents",
    "chunk_stats",
    "corpus_stats",
    "cosine_similarity",
    "ingest_chunks",
    "ingest_mp1",
    "load_corpus",
    "load_documents",
    "make_chunk_id",
    "make_ctx_chunk_id",
    "missing_from_course_cache",
    "model_size_gb",
    "open_retriever",
    "open_store",
    "pg_engine",
    "read_index_meta",
    "replace_table",
    "rrf_fuse",
    "search",
    "similarity_from_score",
    "tokenize",
]
