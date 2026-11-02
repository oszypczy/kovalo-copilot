"""Pakiet `knowledge` (MP1, rozszerzany w F2): korpus, chunki, embeddingi, wyszukiwanie, indeks.

Import pakietu nie ładuje fastembed ani langchain_postgres (oba importowane leniwie).
Kontrakt: docs/contracts/f1.md, sekcja 5.
"""

from kovalo.knowledge.chunking import ChunkStats, chunk_documents, chunk_stats, make_chunk_id
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
from kovalo.knowledge.ingest import IngestStats, ingest_mp1
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

__all__ = [
    "ChunkStats",
    "CorpusError",
    "CorpusStats",
    "CountingEmbeddings",
    "CourseCacheMissError",
    "CourseCacheOnlyEmbeddings",
    "FastEmbedEmbeddings",
    "IndexMeta",
    "IndexMismatchError",
    "IndexUnavailableError",
    "IngestStats",
    "ScoredChunk",
    "build_cached_embeddings",
    "build_course_cache_embeddings",
    "build_prompt",
    "build_vector_store",
    "cache_namespace",
    "check_index",
    "chunk_documents",
    "chunk_stats",
    "corpus_stats",
    "cosine_similarity",
    "ingest_mp1",
    "load_corpus",
    "load_documents",
    "make_chunk_id",
    "missing_from_course_cache",
    "model_size_gb",
    "open_store",
    "pg_engine",
    "read_index_meta",
    "replace_table",
    "search",
    "similarity_from_score",
]
