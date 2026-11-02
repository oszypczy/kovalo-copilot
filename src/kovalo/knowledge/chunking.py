"""Chunkowanie MP1 (FR-1.3, krok 3 kursu; kontrakt F1, sekcja 5.2)."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from kovalo.config import ChunkingConfig

__all__ = ["CHUNK_ID_DIGITS", "ChunkStats", "chunk_documents", "chunk_stats", "make_chunk_id"]

CHUNK_ID_DIGITS = 3
_MAX_INDEX = 10**CHUNK_ID_DIGITS - 1


def make_chunk_id(doc_id: str, index: int) -> str:
    """'POL-001', 2 → 'POL-001#002'. index < 0 albo > 999 → ValueError."""
    if not 0 <= index <= _MAX_INDEX:
        raise ValueError(f"Numer chunku poza zakresem 0..{_MAX_INDEX}: {index} ({doc_id})")
    return f"{doc_id}#{index:0{CHUNK_ID_DIGITS}d}"


def chunk_documents(docs: Sequence[Document], chunking: ChunkingConfig) -> list[Document]:
    """RecursiveCharacterTextSplitter z ustawień PINNED; metadane + chunk_index i chunk_id.

    Tekst chunku dokładnie taki, jaki zwraca splitter; kolejność dokumentów zachowana.
    """
    for doc in docs:
        if not doc.metadata.get("doc_id"):
            raise ValueError("Dokument bez metadata['doc_id'] nie może zostać podzielony.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunking.chunk_size,
        chunk_overlap=chunking.chunk_overlap,
        separators=list(chunking.separators),
    )
    chunks = splitter.split_documents(list(docs))
    counters: dict[str, int] = {}
    for chunk in chunks:
        doc_id = chunk.metadata["doc_id"]
        index = counters.get(doc_id, 0)
        counters[doc_id] = index + 1
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = make_chunk_id(doc_id, index)
    return chunks


class ChunkStats(BaseModel):
    """Jak sekcja "chunks" w report.json kursu."""

    n_chunks: int
    avg_chars: float
    max_chars: int
    most_chunked_document: str


def chunk_stats(chunks: Sequence[Document]) -> ChunkStats:
    if not chunks:
        raise ValueError("Brak chunków.")
    lengths = [len(c.page_content) for c in chunks]
    per_doc: dict[str, int] = {}
    for c in chunks:
        per_doc[c.metadata["doc_id"]] = per_doc.get(c.metadata["doc_id"], 0) + 1
    return ChunkStats(
        n_chunks=len(chunks),
        avg_chars=round(sum(lengths) / len(lengths), 1),
        max_chars=max(lengths),
        most_chunked_document=min(per_doc, key=lambda i: (-per_doc[i], i)),
    )
