"""Chunkowanie: MP1 (FR-1.3, krok 3 kursu MP1) i chunki z kontekstem (FR-2.1, krok 1 kursu MP2).

Kontrakty: F1 sekcja 5.2, F2 sekcja 5.1. Jedyny moduł, który importuje langchain_text_splitters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel

from kovalo.config import ChunkingConfig

__all__ = [
    "CHUNK_ID_DIGITS",
    "CTX_PREFIX",
    "HEADERS_TO_SPLIT_ON",
    "ChunkStats",
    "ChunkVariant",
    "build_chunks",
    "build_contextual_chunks",
    "chunk_documents",
    "chunk_stats",
    "make_chunk_id",
    "make_ctx_chunk_id",
]

CHUNK_ID_DIGITS = 3
_MAX_INDEX = 10**CHUNK_ID_DIGITS - 1
CTX_PREFIX = "ctx"
HEADERS_TO_SPLIT_ON: tuple[tuple[str, str], ...] = (("#", "h1"), ("##", "h2"))
ChunkVariant = Literal["mp1", "ctx"]


def make_chunk_id(doc_id: str, index: int) -> str:
    """'POL-001', 2 → 'POL-001#002'. index < 0 albo > 999 → ValueError."""
    if not 0 <= index <= _MAX_INDEX:
        raise ValueError(f"Numer chunku poza zakresem 0..{_MAX_INDEX}: {index} ({doc_id})")
    return f"{doc_id}#{index:0{CHUNK_ID_DIGITS}d}"


def make_ctx_chunk_id(doc_id: str, index: int) -> str:
    """'CON-101', 3 → 'CON-101#ctx003'. index < 0 albo > 999 → ValueError."""
    if not 0 <= index <= _MAX_INDEX:
        raise ValueError(f"Numer chunku poza zakresem 0..{_MAX_INDEX}: {index} ({doc_id})")
    return f"{doc_id}#{CTX_PREFIX}{index:0{CHUNK_ID_DIGITS}d}"


def _splitter(chunking: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunking.chunk_size,
        chunk_overlap=chunking.chunk_overlap,
        separators=list(chunking.separators),
    )


def chunk_documents(docs: Sequence[Document], chunking: ChunkingConfig) -> list[Document]:
    """RecursiveCharacterTextSplitter z ustawień PINNED; metadane + chunk_index i chunk_id.

    Tekst chunku dokładnie taki, jaki zwraca splitter; kolejność dokumentów zachowana.
    """
    for doc in docs:
        if not doc.metadata.get("doc_id"):
            raise ValueError("Dokument bez metadata['doc_id'] nie może zostać podzielony.")
    chunks = _splitter(chunking).split_documents(list(docs))
    counters: dict[str, int] = {}
    for chunk in chunks:
        doc_id = chunk.metadata["doc_id"]
        index = counters.get(doc_id, 0)
        counters[doc_id] = index + 1
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = make_chunk_id(doc_id, index)
    return chunks


def build_contextual_chunks(docs: Sequence[Document], chunking: ChunkingConfig) -> list[Document]:
    """Sekcje Markdown (# i ##, nagłówki zostają w tekście) → podział z ustawień PINNED.

    page_content = '<title> > <sekcja>' (albo samo '<title>') + pusty wiersz + tekst części;
    metadata = pola dokumentu + section, chunk_index, chunk_id ('<doc_id>#ctxNNN', od 000
    w każdym dokumencie). Klucze splittera (h1, h2) nie trafiają do metadanych.
    """
    for doc in docs:
        for key in ("doc_id", "title"):
            if not doc.metadata.get(key):
                raise ValueError(
                    f"Dokument bez metadata[{key!r}] nie może zostać podzielony na chunki "
                    "z kontekstem."
                )
    headers = MarkdownHeaderTextSplitter(list(HEADERS_TO_SPLIT_ON), strip_headers=False)
    splitter = _splitter(chunking)
    chunks: list[Document] = []
    for doc in docs:
        doc_id, title = doc.metadata["doc_id"], doc.metadata["title"]
        parts = splitter.split_documents(headers.split_text(doc.page_content))
        for index, part in enumerate(parts):
            section = str(part.metadata.get("h2", ""))
            heading = title if section == "" else f"{title} > {section}"
            metadata = {
                **doc.metadata,
                "section": section,
                "chunk_index": index,
                "chunk_id": make_ctx_chunk_id(doc_id, index),
            }
            chunks.append(
                Document(page_content=f"{heading}\n\n{part.page_content}", metadata=metadata)
            )
    return chunks


def build_chunks(
    docs: Sequence[Document], chunking: ChunkingConfig, variant: ChunkVariant
) -> list[Document]:
    """'mp1' → chunk_documents, 'ctx' → build_contextual_chunks; inny wariant → ValueError."""
    if variant == "mp1":
        return chunk_documents(docs, chunking)
    if variant == "ctx":
        return build_contextual_chunks(docs, chunking)
    raise ValueError(f"Nieznany wariant chunków: {variant!r} (dozwolone: mp1, ctx)")


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
