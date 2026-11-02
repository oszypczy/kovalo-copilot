"""Chunkowanie (FR-1.3; port 6 testów kroku 3 kursu + kontrakt F1, 5.2)."""

from __future__ import annotations

import re
from itertools import pairwise

import pytest
from langchain_core.documents import Document


def test_returns_documents_no_longer_than_chunk_size(chunks, settings):
    size = settings.chunking.chunk_size
    assert all(isinstance(c, Document) for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks if len(c.page_content) > size] == []


def test_every_document_is_chunked_in_order(chunks, documents):
    doc_order = [d.metadata["doc_id"] for d in documents]
    assert list(dict.fromkeys(c.metadata["doc_id"] for c in chunks)) == doc_order


def test_chunks_keep_their_document_metadata(chunks, documents):
    missing = {k: v for k, v in documents[0].metadata.items() if chunks[0].metadata.get(k) != v}
    assert not missing


def test_chunk_ids_and_indexes(chunks):
    for chunk in chunks:
        doc_id, index = chunk.metadata["doc_id"], chunk.metadata["chunk_index"]
        assert chunk.metadata["chunk_id"] == f"{doc_id}#{index:03d}"
    per_doc: dict[str, list[int]] = {}
    for chunk in chunks:
        per_doc.setdefault(chunk.metadata["doc_id"], []).append(chunk.metadata["chunk_index"])
    for indexes in per_doc.values():
        assert indexes == list(range(len(indexes)))
    assert re.fullmatch(r"[A-Z]+-\d{3}#\d{3}", chunks[0].metadata["chunk_id"])


def test_chunks_match_the_course_cache(chunks, settings):
    from kovalo.knowledge.embeddings import missing_from_course_cache

    missing = missing_from_course_cache(
        settings.paths.embedding_cache,
        settings.embeddings.model,
        [c.page_content for c in chunks],
    )
    assert [chunks[i].metadata["chunk_id"] for i in missing] == []


def test_overlap_repeats_text_between_neighbours(settings):
    from kovalo.knowledge.chunking import chunk_documents

    text = " ".join(f"word{i}" for i in range(600))
    pieces = chunk_documents(
        [Document(page_content=text, metadata={"doc_id": "TST-001"})], settings.chunking
    )
    assert len(pieces) > 1
    for left, right in pairwise(pieces):
        assert right.page_content.split()[0] in left.page_content.split()


def test_chunk_stats_reference(chunks):
    from kovalo.knowledge.chunking import chunk_stats

    stats = chunk_stats(chunks)
    assert len(chunks) == stats.n_chunks == 47
    assert stats.avg_chars == 603.7
    assert stats.max_chars == 798
    assert stats.most_chunked_document == "CON-101"


def test_make_chunk_id_bounds():
    from kovalo.knowledge.chunking import make_chunk_id

    assert make_chunk_id("POL-001", 2) == "POL-001#002"
    assert make_chunk_id("POL-001", 999) == "POL-001#999"
    for bad in (-1, 1000):
        with pytest.raises(ValueError):
            make_chunk_id("POL-001", bad)


def test_document_without_doc_id_raises(settings):
    from kovalo.knowledge.chunking import chunk_documents

    with pytest.raises(ValueError):
        chunk_documents([Document(page_content="tekst", metadata={})], settings.chunking)
