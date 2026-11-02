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


# ---------------------------------------------------------------------------
# F2: chunki z kontekstem (kontrakt F2, 5.1; port 4 testów kroku 1 MP2 kursu)
# ---------------------------------------------------------------------------


def test_ctx_toy_document_gets_title_and_section_headers(settings):
    from kovalo.knowledge.chunking import build_contextual_chunks

    doc = Document(
        page_content=(
            "# Zasady floty\n\nDotyczą wszystkich furgonetek.\n\n"
            "## Kierowcy\n\nTylko kierowcy z prawem jazdy."
        ),
        metadata={"doc_id": "TST-001", "title": "Zasady floty", "language": "pl"},
    )
    out = build_contextual_chunks([doc], settings.chunking)
    heads = [c.page_content.split("\n\n")[0] for c in out]
    assert heads == ["Zasady floty", "Zasady floty > Kierowcy"]
    assert [c.metadata["section"] for c in out] == ["", "Kierowcy"]
    assert [c.metadata["chunk_id"] for c in out] == ["TST-001#ctx000", "TST-001#ctx001"]
    assert [c.metadata["chunk_index"] for c in out] == [0, 1]
    assert out[1].metadata["language"] == "pl"  # pola dokumentu zachowane
    assert "## Kierowcy" in out[1].page_content  # nagłówki zostają w tekście
    assert "h1" not in out[0].metadata and "h2" not in out[1].metadata


def test_ctx_chunk_ids_are_numbered_per_document(ctx_chunks):
    assert all(re.fullmatch(r"[A-Z]+-\d{3}#ctx\d{3}", c.metadata["chunk_id"]) for c in ctx_chunks)
    firsts = {
        c.metadata["doc_id"] for c in ctx_chunks if c.metadata["chunk_id"].endswith("#ctx000")
    }
    assert firsts == {c.metadata["doc_id"] for c in ctx_chunks}
    assert len({c.metadata["chunk_id"] for c in ctx_chunks}) == len(ctx_chunks)
    per_doc: dict[str, list[int]] = {}
    for c in ctx_chunks:
        per_doc.setdefault(c.metadata["doc_id"], []).append(c.metadata["chunk_index"])
    assert all(idx == list(range(len(idx))) for idx in per_doc.values())


def test_ctx_every_chunk_starts_with_its_document_title(ctx_chunks):
    bad = [
        c.metadata["chunk_id"]
        for c in ctx_chunks
        if not c.page_content.startswith(c.metadata["title"])
    ]
    assert bad == []


def test_ctx_chunks_match_the_course_cache(ctx_chunks, settings):
    from kovalo.knowledge.embeddings import missing_from_course_cache

    missing = missing_from_course_cache(
        settings.paths.embedding_cache,
        settings.embeddings.model,
        [c.page_content for c in ctx_chunks],
    )
    assert [ctx_chunks[i].metadata["chunk_id"] for i in missing] == []


def test_contextual_chunk_stats_reference(ctx_chunks):
    from kovalo.knowledge.chunking import chunk_stats

    stats = chunk_stats(ctx_chunks)
    assert len(ctx_chunks) == stats.n_chunks == 87
    assert stats.avg_chars == 385.6
    assert stats.max_chars == 775
    assert stats.most_chunked_document == "MTG-401"


def test_contextual_chunk_example(ctx_chunks):
    chunk = ctx_chunks[3]
    assert chunk.metadata["chunk_id"] == "CON-101#ctx003"
    assert chunk.page_content.startswith(
        "Streszczenie umowy z dostawcą: Mleczna Dolina (SUP-114) > Kara umowna\n\n"
        "## Kara umowna  \nJeśli miesięczny"
    )
    assert chunk.metadata["section"] == "Kara umowna"


def test_contextual_metadata_keys(ctx_chunks):
    expected = {
        "chunk_id",
        "chunk_index",
        "department",
        "doc_id",
        "doc_type",
        "effective_date",
        "language",
        "owner",
        "section",
        "source",
        "title",
    }
    assert all(set(c.metadata) == expected for c in ctx_chunks)


def test_build_chunks_variants(documents, settings):
    from kovalo.knowledge.chunking import build_chunks

    assert len(build_chunks(documents, settings.chunking, "mp1")) == 47
    assert len(build_chunks(documents, settings.chunking, "ctx")) == 87
    with pytest.raises(ValueError):
        build_chunks(documents, settings.chunking, "nieznany")  # type: ignore[arg-type]


def test_make_ctx_chunk_id_bounds():
    from kovalo.knowledge.chunking import make_ctx_chunk_id

    assert make_ctx_chunk_id("CON-101", 3) == "CON-101#ctx003"
    assert make_ctx_chunk_id("CON-101", 999) == "CON-101#ctx999"
    for bad in (-1, 1000):
        with pytest.raises(ValueError):
            make_ctx_chunk_id("CON-101", bad)


@pytest.mark.parametrize("missing", ["doc_id", "title"])
def test_contextual_document_without_required_metadata_raises(settings, missing):
    from kovalo.knowledge.chunking import build_contextual_chunks

    metadata = {"doc_id": "TST-001", "title": "Tytuł"}
    del metadata[missing]
    with pytest.raises(ValueError):
        build_contextual_chunks(
            [Document(page_content="tekst", metadata=metadata)], settings.chunking
        )
