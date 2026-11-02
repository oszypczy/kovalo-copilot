"""Wczytywanie korpusu (FR-1.1; port 9 testów kroku 1 kursu + rozszerzenia z kontraktu F1, 5.1)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

REQUIRED = ["doc_id", "title", "doc_type", "department", "owner", "effective_date", "language"]


def _load(toy_corpus):
    from kovalo.knowledge.corpus import load_documents

    _, corpus_dir, catalog = toy_corpus
    return load_documents(corpus_dir, catalog, REQUIRED)


def test_returns_one_document_per_markdown_file(documents, settings):
    n_files = len(list(settings.paths.corpus_dir.glob("*.md")))
    assert isinstance(documents, list)
    assert all(isinstance(d, Document) for d in documents)
    assert len(documents) == n_files == 18


def test_sorted_by_doc_id(documents):
    ids = [d.metadata["doc_id"] for d in documents]
    assert ids == sorted(ids)


def test_metadata_is_the_catalog_row_plus_source(documents, settings):
    first = documents[0]
    assert set(first.metadata) == {*settings.corpus.required_metadata, "source"}
    assert "file" not in first.metadata
    assert first.metadata["doc_id"] == "CON-101" and first.metadata["doc_type"] == "contract"
    assert first.metadata["source"] == "CON-101_supplier_agreement_mleczna_dolina.md"


def test_page_content_is_the_file_text_unchanged(documents, settings):
    for doc in documents:
        path = settings.paths.corpus_dir / doc.metadata["source"]
        assert doc.page_content == path.read_text(encoding="utf-8")


def test_polish_document_is_loaded(documents):
    assert {d.metadata["language"] for d in documents} == {"en", "pl"}


def test_load_corpus_uses_settings(documents, settings):
    from kovalo.knowledge.corpus import load_corpus

    again = load_corpus(settings)
    assert [d.metadata for d in again] == [d.metadata for d in documents]


def test_works_on_another_corpus(toy_corpus):
    add = toy_corpus[0]
    add("b.md", "Drugi.", doc_id="BBB-001")
    add("a.md", "Pierwszy.", doc_id="AAA-001", doc_type="report")
    docs = _load(toy_corpus)
    assert [d.page_content for d in docs] == ["Pierwszy.", "Drugi."]
    assert docs[0].metadata["doc_type"] == "report"


def test_catalog_csv_is_not_a_document(toy_corpus):
    toy_corpus[0]("a.md")
    assert len(_load(toy_corpus)) == 1


def test_file_without_catalog_row_raises(toy_corpus):
    add = toy_corpus[0]
    add("listed.md")
    add("stray.md", in_catalog=False)
    with pytest.raises(ValueError, match=r"stray\.md"):
        _load(toy_corpus)


def test_empty_catalog_field_raises(toy_corpus):
    toy_corpus[0]("no_owner.md", owner="")
    with pytest.raises(ValueError, match=r"no_owner\.md"):
        _load(toy_corpus)


def test_catalog_row_without_file_raises(toy_corpus):
    from kovalo.knowledge.corpus import CorpusError

    add, corpus_dir, _ = toy_corpus
    add("kept.md", doc_id="AAA-001")
    add("gone.md", doc_id="BBB-001")
    (corpus_dir / "gone.md").unlink()
    with pytest.raises(CorpusError, match=r"gone\.md"):
        _load(toy_corpus)


def test_duplicate_doc_id_raises(toy_corpus):
    from kovalo.knowledge.corpus import CorpusError

    add = toy_corpus[0]
    add("a.md", doc_id="DUP-001")
    add("b.md", doc_id="DUP-001")
    with pytest.raises(CorpusError, match="DUP-001"):
        _load(toy_corpus)


def test_missing_directory_raises_corpus_error_with_path(tmp_path):
    from kovalo.knowledge.corpus import CorpusError, load_documents

    missing = tmp_path / "nie-ma-takiego-katalogu"
    with pytest.raises(CorpusError, match="nie-ma-takiego-katalogu"):
        load_documents(missing, missing / "catalog.csv", REQUIRED)


def test_corpus_error_is_value_error():
    from kovalo.knowledge.corpus import CorpusError

    assert issubclass(CorpusError, ValueError)


def test_corpus_stats_reference(documents, settings):
    from kovalo.knowledge.corpus import corpus_stats

    stats = corpus_stats(documents, encoding=settings.tokenizer.encoding)
    assert stats.n_documents == 18
    assert stats.total_tokens == 8762
    assert stats.largest_document == "POL-001"
    assert stats.tokens_per_word_by_language == {"en": 1.261, "pl": 2.249}
    assert sum(stats.tokens_by_doc_type.values()) == stats.total_tokens
