"""search_documents: surowe fragmenty z retrievera (port 2 testów wyszukiwania z kroku 2 MP4;
`CountingStore` → `CountingRetriever`; kontrakt F4, 6.2)."""

from __future__ import annotations

import pytest


def test_search_returns_raw_passages_with_ids(retriever):
    from kovalo.tools.search import search_documents

    result = search_documents(retriever, "TransBałt terminowe dostawy palet", 2)
    assert result.status == "ok" and result.query == "TransBałt terminowe dostawy palet"
    assert len(result.passages) == 2 and retriever.ks == [2]
    for passage in result.passages:
        assert passage.doc_id and passage.chunk_id.startswith(passage.doc_id) and passage.text


def test_an_empty_question(retriever):
    from kovalo.tools.search import search_documents

    for blank in ("  ", "", "\n\t"):
        result = search_documents(retriever, blank, 3)
        assert result.status == "invalid_request" and result.passages == [] and result.message
    assert retriever.ks == [], "pusty tekst nie dociera do indeksu"


def test_no_results(retriever):
    from kovalo.tools.search import search_documents

    class Empty:
        name = "empty"

        def retrieve(self, question, k):
            return []

    result = search_documents(Empty(), "cokolwiek", 3)
    assert result.status == "no_results" and result.passages == [] and result.query == "cokolwiek"


def test_passages_keep_the_retriever_order_and_metadata(retriever):
    from kovalo.tools.search import search_documents

    expected = [d.metadata["chunk_id"] for d, _ in retriever.retrieve("palety", 3)]
    result = search_documents(retriever, "palety", 3)
    assert [p.chunk_id for p in result.passages] == expected, "od najlepszego"
    first_doc, _ = retriever.retrieve("palety", 3)[0]
    assert result.passages[0].text == first_doc.page_content
    assert result.passages[0].doc_id == first_doc.metadata["doc_id"]


def test_retriever_errors_go_up(retriever):
    """Błąd wyszukiwania nie jest 'brak wyników': narzędzie zgłasza wyjątek (tool_errors → trace)."""  # noqa: E501
    from kovalo.tools.search import search_documents
    from tests.agent_fakes import BrokenRetriever

    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        search_documents(BrokenRetriever(), "palety", 3)
