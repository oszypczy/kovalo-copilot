"""search_documents: surowe fragmenty dokumentów z retrievera (FR-4.3; krok 2 MP4).

Kontrakt: docs/contracts/f4.md, sekcja 6.2. Bez LangChaina i bez wywołania LLM-a: odpowiedź
składa agent. Błąd wyszukiwania (np. indeks niedostępny) to wyjątek, nigdy „brak wyników”.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from kovalo.knowledge.retrievers import Retriever

__all__ = ["Passage", "Retriever", "SearchResult", "search_documents"]


class Passage(BaseModel):
    doc_id: str
    chunk_id: str
    text: str


class SearchResult(BaseModel):
    """search_documents: surowe fragmenty dokumentów Kovalo."""

    status: Literal["ok", "no_results", "invalid_request"]
    query: str
    passages: list[Passage] = Field(default_factory=list)
    message: str | None = None


def search_documents(retriever: Retriever, question: str, k: int) -> SearchResult:
    """Pytanie puste po strip() → invalid_request; retriever.retrieve(question, k) → fragmenty
    od najlepszego; pusta lista → no_results. Wyjątek retrievera idzie wyżej."""
    if not question.strip():
        return SearchResult(
            status="invalid_request",
            query=question,
            message="pytanie do wyszukiwania jest puste: podaj, czego szukać w dokumentach",
        )
    hits = retriever.retrieve(question, k)
    if not hits:
        return SearchResult(
            status="no_results",
            query=question,
            message="w dokumentach Kovalo nie znaleziono fragmentów pasujących do pytania",
        )
    return SearchResult(
        status="ok",
        query=question,
        passages=[
            Passage(
                doc_id=str(doc.metadata["doc_id"]),
                chunk_id=str(doc.metadata["chunk_id"]),
                text=doc.page_content,
            )
            for doc, _score in hits
        ],
    )
