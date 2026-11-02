"""Fabryka odpowiadacza dokumentowego (FR-2.2; kontrakt F2, 7.4).

Jedyny moduł `rag/`, który importuje kovalo.llm. F3 (węzeł answer_documents) dostaje
DocumentAnswerer przez parametr fabryki grafu.
"""

from __future__ import annotations

import functools

from langchain_core.caches import BaseCache

from kovalo.config import Settings
from kovalo.knowledge.retrievers import Retriever
from kovalo.llm import get_structured_model
from kovalo.rag.answer import DocumentAnswerer, answer_question
from kovalo.rag.records import Answer

__all__ = ["build_document_answerer"]


def build_document_answerer(
    settings: Settings, *, retriever: Retriever, cache: BaseCache | None = None
) -> DocumentAnswerer:
    """Pytanie → RunRecord: model 'answer' z configu (fallback, structured output),
    prompts.answer_system i k = retrieval.top_k. Brak klucza → MissingConfigError od razu."""
    model = get_structured_model(Answer, "answer", settings=settings, cache=cache)
    return functools.partial(
        answer_question,
        retriever=retriever,
        model=model,
        system_prompt=settings.prompts.answer_system,
        k=settings.retrieval.top_k,
    )
