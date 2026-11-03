"""Okablowanie produkcyjne copilota (ADR-0009 pkt 8; kontrakt F3, sekcja 6.8).

Jedyny moduł workflow/, który importuje kovalo.llm, kovalo.rag.factory i kovalo.knowledge.
Reszta pakietu dostaje model i odpowiadacza przez CopilotDeps (testowalność).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langchain_core.caches import BaseCache
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from kovalo.config import ConfigError, Settings
from kovalo.knowledge.corpus import CorpusError
from kovalo.knowledge.pgvector import IndexUnavailableError
from kovalo.knowledge.retrievers import open_retriever
from kovalo.llm import StructuredResult, get_embeddings, get_structured_model
from kovalo.rag.answer import DocumentAnswerer, error_text
from kovalo.rag.factory import build_document_answerer
from kovalo.rag.records import RunRecord
from kovalo.workflow.checkpoint import open_checkpointer
from kovalo.workflow.graph import CopilotDeps, build_copilot_graph
from kovalo.workflow.schemas import Turn
from kovalo.workflow.service import CopilotService

__all__ = [
    "LazyDocumentAnswerer",
    "build_copilot",
    "build_understand_model",
    "open_copilot_service",
]

_BUILD_ERRORS: tuple[type[BaseException], ...] = (
    ConfigError,  # w tym IndexMismatchError, MissingConfigError
    IndexUnavailableError,
    CorpusError,
)


def build_understand_model(
    settings: Settings, *, cache: BaseCache | None = None, offline: bool = False
) -> Runnable[LanguageModelInput, StructuredResult]:
    """llm.get_structured_model(Turn, 'understand'); brak klucza (offline=False) →
    MissingConfigError przy budowie; offline=True bez cache → ValueError."""
    return get_structured_model(Turn, "understand", settings=settings, cache=cache, offline=offline)


class LazyDocumentAnswerer:
    """DocumentAnswerer budowany przy pierwszym pytaniu o dokumenty.

    open_retriever(settings, get_embeddings()) + rag.build_document_answerer. Błąd budowy
    (IndexMismatchError, IndexUnavailableError, ConfigError, CorpusError) → RunRecord z `error`
    (tura kończy się statusem 'error'); kolejne pytanie próbuje zbudować ponownie. Wątkowo
    bezpieczny: budowa pod blokadą, odpowiedzi równolegle.
    """

    def __init__(self, settings: Settings, *, cache: BaseCache | None = None) -> None:
        self.settings = settings
        self.cache = cache
        self._answerer: DocumentAnswerer | None = None
        self._lock = threading.Lock()

    def _build(self) -> DocumentAnswerer:
        retriever = open_retriever(self.settings, get_embeddings(settings=self.settings))
        return build_document_answerer(self.settings, retriever=retriever, cache=self.cache)

    def __call__(self, question: str) -> RunRecord:
        with self._lock:
            if self._answerer is None:
                try:
                    self._answerer = self._build()
                except _BUILD_ERRORS as exc:
                    return RunRecord(question=question, context=[], error=error_text(exc))
            answerer = self._answerer
        return answerer(question)


def build_copilot(
    settings: Settings,
    *,
    checkpointer: BaseCheckpointSaver[Any],
    document_answerer: DocumentAnswerer | None = None,
    cache: BaseCache | None = None,
    offline: bool = False,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Pełny graf copilota z modelem 'understand', polityką i promptem z configu."""
    deps = CopilotDeps(
        understand_model=build_understand_model(settings, cache=cache, offline=offline),
        document_answerer=document_answerer or LazyDocumentAnswerer(settings, cache=cache),
        policy=settings.policy,
        understand_prompt=settings.prompts.understand_system,
    )
    return build_copilot_graph(deps, checkpointer=checkpointer)


@contextmanager
def open_copilot_service(settings: Settings) -> Iterator[CopilotService]:
    """open_checkpointer + build_copilot + CopilotService (dla api).

    MissingConfigError i CheckpointerUnavailableError lecą do wywołującego. Model budujemy
    przed otwarciem puli Postgresa, żeby brak klucza nie otwierał połączeń.
    """
    understand_model = build_understand_model(settings)
    deps = CopilotDeps(
        understand_model=understand_model,
        document_answerer=LazyDocumentAnswerer(settings),
        policy=settings.policy,
        understand_prompt=settings.prompts.understand_system,
    )
    with open_checkpointer(settings) as saver:
        graph = build_copilot_graph(deps, checkpointer=saver)
        yield CopilotService(graph, max_message_chars=settings.workflow.max_message_chars)
