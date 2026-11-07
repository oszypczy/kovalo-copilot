"""Okablowanie produkcyjne copilota (ADR-0009 pkt 8; kontrakt F3, sekcja 6.8; F4, sekcja 8.2;
F5, sekcja 9.1).

Jedyny moduł workflow/, który importuje kovalo.llm, kovalo.rag.factory, kovalo.knowledge,
kovalo.tools, kovalo.company, kovalo.mcp_client i agents.agent.build_agent. Reszta pakietu
dostaje model, odpowiadacza i agenta przez CopilotDeps (testowalność).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from kovalo.agents import Middleware
from kovalo.agents.agent import AgentRunner, build_agent
from kovalo.agents.trace import EventCallback, Trace
from kovalo.company import load_company_data
from kovalo.config import ConfigError, Settings
from kovalo.knowledge.corpus import CorpusError
from kovalo.knowledge.pgvector import IndexUnavailableError
from kovalo.knowledge.retrievers import LazyRetriever, Retriever, open_retriever
from kovalo.llm import (
    StructuredResult,
    fallback_exceptions,
    get_chat_model,
    get_embeddings,
    get_fallback_models,
    get_structured_model,
)
from kovalo.mcp_client import McpStartupError, McpToolSource, open_tool_source
from kovalo.rag.answer import DocumentAnswerer, error_text
from kovalo.rag.factory import build_document_answerer
from kovalo.rag.records import RunRecord
from kovalo.tools.langchain import langchain_tools
from kovalo.tools.specs import ToolContext, tool_specs
from kovalo.workflow.checkpoint import open_checkpointer
from kovalo.workflow.graph import AgentAnswerer, CopilotDeps, build_copilot_graph
from kovalo.workflow.schemas import Turn
from kovalo.workflow.service import CopilotService

__all__ = [
    "LazyAgentRunner",
    "LazyDocumentAnswerer",
    "build_agent_runner",
    "build_copilot",
    "build_tool_context",
    "build_understand_model",
    "open_copilot_service",
]

_BUILD_ERRORS: tuple[type[BaseException], ...] = (
    ConfigError,  # w tym IndexMismatchError, MissingConfigError, CompanyDataError
    McpStartupError,  # podklasa ConfigError; jawnie: serwer MCP nie wystartował (F5)
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


def _retriever_opener(settings: Settings) -> Callable[[], Retriever]:
    """Funkcja otwierająca indeks dla knowledge.LazyRetriever (nazwy modułu czytane w chwili
    wywołania: testy podmieniają open_retriever)."""
    return lambda: open_retriever(settings, get_embeddings(settings=settings))


def build_tool_context(settings: Settings, *, retriever: Retriever | None) -> ToolContext:
    """Dane firmy (CompanyDataError leci wyżej), polityka, warunki z configu, retriever, top_k."""
    return ToolContext(
        data=load_company_data(settings.paths.company_data_dir),
        policy=settings.policy,
        company=settings.company,
        retriever=retriever,
        top_k=settings.retrieval.top_k,
    )


def build_agent_runner(
    settings: Settings,
    *,
    retriever: Retriever | None = None,
    tools: Sequence[BaseTool] | None = None,
    cache: BaseCache | None = None,
    offline: bool = False,
    variant: Literal["course", "ours"] | None = None,
    extra_middleware: Sequence[Middleware] = (),
) -> AgentRunner:
    """Agent z modelem 'agent' i fallbackami z configu (tylko błędy dostępności) i promptem
    agent_system (PINNED). `tools` podane (np. McpToolSource.tools()) → używa ich i nie wczytuje
    danych firmy; None → narzędzia bezpośrednie wariantu (domyślnie tools.variant) nad
    build_tool_context(settings, retriever=retriever). Brak klucza (offline=False) →
    MissingConfigError."""
    model = get_chat_model("agent", settings=settings, cache=cache, offline=offline)
    fallbacks = get_fallback_models("agent", settings=settings, cache=cache, offline=offline)
    if tools is None:
        ctx = build_tool_context(settings, retriever=retriever)
        tools = langchain_tools(ctx, variant or settings.tools.variant)
    agent = build_agent(
        model,
        tools,
        system_prompt=settings.prompts.agent_system,
        fallbacks=fallbacks,
        fallback_exceptions=fallback_exceptions(offline),  # offline: też brak w cache
        extra_middleware=extra_middleware,
    )
    return AgentRunner(agent, settings.agent)


class LazyAgentRunner:
    """AgentAnswerer budowany przy pierwszym pytaniu (jak LazyDocumentAnswerer).

    tools.source == 'direct' → build_agent_runner(settings, retriever=LazyRetriever(opener)) jak
    F4; 'mcp' → self.source = open_tool_source(settings, names=nazwy wariantu tools.variant) i
    build_agent_runner(settings, tools=self.source.tools()). Błąd budowy (ConfigError, w tym
    McpStartupError, CompanyDataError i MissingConfigError) → Trace(status='error'), źródło
    zamknięte; kolejne pytanie próbuje ponownie. close() zamyka źródło (idempotentne).
    """

    def __init__(self, settings: Settings, *, cache: BaseCache | None = None) -> None:
        self.settings = settings
        self.cache = cache
        self.source: McpToolSource | None = None
        self._runner: AgentRunner | None = None
        self._lock = threading.Lock()

    def _build(self) -> AgentRunner:
        settings = self.settings
        if settings.tools.source == "direct":
            lazy = LazyRetriever(_retriever_opener(settings))
            return build_agent_runner(settings, retriever=lazy, cache=self.cache)
        names = [spec.name for spec in tool_specs(settings.tools.variant)]
        # nazwa modułu czytana w chwili wywołania (testy podmieniają open_tool_source; P5)
        self.source = open_tool_source(settings, names=names)
        return build_agent_runner(settings, tools=self.source.tools(), cache=self.cache)

    def _close_source(self) -> None:
        source, self.source = self.source, None
        if source is not None:
            source.close()

    def __call__(
        self, question: str, *, on_event: EventCallback | None = None, callbacks: Callbacks = None
    ) -> Trace:
        with self._lock:
            if self._runner is None:
                try:
                    self._runner = self._build()
                except _BUILD_ERRORS as exc:
                    self._close_source()
                    return Trace(question=question, status="error", error=error_text(exc))
            runner = self._runner
        return runner(question, on_event=on_event, callbacks=callbacks)

    def close(self) -> None:
        with self._lock:
            self._close_source()
            self._runner = None


def _deps(
    settings: Settings,
    understand_model: Runnable[LanguageModelInput, Any],
    *,
    document_answerer: DocumentAnswerer | None,
    agent: AgentAnswerer | None,
    cache: BaseCache | None,
    understand_prompt: str | None,
) -> CopilotDeps:
    if document_answerer is None and agent is None:
        # nazwy modułu czytane w chwili wywołania (testy podmieniają je monkeypatchem; P7)
        if settings.workflow.answerer == "agent":
            agent = LazyAgentRunner(settings, cache=cache)
        else:
            document_answerer = LazyDocumentAnswerer(settings, cache=cache)
    return CopilotDeps(
        understand_model=understand_model,
        policy=settings.policy,
        understand_prompt=(
            understand_prompt
            if understand_prompt is not None
            else settings.prompts.by_name(settings.workflow.understand_prompt)
        ),
        document_answerer=document_answerer,
        agent=agent,
    )


def build_copilot(
    settings: Settings,
    *,
    checkpointer: BaseCheckpointSaver[Any],
    document_answerer: DocumentAnswerer | None = None,
    agent: AgentAnswerer | None = None,
    cache: BaseCache | None = None,
    offline: bool = False,
    understand_prompt: str | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Pełny graf copilota. Bez document_answerer i agent: według workflow.answerer (agent →
    LazyAgentRunner, rag → LazyDocumentAnswerer). understand_prompt None →
    prompts.by_name(workflow.understand_prompt)."""
    deps = _deps(
        settings,
        build_understand_model(settings, cache=cache, offline=offline),
        document_answerer=document_answerer,
        agent=agent,
        cache=cache,
        understand_prompt=understand_prompt,
    )
    return build_copilot_graph(deps, checkpointer=checkpointer)


@contextmanager
def open_copilot_service(settings: Settings) -> Iterator[CopilotService]:
    """open_checkpointer + graf (odpowiadający i prompt z configu) + CopilotService (dla api).

    MissingConfigError i CheckpointerUnavailableError lecą do wywołującego. Model budujemy
    przed otwarciem puli Postgresa, żeby brak klucza nie otwierał połączeń. Na wyjściu serwis
    czeka na trwające tury i zamyka wątki robocze, potem zamyka źródło narzędzi agenta (MCP).
    """
    deps = _deps(
        settings,
        build_understand_model(settings),
        document_answerer=None,
        agent=None,
        cache=None,
        understand_prompt=None,
    )
    with open_checkpointer(settings) as saver:
        graph = build_copilot_graph(deps, checkpointer=saver)
        service = CopilotService(graph, max_message_chars=settings.workflow.max_message_chars)
        try:
            yield service
        finally:
            try:
                service.close()
            finally:
                close = getattr(deps.agent, "close", None)  # LazyAgentRunner: źródło narzędzi
                if callable(close):
                    close()
