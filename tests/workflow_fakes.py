"""Atrapy i pomocniki testów F3 (kontrakt: docs/contracts/f3.md, 14.1).

Moduł nie importuje kodu produkcyjnego `kovalo.workflow` na poziomie modułu (tylko wewnątrz
funkcji), więc importuje się przed implementacją: brak kodu = czerwone testy, nie błąd zbierania.
Współdzielony przez testy `workflow/`, `api/`, `eval/workflow/` i `test_cli.py`
(conftest.py w jednym katalogu nie jest widoczny w pozostałych).

Port `ReplayLLM`, `turn`, `NoStore` i `graph_with` z `reference/course/mini_project_3/tests/`.
"""

from __future__ import annotations

import functools
import json
import threading
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

TOY_WATER_CHUNK = Document(
    page_content="Komitet cenowy > Woda\n\nKomitet utrzymał ceny wody.",
    metadata={"doc_id": "TST-001", "chunk_id": "TST-001#ctx000"},
)


class ReplayLLM:
    """Zastępuje model czatu: zwraca po kolei zaplanowane obiekty Turn, a na pytania
    o dokumenty stałą odpowiedź Answer. Liczy każde wywołanie, więc testy widzą ponowne
    uruchomienia. `gate` = (wszedł, zwolniony): pierwsze wywołanie czeka na zwolnienie
    (test blokady wątku)."""

    def __init__(
        self,
        turns: Sequence[Any] = (),
        answer: Any = None,
        fail: bool = False,
        gate: tuple[threading.Event, threading.Event] | None = None,
    ) -> None:
        from kovalo.rag.records import Answer

        self.turns = list(turns)
        self.answer = answer or Answer(
            answer="(odtworzona odpowiedź)", citations=[], answerable=True
        )
        self.fail = fail
        self.gate = gate
        self.calls = {"Turn": 0, "Answer": 0}
        self.prompts: list[Any] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        from kovalo.rag.records import Answer
        from kovalo.workflow.schemas import Turn

        def respond(messages: Any) -> Any:
            self.prompts.append(messages)
            if self.fail:
                raise ValueError("sztuczny błąd")
            if schema is Turn:
                self.calls["Turn"] += 1
                result = self.turns.pop(0)
                if self.gate is not None and self.calls["Turn"] == 1:
                    entered, release = self.gate
                    entered.set()
                    release.wait(timeout=10)
                return result
            if schema is Answer:
                self.calls["Answer"] += 1
                return self.answer
            raise AssertionError(f"nieoczekiwany schemat {schema}")

        return RunnableLambda(respond)


def turn(**fields: Any) -> Any:
    from kovalo.workflow.schemas import Turn

    base: dict[str, Any] = {"kind": "other", "request_action": "none", "standalone": "(standalone)"}
    base.update(fields)
    return Turn(**base)


class NoAnswerer:
    """Document answerer, którego nie wolno użyć (wszystkie trasy poza pytaniami o dokumenty)."""

    def __call__(self, question: str) -> Any:
        raise AssertionError("ta trasa nie może przeszukiwać dokumentów")


def toy_answerer(
    settings: Any, llm: Any, *, chunks: Sequence[Document] | None = None, k: int = 1
) -> Callable[[str], Any]:
    """`rag.answer_question` na VectorRetriever z atrapą embeddingów i modelem `llm` (ReplayLLM
    albo FakeStructuredChatModel). Domyślnie jeden chunk TST-001#ctx000."""
    from kovalo.knowledge.retrieval import build_vector_store
    from kovalo.knowledge.retrievers import VectorRetriever
    from kovalo.rag.answer import answer_question
    from kovalo.rag.records import Answer

    store = build_vector_store(
        list(chunks or [TOY_WATER_CHUNK]), DeterministicFakeEmbedding(size=16)
    )
    return functools.partial(
        answer_question,
        retriever=VectorRetriever(store),
        model=llm.with_structured_output(Answer),
        system_prompt=settings.prompts.answer_system,
        k=k,
    )


def make_deps(
    settings: Any,
    turns: Sequence[Any] = (),
    *,
    llm: Any = None,
    answerer: Callable[[str], Any] | None = None,
    understand_model: Runnable[Any, Any] | None = None,
) -> tuple[Any, Any]:
    from kovalo.workflow.graph import CopilotDeps
    from kovalo.workflow.schemas import Turn

    llm = llm or ReplayLLM(turns)
    deps = CopilotDeps(
        understand_model=understand_model or llm.with_structured_output(Turn),
        document_answerer=answerer or NoAnswerer(),
        policy=settings.policy,
        understand_prompt=settings.prompts.understand_system,
    )
    return deps, llm


def graph_with(
    settings: Any,
    turns: Sequence[Any] = (),
    *,
    llm: Any = None,
    answerer: Callable[[str], Any] | None = None,
) -> tuple[Any, Any]:
    """(pełny graf copilota na memory_checkpointer, atrapa modelu)."""
    from kovalo.workflow.checkpoint import memory_checkpointer
    from kovalo.workflow.graph import build_copilot_graph

    deps, llm = make_deps(settings, turns, llm=llm, answerer=answerer)
    return build_copilot_graph(deps, checkpointer=memory_checkpointer()), llm


def service_and_llm(
    settings: Any,
    turns: Sequence[Any] = (),
    *,
    llm: Any = None,
    answerer: Callable[[str], Any] | None = None,
) -> tuple[Any, Any]:
    from kovalo.workflow.service import CopilotService

    graph, llm = graph_with(settings, turns, llm=llm, answerer=answerer)
    service = CopilotService(graph, max_message_chars=settings.workflow.max_message_chars)
    return service, llm


def service_with(
    settings: Any,
    turns: Sequence[Any] = (),
    *,
    llm: Any = None,
    answerer: Callable[[str], Any] | None = None,
) -> Any:
    return service_and_llm(settings, turns, llm=llm, answerer=answerer)[0]


def answers(*values: Any) -> Callable[[dict], Any]:
    """Odpowiedzi na kolejne pauzy; wyczerpanie listy to StopIteration (nieoczekiwana pauza)."""
    it = iter(values)
    return lambda payload: next(it)


def gold_turns(dialogues: Sequence[Any], dialogue_id: str) -> list[Any]:
    from kovalo.workflow.schemas import Turn

    dialogue = next(d for d in dialogues if d.id == dialogue_id)
    return [Turn.model_validate(t.understand) for t in dialogue.turns]


class ExplodingChatModel(BaseChatModel):
    """Każde użycie to AssertionError (dowód, że graf/E3 nie wywołał modelu)."""

    calls: list[int] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "exploding"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(1)
        raise AssertionError("model nie powinien być wywołany (odtworzenie bez LLM-a)")

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:  # type: ignore[override]
        self.calls.append(1)
        raise AssertionError("model nie powinien być zbudowany ani związany ze schematem")


def ideal_chat_model(
    dialogues: Sequence[Any],
    settings: Any,
    *,
    cache: Any = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """Prawdziwy BaseChatModel (działa z `cache=`), który na wiadomość z dialogu zwraca JSON
    idealnego Turn, a na prompt RAG stałą odpowiedź Answer. `overrides`: tekst użytkownika →
    inny `understand` (np. zły Turn). Liczy wywołania w `.sent`."""
    from tests.fakes import CachingFakeChatModel

    by_user: dict[str, dict[str, Any]] = {}
    for dialogue in dialogues:
        for t in dialogue.turns:
            assert t.user not in by_user, f"powtórzona wiadomość w dialogach: {t.user!r}"
            by_user[t.user] = t.understand
    by_user.update(overrides or {})
    system = settings.prompts.understand_system

    def reply(prompt: str) -> str:
        if prompt.startswith(system):
            last = prompt.rsplit("\n", 1)[-1]
            return json.dumps(by_user[last])
        return json.dumps(
            {"answer": "Komitet utrzymał ceny wody.", "citations": ["MTG-402"], "answerable": True}
        )

    kwargs = {} if cache is None else {"cache": cache}
    return CachingFakeChatModel(reply=reply, sent=[], **kwargs)
