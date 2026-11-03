"""Serwis rozmowy: jedyny punkt wejścia do grafu copilota dla API i E3 (FR-3.6, FR-6.1).

Kontrakt: docs/contracts/f3.md, sekcja 6.7; decyzja: ADR-0009 pkt 7. Pilnuje strażnika pauzy
(wątek w pauzie odrzuca nową wiadomość), jednej tury naraz na wątek (blokada w procesie),
zamienia tekst człowieka na wartość wznowienia (text_pl.human_answer) i składa TurnResult.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt
from pydantic import BaseModel

from kovalo.text_pl import PauseType, human_answer
from kovalo.workflow.graph import APPROVAL_OPTIONS, is_valid_margin
from kovalo.workflow.replies import describe_request
from kovalo.workflow.schemas import Status

__all__ = [
    "ChatLine",
    "CopilotService",
    "NotPausedError",
    "PauseRef",
    "PendingPause",
    "ResumeValue",
    "Source",
    "StalePauseError",
    "ThreadBusyError",
    "ThreadPausedError",
    "ThreadState",
    "ThreadStuckError",
    "TurnResult",
    "TurnUsage",
]

ResumeValue = float | int | Literal["approved", "rejected", "cancel"]


class Source(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    score: float | None = None


class TurnUsage(BaseModel):
    calls: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None = None
    models: list[str]  # faktyczne modele (fallback widoczny)


class PauseRef(BaseModel):
    type: PauseType
    key: str  # rola (approval) albo pole (clarification)
    revision: int | None = None


class PendingPause(BaseModel):
    type: PauseType
    field: str | None = None  # clarification
    question: str | None = None  # clarification
    role: str | None = None  # approval
    revision: int | None = None
    approvers: list[str] = []
    request: dict[str, Any] | None = None
    summary: str | None = None  # describe_request(request)
    options: list[str]  # approval: approved/rejected/cancel; clarification: cancel

    @property
    def key(self) -> str:
        return (self.role if self.type == "approval" else self.field) or ""

    @property
    def ref(self) -> PauseRef:
        return PauseRef(type=self.type, key=self.key, revision=self.revision)


class TurnResult(BaseModel):
    thread_id: str
    status: Status | None
    replies: list[str]  # treść AIMessage dodanych w tym wywołaniu, po kolei
    pending: PendingPause | None
    resume_rejected: bool = False  # wartość wznowienia nieprawidłowa: graf pyta ponownie
    turn: dict[str, Any] | None = None
    request: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    approvals: dict[str, str] = {}
    sources: list[Source] = []  # cytowane chunki kontekstu pytania o dokumenty
    usage: TurnUsage | None = None  # wywołania LLM-a w tym wywołaniu (FR-6.5)
    error: str | None = None


class ChatLine(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ThreadState(BaseModel):
    thread_id: str
    status: Status | None  # None: wątek bez checkpointu
    pending: PendingPause | None
    request: dict[str, Any] | None
    decision: dict[str, Any] | None
    approvals: dict[str, str]
    messages: list[ChatLine]


class ThreadPausedError(RuntimeError):
    """Nowa wiadomość w wątku, który czeka na odpowiedź na pauzę (FR-3.6)."""

    def __init__(self, pending: PendingPause) -> None:
        who = pending.role if pending.type == "approval" else "doprecyzowanie marży"
        super().__init__(f"Wątek czeka na odpowiedź: {who}. Odpowiedz na pauzę albo ją anuluj.")
        self.pending = pending


class NotPausedError(RuntimeError):
    """Wznowienie wątku, który nie czeka na pauzie."""


class StalePauseError(RuntimeError):
    """pause_ref nie wskazuje bieżącej pauzy (np. drugie kliknięcie starego przycisku)."""

    def __init__(self, pending: PendingPause) -> None:
        super().__init__(
            f"Ta decyzja jest nieaktualna: wątek czeka teraz na {pending.key or pending.type}."
        )
        self.pending = pending


class ThreadBusyError(RuntimeError):
    """W tym wątku trwa inna tura."""


class ThreadStuckError(RuntimeError):
    """Checkpoint z niedokończonym krokiem bez pauzy (wyjątek w węźle przerwał turę)."""


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _dump(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)


def _pending(payload: Mapping[str, Any]) -> PendingPause:
    """Ładunek interrupt() grafu (klucz `type`, jak w kursie) → PendingPause."""
    if payload["type"] == "approval":
        request = payload.get("request")
        return PendingPause(
            type="approval",
            role=payload.get("role"),
            revision=payload.get("revision"),
            approvers=list(payload.get("approvers") or []),
            request=request,
            summary=describe_request(request) if request else None,
            options=list(APPROVAL_OPTIONS),
        )
    return PendingPause(
        type="clarification",
        field=payload.get("field"),
        question=payload.get("question"),
        options=["cancel"],
    )


def _first_pending(interrupts: Sequence[Interrupt]) -> PendingPause | None:
    return _pending(interrupts[0].value) if interrupts else None


def _is_valid_resume(pause_type: str, value: Any) -> bool:
    if value == "cancel":
        return True
    if pause_type == "clarification":
        return is_valid_margin(value)
    return value in ("approved", "rejected")


def _sources(record: Mapping[str, Any] | None) -> list[Source]:
    """Chunki kontekstu z doc_id wymienionym w output.citations."""
    if not record or not record.get("output"):
        return []
    cited = set(record["output"].get("citations") or [])
    return [
        Source(doc_id=c["doc_id"], chunk_id=c["chunk_id"], text=c["text"], score=c.get("score"))
        for c in record.get("context") or []
        if c["doc_id"] in cited
    ]


def _usage(calls: Sequence[Mapping[str, Any]]) -> TurnUsage | None:
    if not calls:
        return None
    usages = [c["usage"] for c in calls if c.get("usage")]
    reasoning = [u["reasoning_tokens"] for u in usages if u.get("reasoning_tokens") is not None]
    return TurnUsage(
        calls=len(calls),
        input_tokens=sum(int(u["input_tokens"]) for u in usages),
        output_tokens=sum(int(u["output_tokens"]) for u in usages),
        reasoning_tokens=sum(reasoning) if reasoning else None,
        models=[str(c["model"]) for c in calls if c.get("model")],
    )


def _text(message: AnyMessage) -> str:
    return message.content if isinstance(message.content, str) else message.text


class CopilotService:
    """send_message / resume / get_thread nad skompilowanym grafem z checkpointerem."""

    def __init__(
        self, graph: CompiledStateGraph[Any, Any, Any, Any], *, max_message_chars: int = 4000
    ) -> None:
        self.graph = graph
        self.max_message_chars = max_message_chars
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    @contextmanager
    def _turn(self, thread_id: str) -> Iterator[None]:
        with self._locks_guard:
            lock = self._locks.setdefault(thread_id, threading.Lock())
        if not lock.acquire(blocking=False):
            raise ThreadBusyError(
                f"W wątku {thread_id} trwa inna tura. Poczekaj na odpowiedź i spróbuj ponownie."
            )
        try:
            yield
        finally:
            lock.release()

    def send_message(self, thread_id: str, text: str) -> TurnResult:
        """Nowa wiadomość menedżera.

        Pusta albo za długa → ValueError; wątek w pauzie → ThreadPausedError; checkpoint
        z przerwaną turą → ThreadStuckError; inna tura w wątku → ThreadBusyError.
        """
        if not text.strip():
            raise ValueError("Wiadomość jest pusta.")
        if len(text) > self.max_message_chars:
            raise ValueError(f"Wiadomość ma {len(text)} znaków; limit to {self.max_message_chars}.")
        cfg = _config(thread_id)
        with self._turn(thread_id):
            snapshot = self.graph.get_state(cfg)
            pending = _first_pending(snapshot.interrupts)
            if pending is not None:
                raise ThreadPausedError(pending)
            if snapshot.next:
                raise ThreadStuckError(
                    f"Wątek {thread_id} ma przerwaną turę (krok {', '.join(snapshot.next)})."
                )
            before = len(snapshot.values.get("messages") or [])
            output = self.graph.invoke(
                {"messages": [HumanMessage(text)]}, cfg, version="v2", durability="sync"
            )
            return self._result(thread_id, output.interrupts, before, fresh=True)

    def resume(
        self,
        thread_id: str,
        *,
        value: ResumeValue | None = None,
        text: str | None = None,
        pause_ref: PauseRef | None = None,
    ) -> TurnResult:
        """Odpowiedź na bieżącą pauzę: dokładnie jedno z value/text (inaczej ValueError).

        Brak pauzy → NotPausedError; pause_ref ≠ bieżąca pauza → StalePauseError (stan bez
        zmian). Nieprawidłowa wartość i tak trafia do grafu (pyta ponownie, jak w kursie),
        a TurnResult.resume_rejected = True.
        """
        if (value is None) == (text is None):
            raise ValueError("Podaj dokładnie jedno: value albo text.")
        cfg = _config(thread_id)
        with self._turn(thread_id):
            snapshot = self.graph.get_state(cfg)
            pending = _first_pending(snapshot.interrupts)
            if pending is None:
                if snapshot.next:
                    raise ThreadStuckError(
                        f"Wątek {thread_id} ma przerwaną turę (krok {', '.join(snapshot.next)})."
                    )
                raise NotPausedError(f"Wątek {thread_id} nie czeka na odpowiedź.")
            if pause_ref is not None and pause_ref != pending.ref:
                raise StalePauseError(pending)
            answer: Any = human_answer(pending.type, text) if text is not None else value
            valid = _is_valid_resume(pending.type, answer)
            before = len(snapshot.values.get("messages") or [])
            output = self.graph.invoke(Command(resume=answer), cfg, version="v2", durability="sync")
            result = self._result(thread_id, output.interrupts, before, fresh=False)
            return result.model_copy(update={"resume_rejected": not valid})

    def get_thread(self, thread_id: str) -> ThreadState:
        """Stan wątku bez wywoływania grafu; wątek bez checkpointu → status None, puste listy."""
        snapshot = self.graph.get_state(_config(thread_id))
        values = snapshot.values or {}
        messages = [
            ChatLine(role="user" if isinstance(m, HumanMessage) else "assistant", content=_text(m))
            for m in values.get("messages") or []
            if isinstance(m, HumanMessage | AIMessage)
        ]
        return ThreadState(
            thread_id=thread_id,
            status=values.get("status"),
            pending=_first_pending(snapshot.interrupts),
            request=_dump(values.get("request")),
            decision=_dump(values.get("decision")),
            approvals=dict(values.get("approvals") or {}),
            messages=messages,
        )

    def get_values(self, thread_id: str) -> dict[str, Any]:
        """Surowe wartości stanu wątku (turn, record, calls…): driver E3, diagnostyka."""
        return dict(self.graph.get_state(_config(thread_id)).values or {})

    def _result(
        self,
        thread_id: str,
        interrupts: Sequence[Interrupt],
        before: int,
        *,
        fresh: bool,
    ) -> TurnResult:
        values = self.get_values(thread_id)
        messages = values.get("messages") or []
        return TurnResult(
            thread_id=thread_id,
            status=values.get("status"),
            replies=[_text(m) for m in messages[before:] if isinstance(m, AIMessage)],
            pending=_first_pending(interrupts),
            turn=_dump(values.get("turn")),
            request=_dump(values.get("request")),
            decision=_dump(values.get("decision")),
            approvals=dict(values.get("approvals") or {}),
            sources=_sources(values.get("record")),
            # wznowienie nie wywołuje modelu: tokeny tury raportuje tylko send_message
            usage=_usage(values.get("calls") or []) if fresh else None,
            error=values.get("error"),
        )
