"""Serwis rozmowy: jedyny punkt wejścia do grafu copilota dla API i E3 (FR-3.6, FR-6.1).

Kontrakt: docs/contracts/f3.md, sekcja 6.7; docs/contracts/f4.md, sekcje 8.3-8.5; decyzje:
ADR-0009 pkt 7, ADR-0012. Pilnuje strażnika pauzy (wątek w pauzie odrzuca nową wiadomość),
jednej tury naraz na wątek (blokada w procesie, wpis znika, gdy nikt jej nie trzyma), zamienia
tekst człowieka na wartość wznowienia (text_pl.human_answer) i składa TurnResult.

F4: tura biegnie w wątku roboczym serwisu i zawsze dobiega do końca, niezależnie od tego, czy
ktoś czyta jej strumień (TurnStream: tool_start, tool_end, final albo error). send_message
i resume czytają ten sam strumień do końca. Utrata Postgresa → CheckpointerUnavailableError.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt, StateSnapshot
from pydantic import BaseModel

from kovalo.agents.show import brief, format_trace
from kovalo.agents.trace import Trace, trace_from_record
from kovalo.text_pl import PauseType, human_answer
from kovalo.workflow.checkpoint import CONNECTION_ERRORS, CheckpointerUnavailableError
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
    "TraceView",
    "TurnEvent",
    "TurnResult",
    "TurnStream",
    "TurnUsage",
]

logger = logging.getLogger(__name__)

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
    trace: dict[str, Any] | None = None  # F4: Trace.model_dump(mode='json') tury z agentem
    evidence_path: str | None = None  # F4: agents.show.format_trace(trace)


class TurnEvent(BaseModel):
    """Zdarzenie strumienia tury (ADR-0012 pkt 2).

    tool_start: {call_id, parent_call_id, agent, tool, arguments, origin}
    tool_end:   {call_id, parent_call_id, tool, status, brief, result, error, transport,
                 invocation_id}  (transport 'mcp' albo None; F5, kontrakt 9.2)
    final:      TurnResult.model_dump(mode='json')
    error:      {error: 'internal_error' | 'copilot_unavailable', detail}
    """

    type: Literal["tool_start", "tool_end", "final", "error"]
    data: dict[str, Any]


class TraceView(BaseModel):
    """Ostatni trace agenta w wątku (GET /threads/{id}/trace)."""

    thread_id: str
    trace_id: str | None = None
    created_at: datetime | None = None
    question: str | None = None
    trace: dict[str, Any] | None = None  # None: w wątku nie było tury z agentem
    evidence_path: str | None = None


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


def _agent_sources(trace: Mapping[str, Any]) -> list[Source]:
    """Fragmenty z udanych wywołań search_documents (kolejność wywołań, bez powtórzeń chunk_id)."""
    seen: set[str] = set()
    sources: list[Source] = []
    for event in trace.get("events") or []:
        result = event.get("result")
        if event.get("tool") != "search_documents" or event.get("status") != "ok":
            continue
        if not isinstance(result, dict):
            continue
        for passage in result.get("passages") or []:
            if passage["chunk_id"] in seen:
                continue
            seen.add(passage["chunk_id"])
            sources.append(
                Source(doc_id=passage["doc_id"], chunk_id=passage["chunk_id"], text=passage["text"])
            )
    return sources


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


_UNAVAILABLE = (
    "Postgres niedostępny dla checkpointów ({name}). Uruchom: docker compose up -d postgres"
)
_INTERNAL = "Wewnętrzny błąd copilota; szczegóły w logu API."


def _unavailable(exc: BaseException) -> CheckpointerUnavailableError:
    # komunikaty sterownika mogą zawierać DSN, więc zostaje tylko klasa błędu
    return CheckpointerUnavailableError(_UNAVAILABLE.format(name=type(exc).__name__))


def _tool_start(event: Mapping[str, Any]) -> TurnEvent:
    return TurnEvent(
        type="tool_start",
        data={
            "call_id": event.get("call_id"),
            "parent_call_id": event.get("parent_call_id"),
            "agent": event.get("agent"),
            "tool": event.get("tool"),
            "arguments": event.get("arguments") or {},
            "origin": event.get("origin"),
        },
    )


def _tool_end(event: Mapping[str, Any]) -> TurnEvent:
    ok = event.get("status") == "ok"
    return TurnEvent(
        type="tool_end",
        data={
            "call_id": event.get("call_id"),
            "parent_call_id": event.get("parent_call_id"),
            "tool": event.get("tool"),
            "status": event.get("status"),
            "brief": brief(event.get("result")) if ok else None,
            "result": event.get("result"),
            "error": event.get("error"),
            "transport": event.get("transport"),
            "invocation_id": event.get("invocation_id"),
        },
    )


_END = object()  # znacznik końca kolejki zdarzeń


class TurnStream:
    """Iterator TurnEvent jednej tury. Tura biegnie w wątku roboczym serwisu i zawsze dobiega
    do końca, niezależnie od czytającego (ADR-0012 pkt 3). Ostatnie zdarzenie to final albo
    error."""

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self._queue: queue.Queue[Any] = queue.Queue()
        self._done = threading.Event()
        self._finished = False  # czytający dostał final/error
        self._closed = False
        self._result: TurnResult | None = None
        self._exception: BaseException | None = None

    # --- strona wątku roboczego ---
    def _emit(self, event: TurnEvent) -> None:
        if not self._closed:
            self._queue.put(event)

    def _finish(
        self, final: TurnEvent, result: TurnResult | None, exc: BaseException | None
    ) -> None:
        self._result, self._exception = result, exc
        self._queue.put(final)
        self._done.set()

    # --- strona czytającego ---
    def next_event(self, timeout: float | None) -> TurnEvent | None:
        """Następne zdarzenie albo None, gdy nic nie przyszło w `timeout` (ping SSE) albo
        strumień już się skończył."""
        if self._finished:
            return None
        try:
            event = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if event.type in ("final", "error"):
            self._finished = True
        return event

    def __iter__(self) -> Iterator[TurnEvent]:
        while not self._finished:
            event = self.next_event(None)
            if event is not None:
                yield event

    def result(self) -> TurnResult:
        """Czeka do końca tury; zdarzenie error → odpowiedni wyjątek."""
        self._done.wait()
        if self._exception is not None:
            raise self._exception
        assert self._result is not None
        return self._result

    def close(self) -> None:
        """Przestaje czytać; tura biegnie dalej (wynik zostaje w checkpoincie)."""
        self._closed = True


@dataclass
class _Slot:
    lock: threading.Lock
    users: int = 0


class CopilotService:
    """send_message / resume / get_thread / get_trace nad skompilowanym grafem z checkpointerem.

    start_message / start_resume sprawdzają wszystko, co da się sprawdzić przed turą,
    synchronicznie (te same wyjątki co F3) i biorą blokadę wątku; turę wykonuje wątek roboczy,
    który zwalnia blokadę na końcu.
    """

    def __init__(
        self,
        graph: CompiledStateGraph[Any, Any, Any, Any],
        *,
        max_message_chars: int = 4000,
        max_workers: int = 4,
    ) -> None:
        self.graph = graph
        self.max_message_chars = max_message_chars
        self._locks: dict[str, _Slot] = {}
        self._locks_guard = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="turn")

    # --- blokady wątków (R1-3) ---
    def _leave(self, thread_id: str, slot: _Slot) -> None:
        with self._locks_guard:
            slot.users -= 1
            if slot.users == 0 and self._locks.get(thread_id) is slot:
                del self._locks[thread_id]

    def _acquire(self, thread_id: str) -> _Slot:
        with self._locks_guard:
            slot = self._locks.get(thread_id)
            if slot is None:
                slot = self._locks[thread_id] = _Slot(threading.Lock())
            slot.users += 1
        if not slot.lock.acquire(blocking=False):
            self._leave(thread_id, slot)
            raise ThreadBusyError(
                f"W wątku {thread_id} trwa inna tura. Poczekaj na odpowiedź i spróbuj ponownie."
            )
        return slot

    def _release(self, thread_id: str, slot: _Slot) -> None:
        slot.lock.release()
        self._leave(thread_id, slot)

    # --- odczyt stanu (R1-2) ---
    def _state(self, cfg: RunnableConfig) -> StateSnapshot:
        try:
            return self.graph.get_state(cfg)
        except CONNECTION_ERRORS as exc:
            raise _unavailable(exc) from None

    # --- tura w wątku roboczym ---
    def _start(
        self,
        thread_id: str,
        slot: _Slot,
        graph_input: Any,
        before: int,
        *,
        fresh: bool,
        adjust: Callable[[TurnResult], TurnResult] | None = None,
    ) -> TurnStream:
        stream = TurnStream(thread_id)
        cfg = _config(thread_id)

        def work() -> None:
            try:
                for part in self.graph.stream(
                    graph_input,
                    cfg,
                    stream_mode=["custom", "updates"],
                    version="v2",
                    durability="sync",
                ):
                    if part["type"] != "custom":
                        continue
                    data = part["data"]
                    if not isinstance(data, dict) or data.get("kind") != "agent_event":
                        continue
                    event = data.get("event") or {}
                    phase = data.get("phase")
                    stream._emit(_tool_start(event) if phase == "start" else _tool_end(event))
                interrupts = self._state(cfg).interrupts
                result = self._result(thread_id, interrupts, before, fresh=fresh)
                if adjust is not None:
                    result = adjust(result)
                stream._finish(
                    TurnEvent(type="final", data=result.model_dump(mode="json")), result, None
                )
            except CheckpointerUnavailableError as exc:
                stream._finish(_error_event("copilot_unavailable", str(exc)), None, exc)
            except CONNECTION_ERRORS as exc:
                unavailable = _unavailable(exc)
                stream._finish(
                    _error_event("copilot_unavailable", str(unavailable)), None, unavailable
                )
            except BaseException as exc:  # także spoza Exception: tura zawsze się kończy
                logger.exception("Tura w wątku %s zakończona wyjątkiem", thread_id)
                stream._finish(_error_event("internal_error", _INTERNAL), None, exc)
            finally:
                self._release(thread_id, slot)

        try:
            self._executor.submit(work)
        except BaseException:
            self._release(thread_id, slot)
            raise
        return stream

    def start_message(self, thread_id: str, text: str) -> TurnStream:
        """Nowa wiadomość menedżera jako strumień tury.

        Pusta albo za długa → ValueError; wątek w pauzie → ThreadPausedError; checkpoint
        z przerwaną turą → ThreadStuckError; inna tura w wątku → ThreadBusyError; utrata
        Postgresa → CheckpointerUnavailableError. Wszystko przed startem strumienia.
        """
        if not text.strip():
            raise ValueError("Wiadomość jest pusta.")
        if len(text) > self.max_message_chars:
            raise ValueError(f"Wiadomość ma {len(text)} znaków; limit to {self.max_message_chars}.")
        slot = self._acquire(thread_id)
        try:
            snapshot = self._state(_config(thread_id))
            pending = _first_pending(snapshot.interrupts)
            if pending is not None:
                raise ThreadPausedError(pending)
            if snapshot.next:
                raise ThreadStuckError(
                    f"Wątek {thread_id} ma przerwaną turę (krok {', '.join(snapshot.next)})."
                )
            before = len(snapshot.values.get("messages") or [])
        except BaseException:
            self._release(thread_id, slot)
            raise
        return self._start(thread_id, slot, {"messages": [HumanMessage(text)]}, before, fresh=True)

    def start_resume(
        self,
        thread_id: str,
        *,
        value: ResumeValue | None = None,
        text: str | None = None,
        pause_ref: PauseRef | None = None,
    ) -> TurnStream:
        """Odpowiedź na bieżącą pauzę jako strumień tury: dokładnie jedno z value/text.

        Brak pauzy → NotPausedError; pause_ref ≠ bieżąca pauza → StalePauseError (stan bez
        zmian). Nieprawidłowa wartość i tak trafia do grafu (pyta ponownie, jak w kursie),
        a TurnResult.resume_rejected = True.
        """
        if (value is None) == (text is None):
            raise ValueError("Podaj dokładnie jedno: value albo text.")
        slot = self._acquire(thread_id)
        try:
            snapshot = self._state(_config(thread_id))
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
        except BaseException:
            self._release(thread_id, slot)
            raise
        return self._start(
            thread_id,
            slot,
            Command(resume=answer),
            before,
            fresh=False,
            adjust=lambda result: result.model_copy(update={"resume_rejected": not valid}),
        )

    def send_message(self, thread_id: str, text: str) -> TurnResult:
        """start_message(...).result(): E3, testy F3 i API JSON idą ścieżką strumienia."""
        return self.start_message(thread_id, text).result()

    def resume(
        self,
        thread_id: str,
        *,
        value: ResumeValue | None = None,
        text: str | None = None,
        pause_ref: PauseRef | None = None,
    ) -> TurnResult:
        """start_resume(...).result()."""
        return self.start_resume(thread_id, value=value, text=text, pause_ref=pause_ref).result()

    def get_thread(self, thread_id: str) -> ThreadState:
        """Stan wątku bez wywoływania grafu; wątek bez checkpointu → status None, puste listy."""
        snapshot = self._state(_config(thread_id))
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
        return dict(self._state(_config(thread_id)).values or {})

    def get_trace(self, thread_id: str) -> TraceView:
        """Ostatnia koperta agent_trace w wątku: bieżący stan, a gdy `record` nie jest kopertą,
        historia checkpointów od najnowszego. Bez wywołania grafu."""
        cfg = _config(thread_id)
        record = _envelope((self._state(cfg).values or {}).get("record"))
        if record is None:
            try:
                for snapshot in self.graph.get_state_history(cfg):
                    record = _envelope((snapshot.values or {}).get("record"))
                    if record is not None:
                        break
            except CONNECTION_ERRORS as exc:
                raise _unavailable(exc) from None
        if record is None:
            return TraceView(thread_id=thread_id)
        trace = trace_from_record(record)
        assert trace is not None
        return TraceView(
            thread_id=thread_id,
            trace_id=record.get("trace_id"),
            created_at=record.get("created_at"),
            question=trace.question,
            trace=record["trace"],
            evidence_path=format_trace(trace),
        )

    def close(self) -> None:
        """Czeka na trwające tury i zamyka wątki robocze."""
        self._executor.shutdown(wait=True)

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
        record = values.get("record")
        envelope = _envelope(record)
        trace = envelope["trace"] if envelope is not None else None
        return TurnResult(
            thread_id=thread_id,
            status=values.get("status"),
            replies=[_text(m) for m in messages[before:] if isinstance(m, AIMessage)],
            pending=_first_pending(interrupts),
            turn=_dump(values.get("turn")),
            request=_dump(values.get("request")),
            decision=_dump(values.get("decision")),
            approvals=dict(values.get("approvals") or {}),
            sources=_agent_sources(trace) if trace is not None else _sources(record),
            # wznowienie nie wywołuje modelu: tokeny tury raportuje tylko send_message
            usage=_usage(values.get("calls") or []) if fresh else None,
            error=values.get("error"),
            trace=trace,
            evidence_path=format_trace(Trace.model_validate(trace)) if trace is not None else None,
        )


def _error_event(error: str, detail: str) -> TurnEvent:
    return TurnEvent(type="error", data={"error": error, "detail": detail})


def _envelope(record: Any) -> dict[str, Any] | None:
    """Koperta agent_trace albo None (rekord RAG, brak rekordu)."""
    if isinstance(record, Mapping) and record.get("kind") == "agent_trace":
        return dict(record)
    return None
