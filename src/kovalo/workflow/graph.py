"""Graf copilota: węzły, kierowanie, pauzy (FR-3.3, FR-3.4; kroki 3-4 MP3 kursu).

Kontrakt: docs/contracts/f3.md, sekcja 6.5; decyzja: ADR-0009 pkt 4-5. F4: węzeł
answer_documents z agentem (docs/contracts/f4.md, sekcja 8.1; ADR-0011 pkt 6, ADR-0012 pkt 4).

    START → understand ─route_by_kind─┬─ document_question → answer_documents → END
                                      ├─ policy_question   → answer_policy    → END
                                      ├─ other | error     → reply_other      → END
                                      └─ approval_request  → update_request → clarify ⟲ → decide
                                                             → approve ⟲ → END

Zasady LangGraph: jedno interrupt() na wywołanie węzła, bez try/except wokół niego, ponowne
pytanie przez krawędź warunkową; przed interrupt() tylko odczyt stanu (węzeł po wznowieniu
startuje od początku). Model wywołuje wyłącznie `understand` (przed pauzami) oraz
document_answerer albo agent. Reguły polityki liczy kovalo.policy, nie model.
"""

from __future__ import annotations

import contextvars
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from kovalo.agents.trace import Trace, TraceEvent, trace_envelope
from kovalo.config import PolicyConfig
from kovalo.policy import required_approvers
from kovalo.policy.models import DiscountRequest
from kovalo.rag.answer import DocumentAnswerer, error_text
from kovalo.rag.records import TokenUsage
from kovalo.workflow import replies
from kovalo.workflow.schemas import TERMINAL_APPROVAL_STATUSES, LlmCall, Turn
from kovalo.workflow.state import CopilotState
from kovalo.workflow.understand import FIELDS, FLAGS, understand_turn, update_request

__all__ = [
    "APPROVAL_OPTIONS",
    "NODES",
    "AgentAnswerer",
    "CopilotDeps",
    "Node",
    "build_conversation_graph",
    "build_copilot_graph",
    "conversation_nodes",
    "is_valid_margin",
    "route_after_approve",
    "route_after_clarify",
    "route_after_decide",
    "route_after_update",
    "route_by_kind",
]

NODES = (
    "understand",
    "answer_documents",
    "answer_policy",
    "reply_other",
    "update_request",
    "clarify",
    "decide",
    "approve",
)
APPROVAL_OPTIONS = ("approved", "rejected", "cancel")

_END: Final = "__end__"  # = langgraph.graph.END, jako literał dla typów krawędzi


class Node(Protocol):
    """Funkcja węzła: stan → słownik zmian (`messages` to lista nowych wiadomości)."""

    def __call__(self, state: CopilotState) -> dict[str, Any]: ...


# (question, *, on_event=None, callbacks=None) -> Trace; produkcja: AgentRunner / LazyAgentRunner
AgentAnswerer = Callable[..., Trace]


@dataclass(frozen=True)
class CopilotDeps:
    """Czego potrzebują węzły. Testy podają atrapy, produkcja: workflow.factory.

    Dokładnie jedno z `document_answerer` (RAG z F2/F3, zawsze w E3) i `agent` (F4: agent
    z narzędziami, aplikacja); inaczej ValueError.
    """

    understand_model: Runnable[LanguageModelInput, Any]  # get_structured_model(Turn, 'understand')
    policy: PolicyConfig
    understand_prompt: str  # settings.prompts.by_name(workflow.understand_prompt)
    document_answerer: DocumentAnswerer | None = None
    agent: AgentAnswerer | None = None

    def __post_init__(self) -> None:
        if (self.document_answerer is None) == (self.agent is None):
            raise ValueError("CopilotDeps: podaj dokładnie jedno z document_answerer i agent.")


def is_valid_margin(value: Any) -> bool:
    """Liczba (nie bool), skończona, w [-100, 100]: poprawna odpowiedź na doprecyzowanie marży."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and -100 <= value <= 100
    )


def _say(text: str) -> AIMessage:
    return AIMessage(content=text)


def _turn(state: CopilotState) -> Turn:
    turn = state.get("turn")
    if turn is None:  # graf zawsze zaczyna od understand
        raise RuntimeError("Brak 'turn' w stanie: węzeł uruchomiony bez understand")
    return turn


def _request(state: CopilotState) -> DiscountRequest:
    request = state.get("request")
    if request is None:
        raise RuntimeError("Brak 'request' w stanie: węzeł uruchomiony bez update_request")
    return request


def _request_from_turn(turn: Turn) -> DiscountRequest:
    """Tymczasowy wniosek dla pytania o politykę (flagi None → False); nigdy nie trafia do stanu."""
    fields: dict[str, Any] = {name: getattr(turn, name) for name in FIELDS}
    for flag in FLAGS:
        fields[flag] = bool(fields[flag])
    return DiscountRequest(**fields)


def conversation_nodes(deps: CopilotDeps) -> dict[str, Node]:
    """Funkcje węzłów {nazwa: funkcja}; te same dla grafu 3a i pełnego grafu 3b."""

    def understand(state: CopilotState) -> dict[str, Any]:
        messages = state.get("messages") or []
        last = messages[-1]
        text = last.content if isinstance(last.content, str) else last.text
        result = understand_turn(
            messages[:-1],
            text,
            deps.understand_model,
            system_prompt=deps.understand_prompt,
        )
        return {
            "turn": result.turn,
            "decision": None,
            "record": None,
            "error": result.call.error,
            "calls": [result.call.model_dump(mode="json")],
        }

    def answer_with_agent(
        state: CopilotState, agent: AgentAnswerer, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """Port `answer_with_tools` (krok 3 MP5): odpowiedź agenta albo krótkie przeprosiny,
        koperta trace w `record`; kroki agenta do strumienia `custom` (ADR-0012 pkt 4).
        Agent dostaje tylko callbacki konfiguracji węzła (F5, R1-2: run agenta pod runem grafu)."""
        turn = _turn(state)
        writer = get_stream_writer()
        node_context = contextvars.copy_context()  # writer działa tylko w kontekście węzła

        def on_event(phase: str, event: TraceEvent) -> None:
            payload = {
                "kind": "agent_event",
                "phase": phase,
                "event": event.model_dump(mode="json"),
            }
            node_context.run(writer, payload)

        try:
            # Pusty kontekst: agent (create_agent, checkpointer=False) nie dziedziczy konfiguracji
            # zewnętrznego grafu (durability="sync", przestrzeń checkpointów), więc biegnie jak
            # zwykła funkcja; zdarzenia idą do strumienia przez writer pobrany wyżej. Przekazujemy
            # tylko callbacki węzła (R1-2), żeby run agenta zagnieżdżał się pod turą grafu.
            callbacks = (config or {}).get("callbacks")
            trace = contextvars.Context().run(
                agent, turn.standalone, on_event=on_event, callbacks=callbacks
            )
        except Exception as exc:  # run_question nie zgłasza wyjątków; to tylko zabezpieczenie
            trace = Trace(question=turn.standalone, status="error", error=error_text(exc))
        if trace.answer is not None:
            text = trace.answer
        elif trace.status == "limit":
            text = replies.AGENT_LIMIT_REPLY
        else:
            text = replies.AGENT_ERROR_REPLY
        calls = [
            LlmCall(
                purpose="agent",
                model=c.model,
                usage=TokenUsage(
                    input_tokens=c.input_tokens,
                    output_tokens=c.output_tokens,
                    reasoning_tokens=c.reasoning_tokens,
                ),
            ).model_dump(mode="json")
            for c in trace.model_calls
        ]
        return {
            "status": "answered" if trace.status == "answered" else "error",
            "error": trace.error,
            "record": trace_envelope(trace),
            "calls": [*(state.get("calls") or []), *calls],
            "messages": [_say(text)],
        }

    def answer_documents(state: CopilotState, config: RunnableConfig) -> dict[str, Any]:
        # `config` wymagany i typu RunnableConfig: tylko wtedy LangGraph go wstrzykuje (R1-2)
        if deps.agent is not None:
            return answer_with_agent(state, deps.agent, config)
        turn = _turn(state)
        answerer = deps.document_answerer
        if answerer is None:  # CopilotDeps.__post_init__ pilnuje, że jest jedno z dwóch
            raise RuntimeError("Brak document_answerer i agenta w CopilotDeps")
        try:
            record = answerer(turn.standalone)
        except Exception as exc:  # np. IndexUnavailableError: tura 'error', nie awaria
            return {
                "status": "error",
                "error": error_text(exc),
                "record": None,
                "messages": [_say(replies.DOCUMENT_ERROR_REPLY)],
            }
        text, status = replies.document_reply(record)
        call = LlmCall(
            purpose="answer",
            model=record.answered_by,
            usage=record.usage,
            seconds=record.seconds,
            error=record.error,
        )
        return {
            "status": status,
            "error": record.error,
            "record": record.model_dump(mode="json"),
            "calls": [*(state.get("calls") or []), call.model_dump(mode="json")],
            "messages": [_say(text)],
        }

    def answer_policy(state: CopilotState) -> dict[str, Any]:
        turn = _turn(state)
        if turn.discount_pct is None:
            return {
                "status": "answered",
                "decision": None,
                "messages": [_say(replies.MISSING_DISCOUNT)],
            }
        request = _request_from_turn(turn)
        try:
            decision = required_approvers(request, deps.policy)
        except ValueError as exc:
            return {"status": "error", "error": str(exc), "messages": [_say(str(exc))]}
        return {
            "decision": decision,
            "status": "answered",
            "messages": [_say(replies.policy_answer(request, decision))],
        }

    def reply_other(state: CopilotState) -> dict[str, Any]:
        if _turn(state).kind == "error":
            return {"status": "error", "messages": [_say(replies.ERROR_REPLY)]}
        return {"status": "answered", "messages": [_say(replies.OTHER_REPLY)]}

    def update_request_node(state: CopilotState) -> dict[str, Any]:
        current = state.get("request")
        try:
            new = update_request(current, _turn(state))
            if new is None:  # kind == approval_request zawsze daje wniosek
                raise ValueError(replies.MISSING_DISCOUNT)
            decision = required_approvers(new, deps.policy)
        except ValueError as exc:
            return {"status": "error", "error": str(exc), "messages": [_say(str(exc))]}
        # Cykl życia wniosku w jednym miejscu: zgody zostają tylko dla niezmienionego wniosku,
        # który każda wymagana rola już zatwierdziła; inaczej nowy cykl.
        approvals = state.get("approvals") or {}
        keep = (
            current is not None
            and new == current
            and all(approvals.get(role) == "approved" for role in decision.approvers)
        )
        missing_margin = new.margin_after_pct is None and not new.clearance
        return {
            "request": new,
            "approvals": dict(approvals) if keep else {},
            "status": "pending_clarification" if missing_margin else "pending_approval",
        }

    def clarify(state: CopilotState) -> dict[str, Any]:
        request = _request(state)
        answer = interrupt(
            {
                "type": "clarification",
                "field": "margin_after_pct",
                "question": replies.CLARIFY_QUESTION,
            }
        )
        if answer == "cancel":
            return {"status": "cancelled", "messages": [_say(replies.CANCELLED_REPLY)]}
        if is_valid_margin(answer):
            # brakująca informacja, nie zmiana: ta sama wersja wniosku
            return {
                "request": request.model_copy(update={"margin_after_pct": float(answer)}),
                "status": "pending_approval",
                "messages": [HumanMessage(replies.margin_message(answer))],
            }
        return {}  # nieprawidłowa odpowiedź: route_after_clarify pyta ponownie

    def decide(state: CopilotState) -> dict[str, Any]:
        request = _request(state)
        try:
            decision = required_approvers(request, deps.policy)
        except ValueError as exc:
            return {"status": "error", "error": str(exc), "messages": [_say(str(exc))]}
        approvals = state.get("approvals") or {}
        if all(approvals.get(role) == "approved" for role in decision.approvers):
            return {
                "decision": decision,
                "status": "approved",
                "messages": [_say(replies.already_approved_reply(request, decision))],
            }
        return {
            "decision": decision,
            "status": "pending_approval",
            "messages": [_say(replies.decision_reply(request, decision))],
        }

    def approve(state: CopilotState) -> dict[str, Any]:
        request = _request(state)
        decision = state.get("decision")
        if decision is None:
            raise RuntimeError("Brak 'decision' w stanie: approve uruchomiony bez decide")
        approvals = state.get("approvals") or {}
        role = next(r for r in decision.approvers if r not in approvals)
        answer = interrupt(
            {
                "type": "approval",
                "role": role,
                "revision": request.revision,
                "approvers": list(decision.approvers),
                "request": request.model_dump(),
            }
        )
        if answer == "approved":
            updated = {**approvals, role: "approved"}
            if all(updated.get(r) == "approved" for r in decision.approvers):
                return {
                    "approvals": updated,
                    "status": "approved",
                    "messages": [_say(replies.approved_reply(request, decision))],
                }
            return {"approvals": updated}
        if answer == "rejected":
            return {
                "approvals": {**approvals, role: "rejected"},
                "status": "rejected",
                "messages": [_say(replies.rejected_reply(request, role))],
            }
        if answer == "cancel":
            return {"status": "cancelled", "messages": [_say(replies.CANCELLED_REPLY)]}
        return {}  # nieprawidłowa odpowiedź: ta sama rola pytana ponownie

    return {
        "understand": understand,
        "answer_documents": cast(Node, answer_documents),  # (state, config): config z LangGraph
        "answer_policy": answer_policy,
        "reply_other": reply_other,
        "update_request": update_request_node,
        "clarify": clarify,
        "decide": decide,
        "approve": approve,
    }


# --- Kierowanie (czyste funkcje stanu) -------------------------------------------------------

_BY_KIND: dict[str, Literal["answer_documents", "answer_policy", "reply_other", "update_request"]]
_BY_KIND = {
    "document_question": "answer_documents",
    "policy_question": "answer_policy",
    "approval_request": "update_request",
    "other": "reply_other",
    "error": "reply_other",
}


def route_by_kind(
    state: CopilotState,
) -> Literal["answer_documents", "answer_policy", "reply_other", "update_request"]:
    return _BY_KIND[_turn(state).kind]


def route_after_update(state: CopilotState) -> Literal["clarify", "decide", "__end__"]:
    status = state.get("status")
    if status == "error":
        return _END
    if status == "pending_clarification":
        return "clarify"
    return "decide"


def _route_after_update_without_pauses(state: CopilotState) -> Literal["decide", "__end__"]:
    return _END if state.get("status") == "error" else "decide"


def route_after_clarify(state: CopilotState) -> Literal["clarify", "decide", "__end__"]:
    if state.get("status") == "cancelled":
        return _END
    if _request(state).margin_after_pct is None:
        return "clarify"
    return "decide"


def route_after_decide(state: CopilotState) -> Literal["approve", "__end__"]:
    return _END if state.get("status") in ("approved", "error") else "approve"


def route_after_approve(state: CopilotState) -> Literal["approve", "__end__"]:
    return _END if state.get("status") in TERMINAL_APPROVAL_STATUSES else "approve"


# --- Grafy -----------------------------------------------------------------------------------


def _base_builder(deps: CopilotDeps, names: tuple[str, ...]) -> StateGraph[Any, Any, Any, Any]:
    nodes = conversation_nodes(deps)
    builder: StateGraph[Any, Any, Any, Any] = StateGraph(CopilotState)
    for name in names:
        builder.add_node(name, nodes[name])
    builder.add_edge(START, "understand")
    builder.add_conditional_edges(
        "understand",
        route_by_kind,
        ["answer_documents", "answer_policy", "reply_other", "update_request"],
    )
    for name in ("answer_documents", "answer_policy", "reply_other"):
        builder.add_edge(name, END)
    return builder


def build_conversation_graph(
    deps: CopilotDeps, *, checkpointer: BaseCheckpointSaver[Any]
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Punkt kontrolny 3a kursu: bez pauz. update_request → (error → END) → decide → END."""
    builder = _base_builder(deps, (*NODES[:5], "decide"))
    builder.add_conditional_edges(
        "update_request", _route_after_update_without_pauses, ["decide", END]
    )
    builder.add_edge("decide", END)
    return builder.compile(checkpointer=checkpointer)


def build_copilot_graph(
    deps: CopilotDeps, *, checkpointer: BaseCheckpointSaver[Any]
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Pełny graf (punkt kontrolny 3b): doprecyzowanie marży i akceptacja rola po roli."""
    builder = _base_builder(deps, NODES)
    builder.add_conditional_edges("update_request", route_after_update, ["clarify", "decide", END])
    builder.add_conditional_edges("clarify", route_after_clarify, ["clarify", "decide", END])
    builder.add_conditional_edges("decide", route_after_decide, ["approve", END])
    builder.add_conditional_edges("approve", route_after_approve, ["approve", END])
    return builder.compile(checkpointer=checkpointer)
