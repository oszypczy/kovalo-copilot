"""Graf copilota: węzły, kierowanie, pauzy (FR-3.3, FR-3.4; kroki 3-4 MP3 kursu).

Kontrakt: docs/contracts/f3.md, sekcja 6.5; decyzja: ADR-0009 pkt 4-5.

    START → understand ─route_by_kind─┬─ document_question → answer_documents → END
                                      ├─ policy_question   → answer_policy    → END
                                      ├─ other | error     → reply_other      → END
                                      └─ approval_request  → update_request → clarify ⟲ → decide
                                                             → approve ⟲ → END

Zasady LangGraph: jedno interrupt() na wywołanie węzła, bez try/except wokół niego, ponowne
pytanie przez krawędź warunkową; przed interrupt() tylko odczyt stanu (węzeł po wznowieniu
startuje od początku). Model wywołuje wyłącznie `understand` (przed pauzami) oraz
document_answerer. Reguły polityki liczy kovalo.policy, nie model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from kovalo.config import PolicyConfig
from kovalo.policy import required_approvers
from kovalo.policy.models import DiscountRequest
from kovalo.rag.answer import DocumentAnswerer, error_text
from kovalo.workflow import replies
from kovalo.workflow.schemas import TERMINAL_APPROVAL_STATUSES, LlmCall, Turn
from kovalo.workflow.state import CopilotState
from kovalo.workflow.understand import FIELDS, FLAGS, understand_turn, update_request

__all__ = [
    "APPROVAL_OPTIONS",
    "NODES",
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


@dataclass(frozen=True)
class CopilotDeps:
    """Czego potrzebują węzły. Testy podają atrapy, produkcja: workflow.factory."""

    understand_model: Runnable[LanguageModelInput, Any]  # get_structured_model(Turn, 'understand')
    document_answerer: DocumentAnswerer  # F3: RAG; F5: agent przez MCP
    policy: PolicyConfig
    understand_prompt: str  # settings.prompts.understand_system


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

    def answer_documents(state: CopilotState) -> dict[str, Any]:
        turn = _turn(state)
        try:
            record = deps.document_answerer(turn.standalone)
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
        "answer_documents": answer_documents,
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
