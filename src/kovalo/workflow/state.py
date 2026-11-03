"""Stan grafu copilota (FR-3.3; helpers.workflow.CopilotState kursu + pole NASZE `calls`).

Kontrakt: docs/contracts/f3.md, sekcja 6.3. `messages` rośnie z każdą turą (reducer
add_messages); pozostałe pola są nadpisywane i opisują bieżącą sytuację.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from kovalo.policy.models import Decision, DiscountRequest
from kovalo.workflow.schemas import Status, Turn

__all__ = ["CopilotState"]


class CopilotState(TypedDict, total=False):
    """Co przenosi graf. `messages` rośnie z każdą turą; reszta opisuje bieżącą sytuację."""

    messages: Annotated[list[AnyMessage], add_messages]
    turn: Turn | None  # najnowsze zrozumienie wiadomości
    request: DiscountRequest | None  # kanoniczny wniosek
    decision: Decision | None  # wynik polityki dla `request` (albo dla pytania o politykę)
    approvals: dict[str, str]  # rola → 'approved' / 'rejected', tylko bieżąca wersja wniosku
    status: Status
    record: dict[str, Any] | None  # RunRecord.model_dump(mode='json') dla pytania o dokumenty
    error: str | None
    calls: list[dict[str, Any]]  # NASZE: LlmCall.model_dump(mode='json') z bieżącej tury
