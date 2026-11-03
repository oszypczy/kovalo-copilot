"""Zrozumienie wiadomości i aktualizacja wniosku (FR-3.2, FR-3.5; krok 2 MP3 kursu).

Kontrakt: docs/contracts/f3.md, sekcja 6.2. Model przychodzi przez parametr jako Runnable już
związany ze schematem Turn (produkcja: llm.get_structured_model(Turn, 'understand'), testy:
atrapy). Błąd modelu nigdy nie jest wyjątkiem: kończy się Turn(kind='error').
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from kovalo.policy.models import DiscountRequest
from kovalo.rag.answer import error_text, unpack_structured
from kovalo.workflow.schemas import LlmCall, Turn

__all__ = [
    "AMBIGUOUS_ACTION",
    "FIELDS",
    "FLAGS",
    "Understanding",
    "understand_turn",
    "update_request",
]

FIELDS = (
    "discount_pct",
    "margin_after_pct",
    "clearance",
    "strategic_account",
    "competitor_offer",
    "customer",
    "product",
)
FLAGS = ("clearance", "strategic_account", "competitor_offer")

AMBIGUOUS_ACTION = (
    "Nie wiem, czy to nowy wniosek, czy zmiana obecnego. "
    "Napisz „nowy wniosek” albo „zmień obecny wniosek”."
)


@dataclass(frozen=True)
class Understanding:
    turn: Turn
    call: LlmCall  # purpose='understand'; error ustawiony przy Turn błędu z powodu modelu


def _error_turn(message: str) -> Turn:
    return Turn(kind="error", request_action="none", standalone=message)


def understand_turn(
    history: Sequence[AnyMessage | BaseMessage],
    message: str,
    model: Runnable[LanguageModelInput, Any],
    *,
    system_prompt: str,
    clock: Callable[[], float] = time.perf_counter,
) -> Understanding:
    """[SystemMessage(system_prompt), *history, HumanMessage(message)] → model → Turn.

    Każdy Exception modelu (także AssertionError atrapy), błąd parsowania albo brak obiektu Turn
    → Turn(kind='error', request_action='none', standalone=message) z call.error. Tokeny
    i model (include_raw) trafiają do call także przy błędzie parsowania.
    """
    messages: list[BaseMessage] = [
        SystemMessage(system_prompt),
        *history,
        HumanMessage(message),
    ]
    start = clock()
    try:
        result = model.invoke(messages)
    except Exception as exc:  # błąd dostawcy, LlmCacheMiss, wyczerpana atrapa
        call = LlmCall(purpose="understand", seconds=clock() - start, error=error_text(exc))
        return Understanding(turn=_error_turn(message), call=call)
    seconds = clock() - start
    outcome = unpack_structured(result, Turn)
    call = LlmCall(
        purpose="understand",
        model=outcome.answered_by,
        usage=outcome.usage,
        seconds=seconds,
        error=outcome.error,
    )
    if outcome.error is not None:
        return Understanding(turn=_error_turn(message), call=call)
    return Understanding(turn=outcome.parsed, call=call)


def update_request(current: DiscountRequest | None, turn: Turn) -> DiscountRequest | None:
    """Kanoniczny wniosek po wiadomości (krok 2 kursu).

    - kind != 'approval_request' → current bez zmian (ten sam obiekt, także None);
    - request_action 'none' → ValueError z pytaniem (graf zamienia je w odpowiedź);
    - 'new' (także 'update' bez current) → nowy wniosek, revision 1, flagi None → False;
    - 'update' → kopia current z polami Turn różnymi od None; revision + 1 tylko przy zmianie.
    """
    if turn.kind != "approval_request":
        return current
    if turn.request_action == "none":
        raise ValueError(AMBIGUOUS_ACTION)
    given = {name: getattr(turn, name) for name in FIELDS if getattr(turn, name) is not None}
    if turn.request_action == "new" or current is None:
        fields: dict[str, Any] = {name: False for name in FLAGS}
        fields.update(given)
        return DiscountRequest(revision=1, **fields)
    updated = current.model_copy(update=given)
    if updated == current:
        return current
    return updated.model_copy(update={"revision": current.revision + 1})
