"""Driver tury i atrapy modeli E3 (kontrakt F3, 10.1; port helpers.workflow.run_turn kursu).

run_turn używa CopilotService (tej samej ścieżki co API), nie grafu wprost.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from kovalo.rag.records import Answer
from kovalo.workflow.schemas import Turn
from kovalo.workflow.service import CopilotService, ThreadPausedError, TurnResult

__all__ = ["ScriptedAnswerModel", "ScriptedUnderstanding", "run_turn"]


def run_turn(
    service: CopilotService,
    thread_id: str,
    text: str,
    answer_pause: Callable[[dict[str, Any]], Any],
    *,
    max_pauses: int = 10,
) -> dict[str, Any]:
    """Wysyła jedną wiadomość, odpowiada na każdą pauzę i zwraca wpis transkryptu (JSON).

    Klucze kursu: user, turn, request, approvers, reasons, pauses, approvals, status, record,
    error, reply; NASZE: calls, seconds. Wątek w pauzie → RuntimeError z „pauza” (jak kurs).
    Wyjątek w answer_pause albo więcej niż max_pauses pauz → pauza ANULOWANA (wątek nie wisi),
    status 'error', powód w 'error' („… przy pauzie <typ> <rola|pole>: …”).
    """
    start = time.perf_counter()
    try:
        result: TurnResult = service.send_message(thread_id, text)
    except ThreadPausedError as exc:
        raise RuntimeError(
            f"wątek {thread_id!r} czeka na odpowiedź (pauza): odpowiedz na pauzę, "
            "zanim wyślesz nową wiadomość"
        ) from exc
    pauses: list[list[Any]] = []
    failure: str | None = None
    while result.pending is not None:
        pending = result.pending
        payload = pending.model_dump()
        pauses.append([pending.type, pending.key])
        try:
            if len(pauses) > max_pauses:
                raise RuntimeError(
                    f"więcej niż {max_pauses} pauz w jednej turze: graf chyba się zapętlił"
                )
            answer = answer_pause(payload)
        except Exception as exc:  # brak odpowiedzi: anuluj pauzę, zachowaj to, co zaobserwowano
            failure = f"{type(exc).__name__} przy pauzie {pending.type} {pending.key}: {exc}"
            service.resume(thread_id, value="cancel")
            break
        result = service.resume(thread_id, value=answer)

    state = service.get_values(thread_id)
    turn, request, decision = state.get("turn"), state.get("request"), state.get("decision")
    replies = [m.content for m in state.get("messages") or [] if m.type == "ai"]
    return {
        "user": text,
        "turn": turn.model_dump() if turn else None,
        "request": request.model_dump() if request else None,
        "approvers": list(decision.approvers) if decision else None,
        "reasons": list(decision.reasons) if decision else None,
        "pauses": pauses,
        "approvals": dict(state.get("approvals") or {}),
        "status": "error" if failure else state.get("status"),
        "record": state.get("record"),
        "error": failure or state.get("error"),
        "reply": replies[-1] if replies else None,
        "calls": list(state.get("calls") or []),
        "seconds": time.perf_counter() - start,
    }


class ScriptedUnderstanding:
    """Idealne `understand`: Runnable zwracający kolejne Turn z listy (jak ReplayLLM kursu).

    Liczy wywołania; wyczerpana lista → AssertionError (wznowienie nie może ponownie wywołać
    understand). understand_turn zamienia ten wyjątek w turę 'error', a licznik zostaje.
    """

    def __init__(self, turns: Sequence[Turn]) -> None:
        self.turns = list(turns)
        self.calls = 0

    def _next(self, messages: Any) -> Turn:
        self.calls += 1
        if not self.turns:
            raise AssertionError(
                f"ScriptedUnderstanding: wywołanie nr {self.calls}, a skrypt jest wyczerpany"
            )
        return self.turns.pop(0)

    def as_runnable(self) -> Runnable[LanguageModelInput, Turn]:
        return RunnableLambda(self._next, name="scripted_understanding")


class ScriptedAnswerModel(Runnable[LanguageModelInput, Answer]):
    """Structured model Answer dla odtworzenia: stała odpowiedź (domyślnie answerable=True,
    citations ['MTG-402']), bez sieci. Liczy wywołania."""

    def __init__(self, answer: Answer | None = None) -> None:
        self.answer = answer or Answer(
            answer="Komitet utrzymał ceny wody.", citations=["MTG-402"], answerable=True
        )
        self.calls = 0

    def invoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Answer:
        self.calls += 1
        return self.answer
