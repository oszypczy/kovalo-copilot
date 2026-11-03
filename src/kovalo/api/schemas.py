"""Modele żądań i błędów API wątków (kontrakt F3, 8.2; ADR-0010).

Odpowiedzi 200 to modele serwisu (`TurnResult`, `ThreadState` z kovalo.workflow.service);
tu są tylko wejścia i wspólne ciało błędu.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, StrictFloat, StrictInt, model_validator

from kovalo.workflow.service import PauseRef, PendingPause

__all__ = ["ApiError", "MessageIn", "PauseRef", "ResumeIn", "ThreadCreated"]


class ThreadCreated(BaseModel):
    thread_id: str


class MessageIn(BaseModel):
    """Nowa wiadomość menedżera. Limit długości pilnuje serwis (workflow.max_message_chars)."""

    text: str = Field(min_length=1)


class ResumeIn(BaseModel):
    """Odpowiedź na pauzę: wartość z przycisku (`value`) albo tekst z czatu (`text`).

    `pause_ref` (opcjonalny) wskazuje pauzę, na którą odpowiadamy; inna bieżąca pauza → 409.
    """

    # Strict: JSON `true` albo "1.5" nie mogą stać się marżą (przegląd R1, R1-1)
    value: StrictFloat | StrictInt | Literal["approved", "rejected", "cancel"] | None = None
    text: str | None = None
    pause_ref: PauseRef | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> ResumeIn:
        if (self.value is None) == (self.text is None):
            raise ValueError("Podaj dokładnie jedno: value albo text.")
        return self


class ApiError(BaseModel):
    """Ciało każdej odpowiedzi z błędem (`detail` po polsku)."""

    error: str
    detail: str
    pending: PendingPause | None = None
