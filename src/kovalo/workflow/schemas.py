"""Schematy workflow (FR-3.2, FR-3.8; Turn 1:1 z reference/course/helpers/workflow.py).

Kontrakt: docs/contracts/f3.md, sekcja 6.1. Docstring i opisy pól `Turn` są częścią schematu
wysyłanego do modelu (structured output), więc traktujemy je jak prompt kursu: zmiana = ADR.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from kovalo.rag.records import TokenUsage
from kovalo.text_pl import PauseType

__all__ = [
    "TERMINAL_APPROVAL_STATUSES",
    "LlmCall",
    "PauseType",
    "Status",
    "Turn",
    "TurnKind",
]

TurnKind = Literal["document_question", "policy_question", "approval_request", "other", "error"]
Status = Literal[
    "answered",
    "pending_clarification",
    "pending_approval",
    "approved",
    "rejected",
    "cancelled",
    "error",
]
TERMINAL_APPROVAL_STATUSES: frozenset[str] = frozenset({"approved", "rejected", "cancelled"})


class Turn(BaseModel):
    """Jak obsłużyć najnowszą wiadomość menedżera."""

    kind: Literal["document_question", "policy_question", "approval_request", "other", "error"] = (
        Field(description="zobacz instrukcje; 'error' ustawia program, nigdy go nie wybieraj")
    )
    request_action: Literal["new", "update", "none"] = Field(
        description="new: nowy wniosek o rabat; update: zmienia bieżący wniosek; none: ani jedno, ani drugie"  # noqa: E501
    )
    standalone: str = Field(
        description="wiadomość przepisana tak, żeby była zrozumiała bez wcześniejszej rozmowy"
    )
    discount_pct: float | None = Field(
        default=None,
        description="rabat w procentach, jeśli został podany w wiadomości lub przejęty z wcześniejszej rozmowy",  # noqa: E501
    )
    margin_after_pct: float | None = Field(
        default=None,
        description="marża brutto PO rabacie, w procentach, jeśli została podana w wiadomości lub przejęta z wcześniejszej rozmowy",  # noqa: E501
    )
    clearance: bool | None = Field(
        default=None, description="wyprzedaż końcówek? null, jeśli nie wspomniano"
    )
    strategic_account: bool | None = Field(
        default=None, description="klient strategiczny? null, jeśli nie wspomniano"
    )
    competitor_offer: bool | None = Field(
        default=None, description="udokumentowana oferta konkurencji? null, jeśli nie wspomniano"
    )
    customer: str | None = Field(default=None, description="klient, jeśli został wymieniony")
    product: str | None = Field(default=None, description="produkt, jeśli został wymieniony")


class LlmCall(BaseModel):
    """NASZE: jedno wywołanie modelu w turze (FR-6.5)."""

    purpose: Literal["understand", "answer"]
    model: str | None = None  # response_metadata['model_name'] (fallback widoczny)
    usage: TokenUsage | None = None
    seconds: float | None = None
    error: str | None = None
