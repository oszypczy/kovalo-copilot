"""Wniosek o rabat i decyzja polityki (FR-3.1; 1:1 z reference/course/helpers/workflow.py).

Kontrakt: docs/contracts/f3.md, sekcja 5.2. Czysty Pydantic: bez LangChaina i LangGrapha.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["EXCEPTION_NAMES", "Decision", "DiscountRequest"]

EXCEPTION_NAMES: dict[str, str] = {
    "clearance": "wyprzedaż końcówek",
    "strategic_account": "klient strategiczny",
    "competitor_offer": "oferta konkurencji",
}


class DiscountRequest(BaseModel):
    """Wniosek o rabat, którego dotyczy rozmowa. Flagi mają tu zawsze wartość true/false."""

    revision: int = 1
    discount_pct: float | None = None
    margin_after_pct: float | None = None
    clearance: bool = False
    strategic_account: bool = False
    competitor_offer: bool = False
    customer: str | None = None
    product: str | None = None


class Decision(BaseModel):
    """Czego kursowa polityka wymaga dla jednego wniosku."""

    approvers: list[str]  # role w kolejności pytania o zgodę, np. ["CEO", "CFO"]
    reasons: list[str]  # jeden wiersz na każdą regułę, która wpłynęła na wynik
    margin_checked: bool  # False, gdy marża po rabacie była nieznana
    version: str  # interpretacja polityki, np. "course-v1"
