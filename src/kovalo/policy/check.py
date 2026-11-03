"""check_discount_policy: sprawdzenie polityki rabatowej tylko do odczytu (FR-4.3; krok 2 MP4).

Kontrakt: docs/contracts/f4.md, sekcja 6.1. Nad silnikiem z F3 (`required_approvers`); różnica
wobec kursu: pierwszy argument to `PolicyConfig`. Nigdy niczego nie zatwierdza.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from kovalo.config import PolicyConfig
from kovalo.policy.engine import required_approvers
from kovalo.policy.models import DiscountRequest

__all__ = ["PolicyCheck", "check_discount_policy"]

_OUT_OF_SCOPE = (
    "rabat wolumenowy z umowy z klientem nie podlega tej polityce: według POL-001 takie rabaty "
    "przechodzą przez proces warunków kredytowych i handlowych"
)


class PolicyCheck(BaseModel):
    """check_discount_policy: kto musi zatwierdzić rabat (tylko informacja)."""

    status: Literal["ok", "out_of_scope", "invalid_request"]
    approvers: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    version: str | None = None  # interpretacja polityki, np. course-v1
    margin_checked: bool = False
    partial: bool = False  # True, gdy marża była nieznana: minimalnej marży nie sprawdzono
    note: str = (
        "tylko informacja: to sprawdzenie niczego nie zatwierdza; wnioski o akceptację "
        "przechodzą przez workflow akceptacji (Miniprojekt 3)"
    )
    message: str | None = None


def check_discount_policy(
    policy: PolicyConfig,
    discount_pct: float,
    margin_after_pct: float | None = None,
    clearance: bool = False,
    strategic_account: bool = False,
    competitor_offer: bool = False,
    contractual_volume_rebate: bool = False,
) -> PolicyCheck:
    """Krok 2 kursu: rabat wolumenowy z umowy → out_of_scope; inaczej required_approvers
    (approvers, reasons, version, margin_checked; partial = not margin_checked);
    ValueError → invalid_request z jego komunikatem. Nigdy niczego nie zatwierdza."""
    if contractual_volume_rebate:
        return PolicyCheck(status="out_of_scope", message=_OUT_OF_SCOPE)
    try:
        request = DiscountRequest(
            discount_pct=discount_pct,
            margin_after_pct=margin_after_pct,
            clearance=bool(clearance),
            strategic_account=bool(strategic_account),
            competitor_offer=bool(competitor_offer),
        )
        decision = required_approvers(request, policy)
    except (ValueError, TypeError) as exc:
        return PolicyCheck(status="invalid_request", message=str(exc))
    return PolicyCheck(
        status="ok",
        approvers=decision.approvers,
        reasons=decision.reasons,
        version=decision.version,
        margin_checked=decision.margin_checked,
        partial=not decision.margin_checked,
    )
