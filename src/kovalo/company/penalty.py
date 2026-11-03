"""Kalkulator kary OTIF: dokładna arytmetyka dziesiętna (FR-4.2; krok 2 MP4).

Kontrakt: docs/contracts/f4.md, sekcja 5.4. Każda wartość jako Decimal(str(x)), zaokrąglenie
do pełnych groszy (HALF_UP) raz, na końcu. Kalkulator tylko liczy: to, czy kara się należy,
wynika z warunków umowy (get_contract_terms).
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

from kovalo.company.records import PenaltyCalculation
from kovalo.quantities import round_grosze

__all__ = ["calculate_otif_penalty"]


class _Invalid(ValueError):
    pass


def _text(value: Any) -> str:
    """101 i 101.0 → „101” (komunikat nie zależy od int/float)."""
    try:
        return f"{Decimal(str(value)).normalize():f}"
    except ArithmeticError:
        return str(value)


def _number(name: str, value: Any, low: float | None, high: float | None) -> Decimal:
    if value is None:
        raise _Invalid(f"brak wartości {name}")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _Invalid(f"{name} musi być liczbą, a jest {value!r}")
    if not math.isfinite(value):
        raise _Invalid(f"{name} musi być liczbą skończoną, a jest {value}")
    if (low is not None and value < low) or (high is not None and value > high):
        if high is None:
            raise _Invalid(f"{name} nie może być ujemne, a jest {_text(value)}")
        raise _Invalid(f"{name} musi mieścić się w zakresie 0–100, a jest {_text(value)}")  # noqa: RUF001
    return Decimal(str(value))


def _invoice(value: Any) -> int:
    number = _number("invoice_grosze", value, None, None)
    if number != number.to_integral_value() or number < 0:
        raise _Invalid(
            f"invoice_grosze musi być całkowitą liczbą groszy >= 0, a jest {_text(value)}"
        )
    return int(number)


def calculate_otif_penalty(
    otif_pct: float,
    target_pct: float,
    rate_per_full_point_pct: float,
    cap_pct: float,
    invoice_grosze: int,
) -> PenaltyCalculation:
    """shortfall = max(0, cel - OTIF); full_points = część całkowita; uncapped = full_points x
    stawka; pct = min(uncapped, limit); penalty_grosze = round_grosze(faktura x pct / 100)."""
    try:
        otif = _number("otif_pct", otif_pct, 0, 100)
        target = _number("target_pct", target_pct, 0, 100)
        rate = _number("rate_per_full_point_pct", rate_per_full_point_pct, 0, None)
        cap = _number("cap_pct", cap_pct, 0, 100)
        invoice = _invoice(invoice_grosze)
    except _Invalid as exc:
        return PenaltyCalculation(status="invalid_request", message=str(exc))
    shortfall = max(Decimal(0), target - otif)
    full_points = int(shortfall)
    uncapped = full_points * rate
    pct = min(uncapped, cap)
    return PenaltyCalculation(
        status="ok",
        otif_pct=float(otif),
        target_pct=float(target),
        rate_per_full_point_pct=float(rate),
        cap_pct=float(cap),
        invoice_grosze=invoice,
        shortfall_points=float(shortfall),
        full_points=full_points,
        penalty_pct_uncapped=float(uncapped),
        penalty_pct=float(pct),
        penalty_grosze=round_grosze(Decimal(invoice) * pct / 100),
    )
