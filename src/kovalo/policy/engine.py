"""Silnik polityki rabatowej course-v1 (FR-3.1; krok 1 MP3 kursu).

Kontrakt: docs/contracts/f3.md, sekcja 5.3. Reguły w kodzie, nie w modelu: progi z
`settings.policy` (PINNED), porównania na Decimal(str(x)), więc granice 5 / 5,01 i 9 / 8,99
nie zależą od reprezentacji binarnej float.
"""

from __future__ import annotations

import math
from decimal import Decimal

from kovalo.config import PolicyConfig
from kovalo.policy.models import EXCEPTION_NAMES, Decision, DiscountRequest
from kovalo.text_pl import number

__all__ = ["required_approvers"]


_LABELS = {"discount_pct": "Rabat", "margin_after_pct": "Marża po rabacie"}


def _decimal(value: float, field: str, low: int, high: int) -> Decimal:
    if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(
            f"{_LABELS[field]} ({field}) musi być liczbą z zakresu {low}–{high}%, "  # noqa: RUF001
            f"a jest {value}."
        )
    return Decimal(str(value))


def _roles(roles: list[str]) -> str:
    return ", ".join(roles)


def required_approvers(request: DiscountRequest, policy: PolicyConfig) -> Decision:
    """Kursowa interpretacja course-v1 (krok 1 MP3), w tej kolejności:

    0. Walidacja: discount_pct None, nieskończony/NaN albo poza [0, 100] → ValueError
       z 'discount_pct'; margin_after_pct (jeśli nie None) nieskończony/NaN albo poza
       [-100, 100] → ValueError z 'margin_after_pct'.
    1. Przedział: pierwszy z d <= up_to (5 → do 5%; 5,01 → następny).
    2. Wyjątki (kolejność z configu): flaga True ORAZ d <= cap.
    3. Najniższa ranga z przedziału i wyjątków; wyjątek nigdy nie podnosi poziomu.
    4. Minimalna marża: marża < margin_floor i bez wyprzedaży końcówek → floor_role na końcu
       (o ile go nie ma). Wyprzedaż końcówek jest zwolniona także powyżej swojego limitu.
    5. Jeden wiersz uzasadnienia na każdą regułę; 6. version = policy.version. Funkcja czysta.
    """
    if request.discount_pct is None:
        raise ValueError("Brak wysokości rabatu (discount_pct). Podaj rabat w procentach.")
    discount = _decimal(request.discount_pct, "discount_pct", 0, 100)
    margin = (
        None
        if request.margin_after_pct is None
        else _decimal(request.margin_after_pct, "margin_after_pct", -100, 100)
    )

    bands = policy.bands
    base = next(i for i, band in enumerate(bands) if discount <= band.up_to)
    reasons = [
        f"rabat {number(discount)}% mieści się w przedziale do {number(bands[base].up_to)}%; "
        f"zatwierdza: {_roles(bands[base].roles)} (tabela POL-001)"
    ]

    rank = base
    for name, exception in policy.exceptions.items():
        if not getattr(request, name):
            continue
        label = EXCEPTION_NAMES.get(name, name)
        if discount > exception.cap:
            reasons.append(
                f"wyjątek: {label} nie ma zastosowania: rabat {number(discount)}% "
                f"przekracza limit {number(exception.cap)}%"
            )
            continue
        exception_rank = policy.band_rank(exception.roles)
        if exception_rank < rank:
            rank = exception_rank
            reasons.append(
                f"wyjątek: {label} (do {number(exception.cap)}%); "
                f"zatwierdza: {_roles(exception.roles)} (POL-001)"
            )
    approvers = list(bands[rank].roles)

    margin_checked = margin is not None
    if margin is None:
        reasons.append("marża po rabacie nieznana: minimalnej marży nie sprawdzono")
    elif margin < policy.margin_floor:
        if request.clearance:
            reasons.append("wyprzedaż końcówek jest zwolniona z minimalnej marży (POL-001)")
        else:
            if policy.floor_role not in approvers:
                approvers.append(policy.floor_role)
            reasons.append(
                f"marża {number(margin)}% jest niższa od minimalnej "
                f"({number(policy.margin_floor)}%); dochodzi: {policy.floor_role} "
                "(minimalna marża, POL-001)"
            )

    return Decision(
        approvers=approvers,
        reasons=reasons,
        margin_checked=margin_checked,
        version=policy.version,
    )
