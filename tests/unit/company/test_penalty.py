"""Dokładna arytmetyka kary (port testów kalkulatora z kroku 2 MP4; kontrakt F4, 5.4)."""

from __future__ import annotations

import pytest


@pytest.fixture
def penalty(calc):
    def make(otif, invoice=18240000, target=97, rate=2, cap=10):
        return calc.calculate_otif_penalty(
            otif_pct=otif,
            target_pct=target,
            rate_per_full_point_pct=rate,
            cap_pct=cap,
            invoice_grosze=invoice,
        )

    return make


def test_july_with_every_intermediate_value(penalty):
    result = penalty(94.6)
    assert result.status == "ok"
    assert result.shortfall_points == 2.4, (
        "97 - 94.6 to dokładnie 2.4 punktu (na floatach: 2.4000000000000057): użyj Decimal(str(x))"
    )
    assert result.full_points == 2
    assert (result.penalty_pct_uncapped, result.penalty_pct) == (4, 4)
    assert result.penalty_grosze == 729600, "4% z 18 240 000 groszy"
    assert result.currency == "PLN"
    assert (
        result.otif_pct,
        result.target_pct,
        result.rate_per_full_point_pct,
        result.cap_pct,
        result.invoice_grosze,
    ) == (94.6, 97, 2, 10, 18240000), "powtórz dane wejściowe"


@pytest.mark.parametrize(
    ("otif", "full_points", "grosze"),
    [
        (95.2, 1, 364800),  # brakuje 1,8: 1 pełny punkt, 2% z 18 240 000 groszy
        (96.4, 0, 0),  # maj: brakuje 0,6
        (96.01, 0, 0),  # brakuje 0,99: to nie pełny punkt
        (96.0, 1, 364800),  # dokładnie 1 punkt
        (97.0, 0, 0),  # równo z celem
        (98.5, 0, 0),  # powyżej celu: brak nigdy nie jest ujemny
    ],
)
def test_full_points_are_whole_points(penalty, otif, full_points, grosze):
    result = penalty(otif)
    assert result.full_points == full_points, f"OTIF {otif}% vs 97%"
    assert result.penalty_grosze == grosze
    assert result.shortfall_points >= 0


def test_a_full_point_floats_would_lose(penalty):
    # na floatach 64.1 - 63.1 = 0.9999999999999929, a część całkowita to 0: Decimal daje dokładnie 1.0  # noqa: E501
    result = penalty(63.1, invoice=10000, target=64.1)
    assert (result.shortfall_points, result.full_points, result.penalty_grosze) == (1.0, 1, 200)


def test_exact_values_without_float_noise(penalty):
    # na floatach 96 - 95.1 = 0.9000000000000057: rekord musi podać 0.9
    result = penalty(95.1, target=96)
    assert result.shortfall_points == 0.9 and result.full_points == 0


def test_the_cap(penalty):
    result = penalty(85.0)  # 12 pełnych punktów -> 24%, limit 10%
    assert (result.full_points, result.penalty_pct_uncapped, result.penalty_pct) == (12, 24, 10)
    assert result.penalty_grosze == 1824000


@pytest.mark.parametrize(("invoice", "grosze"), [(125, 3), (124, 2), (175, 4), (25, 1)])
def test_rounding_half_up_to_whole_grosze(penalty, invoice, grosze):
    # 1 pełny punkt x 2%: 125 groszy -> 2.5 -> 3 (round(2.5) w Pythonie daje 2: połówki do parzystej);  # noqa: E501
    # 124 -> 2.48 -> 2; 175 -> 3.5 -> 4; 25 -> 0.5 -> 1
    assert penalty(96.0, invoice=invoice).penalty_grosze == grosze


@pytest.mark.parametrize(
    "change",
    [
        {"otif_pct": float("nan")},
        {"otif_pct": 101},
        {"target_pct": -1},
        {"rate_per_full_point_pct": -2},
        {"cap_pct": 120},
        {"invoice_grosze": 100.5},
        {"invoice_grosze": -1},
        {"otif_pct": None},
        {"otif_pct": True},
        {"otif_pct": float("inf")},
        {"otif_pct": "94,6"},
    ],
)
def test_invalid_inputs(calc, change):
    args = dict(
        otif_pct=94.6, target_pct=97, rate_per_full_point_pct=2, cap_pct=10, invoice_grosze=18240000
    )
    args.update(change)
    result = calc.calculate_otif_penalty(**args)
    assert result.status == "invalid_request", f"{change} musi zostać odrzucone"
    assert result.penalty_grosze is None and result.message


@pytest.mark.parametrize(
    ("name", "value"),
    [("otif_pct", 101), ("target_pct", -1), ("rate_per_full_point_pct", -2), ("cap_pct", 120)],
)
def test_a_message_does_not_depend_on_int_or_float(calc, name, value):
    args = dict(
        otif_pct=94.6, target_pct=97, rate_per_full_point_pct=2, cap_pct=10, invoice_grosze=18240000
    )
    as_int = calc.calculate_otif_penalty(**{**args, name: value})
    as_float = calc.calculate_otif_penalty(**{**args, name: float(value)})
    assert as_int.message == as_float.message, (
        f"{name}={value} i {float(value)} muszą dać ten sam komunikat, a dały "
        f"{as_int.message!r} i {as_float.message!r}: klient MCP może wysłać każdą z tych form"
    )


def test_message_names_the_field_and_the_value(calc):
    """Kontrakt F4, 5.4: „otif_pct musi mieścić się w zakresie 0-100, a jest 101”."""
    result = calc.calculate_otif_penalty(
        otif_pct=101.0, target_pct=97, rate_per_full_point_pct=2, cap_pct=10, invoice_grosze=1
    )
    assert "otif_pct" in result.message and "101" in result.message
    assert "101.0" not in result.message, "101 i 101.0 dają ten sam komunikat"


def test_integral_float_invoice_accepted(calc):
    """Schemat narzędzia i MCP mogą przysłać 18240000.0: to całkowita liczba groszy."""
    result = calc.calculate_otif_penalty(
        otif_pct=94.6,
        target_pct=97,
        rate_per_full_point_pct=2,
        cap_pct=10,
        invoice_grosze=18240000.0,
    )
    assert result.status == "ok" and result.penalty_grosze == 729600
    assert isinstance(result.invoice_grosze, int), "wynik powtarza dane wejściowe jako int"
