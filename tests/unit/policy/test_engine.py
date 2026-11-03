"""Silnik polityki rabatowej course-v1 (kontrakt F3, 5.3; krok 1 MP3 kursu).

Tabela przypadków: 22 przypadki kroku 1 kursu (`CASES`) + 7 z docs/05-domain.md.
Druga strona porównania to role policzone RĘCZNIE (kurs, domena), nie kod.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest


def R(**fields):
    from kovalo.policy.models import DiscountRequest

    return DiscountRequest(**fields)


RKS, DH, PH = "Regionalny kierownik sprzedaży", "Dyrektor handlowy", "Przedstawiciel handlowy"

# Sprawdzone ręcznie według kursowej interpretacji course-v1 (kurs, krok 1: CASES, 22 przypadki).
CASES = [
    (dict(discount_pct=5, margin_after_pct=10), [PH]),
    (dict(discount_pct=5.01, margin_after_pct=10), [RKS]),
    (dict(discount_pct=12, margin_after_pct=10), [RKS]),
    (dict(discount_pct=12.01, margin_after_pct=10), [DH]),
    (dict(discount_pct=20, margin_after_pct=10), [DH]),
    (dict(discount_pct=20.01, margin_after_pct=10), ["CEO", "CFO"]),
    (dict(discount_pct=8, margin_after_pct=9), [RKS]),
    (dict(discount_pct=8, margin_after_pct=8.99), [RKS, "CFO"]),
    (dict(discount_pct=3, margin_after_pct=7), [PH, "CFO"]),
    (dict(discount_pct=15, competitor_offer=True, margin_after_pct=10), [RKS]),
    (dict(discount_pct=15.01, competitor_offer=True, margin_after_pct=10), [DH]),
    (dict(discount_pct=25, strategic_account=True, margin_after_pct=10), [DH]),
    (dict(discount_pct=25.01, strategic_account=True, margin_after_pct=10), ["CEO", "CFO"]),
    (dict(discount_pct=35, clearance=True, margin_after_pct=4), [RKS]),
    (dict(discount_pct=35.01, clearance=True, margin_after_pct=10), ["CEO", "CFO"]),
    (dict(discount_pct=3, strategic_account=True, margin_after_pct=10), [PH]),
    (dict(discount_pct=22, margin_after_pct=10), ["CEO", "CFO"]),
    (dict(discount_pct=22, strategic_account=True, margin_after_pct=7), [DH, "CFO"]),
    (
        dict(discount_pct=14, strategic_account=True, competitor_offer=True, margin_after_pct=7),
        [RKS, "CFO"],
    ),
    (dict(discount_pct=30, clearance=True, margin_after_pct=4), [RKS]),
    (dict(discount_pct=36, clearance=True, margin_after_pct=4), ["CEO", "CFO"]),
    (dict(discount_pct=25, margin_after_pct=5), ["CEO", "CFO"]),
]

# Tabela z docs/05-domain.md („Przypadki testowe”): 7 wierszy.
DOMAIN_TABLE = [
    (
        "3% strategiczny, marża 10%",
        dict(discount_pct=3, strategic_account=True, margin_after_pct=10),
        [PH],
    ),
    ("15% zwykły, marża 10%", dict(discount_pct=15, margin_after_pct=10), [DH]),
    ("22% zwykły, marża 10%", dict(discount_pct=22, margin_after_pct=10), ["CEO", "CFO"]),
    (
        "22% strategiczny, marża 7%",
        dict(discount_pct=22, strategic_account=True, margin_after_pct=7),
        [DH, "CFO"],
    ),
    (
        "14% strategiczny + konkurencja, marża 7%",
        dict(discount_pct=14, strategic_account=True, competitor_offer=True, margin_after_pct=7),
        [RKS, "CFO"],
    ),
    ("30% wyprzedaż, marża 4%", dict(discount_pct=30, clearance=True, margin_after_pct=4), [RKS]),
    (
        "36% wyprzedaż, marża 4%",
        dict(discount_pct=36, clearance=True, margin_after_pct=4),
        ["CEO", "CFO"],
    ),
]


def test_course_cases_count():
    assert len(CASES) == 22 and len(DOMAIN_TABLE) == 7


@pytest.mark.parametrize(
    "fields, expected", CASES, ids=[f"course-{i:02d}" for i in range(len(CASES))]
)
def test_hand_checked_cases(fields, expected, settings):
    from kovalo.policy import required_approvers

    decision = required_approvers(R(**fields), settings.policy)
    assert decision.approvers == expected, (
        f"{fields} -> oczekiwano {expected}, jest {decision.approvers}"
    )


@pytest.mark.parametrize(
    "label, fields, expected", DOMAIN_TABLE, ids=[row[0] for row in DOMAIN_TABLE]
)
def test_domain_table(label, fields, expected, settings):
    from kovalo.policy import required_approvers

    assert required_approvers(R(**fields), settings.policy).approvers == expected


def test_decision_carries_reasons_and_version(settings):
    from kovalo.policy import required_approvers

    decision = required_approvers(
        R(discount_pct=22, strategic_account=True, margin_after_pct=7), settings.policy
    )
    assert decision.version == settings.policy.version == "course-v1"
    assert len(decision.reasons) >= 3, "przedział, wyjątek i minimalna marża"
    assert decision.margin_checked is True


def test_missing_margin_means_floor_not_checked(settings):
    from kovalo.policy import required_approvers

    decision = required_approvers(R(discount_pct=8), settings.policy)
    assert decision.approvers == [RKS] and decision.margin_checked is False
    assert "marża po rabacie nieznana: minimalnej marży nie sprawdzono" in decision.reasons


@pytest.mark.parametrize(
    ("fields", "key"),
    [
        (dict(discount_pct=None), "discount_pct"),
        (dict(discount_pct=math.nan), "discount_pct"),
        (dict(discount_pct=math.inf), "discount_pct"),
        (dict(discount_pct=101), "discount_pct"),
        (dict(discount_pct=-1), "discount_pct"),
        (dict(discount_pct=10, margin_after_pct=150), "margin_after_pct"),
        (dict(discount_pct=10, margin_after_pct=-100.5), "margin_after_pct"),
        (dict(discount_pct=10, margin_after_pct=math.inf), "margin_after_pct"),
        (dict(discount_pct=10, margin_after_pct=math.nan), "margin_after_pct"),
    ],
    ids=lambda v: str(v) if not isinstance(v, dict) else ",".join(f"{k}={x}" for k, x in v.items()),
)
def test_invalid_input_raises(fields, key, settings):
    from kovalo.policy import required_approvers

    with pytest.raises(ValueError, match=key):
        required_approvers(R(**fields), settings.policy)


def test_extreme_valid_values_accepted(settings):
    from kovalo.policy import required_approvers

    assert required_approvers(
        R(discount_pct=0, margin_after_pct=-100), settings.policy
    ).approvers == [PH, "CFO"]
    assert required_approvers(
        R(discount_pct=100, margin_after_pct=100), settings.policy
    ).approvers == ["CEO", "CFO"]


def test_numbers_come_from_config(settings):
    from kovalo.policy import required_approvers

    stricter = settings.policy.model_copy(update={"margin_floor": Decimal(10)})
    assert required_approvers(R(discount_pct=8, margin_after_pct=9.5), stricter).approvers == [
        RKS,
        "CFO",
    ]
    assert required_approvers(
        R(discount_pct=8, margin_after_pct=9.5), settings.policy
    ).approvers == [RKS]


def test_reasons_wording(settings):
    from kovalo.policy import required_approvers

    low = required_approvers(R(discount_pct=7, margin_after_pct=8), settings.policy)
    assert (
        "rabat 7% mieści się w przedziale do 12%; "
        "zatwierdza: Regionalny kierownik sprzedaży (tabela POL-001)" in low.reasons
    )
    assert (
        "marża 8% jest niższa od minimalnej (9%); dochodzi: CFO (minimalna marża, POL-001)"
        in low.reasons
    )

    strategic = required_approvers(
        R(discount_pct=22, strategic_account=True, margin_after_pct=10), settings.policy
    )
    assert (
        "wyjątek: klient strategiczny (do 25%); zatwierdza: Dyrektor handlowy (POL-001)"
        in strategic.reasons
    )

    over = required_approvers(
        R(discount_pct=36, clearance=True, margin_after_pct=4), settings.policy
    )
    assert (
        "wyjątek: wyprzedaż końcówek nie ma zastosowania: rabat 36% przekracza limit 35%"
        in over.reasons
    )
    assert "wyprzedaż końcówek jest zwolniona z minimalnej marży (POL-001)" in over.reasons


def test_floor_reason_even_when_cfo_already_listed(settings):
    from kovalo.policy import required_approvers

    decision = required_approvers(R(discount_pct=22, margin_after_pct=7), settings.policy)
    assert decision.approvers == ["CEO", "CFO"]
    assert any("marża 7% jest niższa od minimalnej (9%)" in r for r in decision.reasons)


def test_function_is_pure(settings):
    from kovalo.policy import required_approvers

    request = R(discount_pct=22, strategic_account=True, margin_after_pct=7)
    before = request.model_copy()
    assert required_approvers(request, settings.policy) == required_approvers(
        request, settings.policy
    )
    assert request == before


@pytest.mark.parametrize(
    ("discount", "margin", "flags", "expected"),
    [
        ("5", None, {}, [PH]),
        ("5.01", None, {}, [RKS]),
        ("12", None, {}, [RKS]),
        ("12.01", None, {}, [DH]),
        ("20.01", None, {}, ["CEO", "CFO"]),
        ("8", "9", {}, [RKS]),
        ("8", "8.99", {}, [RKS, "CFO"]),
        ("35", "4", {"clearance": True}, [RKS]),
        ("35.01", "4", {"clearance": True}, ["CEO", "CFO"]),
        ("25", "10", {"strategic_account": True}, [DH]),
        ("25.01", "10", {"strategic_account": True}, ["CEO", "CFO"]),
        ("15", "10", {"competitor_offer": True}, [RKS]),
        ("15.01", "10", {"competitor_offer": True}, [DH]),
    ],
)
def test_decimal_boundaries(discount, margin, flags, expected, settings):
    """Granice liczone na Decimal(str(x)): wartość podana jako float nie psuje porównania."""
    from kovalo.policy import required_approvers

    request = R(
        discount_pct=float(discount),
        margin_after_pct=None if margin is None else float(margin),
        **flags,
    )
    assert required_approvers(request, settings.policy).approvers == expected


def test_margin_just_below_floor_is_below(settings):
    from kovalo.policy import required_approvers

    # Decimal(str(9 - 1e-12)) = 8.999999999999 < 9: reprezentacja nie zaokrągla do granicy
    assert required_approvers(
        R(discount_pct=12, margin_after_pct=9 - 1e-12), settings.policy
    ).approvers == [RKS, "CFO"]


def test_policy_package_reexports():
    import kovalo.policy as policy

    for name in ("DiscountRequest", "Decision", "required_approvers", "EXCEPTION_NAMES"):
        assert hasattr(policy, name), name
    assert policy.EXCEPTION_NAMES == {
        "clearance": "wyprzedaż końcówek",
        "strategic_account": "klient strategiczny",
        "competitor_offer": "oferta konkurencji",
    }


def test_models_match_course_defaults():
    from kovalo.policy.models import DiscountRequest

    request = DiscountRequest()
    assert request.revision == 1 and request.discount_pct is None
    assert (request.clearance, request.strategic_account, request.competitor_offer) == (False,) * 3
    assert request.customer is None and request.product is None


def test_last_band_wording(settings):
    """A0 (kontrakt F4, 6.1): ostatni przedział to „powyżej 20%”, nie „do 100%”; reszta bez zmian."""  # noqa: E501
    from kovalo.policy import required_approvers

    top = required_approvers(R(discount_pct=22, margin_after_pct=10), settings.policy)
    # P5 (kontrakt F4, 17): role łączone jak w silniku F3, czyli „CEO, CFO”
    assert top.reasons[0] == (
        "rabat 22% mieści się w przedziale powyżej 20%; zatwierdza: CEO, CFO (tabela POL-001)"
    )
    assert not any("do 100%" in reason for reason in top.reasons)
    low = required_approvers(R(discount_pct=7, margin_after_pct=10), settings.policy)
    assert "rabat 7% mieści się w przedziale do 12%" in low.reasons[0]
    edge = required_approvers(R(discount_pct=100, margin_after_pct=10), settings.policy)
    assert "przedziale powyżej 20%" in edge.reasons[0]
    just_over = required_approvers(R(discount_pct=20.01, margin_after_pct=10), settings.policy)
    assert "przedziale powyżej 20%" in just_over.reasons[0]
    at_limit = required_approvers(R(discount_pct=20, margin_after_pct=10), settings.policy)
    assert "przedziale do 20%" in at_limit.reasons[0]
