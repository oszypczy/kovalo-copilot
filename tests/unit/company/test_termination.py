"""check_termination_condition: warunek rozwiązania umowy z powodu OTIF policzony w kodzie
(NASZE, FR-4.7; kontrakt F4, 5.5)."""

from __future__ import annotations

from decimal import Decimal

import pytest


@pytest.fixture
def check():
    from kovalo.company.termination import check_termination_condition

    return check_termination_condition


@pytest.fixture
def terms(settings):
    return settings.company


def test_t1_july_breaks_the_series(check, company_data, terms):
    """Kryterium 1 F4: lipiec 94,6% nie jest poniżej progu 94%, więc warunek nie jest spełniony."""
    result = check(company_data, terms, "Mleczna Dolina", "2026-09")
    assert result.status == "ok"
    assert (result.supplier_id, result.supplier) == ("SUP-114", "Mleczna Dolina")
    assert [(m.month, m.otif_pct, m.below_threshold) for m in result.months] == [
        ("2026-07", 94.6, False),
        ("2026-08", 93.8, True),
        ("2026-09", 93.5, True),
    ]
    assert result.months_below == 2 and result.condition_met is False
    assert (result.source_doc_id, result.otif_below_pct) == ("CON-101", 94.0)
    assert (result.consecutive_months, result.notice_days) == (3, 60)
    assert result.as_of_month == "2026-09" and result.missing_months == []
    assert result.terms_version == "termination-v1"
    assert "nie rozwiązuje umowy" in result.note


def test_missing_last_month_is_unknown(check, company_data, terms):
    result = check(company_data, terms, "SUP-114", "2026-10")
    assert result.status == "ok"
    assert [(m.month, m.below_threshold) for m in result.months] == [
        ("2026-08", True),
        ("2026-09", True),
        ("2026-10", None),
    ]
    assert result.months[2].otif_pct is None
    assert result.missing_months == ["2026-10"]
    assert result.condition_met is None, "brak danych bez przerwania serii: nie wiadomo"
    assert result.months_below == 2 and result.message


def test_no_data(check, company_data, terms):
    result = check(company_data, terms, "SUP-114", "2027-03")
    assert result.status == "no_data"
    assert result.condition_met is None and result.months_below == 0
    assert result.missing_months == ["2027-01", "2027-02", "2027-03"]


def test_exactly_threshold_is_not_below(check, small_data, settings):
    """Dokładnie 94,0 NIE spełnia warunku „mniej niż 94%” (small_data: sierpień 94,0)."""
    terms = settings.company
    result = check(small_data, terms, "SUP-114", "2026-09")
    august = next(m for m in result.months if m.month == "2026-08")
    assert august.otif_pct == 94.0 and august.below_threshold is False
    assert result.condition_met is False, "seria przerwana także przy brakującym lipcu"
    assert [m.month for m in result.months if m.otif_pct is None] == ["2026-07"]
    assert result.missing_months == ["2026-07"]
    assert terms.termination_terms["SUP-114"].otif_below_pct == Decimal(94)


def test_window_across_new_year(check, company_data, terms):
    result = check(company_data, terms, "SUP-114", "2027-01")
    assert [m.month for m in result.months] == ["2026-11", "2026-12", "2027-01"]
    assert result.status == "no_data"


def test_aqua_and_transbalt_not_applicable(check, company_data, terms):
    aqua = check(company_data, terms, "Aqua Kryształ", "2026-09")
    assert aqua.status == "not_applicable" and aqua.supplier_id == "SUP-207"
    assert aqua.message and "Aqua" in aqua.message and aqua.condition_met is None
    trans = check(company_data, terms, "TransBałt", "2026-09")
    assert trans.status == "not_applicable" and trans.source_doc_id == "CON-103"
    assert "CON-103" in trans.message and "search_documents" in trans.message
    assert trans.condition_met is None


def test_unknown_supplier_and_bad_month(check, company_data, terms):
    unknown = check(company_data, terms, "Nieznany", "2026-09")
    assert unknown.status == "unknown_supplier" and unknown.supplier_id is None
    bad = check(company_data, terms, "Mleczna Dolina", "2026-9")
    assert bad.status == "invalid_request" and bad.as_of_month == "2026-9"
    assert "YYYY-MM" in bad.message and bad.months == []


def test_carries_the_otif_record(check, company_data, terms):
    """Rekord odczytu (zdarzenie składowe w trace, §7.1): ten sam kod co narzędzie get_supplier_otif."""  # noqa: E501
    from kovalo.company.lookup import get_supplier_otif

    result = check(company_data, terms, "Mleczna Dolina", "2026-09")
    assert result.otif is not None and len(result.otif.observations) == 3
    assert result.otif == get_supplier_otif(company_data, "SUP-114", "2026-07", "2026-09")
    assert result.otif.requested == {"from": "2026-07", "to": "2026-09"}
    dumped = result.model_dump(mode="json")
    assert dumped["otif"]["supplier_id"] == "SUP-114", "pole `otif` jest w JSON wyniku narzędzia"


def _otif_data(small_data, values):
    """small_data z OTIF SUP-114 zastąpionym podanymi miesiącami (None = brak wiersza)."""
    import dataclasses

    otif = {k: v for k, v in small_data.otif.items() if k[0] != "SUP-114"}
    otif.update({("SUP-114", m): v for m, v in values.items() if v is not None})
    return dataclasses.replace(small_data, otif=otif)


@pytest.mark.parametrize(
    ("values", "met", "below"),
    [
        # granica progu 94%: poniżej tylko wartość mniejsza (float 93.99 i 94.01 w Decimal(str(x)))
        ({"2026-07": 93.99, "2026-08": 93.99, "2026-09": 93.99}, True, 3),
        ({"2026-07": 93.0, "2026-08": 93.0, "2026-09": 94.0}, False, 2),
        ({"2026-07": 93.0, "2026-08": 93.0, "2026-09": 94.01}, False, 2),
        ({"2026-07": 94.0, "2026-08": 93.0, "2026-09": 93.0}, False, 2),
        # trzy kolejne miesiące poniżej wystarczą także po dobrym miesiącu wcześniej
        ({"2026-06": 99.0, "2026-07": 93.5, "2026-08": 93.5, "2026-09": 93.5}, True, 3),
        # luka w środku bez miesiąca „w normie”: nie wiadomo; z takim miesiącem: False
        ({"2026-07": 93.0, "2026-08": None, "2026-09": 93.0}, None, 2),
        ({"2026-07": 95.0, "2026-08": None, "2026-09": 93.0}, False, 1),
    ],
)
def test_window_and_threshold_table(check, small_data, settings, values, met, below):
    result = check(_otif_data(small_data, values), settings.company, "SUP-114", "2026-09")
    assert result.condition_met is met, [(m.month, m.otif_pct) for m in result.months]
    assert result.months_below == below
    assert [m.month for m in result.months] == ["2026-07", "2026-08", "2026-09"]
