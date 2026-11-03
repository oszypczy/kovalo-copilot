"""Narzędzia na PRAWDZIWYCH danych firmy dają wartości wzorcowe policzone ręcznie, a uszkodzone
pliki kończą się wyraźnym błędem (port `test_provided_company_data.py` MP4; kryterium 1 F4).

Wzorzec w data/eval/mini_project_4_questions.yaml policzono ręcznie; nic tutaj nie liczy go
testowanym kodem.
"""

from __future__ import annotations

import shutil

import pytest

HAND_WORKED = {"2026-07": 729600, "2026-06": 362400, "2026-05": 0}  # grosze: 7296,00 / 3624,00 / 0


def test_every_gold_calculation_matches_the_data_and_the_hand_worked_penalty(
    lookup, calc, company_data, agent_gold
):
    checked = 0
    for question in agent_gold.values():
        for g in question.calculations:
            otif = lookup.get_supplier_otif(company_data, g.supplier_id, g.month, g.month)
            terms = lookup.get_contract_terms(company_data, g.supplier_id)
            invoice = lookup.get_invoice_value(company_data, g.supplier_id, g.month)
            assert otif.observations[0].otif_pct == g.otif_pct, f"{question.id}: OTIF"
            assert (terms.otif_target_pct, terms.rate_per_full_point_pct, terms.cap_pct) == (
                g.target_pct,
                g.rate_per_full_point_pct,
                g.cap_pct,
            ), f"{question.id}: terms"
            assert invoice.invoiced_grosze == g.invoice_grosze, f"{question.id}: invoice"
            result = calc.calculate_otif_penalty(
                g.otif_pct, g.target_pct, g.rate_per_full_point_pct, g.cap_pct, g.invoice_grosze
            )
            assert result.penalty_grosze == HAND_WORKED[g.month], f"{question.id}: penalty"
            checked += 1
    assert checked == 3, "P1, P2, P3"


def test_p1_p2_p3_penalties_from_the_real_data(lookup, calc, company_data):
    """Kryterium 1: 729 600 / 362 400 / 0 groszy na prawdziwych danych."""
    got = {}
    for month in ("2026-07", "2026-06", "2026-05"):
        terms = lookup.get_contract_terms(company_data, "Mleczna Dolina")
        otif = lookup.get_supplier_otif(company_data, "Mleczna Dolina", month, month)
        invoice = lookup.get_invoice_value(company_data, "Mleczna Dolina", month)
        got[month] = calc.calculate_otif_penalty(
            otif.observations[0].otif_pct,
            terms.otif_target_pct,
            terms.rate_per_full_point_pct,
            terms.cap_pct,
            invoice.invoiced_grosze,
        ).penalty_grosze
    assert got == {"2026-07": 729600, "2026-06": 362400, "2026-05": 0}


def test_the_termination_months(lookup, company_data):
    result = lookup.get_supplier_otif(company_data, "Mleczna Dolina", "2026-07", "2026-10")
    assert [(o.month, o.otif_pct) for o in result.observations] == [
        ("2026-07", 94.6),
        ("2026-08", 93.8),
        ("2026-09", 93.5),
    ]
    assert result.missing_months == ["2026-10"]


def test_aqua_missed_its_target_without_a_penalty_clause(lookup, company_data):
    terms = lookup.get_contract_terms(company_data, "Aqua Kryształ")
    assert (terms.source_doc_id, terms.otif_target_pct, terms.penalty_clause) == (
        "CON-102",
        96,
        False,
    )
    result = lookup.get_supplier_otif(company_data, "SUP-207", "2026-07", "2026-07")
    assert result.observations[0].otif_pct == 95.1


def test_transbalt_never_feeds_an_otif_calculation(lookup, company_data):
    terms = lookup.get_contract_terms(company_data, "TransBałt")
    assert terms.status == "not_applicable" and terms.source_doc_id == "CON-103"
    assert terms.otif_target_pct is None


def test_october_has_no_data_yet(lookup, company_data):
    assert lookup.get_supplier_otif(company_data, "SUP-114", "2026-10", "2026-10").status == (
        "no_data"
    )


def test_company_data_error_is_a_config_error():
    from kovalo.company.data import CompanyDataError
    from kovalo.config import ConfigError

    assert issubclass(CompanyDataError, ConfigError)


def test_a_missing_directory_is_an_error(tmp_path):
    from kovalo.company.data import CompanyDataError, load_company_data

    with pytest.raises(CompanyDataError, match=r"suppliers\.csv"):
        load_company_data(tmp_path / "nie-ma-takiego-katalogu")


@pytest.fixture
def broken(session_settings, tmp_path):
    """Kopia data/company/, w której można podmienić jeden wiersz jednego pliku."""
    from kovalo.company.data import load_company_data

    folder = tmp_path / "company"
    shutil.copytree(session_settings.paths.company_data_dir, folder)

    def make(file: str, old: str, new: str):
        path = folder / file
        text = path.read_text(encoding="utf-8")
        assert old in text
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return load_company_data(folder)

    return make


@pytest.mark.parametrize(
    ("file", "old", "new", "says"),
    [
        ("supplier_otif.csv", "SUP-114,2026-08,93.8", "SUP-114,2026-07,93.8", "zduplikowany"),
        ("supplier_otif.csv", "2026-08,93.8", "2026-8,93.8", "YYYY-MM"),
        ("supplier_otif.csv", "93.8", "938", "od 0 do 100"),
        ("supplier_otif.csv", "93.8", "n/a", "nie jest liczbą"),
        ("supplier_otif.csv", "SUP-207,2026-05", "SUP-999,2026-05", "nieznany dostawca"),
        ("supplier_invoices.csv", "18240000", "182400.00", "całkowitą liczbą groszy"),
        ("supplier_invoices.csv", "18240000,PLN", "18240000,EUR", "PLN"),
    ],
)
def test_a_broken_file_is_an_error_not_no_data(broken, file, old, new, says):
    from kovalo.company.data import CompanyDataError

    with pytest.raises(CompanyDataError, match=says):
        broken(file, old, new)


def test_month_helpers():
    from kovalo.company.data import is_month, month_range

    assert is_month("2026-07") and not is_month("2026-7") and not is_month(202607)
    assert month_range("2026-11", "2027-01") == ["2026-11", "2026-12", "2027-01"]
    assert month_range("2026-09", "2026-07") == []
    with pytest.raises(ValueError, match="YYYY-MM"):
        month_range("2026-7", "2026-07")
