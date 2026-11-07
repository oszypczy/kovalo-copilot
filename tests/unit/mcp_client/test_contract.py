"""Kontrakt rekordów klienta MCP (kontrakt F5, 8.1 i 13.2): port `check_record` z
`test_provided_mcp_runtime.py` kursu plus rekord zagnieżdżony i reguła dla
`check_termination_condition`. Czyste funkcje, bez sieci."""

from __future__ import annotations

import json

import pytest

from tests.mcp_fakes import direct_record

INVOICE = {
    "status": "ok",
    "supplier_id": "SUP-114",
    "supplier": "Mleczna Dolina",
    "month": "2026-07",
    "invoiced_grosze": 18240000,
    "currency": "PLN",
    "message": None,
}


def check(tool, structured, text=()):
    from kovalo.mcp_client.contract import check_record

    return check_record(tool, structured, list(text))


def error():
    from kovalo.mcp_client.connection import McpCallError

    return McpCallError


def test_a_valid_record_is_returned_unchanged():
    assert check("get_invoice_value", dict(INVOICE)) == INVOICE


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ({"invoiced_grosze": True}, "invoiced_grosze"),  # bool to nie pieniądze
        ({"invoiced_grosze": "18240000"}, "invoiced_grosze"),  # napis z liczbą też nie
        ({"invoiced_grosze": 182400.0}, "invoiced_grosze"),  # ani złotówki jako float
        ({"invoiced_grosze": None}, "nie może być null"),  # null tam, gdzie "ok" wymaga wartości
        ({"extra": 1}, "spoza kontraktu"),
    ],
)
def test_wrong_values_break_the_contract(change, says):
    with pytest.raises(error(), match=says) as info:
        check("get_invoice_value", {**INVOICE, **change})
    assert info.value.layer == "validation"
    assert str(info.value).startswith("[błąd walidacji lokalnej]")


def test_absent_is_not_null():
    record = dict(INVOICE)
    record.pop("message")  # pole z wartością domyślną, po prostu pominięte
    with pytest.raises(error(), match="message: brak"):
        check("get_invoice_value", record)


def test_nested_records_are_checked_item_by_item():
    otif = {
        "status": "ok",
        "supplier_id": "SUP-114",
        "supplier": "M",
        "requested": {"from": "2026-07", "to": "2026-07"},
        "observations": [{"month": "2026-07"}],
        "missing_months": [],
        "complete": True,
        "message": None,
    }
    with pytest.raises(error(), match=r"observations\[0\]\.otif_pct: brak"):
        check("get_supplier_otif", otif)


def test_text_and_structured_content_must_agree():
    with pytest.raises(error(), match="nie zgadzają"):
        check("calculate_otif_penalty", {"penalty_grosze": 729600}, ['{"penalty_grosze": 1}'])


def test_no_structured_content_is_a_validation_error():
    with pytest.raises(error(), match=r"\[błąd walidacji lokalnej\].*nie ma structuredContent"):
        check("get_invoice_value", None, ["{}"])


def test_an_ok_invoice_must_be_in_pln():
    with pytest.raises(error(), match="currency musi mieć wartość 'PLN'"):
        check("get_invoice_value", {**INVOICE, "currency": "USD"})


@pytest.mark.parametrize(
    ("second", "says"),
    [
        ('{"invoiced_grosze": 1}', "nie zgadzają"),  # druga, sprzeczna kopia
        ("szczegóły w logu", "nie jest kopią JSON"),  # drugi blok, który nie jest kopią
    ],
)
def test_every_text_block_must_agree(second, says):
    first = json.dumps(INVOICE)  # kompletny, poprawny rekord: różnią się tylko bloki
    assert check("get_invoice_value", dict(INVOICE), [first]) == INVOICE
    with pytest.raises(error(), match=says):
        check("get_invoice_value", dict(INVOICE), [first, second])


def test_unknown_tool_has_no_canonical_contract():
    with pytest.raises(error(), match="brak kanonicznego kontraktu"):
        check("approve_discount", {"status": "ok"})


# --- NASZE: tabele kontraktu i rekord zagnieżdżony ----------------------------------------------


def test_canonical_covers_all_seven_specs():
    from kovalo.mcp_client.contract import CANONICAL
    from kovalo.tools.specs import TOOL_SPECS

    assert len(CANONICAL) == 7
    assert {spec.name: spec.record for spec in TOOL_SPECS} == CANONICAL


def test_status_tables_keep_the_courses_entries_and_add_one():
    from kovalo.mcp_client.contract import STATUS_RULES, STATUS_VALUES

    assert STATUS_RULES[("get_supplier_otif", "ok")] == ("supplier_id",)
    assert STATUS_RULES[("get_invoice_value", "ok")] == ("supplier_id", "invoiced_grosze")
    assert STATUS_RULES[("get_contract_terms", "ok")] == (
        "supplier_id",
        "source_doc_id",
        "otif_target_pct",
        "penalty_clause",
        "terms_version",
    )
    assert STATUS_RULES[("calculate_otif_penalty", "ok")] == (
        "shortfall_points",
        "full_points",
        "penalty_pct_uncapped",
        "penalty_pct",
        "penalty_grosze",
    )
    assert STATUS_RULES[("check_discount_policy", "ok")] == ("version",)
    assert STATUS_RULES[("check_termination_condition", "ok")] == (
        "supplier_id",
        "source_doc_id",
        "otif_below_pct",
        "consecutive_months",
        "otif",
    )
    assert len(STATUS_RULES) == 6
    assert STATUS_VALUES == {
        ("get_invoice_value", "ok"): {"currency": "PLN"},
        ("calculate_otif_penalty", "ok"): {"currency": "PLN"},
    }


def _termination(settings):
    return direct_record(
        settings,
        "check_termination_condition",
        {"supplier": "Mleczna Dolina", "as_of_month": "2026-09"},
    )


def test_a_real_termination_record_passes(settings):
    record = _termination(settings)
    assert check("check_termination_condition", record) == record


def test_nested_single_record_is_checked(settings):
    """`otif` (pojedynczy zagnieżdżony rekord, `OtifResult | None`) bez `missing_months`."""
    record = _termination(settings)
    del record["otif"]["missing_months"]
    with pytest.raises(error(), match=r"otif\.missing_months: brak"):
        check("check_termination_condition", record)


def test_nested_single_record_extra_field_is_rejected(settings):
    record = _termination(settings)
    record["otif"]["bonus"] = 1
    with pytest.raises(error(), match=r"otif\.bonus: spoza kontraktu"):
        check("check_termination_condition", record)


def test_an_ok_termination_needs_its_otif_record(settings):
    record = _termination(settings)
    record["otif"] = None
    with pytest.raises(error(), match="otif nie może być null"):
        check("check_termination_condition", record)
