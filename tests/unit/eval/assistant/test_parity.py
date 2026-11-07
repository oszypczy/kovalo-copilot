"""Zgodność MCP z wywołaniem bezpośrednim (kontrakt F5, 10.2; port kroku 2 kursu
`test_step2_compare_call.py`): `compare_call`, macierz 20 wywołań i mutanty. Serwer w pamięci."""

from __future__ import annotations

import contextlib
import copy

import pytest

from tests.mcp_fakes import (
    COURSE_MATRIX,
    PARITY_MATRIX,
    FakeRetriever,
    connect,
    direct_record,
    mcp_server,
    mutant_server,
)


def compare(direct, mcp):
    from kovalo.eval.assistant.parity import compare_call

    return compare_call(direct, mcp)


def kinds(result):
    return [(d["path"], d["kind"]) for d in result["differences"]]


def test_identical_records_are_the_same():
    record = {
        "status": "ok",
        "observations": [{"month": "2026-07", "otif_pct": 94.6}],
        "message": None,
    }
    assert compare(record, copy.deepcopy(record)) == {"same": True, "differences": []}


def test_missing_and_extra_fields():
    result = compare({"a": 1, "b": 2}, {"a": 1, "c": 3})
    assert kinds(result) == [("b", "missing"), ("c", "extra")] and result["same"] is False
    assert result["differences"][0] == {"path": "b", "kind": "missing", "direct": 2, "mcp": None}


def test_grosze_turned_into_pln_is_a_unit_difference():
    assert kinds(compare({"invoiced_grosze": 18240000}, {"invoiced_grosze": 182400.0})) == [
        ("invoiced_grosze", "unit")
    ]
    assert kinds(compare({"penalty_grosze": 729600}, {"penalty_grosze": 7296})) == [
        ("penalty_grosze", "unit")
    ]


@pytest.mark.parametrize(("direct", "mcp"), [(97, 97.0), (5, "5"), (True, 1), (None, "None")])
def test_representation_is_part_of_the_contract(direct, mcp):
    assert kinds(compare({"x": direct}, {"x": mcp})) == [("x", "type")]


def test_nested_paths_and_list_items():
    direct = {
        "observations": [
            {"month": "2026-07", "otif_pct": 94.6},
            {"month": "2026-08", "otif_pct": 93.8},
        ]
    }
    mcp = {"observations": [{"month": "2026-07", "otif_pct": 94.5}]}
    assert kinds(compare(direct, mcp)) == [
        ("observations[0].otif_pct", "changed"),
        ("observations[1]", "missing"),
    ]


def test_inputs_are_not_modified():
    direct, mcp = {"a": [1, {"b": 2}]}, {"a": [1, {"b": 3}], "z": 0}
    before = (copy.deepcopy(direct), copy.deepcopy(mcp))
    compare(direct, mcp)
    assert (direct, mcp) == before


def test_grosze_precedence_rules():
    """`*_grosze`: mcp nie-int albo mcp*100 == direct → unit; bool nie jest int; inna liczba
    to zwykła zmiana (changed)."""
    assert kinds(compare({"invoiced_grosze": 100}, {"invoiced_grosze": 1})) == [
        ("invoiced_grosze", "unit")
    ]
    assert kinds(compare({"invoiced_grosze": 100}, {"invoiced_grosze": 99})) == [
        ("invoiced_grosze", "changed")
    ]
    assert kinds(compare({"invoiced_grosze": 1}, {"invoiced_grosze": True})) == [
        ("invoiced_grosze", "unit")
    ]


def test_order_is_direct_fields_then_extras():
    result = compare({"b": 1, "a": 2}, {"c": 1, "z": 2})
    assert kinds(result) == [("b", "missing"), ("a", "missing"), ("c", "extra"), ("z", "extra")]


def raw(connection, tool, args):
    from kovalo.mcp_client.connection import McpCallError

    with contextlib.suppress(McpCallError):  # mutant może łamać kontrakt, a surowe dane i tak są
        connection.call(tool, args)
    return connection.captures[-1]["structured"]


def test_the_matrix_is_the_courses_17_plus_our_3():
    from kovalo.eval.assistant.parity import PARITY_MATRIX as matrix

    assert len(matrix) == 20 and len(COURSE_MATRIX) == 17
    assert [(t, a) for t, a in matrix] == [(t, a) for t, a in PARITY_MATRIX]
    ours = [t for t, _ in matrix[17:]]
    assert ours == ["check_termination_condition"] * 3
    assert matrix[17][1] == {"supplier": "Mleczna Dolina", "as_of_month": "2026-09"}
    assert matrix[18][1] == {"supplier": "SUP-114", "as_of_month": "2026-10"}
    assert matrix[19][1] == {"supplier": "SUP-207", "as_of_month": "2026-07"}


@pytest.mark.parametrize(("tool", "args"), PARITY_MATRIX)
def test_your_server_matches_the_direct_calls(settings, ours_connection, tool, args):
    # serwer ours_connection ma FakeRetriever, tak samo wywołanie bezpośrednie
    direct = direct_record(settings, tool, args, retriever=FakeRetriever())
    result = compare(direct, raw(ours_connection, tool, args))
    assert result["same"], f"{tool}{args}: {result['differences']}"


@pytest.mark.parametrize(
    ("kind", "tool", "args", "expected"),
    [
        (
            "pln_floats",
            "get_invoice_value",
            {"supplier": "SUP-114", "month": "2026-07"},
            ("invoiced_grosze", "unit"),
        ),
        (
            "dropped_missing",
            "get_supplier_otif",
            {"supplier": "SUP-114", "from_month": "2026-08", "to_month": "2026-10"},
            ("missing_months", "missing"),
        ),
        (
            "wrong_function",
            "get_invoice_value",
            {"supplier": "SUP-114", "month": "2026-07"},
            ("invoiced_grosze", "missing"),
        ),
        ("static", "get_contract_terms", {"supplier": "SUP-207"}, ("supplier_id", "changed")),
        ("renamed", "get_contract_terms", {"supplier": "SUP-114"}, ("terms_version", "missing")),
    ],
)
def test_each_mutant_wrapper_is_diagnosed(settings, kind, tool, args, expected):
    with connect(mutant_server(kind, settings)) as conn:
        result = compare(direct_record(settings, tool, args), raw(conn, tool, args))
    assert expected in kinds(result), f"{kind}: oczekiwano {expected}, otrzymano {kinds(result)}"


# --- run_parity ---------------------------------------------------------------------------------


def parity_context(settings, retriever):
    from kovalo.company import load_company_data
    from tests.agent_fakes import tool_context

    return tool_context(settings, load_company_data(settings.paths.company_data_dir), retriever)


def test_direct_record_uses_the_spec_functions(settings):
    from kovalo.eval.assistant.parity import direct_record as direct

    ctx = parity_context(settings, FakeRetriever())
    record = direct(ctx, "get_invoice_value", {"supplier": "SUP-114", "month": "2026-07"})
    assert record["invoiced_grosze"] == 18240000 and record["currency"] == "PLN"
    assert isinstance(record, dict)


def test_run_parity_all_20_rows_are_the_same(settings):
    from kovalo.eval.assistant.parity import ParityRow, run_parity

    retriever = FakeRetriever()
    with connect(mcp_server(settings, "ours", retriever=retriever)) as conn:
        rows = run_parity(parity_context(settings, retriever), conn)
    assert len(rows) == 20 and all(isinstance(r, ParityRow) for r in rows)
    assert all(r.same and r.differences == [] and r.error is None for r in rows)
    assert [(r.tool, r.arguments) for r in rows] == [(t, a) for t, a in PARITY_MATRIX]


def test_run_parity_reports_a_mutant_and_the_validation_error(settings):
    from kovalo.eval.assistant.parity import run_parity

    retriever = FakeRetriever()
    with connect(mutant_server("pln_floats", settings, retriever, "ours")) as conn:
        rows = run_parity(parity_context(settings, retriever), conn)
    broken = [r for r in rows if not r.same]
    assert broken and {r.tool for r in broken} == {"get_invoice_value"}
    ok_invoice = next(r for r in broken if r.arguments["month"] == "2026-07")
    assert ("invoiced_grosze", "unit") in kinds({"differences": ok_invoice.differences})
    assert ok_invoice.error and "invoiced_grosze" in ok_invoice.error, (
        "surowy wynik porównany także po odrzuceniu przez check_record; error = komunikat"
    )


def test_run_parity_static_mutant_names_the_changed_supplier(settings):
    from kovalo.eval.assistant.parity import run_parity

    retriever = FakeRetriever()
    with connect(mutant_server("static", settings, retriever, "ours")) as conn:
        rows = run_parity(parity_context(settings, retriever), conn)
    broken = [r for r in rows if not r.same]
    assert broken and all(r.tool == "get_contract_terms" for r in broken)
    assert any(r.arguments == {"supplier": "SUP-207"} for r in broken)


def test_run_parity_without_local_retriever_skips_search(settings):
    """Bez lokalnego retrievera (np. zgodność przez HTTP z kontenerem na pgvector) wiersze
    `search_documents` są jawnie pominięte: bez wywołania bezpośredniego i bez wywołania MCP;
    pozostałe 19 porównane normalnie."""
    from kovalo.eval.assistant.parity import run_parity

    with connect(mcp_server(settings, "ours", retriever=FakeRetriever())) as conn:
        rows = run_parity(parity_context(settings, None), conn)
        called = [c["tool"] for c in conn.captures]
    search = [r for r in rows if r.tool == "search_documents"]
    assert len(rows) == 20 and len(search) == 1
    assert search[0].skipped and not search[0].same and "pominięte" in (search[0].error or "")
    assert "search_documents" not in called
    others = [r for r in rows if r.tool != "search_documents"]
    assert all(r.same and not r.skipped for r in others)
