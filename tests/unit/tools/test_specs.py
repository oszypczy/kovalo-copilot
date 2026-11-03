"""Specyfikacje narzędzi wspólne dla LangChaina i MCP.

Kontrakt F4, 6.3-6.4; port testów kroku 3 MP4.
"""

from __future__ import annotations

import inspect

import pytest

COURSE_NAMES = {
    "search_documents",
    "get_supplier_otif",
    "get_invoice_value",
    "get_contract_terms",
    "calculate_otif_penalty",
    "check_discount_policy",
}
ARGUMENTS = {
    "search_documents": {"question"},
    "get_supplier_otif": {"supplier", "from_month", "to_month"},
    "get_invoice_value": {"supplier", "month"},
    "get_contract_terms": {"supplier"},
    "calculate_otif_penalty": {
        "otif_pct",
        "target_pct",
        "rate_per_full_point_pct",
        "cap_pct",
        "invoice_grosze",
    },
    "check_discount_policy": {
        "discount_pct",
        "margin_after_pct",
        "clearance",
        "strategic_account",
        "competitor_offer",
        "contractual_volume_rebate",
    },
    "check_termination_condition": {"supplier", "as_of_month"},
}
ORDER = [
    "search_documents",
    "get_supplier_otif",
    "get_invoice_value",
    "get_contract_terms",
    "calculate_otif_penalty",
    "check_discount_policy",
    "check_termination_condition",
]


@pytest.fixture
def make_tools(deps_for):
    from kovalo.tools.langchain import langchain_tools

    def make(variant):
        _, ctx = deps_for([])
        return {t.name: t for t in langchain_tools(ctx, variant)}

    return make


def test_course_variant_is_exactly_allowed_tools(make_tools):
    tools = make_tools("course")
    assert set(tools) == COURSE_NAMES, (
        "dokładnie sześć narzędzi tylko do odczytu: nic, co zatwierdza albo zapisuje"
    )
    assert len(tools) == 6


def test_specs_constants():
    from kovalo.tools.specs import COURSE_TOOLS, ORIGINS, OUR_TOOLS, TOOL_SPECS, tool_specs

    assert [s.name for s in TOOL_SPECS] == ORDER, "7 pozycji, kolejność jak w tabeli 6.3"
    assert frozenset(COURSE_NAMES) == COURSE_TOOLS
    assert COURSE_TOOLS | {"check_termination_condition"} == OUR_TOOLS
    assert ORIGINS == {
        "search_documents": "data",
        "get_supplier_otif": "data",
        "get_invoice_value": "data",
        "get_contract_terms": "data",
        "calculate_otif_penalty": "calculation",
        "check_discount_policy": "calculation",
        "check_termination_condition": "calculation",
    }
    assert [s.name for s in tool_specs("course")] == ORDER[:6]
    assert [s.name for s in tool_specs("ours")] == ORDER
    for spec in TOOL_SPECS:
        assert spec.origin == ORIGINS[spec.name] and spec.record.__name__


def test_ours_adds_only_termination(make_tools):
    from kovalo.tools.specs import OUR_TOOLS

    tools = make_tools("ours")
    assert set(tools) == OUR_TOOLS and len(tools) == 7
    assert set(tools) - COURSE_NAMES == {"check_termination_condition"}
    for name, tool in tools.items():
        text = f"{name} {tool.description}".lower()
        for word in ("approve", "zatwierdź", "zapisz", "send"):
            assert word not in text, f"{name}: narzędzie nie może niczego zatwierdzać ani zapisywać"


def test_the_model_sees_names_descriptions_and_arguments_only(make_tools):
    for variant in ("course", "ours"):
        for name, t in make_tools(variant).items():
            assert len(t.description) > 40, f"{name}: opisz, co zwraca, i formaty argumentów"
            assert set(t.args) == ARGUMENTS[name], (
                f"{name}: argumenty {set(t.args)}; data, retriever i policy nigdy nie są argumentami modelu"  # noqa: E501
            )


def test_descriptions_carry_the_contract_sentences(make_tools):
    tools = make_tools("ours")
    assert "penalty_clause" in tools["calculate_otif_penalty"].description
    assert "niczego nie zatwierdza" in tools["check_discount_policy"].description
    termination = tools["check_termination_condition"].description
    for phrase in ("condition_met", "source_doc_id", "search_documents"):
        assert phrase in termination
    assert "YYYY-MM" in tools["get_supplier_otif"].description


def test_tools_return_json_ready_records(make_tools):
    tools = make_tools("ours")
    otif = tools["get_supplier_otif"].invoke(
        {"supplier": "SUP-114", "from_month": "2026-08", "to_month": "2026-10"}
    )
    assert (
        isinstance(otif, dict) and otif["status"] == "ok" and otif["missing_months"] == ["2026-10"]
    )
    terms = tools["get_contract_terms"].invoke({"supplier": "TransBałt"})
    assert terms["status"] == "not_applicable"
    penalty = tools["calculate_otif_penalty"].invoke(
        {
            "otif_pct": 94.6,
            "target_pct": 97,
            "rate_per_full_point_pct": 2,
            "cap_pct": 10,
            "invoice_grosze": 18240000,
        }
    )
    assert penalty["penalty_grosze"] == 729600
    policy = tools["check_discount_policy"].invoke(
        {"discount_pct": 22, "margin_after_pct": 7, "strategic_account": True}
    )
    assert policy["approvers"] == ["Dyrektor handlowy", "CFO"]
    invoice = tools["get_invoice_value"].invoke({"supplier": "SUP-114", "month": "2026-06"})
    assert invoice["invoiced_grosze"] == 18120000
    # SUP-114 na small_data: sierpień 94,0 nie jest poniżej 94, lipiec brakuje
    ended = tools["check_termination_condition"].invoke(
        {"supplier": "SUP-114", "as_of_month": "2026-09"}
    )
    assert ended["status"] == "ok" and ended["condition_met"] is False
    assert ended["missing_months"] == ["2026-07"]


def test_search_uses_the_pinned_top_k(make_tools, retriever, settings):
    result = make_tools("course")["search_documents"].invoke({"question": "palety"})
    assert result["status"] == "ok" and retriever.ks == [settings.retrieval.top_k]
    assert settings.retrieval.top_k == 3, "PINNED"


def test_search_without_index_reports_an_error(settings, small_data):
    """retriever=None → RuntimeError („indeks niedostępny”) zamiast cichego „brak wyników”."""
    from kovalo.tools.langchain import langchain_tools
    from tests.agent_fakes import tool_context

    ctx = tool_context(settings, small_data, retriever=None)
    tools = {t.name: t for t in langchain_tools(ctx, "ours")}
    with pytest.raises(Exception, match="indeks"):
        tools["search_documents"].invoke({"question": "palety"})


def test_spec_schema_matches_langchain_args(deps_for):
    """Schemat argumentów StructuredTool == schemat z sygnatury `spec.make(ctx)` (F5: FastMCP)."""
    from pydantic import TypeAdapter

    from kovalo.tools.langchain import langchain_tools
    from kovalo.tools.specs import tool_specs

    _, ctx = deps_for([])
    tools = {t.name: t for t in langchain_tools(ctx, "ours")}
    keys = ("type", "anyOf", "description", "default")
    for spec in tool_specs("ours"):
        function = spec.make(ctx)
        assert function.__name__ == spec.name
        assert function.__doc__ and len(function.__doc__) > 40
        assert inspect.signature(function).return_annotation in (spec.record, spec.record.__name__)
        expected = TypeAdapter(function).json_schema()["properties"]
        actual = tools[spec.name].args
        assert set(actual) == set(expected) == ARGUMENTS[spec.name]
        for arg in expected:
            for key in keys:
                assert actual[arg].get(key) == expected[arg].get(key), (spec.name, arg, key)
            assert expected[arg].get("description"), f"{spec.name}.{arg}: opis argumentu"
        assert tools[spec.name].description.strip() == function.__doc__.strip()
