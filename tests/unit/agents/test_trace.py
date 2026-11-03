"""Trace, budżet i run_question (port `test_provided_trace_plumbing.py` MP4 bez testów liczb;
kontrakt F4, 7.1). Zapisy przebiegów kursu wczytują się bez zmian."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from tests.agent_fakes import ScriptedModel, answer, call, parallel_calls

COURSE_RUNS = (
    Path(__file__).resolve().parents[3] / "reference/course/mini_project_4/reference_traces"
)


def supervisor_with_one_worker(worker_script, supervisor_script, budget_box):
    from kovalo.agents.trace import run_question, worker_result

    @tool
    def get_contract_terms(supplier: str) -> dict:
        """Warunki umowy (dubler testowy)."""
        return {
            "status": "ok",
            "supplier_id": "SUP-207",
            "penalty_clause": False,
            "otif_target_pct": 96.0,
        }

    worker = create_agent(ScriptedModel(script=worker_script), [get_contract_terms])

    @tool
    def ask_data_analyst(task: str) -> dict:
        """Zadanie dla specjalisty od danych (dubler testowy)."""
        return worker_result(
            run_question(worker, task, budget_box["budget"], agent_name="data_analyst")
        )

    return create_agent(ScriptedModel(script=supervisor_script), [ask_data_analyst])


def test_worker_calls_are_nested_and_its_summary_is_not_a_source():
    from kovalo.agents.trace import Budget, run_question

    box = {"budget": Budget(20, 60)}
    supervisor = supervisor_with_one_worker(
        [call("get_contract_terms", supplier="SUP-207"), answer("Aqua jest winna 1926 PLN.")],
        [
            call("ask_data_analyst", task="Czy Aqua jest winna karę?"),
            answer("Aqua jest winna 1926 PLN."),
        ],
        box,
    )
    trace = run_question(supervisor, "Czy Aqua jest winna karę?", box["budget"])
    assert trace.status == "answered"
    wrapper, nested = trace.events
    assert (wrapper.tool, wrapper.origin, wrapper.result) == (
        "ask_data_analyst",
        "worker_summary",
        {"summary": "Aqua jest winna 1926 PLN."},
    )
    assert (nested.tool, nested.origin, nested.agent, nested.parent_call_id) == (
        "get_contract_terms",
        "data",
        "data_analyst",
        wrapper.call_id,
    )
    assert trace.steps == 6, "jeden limit liczy kroki supervisora ORAZ specjalisty"


def test_one_budget_for_the_whole_question():
    from kovalo.agents.trace import Budget, run_question

    box = {"budget": Budget(max_steps=5, timeout_s=60)}
    looping = [lambda m: call("get_contract_terms", supplier="SUP-207")] * 20
    supervisor = supervisor_with_one_worker(
        looping, [call("ask_data_analyst", task="warunki?")] * 5, box
    )
    trace = run_question(supervisor, "warunki?", box["budget"])
    assert trace.status == "limit", "specjalista nie może wyzerować limitu pytania"
    assert any(e.agent == "data_analyst" for e in trace.events), (
        "zdarzenia specjalisty zostają zachowane"
    )


def test_a_time_limit():
    from kovalo.agents.trace import Budget

    budget = Budget(max_steps=100, timeout_s=-1)
    with pytest.raises(Exception, match="limit czasu"):
        budget.tick()


def test_budget_from_settings_is_pinned(settings):
    from kovalo.agents.trace import Budget

    budget = Budget.from_settings(settings.agent)
    assert (budget.max_steps, budget.timeout_s) == (24, 180)
    assert budget.steps == 0


def invoice_tool(performed):
    @tool
    def get_invoice_value(supplier: str, month: str = "2026-07") -> dict:
        """Faktura (dubler testowy)."""
        performed.append(supplier)
        return {
            "status": "ok",
            "supplier_id": supplier,
            "month": month,
            "invoiced_grosze": 10000,
            "currency": "PLN",
        }

    return get_invoice_value


def test_parallel_results_straddling_the_limit_are_all_kept():
    from kovalo.agents.trace import Budget, run_question

    # model zużywa krok 1 z 2; paczka trzech narzędzi zostaje dopuszczona, a jej wyniki przekraczają limit  # noqa: E501
    performed: list[str] = []
    model = ScriptedModel(
        script=[
            parallel_calls("get_invoice_value", [{"supplier": s} for s in "ABC"], ["a", "b", "c"]),
            answer("gotowe"),
        ]
    )
    trace = run_question(create_agent(model, [invoice_tool(performed)]), "faktury?", Budget(2, 60))
    assert sorted(performed) == ["A", "B", "C"]
    assert trace.status == "limit"
    assert sorted((e.call_id, e.status) for e in trace.events) == [
        ("a", "ok"),
        ("b", "ok"),
        ("c", "ok"),
    ], "każdy dostarczony wynik jest zapisany"
    assert model.calls == 1, "po wyczerpaniu limitu nie ma nowej tury modelu"


def test_parallel_workers_when_one_spends_the_shared_budget():
    from kovalo.agents.trace import Budget, run_question, worker_result

    box = {"budget": Budget(max_steps=6, timeout_s=60)}
    performed: list[str] = []
    looping = create_agent(
        ScriptedModel(script=[lambda m: call("get_invoice_value", supplier="L")] * 20),
        [invoice_tool(performed)],
    )
    quick = create_agent(
        ScriptedModel(script=[call("get_invoice_value", supplier="Q"), answer("100 PLN")]),
        [invoice_tool(performed)],
    )

    @tool
    def ask_looping(task: str) -> dict:
        """Specjalista, który się zapętla (dubler testowy)."""
        return worker_result(run_question(looping, task, box["budget"], agent_name="looping"))

    @tool
    def ask_quick(task: str) -> dict:
        """Szybki specjalista (dubler testowy)."""
        return worker_result(run_question(quick, task, box["budget"], agent_name="quick"))

    both = AIMessage(
        content="",
        tool_calls=[
            {"id": "w1", "name": "ask_looping", "args": {"task": "t"}},
            {"id": "w2", "name": "ask_quick", "args": {"task": "t"}},
        ],
    )
    supervisor = create_agent(
        ScriptedModel(script=[both, answer("gotowe")]), [ask_looping, ask_quick]
    )
    trace = run_question(supervisor, "q", box["budget"])
    assert trace.status == "limit"
    wrappers = {e.call_id: e for e in trace.events if e.origin == "worker_summary"}
    assert set(wrappers) == {"w1", "w2"}, "wyniki obu specjalistów są zapisane"
    assert {e.agent for e in trace.events if e.origin == "data"} <= {"looping", "quick"}
    assert all(e.parent_call_id in wrappers for e in trace.events if e.origin == "data"), (
        "zagnieżdżone dowody zostają zachowane"
    )


def test_no_new_task_starts_when_the_steps_are_exactly_used():
    from kovalo.agents.trace import Budget, run_question

    # 2 kroki: model (1) i narzędzie (2) zużywają wszystkie, więc kolejne wywołanie modelu się nie zaczyna  # noqa: E501
    model = ScriptedModel(script=[call("get_invoice_value", supplier="A"), answer("gotowe")])
    trace = run_question(create_agent(model, [invoice_tool([])]), "faktura?", Budget(2, 60))
    assert (trace.status, trace.answer, trace.steps, model.calls) == ("limit", None, 2, 1)
    assert [e.status for e in trace.events] == ["ok"], "dostarczony wynik zostaje zachowany"


def test_an_answer_completing_exactly_on_the_limit_succeeds():
    from kovalo.agents.trace import Budget, run_question

    model = ScriptedModel(script=[call("get_invoice_value", supplier="A"), answer("gotowe")])
    trace = run_question(create_agent(model, [invoice_tool([])]), "faktura?", Budget(3, 60))
    assert (trace.status, trace.answer, trace.steps) == ("answered", "gotowe", 3)


@pytest.mark.parametrize("budget_kind", ["zero", "spent"])
def test_a_zero_or_used_up_step_budget_starts_nothing(budget_kind):
    from kovalo.agents.trace import Budget, run_question

    budget = Budget(0, 60)
    if budget_kind == "spent":  # wspólny limit, który inni specjaliści supervisora już wyczerpali
        budget = Budget(24, 60)
        budget.steps = 24
    model = ScriptedModel(script=[answer("za późno")])
    trace = run_question(create_agent(model, []), "dzień dobry", budget)
    assert trace.status == "limit" and model.calls == 0


def test_a_spent_budget_starts_no_model_call():
    from kovalo.agents.trace import Budget, run_question

    model = ScriptedModel(script=[answer("za późno")])
    trace = run_question(create_agent(model, []), "dzień dobry", Budget(20, -1))
    assert trace.status == "limit" and model.calls == 0


def test_a_late_final_answer_is_limit():
    from kovalo.agents.trace import Budget, run_question

    def slow(messages):
        time.sleep(0.05)
        return answer("spóźniona")

    trace = run_question(
        create_agent(ScriptedModel(script=[slow]), []), "dzień dobry", Budget(20, 0.01)
    )
    assert (trace.status, trace.answer) == ("limit", "spóźniona")


def test_two_levels_of_workers_keep_parents_and_origins():
    from kovalo.agents.trace import Budget, run_question, worker_result

    shared, performed = Budget(30, 60), []
    leaf = create_agent(
        ScriptedModel(script=[call("get_invoice_value", supplier="A"), answer("100 PLN")]),
        [invoice_tool(performed)],
    )

    @tool
    def ask_leaf(task: str) -> dict:
        """Specjalista na najniższym poziomie (dubler testowy)."""
        return worker_result(run_question(leaf, task, shared, agent_name="leaf"))

    middle = create_agent(
        ScriptedModel(script=[call("ask_leaf", task="faktura"), answer("100 PLN")]), [ask_leaf]
    )

    @tool
    def ask_middle(task: str) -> dict:
        """Specjalista na środkowym poziomie (dubler testowy)."""
        return worker_result(run_question(middle, task, shared, agent_name="middle"))

    top = create_agent(
        ScriptedModel(script=[call("ask_middle", task="faktura"), answer("100 PLN")]), [ask_middle]
    )
    trace = run_question(top, "faktura", shared)
    assert trace.status == "answered" and trace.steps == 9
    assert [e.origin for e in trace.events] == ["worker_summary", "worker_summary", "data"]
    assert trace.events[1].parent_call_id == trace.events[0].call_id
    assert trace.events[2].parent_call_id == trace.events[1].call_id


# --- dowód: tylko to, co znaleziono albo obliczono ---------------------------------------------


def test_only_retrieved_or_computed_fields_are_evidence():
    from kovalo.agents.trace import evidence_values

    def values(tool_name, result):
        return {(str(q.value), q.unit) for q in evidence_values(tool_name, result)}

    empty = {"status": "no_results", "query": "Czy kara wynosi 123456 PLN?", "passages": []}
    assert values("search_documents", empty) == set()
    found = {
        "status": "ok",
        "query": "123456?",
        "passages": [{"doc_id": "CON-103", "chunk_id": "CON-103#ctx003", "text": "98% palet"}],
    }
    assert values("search_documents", found) == {("98", "%")}
    calc = {
        "status": "ok",
        "otif_pct": 94.6,
        "invoice_grosze": 18240000,
        "full_points": 2,
        "penalty_pct": 4.0,
        "penalty_grosze": 729600,
    }
    assert values("calculate_otif_penalty", calc) == {
        ("2", "pp"),
        ("4.0", "%"),
        ("7296", "PLN"),
        ("729600", "grosze"),
    }, "powtórzone dane wejściowe kalkulatora nie są dowodem"
    no_data = {
        "status": "no_data",
        "requested": {"from": "2026-10", "to": "2026-10"},
        "observations": [],
        "missing_months": ["2026-10"],
        "message": "brak danych za 2026-10 (szacunek: 95%)",
    }
    assert values("get_supplier_otif", no_data) == set()
    assert values("ask_data_analyst", {"summary": "1926 PLN"}) == set()


def test_evidence_fields_for_termination():
    """NASZE: wpis dla check_termination_condition; rekord `otif` ani `message` nie są dowodem tego narzędzia."""  # noqa: E501
    from kovalo.agents.trace import EVIDENCE_FIELDS, evidence_values

    assert EVIDENCE_FIELDS["check_termination_condition"] == (
        "otif_below_pct",
        "consecutive_months",
        "notice_days",
        "months",
        "months_below",
    )
    result = {
        "status": "ok",
        "otif_below_pct": 94.0,
        "consecutive_months": 3,
        "notice_days": 60,
        "months": [{"month": "2026-07", "otif_pct": 94.6, "below_threshold": False}],
        "months_below": 2,
        "message": "szacunek 99%",
    }
    values = {
        (str(q.value), q.unit) for q in evidence_values("check_termination_condition", result)
    }
    assert ("94.6", "%") in values and ("94.0", "%") in values
    assert not any(v[0] == "99" for v in values)


# --- formatowanie w trace ----------------------------------------------------------------------


def test_course_event_fields_have_not_changed():
    from kovalo.agents.trace import ORIGINS, SOURCE_ORIGINS, Trace, TraceEvent

    assert frozenset({"data", "calculation"}) == SOURCE_ORIGINS
    assert set(TraceEvent.model_fields) >= {
        "call_id",
        "parent_call_id",
        "agent",
        "tool",
        "arguments",
        "origin",
        "status",
        "result",
        "error",
        "transport",
        "invocation_id",
    }
    assert set(Trace.model_fields) >= {
        "question_id",
        "question",
        "answer",
        "status",
        "error",
        "events",
        "steps",
        "input_tokens",
        "output_tokens",
        "seconds",
        "model_calls",
    }
    assert ORIGINS["check_termination_condition"] == "calculation"


# --- zdarzenia składowe (FR-4.7, ADR-0011) ------------------------------------------------------


def test_derived_event_for_termination(agent_graph):
    from kovalo.agents.trace import Budget, run_question

    agent, _ = agent_graph(
        [
            call("check_termination_condition", supplier="SUP-114", as_of_month="2026-09"),
            answer("ok"),
        ],
        variant="ours",
    )
    seen: list[tuple[str, str, str]] = []
    budget = Budget(20, 60)
    trace = run_question(
        agent,
        "Czy można rozwiązać umowę?",
        budget,
        on_event=lambda phase, ev: seen.append((phase, ev.tool, ev.call_id)),
    )
    parent, derived = trace.events
    assert (parent.tool, parent.origin, parent.status) == (
        "check_termination_condition",
        "calculation",
        "ok",
    )
    assert derived.call_id == f"{parent.call_id}#get_supplier_otif"
    assert derived.parent_call_id == parent.call_id
    assert (derived.tool, derived.origin, derived.status) == ("get_supplier_otif", "data", "ok")
    assert derived.agent == "check_termination_condition"
    assert derived.arguments == {
        "supplier": "SUP-114",
        "from_month": "2026-07",
        "to_month": "2026-09",
    }
    assert derived.result == parent.result["otif"], (
        "wynik składowego to rekord odczytu z pola `otif`"
    )
    assert (budget.steps, trace.steps) == (3, 3), (
        "składowe nie liczy się do budżetu: tura, wynik, odpowiedź"
    )
    assert [(p, t) for p, t, _ in seen if t == "get_supplier_otif"] == [
        ("start", "get_supplier_otif"),
        ("end", "get_supplier_otif"),
    ]
    assert [(p, t) for p, t, _ in seen if t == "check_termination_condition"] == [
        ("start", "check_termination_condition"),
        ("end", "check_termination_condition"),
    ]
    order = [(p, t) for p, t, _ in seen]
    assert order.index(("start", "check_termination_condition")) < order.index(
        ("start", "get_supplier_otif")
    )


def test_no_derived_event_when_the_tool_fails(agent_graph):
    agent, _ = agent_graph(
        [call("check_termination_condition", supplier="SUP-114"), answer("ok")],  # brak as_of_month
        variant="ours",
    )
    from kovalo.agents.trace import Budget, run_question

    trace = run_question(agent, "?", Budget(20, 60))
    assert [(e.tool, e.status) for e in trace.events] == [("check_termination_condition", "error")]


def test_no_derived_event_for_not_applicable(agent_graph):
    """Wynik bez rekordu odczytu (`otif` None): brak zdarzenia składowego."""
    from kovalo.agents.trace import Budget, run_question

    agent, _ = agent_graph(
        [
            call("check_termination_condition", supplier="Aqua Kryształ", as_of_month="2026-09"),
            answer("ok"),
        ],
        variant="ours",
    )
    trace = run_question(agent, "?", Budget(20, 60))
    assert [e.tool for e in trace.events] == ["check_termination_condition"]
    assert trace.events[0].result["status"] == "not_applicable"


def test_on_event_errors_do_not_break_run(agent_graph):
    from kovalo.agents.trace import Budget, run_question

    agent, _ = agent_graph([call("get_contract_terms", supplier="SUP-114"), answer("Gotowe.")])

    def broken(phase, ev):
        raise RuntimeError("callback padł")

    trace = run_question(agent, "Warunki?", Budget(20, 60), on_event=broken)
    assert trace.status == "answered" and trace.answer == "Gotowe."
    assert [e.status for e in trace.events] == ["ok"]


def test_on_event_start_and_end_order(agent_graph):
    from kovalo.agents.trace import Budget, run_question

    agent, _ = agent_graph(
        [
            call("get_contract_terms", supplier="SUP-114"),
            call("get_invoice_value", supplier="SUP-114", month="2026-06"),
            answer("Gotowe."),
        ]
    )
    log: list[tuple[str, str, str | None]] = []
    run_question(
        agent,
        "?",
        Budget(20, 60),
        on_event=lambda phase, ev: log.append((phase, ev.tool, ev.status)),
    )
    assert [(p, t) for p, t, _ in log] == [
        ("start", "get_contract_terms"),
        ("end", "get_contract_terms"),
        ("start", "get_invoice_value"),
        ("end", "get_invoice_value"),
    ]
    assert log[0][2] == "no_result", "start: wynik jeszcze nieznany"
    assert log[1][2] == "ok"


# --- koperta ------------------------------------------------------------------------------------


def test_envelope_roundtrip():
    from kovalo.agents.trace import (
        ENVELOPE_KIND,
        ENVELOPE_VERSION,
        trace_envelope,
        trace_from_record,
    )
    from tests.agent_fakes import JULY, july_events, make_trace

    trace = make_trace(JULY, "Kara: 7296,00 PLN.", july_events())
    envelope = trace_envelope(trace)
    assert (
        (envelope["kind"], envelope["version"])
        == (ENVELOPE_KIND, ENVELOPE_VERSION)
        == (
            "agent_trace",
            1,
        )
    )
    assert trace_from_record(envelope) == trace
    assert len(envelope["trace_id"]) == 32 and int(envelope["trace_id"], 16) >= 0
    created = datetime.fromisoformat(envelope["created_at"])
    assert created.utcoffset() is not None and created.utcoffset().total_seconds() == 0
    assert trace_envelope(trace)["trace_id"] != envelope["trace_id"]
    assert envelope["trace"] == trace.model_dump(mode="json")
    with pytest.raises(ValueError, match="wersj"):
        trace_from_record({**envelope, "version": 2})
    assert trace_from_record(None) is None
    assert trace_from_record({"question": "q", "context": []}) is None, "rekord RAG to nie koperta"


# --- zapisy kursu ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["agent_run.json", "supervisor_run.json"])
def test_course_traces_load(name):
    from kovalo.agents.trace import load_traces

    run, rows = load_traces(COURSE_RUNS / name)
    assert len(rows) == 9 and run["budget"] == {"max_steps": 24, "timeout_s": 180}
    for row in rows:
        trace = row["trace"]
        assert trace.model_calls == [], "pola NASZE mają wartości domyślne"
        assert "score" in row


def test_save_and_load_roundtrip(tmp_path):
    from kovalo.agents.trace import load_traces, save_traces
    from tests.agent_fakes import JULY, july_events, make_trace

    trace = make_trace(JULY, "Kara: 7296,00 PLN.", july_events())
    path = tmp_path / "x" / "agent_run.json"
    run = {"run_id": "r1", "date": "2026-11-03", "model": "m", "architecture": "single agent"}
    save_traces(path, run, [trace], [{"outcome": "answered"}])
    loaded_run, rows = load_traces(path)
    assert loaded_run == run and rows[0]["trace"] == trace
    assert rows[0]["score"] == {"outcome": "answered"}


def test_origins_match_specs():
    from kovalo.agents import trace as agents_trace
    from kovalo.tools import specs

    assert agents_trace.ORIGINS == specs.ORIGINS
    assert agents_trace.COURSE_ORIGINS == {
        "search_documents": "data",
        "get_supplier_otif": "data",
        "get_invoice_value": "data",
        "get_contract_terms": "data",
        "calculate_otif_penalty": "calculation",
        "check_discount_policy": "calculation",
    }
    assert set(agents_trace.COURSE_ORIGINS) == set(specs.COURSE_TOOLS)
    assert agents_trace.DERIVED_EVENTS == {
        "check_termination_condition": ("otif", "get_supplier_otif")
    }


def test_tool_errors_is_the_langchain_middleware():
    """tool_errors(): „Błąd: <Typ>: <treść>” (1:1 z kursem)."""
    from kovalo.agents.trace import tool_errors

    middleware = tool_errors()
    assert type(middleware).__name__ == "ToolErrorMiddleware"


def test_answer_from_content_blocks_is_plain_text():
    """Gemini 3 zwraca odpowiedź jako bloki z podpisem myśli; w trace ma trafić sam tekst
    (inaczej podpis base64 trafia do odpowiedzi, UI i oceny liczb; E4 na żywo, P1)."""
    from kovalo.agents.trace import Budget, run_question

    blocks = [
        {"type": "text", "text": "Kara wynosi 7 296,00 PLN.", "extras": {"signature": "Em8/8+x"}}
    ]
    model = ScriptedModel(script=[AIMessage(content=blocks)])
    trace = run_question(create_agent(model, []), "kara?", Budget(3, 60))
    assert (trace.status, trace.answer) == ("answered", "Kara wynosi 7 296,00 PLN.")
