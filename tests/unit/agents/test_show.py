"""format_trace i brief (port `show_trace.py` MP4; kontrakt F4, 7.4)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
COURSE_RUNS = ROOT / "reference/course/mini_project_4/reference_traces"
README = ROOT / "reference/course/mini_project_4/README.md"


def readme_block(name: str) -> str:
    text = README.read_text(encoding="utf-8")
    match = re.search(rf"<!-- reference-run:{name} .*?-->\n```\n(.*?)```\n", text, re.DOTALL)
    assert match, f"brak bloku {name} w README MP4"
    return match.group(1).strip()


def course_trace(qid: str, which: str = "agent_run.json"):
    from kovalo.agents.trace import load_traces

    _, rows = load_traces(COURSE_RUNS / which)
    return next(r for r in rows if r["trace"].question_id == qid)


def test_format_trace_matches_course_p1():
    """Linie wywołań i wyników P1 z agent_run.json == blok trace_P1 z README MP4 (cały blok)."""
    from kovalo.agents.show import format_trace

    block = readme_block("trace_P1")
    text = format_trace(course_trace("P1")["trace"])
    assert text.strip() == block
    calls = [line for line in text.splitlines() if re.match(r"\s+\d+\. ", line)]
    assert len(calls) == 4 and "<calculation>" in calls[3]
    assert "penalty_grosze" in text and "729600" in text


def test_format_trace_with_score_appends_it():
    from kovalo.agents.show import format_trace

    row = course_trace("P1")
    text = format_trace(row["trace"], row["score"])
    assert text.splitlines()[-1].startswith("ocena: ")
    assert '"outcome": "answered"' in text


def test_nested_events_are_indented_and_labelled():
    from kovalo.agents.show import format_trace

    trace = course_trace("P1", "supervisor_run.json")["trace"]
    text = format_trace(trace)
    nested = [line for line in text.splitlines() if "[" in line and "]" in line and "<" in line]
    assert nested, "zdarzenia specjalistów mają etykietę [agent]"
    top = next(line for line in text.splitlines() if "ask_data_analyst(" in line)
    child = next(line for line in text.splitlines() if "get_contract_terms(" in line)
    assert len(child) - len(child.lstrip()) > len(top) - len(top.lstrip())


def test_derived_events_are_indented_under_their_parent():
    from kovalo.agents.show import format_trace
    from tests.agent_fakes import T1_QUESTION, event, make_trace, otif_result

    result = otif_result("SUP-114", {"2026-07": 94.6, "2026-08": 93.8, "2026-09": 93.5})
    parent = event(
        "check_termination_condition",
        {"status": "ok", "condition_met": False, "months_below": 2, "otif": result},
        call_id="c1",
        supplier="Mleczna Dolina",
        as_of_month="2026-09",
    )
    derived = event(
        "get_supplier_otif",
        result,
        call_id="c1#get_supplier_otif",
        parent="c1",
        supplier="SUP-114",
        from_month="2026-07",
        to_month="2026-09",
    )
    text = format_trace(make_trace(T1_QUESTION, "Nie.", [parent, derived]))
    lines = text.splitlines()
    p = next(line for line in lines if "check_termination_condition(" in line)
    d = next(line for line in lines if "get_supplier_otif(" in line)
    assert len(d) - len(d.lstrip()) > len(p) - len(p.lstrip())
    assert "<calculation>" in p and "<data>" in d


def test_brief():
    from kovalo.agents.show import brief

    assert brief("surowy tekst") == "surowy tekst"
    passages = {"status": "ok", "passages": [{"chunk_id": "CON-101#ctx004"}, {"chunk_id": "X#1"}]}
    assert brief(passages) == "ok: CON-101#ctx004, X#1"
    assert brief({"summary": "podsumowanie\nwieloliniowe"}).startswith("podsumowanie: ")
    otif = {
        "status": "ok",
        "observations": [{"month": "2026-07", "otif_pct": 94.6}],
        "complete": True,
        "requested": {"from": "2026-07", "to": "2026-07"},
    }
    text = brief(otif)
    assert text.startswith("ok: ") and "2026-07: 94.6" in text
    assert "requested" not in text, "SKIP: pola powtarzające zapytanie"
    with_message = {"status": "no_data", "observations": [], "message": "brak danych za 2026-10"}
    assert brief(with_message).endswith("(brak danych za 2026-10)")


def test_architecture_label():
    from kovalo.agents.show import architecture_label

    assert isinstance(architecture_label("single agent"), str) and architecture_label(
        "single agent"
    )
    assert architecture_label("supervisor + 3 specialists") != architecture_label("single agent")


@pytest.mark.parametrize("status", ["error", "limit"])
def test_failed_trace_shows_status_and_error(status):
    from kovalo.agents.show import format_trace
    from tests.agent_fakes import JULY, make_trace

    trace = make_trace(JULY, None, [], status=status)
    trace.error = "coś poszło nie tak"
    text = format_trace(trace)
    assert f"status: {status} (coś poszło nie tak)" in text
    assert "odpowiedź: (brak)" in text
