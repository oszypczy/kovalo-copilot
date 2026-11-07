"""Adaptery etapów E1-E5: JSON raportu → StageSummary (kontrakt F6, 8.2-8.3).

Fixture'y `tests/fixtures/dashboard/e*.json` to wycinki prawdziwych raportów (generator:
`make_fixtures.py` obok nich). Oczekiwane liczby pochodzą z raportów Markdown etapów
(`reports/mp*/…md`) i z tabeli „Wyniki” w docs/06-evaluation.md, nie z kodu adapterów.
Nazwy wierszy (`MetricRow.name`) mają być po polsku, jak w raportach Markdown, więc wiersze
wyszukujemy po fragmentach nazwy (bez rozróżniania wielkości liter).
"""

from __future__ import annotations

import copy
import datetime

import pytest

from tests.dashboard_fakes import REPORT_PATHS, fixture_json


def make_spec(stage, **extra):
    from kovalo.eval.dashboard.spec import StageSpec

    base = {
        "stage": stage,
        "report": f"{REPORT_PATHS[stage]}.json",
        "markdown": f"{REPORT_PATHS[stage]}.md",
    }
    base.update(extra)
    return StageSpec(**base)


def summarise(stage, data=None, **extra):
    from kovalo.eval.dashboard.stages import ADAPTERS

    return ADAPTERS[stage](
        data if data is not None else fixture_json(stage), make_spec(stage, **extra)
    )


def rows(summary, *needles):
    """Wiersze, w których nazwie są wszystkie fragmenty."""
    found = [r for r in summary.metrics if all(n.lower() in r.name.lower() for n in needles)]
    return found


def row(summary, *needles):
    found = rows(summary, *needles)
    assert found, f"brak wiersza z {needles}; są: {[r.name for r in summary.metrics]}"
    return found[0]


def ours(r):
    """Wartość „nasza”: na żywo, a gdy etap jej nie ma (E1), z przeliczenia."""
    return r.live if r.live is not None else r.replay


def test_adapters_cover_all_stages():
    from kovalo.eval.dashboard.stages import ADAPTERS, TARGETS

    assert set(ADAPTERS) == {"e1", "e2", "e3", "e4", "e5"}
    assert TARGETS, "cele z 06-evaluation są w jednym miejscu"


@pytest.mark.parametrize("stage", ["e1", "e2", "e3", "e4", "e5"])
def test_summary_basics(stage):
    from kovalo.eval.dashboard.models import StageSummary

    spec = make_spec(stage, related=["x/related.md"], notes=["uwaga z dashboard.yaml"])
    from kovalo.eval.dashboard.stages import ADAPTERS

    s = ADAPTERS[stage](fixture_json(stage), spec)
    assert isinstance(s, StageSummary) and s.stage == stage
    assert s.report_md.endswith(f"{REPORT_PATHS[stage]}.md")
    assert s.related_md == ["x/related.md"]
    assert "uwaga z dashboard.yaml" in s.notes
    assert s.headline and stage.upper() in s.headline.upper()
    assert s.title and s.metrics
    assert all(r.name and isinstance(r.met, (bool, type(None))) for r in s.metrics)


def test_adapter_e1():
    s = summarise("e1")
    assert s.date == datetime.date(2026, 11, 2)
    assert s.embedding_model == "intfloat/multilingual-e5-large"
    hit1, hit3 = rows(s, "hit@1"), rows(s, "hit@3")
    assert hit1 and hit3, [r.name for r in s.metrics]
    assert any(r.reference == "0,9" and ours(r) == "0,9" for r in hit1), "kurs 0,9; nasz 0,9"
    assert any(r.reference == "0,9" and ours(r) in ("1,0", "1") for r in hit3), "kurs 0,9; nasz 1,0"
    assert all(r.met is True for r in hit1 + hit3), "cele ≥ 0,8 i ≥ 0,9 spełnione"
    assert row(s, "m03").met is True, "M03 trafione (jedyny model, który trafia SUP-114)"
    cost = row(s, "koszt")
    assert cost.reference == "$17,78 / $69,86" and cost.met is True
    assert cost.target, "cel z 06-evaluation"


def test_adapter_e2():
    s = summarise("e2")
    assert s.date == datetime.date(2026, 11, 2)
    assert s.model == "gemini-3.5-flash-lite"
    assert s.embedding_model == "intfloat/multilingual-e5-large"
    assert s.calls.planned == 48 and s.calls.cached == 42 and s.calls.network == 0
    judge = [r for r in rows(s, "sędzia") if "poprawn" in r.name.lower()]
    assert judge, [r.name for r in s.metrics]
    assert any(r.reference == "8/8" and "9/9" in (r.live or "") for r in judge)
    assert all(r.met is True for r in judge), "cel co najmniej referencja minus 1"
    refusals = row(s, "odmow")
    assert "2/2" in (refusals.reference or "") and "2/2" in (refusals.live or "")
    assert all(r.met is not False for r in s.metrics), "E2 spełnia cele (06-evaluation)"
    assert s.cost.ours_cents is not None or s.cost.reference_cents is not None


def test_adapter_e3():
    s = summarise("e3")
    assert s.date == datetime.date(2026, 11, 3)
    assert s.model == "gemini-3.5-flash-lite"
    routing = row(s, "kierowanie")
    assert (routing.reference, routing.replay, routing.live) == ("13/13", "13/13", "13/13")
    assert routing.target and "12/13" in routing.target and routing.met is True
    fields = row(s, "pola")
    assert (fields.reference, fields.replay, fields.live) == ("11/11", "11/11", "10/11")
    assert fields.met is True, "cel dla pól to odtworzenie 100%; na żywo 10/11 go nie dotyczy"
    for needle, total in (("wniosk", "8/8"), ("zatwierdzaj", "11/11"), ("pauz", "13/13")):
        r = row(s, needle)
        assert (r.reference, r.replay, r.live) == (total, total, total), needle
    assert row(s, "błęd").live == "0"
    assert all(r.met is True for r in s.metrics)
    assert (s.calls.planned, s.calls.cached, s.calls.network) == (14, 14, 0)
    assert s.cost.tokens_in == 6626 and s.cost.tokens_out == 1439
    assert s.cost.ours_cents == pytest.approx(0.5585, abs=0.005)
    assert s.cost.reference_cents == pytest.approx(1.1445, abs=0.005)
    assert s.cost.course_cents == pytest.approx(1.65)
    assert "13/13" in s.headline, "wynik główny w nagłówku"


def test_adapter_e4():
    s = summarise("e4")
    assert s.date == datetime.date(2026, 11, 3)
    assert s.model == "gemini-3.5-flash-lite"
    tools = row(s, "pokrycie")
    assert (tools.reference, tools.live) == ("8/9", "8/9")
    assert tools.target == "9/9" and tools.met is False, "B1: cel 9/9 niespełniony"
    facts = row(s, "fakty")
    assert (facts.reference, facts.live) == ("8/9", "8/9") and facts.met is False
    args = row(s, "argument")
    assert (args.reference, args.live) == ("3/3", "3/3") and args.met is True
    unsourced = row(s, "bez źródła")
    assert unsourced.live.startswith("0") and unsourced.met is True
    claims = row(s, "zatwierdzeni")
    assert claims.live == "0/9" and claims.met is True
    t1 = row(s, "t1")
    assert t1.met is not False, "T1 poprawne (100% narzędzi i faktów)"
    assert s.cost.tokens_in == 62092 and s.cost.tokens_out == 2509
    assert s.cost.ours_cents == pytest.approx(2.49, abs=0.005)
    assert s.cost.reference_cents == pytest.approx(5.79, abs=0.01)
    assert s.cost.course_cents == pytest.approx(3.8)
    assert s.cost.includes_cached is False
    assert s.calls.planned == 22 or s.calls.cached == 24


def test_adapter_e5():
    s = summarise("e5")
    assert s.date == datetime.date(2026, 11, 4)
    assert s.model == "gemini-3.5-flash-lite"
    path = row(s, "ścieżk")
    assert (path.reference, path.replay, path.live) == ("25/25", "25/25", "25/25")
    assert path.target and "24/25" in path.target and path.met is True
    for needle, total in (
        ("zatwierdzaj", "19/19"),
        ("pauz", "22/22"),
        ("wniosk", "15/15"),
        ("pola", "11/11"),
    ):
        r = row(s, needle)
        assert (r.reference, r.live) == (total, total), needle
    data_checks = rows(s, "pytania o dane")
    assert len(data_checks) == 5, "5 sprawdzeń pytań o dane"
    assert {r.live for r in data_checks} == {"4/4", "1/1"}
    mcp = row(s, "wywołania", "mcp")
    assert mcp.live == "7 (0)" and mcp.met is True, "wywołania MCP: 7, nieudanych 0"
    assert all(r.met is True for r in s.metrics), "E5: 100% we wszystkich wymiarach"
    assert (s.calls.planned, s.calls.network, s.calls.cached) == (34, 0, 34)
    assert s.calls.source == "report"
    assert (s.cost.tokens_in, s.cost.tokens_out) == (36347, 3751)
    assert s.cost.ours_cents == pytest.approx(2.0282, abs=0.005)
    assert s.cost.reference_cents == pytest.approx(4.414, abs=0.005)
    assert s.cost.course_cents == pytest.approx(4.4)
    assert s.cost.includes_cached is True, "R1-3: tokeny i koszt E5 liczą też tury z cache"
    assert "25/25" in s.headline


def test_first_live_overrides_calls():
    from kovalo.eval.dashboard.spec import FirstLiveRun

    first = FirstLiveRun(
        run_id="20261104T092904-0f4dd55b",
        network_calls=30,
        cached_calls=4,
        failed_attempts=1,
        seconds=132,
        source="reports/mp5/review.yaml",
    )
    s = summarise("e5", first_live=first)
    assert s.calls.source == "dashboard.yaml"
    assert (s.calls.network, s.calls.cached, s.calls.failed, s.calls.seconds) == (30, 4, 1, 132)
    assert "30" in s.headline and "30 wywołań" in s.headline, "headline z liczb oryginału"
    plain = summarise("e5")
    assert plain.calls.source == "report" and plain.calls.network == 0


@pytest.mark.parametrize(
    ("stage", "key"),
    [("e1", "runs"), ("e2", "retrieval"), ("e3", "replay"), ("e4", "live"), ("e5", "reference")],
)
def test_missing_field_is_data_error(stage, key):
    from kovalo.eval.dashboard.stages import ADAPTERS, DashboardDataError

    data = copy.deepcopy(fixture_json(stage))
    del data[key]
    with pytest.raises(DashboardDataError) as info:
        ADAPTERS[stage](data, make_spec(stage))
    text = str(info.value).lower()
    assert stage in text and key in text, "komunikat podaje etap i ścieżkę brakującego pola"
    assert info.value.stage.lower() == stage and key in str(info.value.path)


def test_data_error_is_not_silent_for_nested_key():
    from kovalo.eval.dashboard.stages import ADAPTERS, DashboardDataError

    data = copy.deepcopy(fixture_json("e5"))
    del data["live"]["totals"]
    with pytest.raises(DashboardDataError) as info:
        ADAPTERS["e5"](data, make_spec("e5"))
    assert "totals" in str(info.value)
