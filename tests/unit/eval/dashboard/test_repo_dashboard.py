"""Dashboard z repozytorium (`reports/dashboard.*` w gicie): spójność z 06-evaluation i aktualność
(kontrakt F6, 12; to samo sprawdza CI krokiem `kovalo eval dashboard --check`)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
REPORTS = REPO / "reports"
EVALUATION = REPO / "docs" / "06-evaluation.md"


def committed_dashboard():
    from kovalo.eval.dashboard.models import Dashboard

    path = REPORTS / "dashboard.json"
    assert path.is_file(), "reports/dashboard.json musi być w repozytorium (A4)"
    return Dashboard.model_validate_json(path.read_text(encoding="utf-8"))


def results_table() -> dict[str, str]:
    """Wiersze tabeli „Wyniki” z 06-evaluation: etap (E1…E5) → tekst wiersza."""
    text = EVALUATION.read_text(encoding="utf-8")
    section = text[text.index("\n## Wyniki") :].split("\n## ", 2)[1]
    return {
        m.group(1): m.group(0)
        for m in re.finditer(r"^\| (E[1-5]) \|.*$", section, flags=re.MULTILINE)
    }


def metric(dashboard, stage, *needles):
    summary = next(s for s in dashboard.stages if s.stage == stage)
    for r in summary.metrics:
        if all(n.lower() in r.name.lower() for n in needles):
            return r
    raise AssertionError(f"{stage}: brak wiersza {needles}: {[r.name for r in summary.metrics]}")


# (etap, fragmenty nazwy wiersza, pole MetricRow, wartość z tabeli „Wyniki” w 06-evaluation)
MAIN_ROWS = [
    ("e1", ("koszt",), "reference", "$17,78 / $69,86"),
    ("e3", ("kierowanie",), "live", "13/13"),
    ("e3", ("pola",), "live", "10/11"),
    ("e3", ("wniosk",), "live", "8/8"),
    ("e3", ("zatwierdzaj",), "live", "11/11"),
    ("e4", ("pokrycie",), "live", "8/9"),
    ("e4", ("argument",), "live", "3/3"),
    ("e5", ("ścieżk",), "live", "25/25"),
    ("e5", ("pola",), "live", "11/11"),
    ("e5", ("pauz",), "live", "22/22"),
    ("e5", ("zatwierdzaj",), "live", "19/19"),
    ("e5", ("wniosk",), "live", "15/15"),
]


@pytest.mark.parametrize(("stage", "needles", "field", "value"), MAIN_ROWS)
def test_dashboard_matches_evaluation_doc(stage, needles, field, value):
    """Liczba w dashboardzie jest równa liczbie z tabeli „Wyniki” (ta sama w obu miejscach)."""
    table = results_table()
    assert set(table) == {"E1", "E2", "E3", "E4", "E5"}, "tabela „Wyniki” ma 5 etapów"
    assert value in table[stage.upper()], f"{value} nie występuje w wierszu {stage} 06-evaluation"
    assert getattr(metric(committed_dashboard(), stage, *needles), field) == value


def joined(dashboard, stage, field, *rows):
    """Wartości pola z kolejnych wierszy (po fragmentach nazwy), połączone przecinkami."""
    return ", ".join(getattr(metric(dashboard, stage, *needles), field) for needles in rows)


# (etap, pole, wiersze, wartość w dashboardzie, fragment wiersza etapu w „Wyniki” 06-evaluation)
E3_ROWS = [("kierowanie",), ("pola",), ("wniosk",), ("zatwierdzaj",), ("pauz",), ("status",)]
E4_ROWS = [("test 1",), ("test 2",), ("test 3",), ("test 4",), ("test 5",)]
E5_ROWS = [
    ("ścieżk",),
    ("status",),
    ("pola",),
    ("pauz",),
    ("zatwierdzaj",),
    ("wniosk",),
    ("trace",),
]
E5_DATA = [("pokrycie",), ("argument",), ("fakty",), ("pokryciem",), ("akceptacji",)]
E2_ROWS = [("poziom 3: udzielone",), ("poziom 3: wszystkie",), ("sędzia, odpowiedź poprawna",)]
SEQUENCES = [
    (
        "e3",
        "replay",
        E3_ROWS,
        "13/13, 11/11, 8/8, 11/11, 13/13, 13/13",
        "13/13, 11/11, 8/8, 11/11, 13/13, 13/13",
    ),
    (
        "e3",
        "live",
        E3_ROWS,
        "13/13, 10/11, 8/8, 11/11, 13/13, 13/13",
        "pola 10/11, wniosek 8/8, zatwierdzający 11/11, pauzy 13/13, status 13/13",
    ),
    (
        "e4",
        "reference",
        E4_ROWS,
        "8/9, 3/3, 8/9, 0 (0), 0/9",
        "jeden agent 8/9, 3/3, 8/9, 0 (0), 0/9",
    ),
    (
        "e4",
        "live",
        E4_ROWS,
        "8/9, 3/3, 8/9, 0 (0), 0/9",
        "fakty 8/9, liczby bez źródła 0, twierdzenia o zatwierdzeniu 0/9",
    ),
    (
        "e4",
        "reference",
        [("zespół",)],
        "8/9, 3/3, 8/9, 3 (2), 0/9",
        "zespół 8/9, 3/3, 8/9, 3 (2), 0/9",
    ),
    ("e4", "live", [("t1",)], "100% / 100%", "100% narzędzi i faktów"),
    ("e4", "reference", [("t1",)], "67% / 0%", "kurs 67% / 0%"),
    ("e4", "live", [("wywołania narzędzi",)], "21 (0)", "21 wywołań narzędzi bez błędów"),
    (
        "e5",
        "live",
        E5_ROWS,
        "25/25, 25/25, 11/11, 22/22, 19/19, 15/15, 25/25",
        "pola 11/11, pauzy 22/22, zatwierdzający 19/19, wniosek 15/15, trace 25/25",
    ),
    ("e5", "live", E5_DATA, "4/4, 1/1, 4/4, 4/4, 4/4", "pytania o dane 4/4, 1/1, 4/4, 4/4, 4/4"),
    ("e5", "live", [("wywołania mcp",)], "7 (0)", "wywołania MCP 7 (0)"),
    ("e5", "replay", [("zgodność mcp",)], "20/20", "zgodność MCP z wywołaniem bezpośrednim 20/20"),
    (
        "e2",
        "live",
        E2_ROWS,
        "9/10 / 9/10, 9/9 / 9/9, 9/9 / 9/9",
        "odpowiedzi 9/10 / 9/10, fakty 9/9 / 9/9, sędzia 9/9 / 9/9",
    ),
    ("e2", "live", [("odmowy",)], "2/2 / 2/2", "odmowy 2/2 / 2/2"),
    ("e1", "replay", [("hit@1 (mp1",), ("hit@3 (mp1",)], "0,9, 1,0", "MP1 0,9 / 1,0"),
    ("e1", "replay", [("hit@1 (mp2",), ("hit@3 (mp2",)], "0,9, 1,0", "MP2 0,9 / 1,0"),
    ("e1", "reference", [("hit@1 (mp1",), ("hit@3 (mp1",)], "0,9, 0,9", "0,9 / 0,9 w MP1 i MP2"),
]


@pytest.mark.parametrize(("stage", "field", "rows", "value", "fragment"), SEQUENCES)
def test_dashboard_number_sequences_match_evaluation_doc(stage, field, rows, value, fragment):
    """Całe ciągi liczb z wiersza etapu w „Wyniki” są w dashboardzie (kolejność, jednostki)."""
    assert fragment in results_table()[stage.upper()], f"{fragment!r} nie ma w wierszu {stage}"
    assert joined(committed_dashboard(), stage, field, *rows) == value


def test_dashboard_monthly_cost_deviation_is_explained():
    """Jedyna zamierzona różnica liczbowa wobec 06-evaluation ($15,58 vs $15,66) jest w notes."""
    table = results_table()["E1"]
    dashboard = committed_dashboard()
    ours = metric(dashboard, "e1", "gemini-3.6-flash").replay
    assert ours == "$15,58 / $67,65" and "$15,66 / $67,65" in table and ours not in table
    notes = " ".join(next(s for s in dashboard.stages if s.stage == "e1").notes)
    assert "$15,66" in notes
    assert "677" in table and "677" in notes and "959" in notes, "677 vs 959 wyjaśnione"


def test_dashboard_stage_dates_match_evaluation_doc():
    table = results_table()
    dashboard = committed_dashboard()
    for summary in dashboard.stages:
        assert summary.date.isoformat() in table[summary.stage.upper()], summary.stage


def test_dashboard_e3_e4_e5_costs_match_evaluation_doc():
    table = results_table()
    stages = {s.stage: s for s in committed_dashboard().stages}
    e3, e4 = stages["e3"].cost, stages["e4"].cost
    assert "0,56 centa" in table["E3"] and e3.ours_cents == pytest.approx(0.56, abs=0.01)
    assert "2,49 centa" in table["E4"] and e4.ours_cents == pytest.approx(2.49, abs=0.01)
    assert e4.reference_cents == pytest.approx(5.79, abs=0.01)
    assert "30 wywołań" in table["E5"]
    assert stages["e5"].calls.source == "dashboard.yaml" and stages["e5"].calls.network == 30


def test_committed_dashboard_is_current():
    """To samo co CI: `check_dashboard` na plikach z repozytorium (w CI bez JSON-ów raportów,
    dane etapów z dashboard.json) nie zgłasza różnic."""
    from kovalo.eval.dashboard import build_dashboard
    from kovalo.eval.dashboard.build import check_dashboard
    from kovalo.eval.dashboard.spec import load_spec

    spec = load_spec(REPORTS / "dashboard.yaml")
    previous = committed_dashboard()
    dashboard = build_dashboard(spec, REPORTS, previous)
    assert not check_dashboard(dashboard, REPORTS)
