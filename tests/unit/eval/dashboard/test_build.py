"""Budowa, zapis i sprawdzanie dashboardu (kontrakt F6, 8.4-8.5)."""

from __future__ import annotations

import datetime
import re

import pytest

from kovalo.cli import main


def build(reports_dir, previous=None):
    from kovalo.eval.dashboard import build_dashboard
    from kovalo.eval.dashboard.spec import load_spec

    spec = load_spec(reports_dir / "dashboard.yaml")
    return build_dashboard(spec, reports_dir, previous)


def write(dashboard, reports_dir):
    from kovalo.eval.dashboard import write_dashboard

    write_dashboard(dashboard, reports_dir)


def read_outputs(reports_dir):
    return (
        (reports_dir / "dashboard.md").read_bytes(),
        (reports_dir / "dashboard.json").read_bytes(),
    )


def test_build_dashboard_structure(reports_dir):
    dashboard = build(reports_dir)
    assert dashboard.version == 1
    assert dashboard.as_of == datetime.date(2026, 11, 4), (
        "data najnowszego raportu, nie dnia budowy"
    )
    assert [s.stage for s in dashboard.stages] == ["e1", "e2", "e3", "e4", "e5"]
    assert [g.stage for g in dashboard.gemini_usage] == ["F3", "F4", "F5", "F6"]
    monthly = dashboard.monthly_cost
    assert monthly.reference_rag_usd == pytest.approx(17.78)
    assert monthly.reference_full_usd == pytest.approx(69.86)
    assert monthly.ours_model == "gemini-3.6-flash", "ceny z pomiaru E1 (06-evaluation, Wyniki)"
    assert monthly.ours_rag_usd == pytest.approx(15.58, abs=0.1)
    assert monthly.ours_full_usd == pytest.approx(67.65, abs=0.01)
    e5 = dashboard.stages[-1]
    assert e5.calls.source == "dashboard.yaml" and e5.calls.network == 30, "first_live z yaml"
    assert any("R1-3" in n for n in e5.notes)


def test_idempotent(reports_dir):
    write(build(reports_dir), reports_dir)
    first = read_outputs(reports_dir)
    previous = _previous(reports_dir)
    write(build(reports_dir, previous), reports_dir)
    assert read_outputs(reports_dir) == first, "drugie uruchomienie daje identyczne bajty"
    write(build(reports_dir), reports_dir)
    assert read_outputs(reports_dir) == first, "poprzedni dashboard.json nie zmienia wyniku"


def _previous(reports_dir):
    from kovalo.eval.dashboard.models import Dashboard

    text = (reports_dir / "dashboard.json").read_text(encoding="utf-8")
    return Dashboard.model_validate_json(text)


def test_fresh_clone_without_report_jsons_gives_same_files(reports_dir):
    """W świeżym klonie JSON-y raportów są poza gitem: dane z dashboard.json dają te same pliki."""
    write(build(reports_dir), reports_dir)
    expected = read_outputs(reports_dir)
    previous = _previous(reports_dir)
    for path in reports_dir.glob("mp*/*.json"):
        path.unlink()
    write(build(reports_dir, previous), reports_dir)
    assert read_outputs(reports_dir) == expected


def test_outputs_format(reports_dir):
    write(build(reports_dir), reports_dir)
    md, js = read_outputs(reports_dir)
    assert js.decode("utf-8").startswith("{\n  "), "model_dump_json(indent=2)"
    assert b"\r" not in md and md.endswith(b"\n")


def test_check_dashboard(reports_dir):
    from kovalo.eval.dashboard import write_dashboard
    from kovalo.eval.dashboard.build import check_dashboard

    dashboard = build(reports_dir)
    assert check_dashboard(dashboard, reports_dir), "brak plików wyjściowych = różnica"
    write_dashboard(dashboard, reports_dir)
    assert not check_dashboard(dashboard, reports_dir)
    md = reports_dir / "dashboard.md"
    md.write_text(md.read_text(encoding="utf-8") + "ręczna poprawka\n", encoding="utf-8")
    diffs = check_dashboard(dashboard, reports_dir)
    assert diffs and "dashboard.md" in " ".join(map(str, diffs))
    assert "dashboard.json" not in " ".join(map(str, diffs))
    assert md.read_text(encoding="utf-8").endswith("ręczna poprawka\n"), (
        "--check niczego nie zapisuje"
    )


def test_missing_json_uses_previous(reports_dir, capsys):
    assert main(["eval", "dashboard", "--reports-dir", str(reports_dir)]) == 0
    expected = read_outputs(reports_dir)
    capsys.readouterr()

    (reports_dir / "mp3" / "2026-11-03-e3.json").unlink()
    assert main(["eval", "dashboard", "--reports-dir", str(reports_dir)]) == 0
    err = capsys.readouterr().err
    assert "E3: brak 2026-11-03-e3.json, dane z dashboard.json" in err
    assert read_outputs(reports_dir) == expected, "wpis z previous daje te same liczby"

    (reports_dir / "dashboard.json").unlink()
    assert main(["eval", "dashboard", "--reports-dir", str(reports_dir)]) == 2
    assert "e3" in capsys.readouterr().err.lower(), "bez JSON-u i bez poprzedniego dashboardu: błąd"


def test_missing_json_for_stage_absent_from_previous_is_exit_2(reports_dir, capsys):
    """Poprzedni dashboard.json bez tego etapu nie ratuje brakującego raportu."""
    import json

    assert main(["eval", "dashboard", "--reports-dir", str(reports_dir)]) == 0
    path = reports_dir / "dashboard.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["stages"] = [s for s in data["stages"] if s["stage"] != "e2"]
    path.write_text(json.dumps(data), encoding="utf-8")
    (reports_dir / "mp2" / "2026-11-02-e2.json").unlink()
    capsys.readouterr()
    assert main(["eval", "dashboard", "--reports-dir", str(reports_dir)]) == 2


def test_markdown_sections_in_order(reports_dir):
    write(build(reports_dir), reports_dir)
    md = (reports_dir / "dashboard.md").read_text(encoding="utf-8")
    lines = md.splitlines()
    assert lines[0].startswith("# Dashboard ewaluacji")
    headings = [line for line in lines if line.startswith("## ")]
    wanted = [
        "## Podsumowanie",
        "## E1",
        "## E2",
        "## E3",
        "## E4",
        "## E5",
        "## Koszty",
        "## Zużycie darmowego limitu Gemini",
        "## Znane ograniczenia",
    ]
    assert len(headings) == len(wanted), headings
    for heading, prefix in zip(headings, wanted, strict=True):
        assert heading.startswith(prefix), (heading, prefix)
    intro = "\n".join(lines[: lines.index(headings[0])])
    assert "Stan na: 2026-11-04" in intro
    assert "kovalo eval dashboard" in intro, "polecenie odtworzenia"
    assert "06-evaluation" in intro, "link do 06-evaluation"
    assert re.search(r"generowan", intro, re.IGNORECASE), "nagłówek: plik jest generowany"
    assert re.search(r"ręcznie", intro), "…i nie wolno go poprawiać ręcznie"


def test_markdown_numbers_notes_and_links(reports_dir):
    write(build(reports_dir), reports_dir)
    md = (reports_dir / "dashboard.md").read_text(encoding="utf-8")
    section = lambda name: md[md.index(f"\n## {name}") :].split("\n## ", 2)[1]  # noqa: E731
    assert "$17,78" in md and "$69,86" in md, "polski zapis kwot referencyjnych"
    assert "13/13" in section("E3") and "25/25" in section("E5")
    assert "8/9" in section("E4") and "9/9" in section("E4"), "kurs 8/9, cel 9/9"
    costs = md[md.index("\n## Koszty") :].split("\n## ", 2)[1]
    assert "62 092" in costs and "36 347" in costs, "tokeny z odstępem tysięcy"
    assert "2,03" in costs, "koszt E5 po cenach Gemini w centach, z przecinkiem"
    assert "w darmowym tierze: 0 USD" in costs
    limits = md[md.index("\n## Zużycie darmowego limitu Gemini") :].split("\n## ", 2)[1]
    assert "18" in limits and "47" in limits and "55" in limits
    assert "uzupełnić po H1" in limits
    known = md[md.index("\n## Znane ograniczenia") :]
    assert "R1-3" in known and "B1" in known, "notes wszystkich etapów w jednym miejscu"
    assert "20261104T092904-0f4dd55b" in section("E5") or "20261104T092904-0f4dd55b" in md
    assert "mp4/2026-11-04-e4.md" in section("E4"), "link do related"
    assert "mp5/2026-11-04-e5.md" in md


def test_data_error_in_report_is_exit_2(reports_dir, capsys):
    import json

    path = reports_dir / "mp5" / "2026-11-04-e5.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["reference"]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert main(["eval", "dashboard", "--reports-dir", str(reports_dir)]) == 2
    err = capsys.readouterr().err
    assert "e5" in err.lower() and "reference" in err
    assert not (reports_dir / "dashboard.md").exists(), "błąd danych nie zostawia połowy wyniku"
