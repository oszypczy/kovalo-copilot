"""Raport E4: Markdown, JSON, agent_run.<tryb>.json (kontrakt F4, 11.4; FR-6.4)."""

from __future__ import annotations

import json
from pathlib import Path

SECTIONS = [
    "# E4: agent z narzędziami (MP4)",
    "## Testy",
    "## Pytanie po pytaniu",
    "## T1: warunek rozwiązania umowy",
    "## Ścieżki dowodowe",
    "## Jeden agent czy zespół?",
    "## Tokeny i koszt",
    "## Rozbieżności przejrzane ręcznie",
]


def md_of(report, review=None) -> str:
    from kovalo.eval.trace.report import render_markdown

    return render_markdown(report, review) if review is not None else render_markdown(report)


def _review(items=(), *, intro="", notes=()):
    from kovalo.eval.review import Review

    return Review.model_validate(
        {
            "version": 1,
            "reviewed_on": "2026-11-03",
            "intro": intro,
            "items": list(items),
            "notes": list(notes),
        }
    )


def test_markdown_sections_and_polish_numbers(e4_live_report, session_settings):
    md = md_of(e4_live_report)
    positions = [md.index(heading) for heading in SECTIONS]
    assert positions == sorted(positions), "sekcje w kolejności z 11.4"
    tests = md[md.index("## Testy") : md.index("## Pytanie po pytaniu")]
    for column in ("Kurs: jeden agent", "Kurs: zespół", "Odtworzenie", "Na żywo", "Cel"):
        assert column in tests, column
    assert "8/9" in tests and "3/3" in tests and "9/9" in tests, "liczby po polsku: n/m"
    assert session_settings.reference_llm.model in md, "model referencyjny z configu"
    assert e4_live_report.live.run["run_id"] in md and e4_live_report.replay.run["run_id"] in md
    assert "check_termination_condition" in md and "agent_system" in md
    assert "24" in md and "180" in md, "budżet 24 kroki / 180 s"
    assert "--max-calls" in md, "nagłówek: limit wywołań"
    assert session_settings.llm.model in md, "model na żywo"


def test_per_question_table_lists_every_question(e4_live_report):
    md = md_of(e4_live_report)
    section = md[md.index("## Pytanie po pytaniu") : md.index("## T1: warunek rozwiązania umowy")]
    for qid in ("P1", "P2", "P3", "T1", "A1", "D1", "S1", "N1", "B1"):
        assert qid in section
    assert "scripted-model" in section or "answered" in section


def test_t1_section_shows_the_derived_read_and_the_course_answer(e4_live_report):
    md = md_of(e4_live_report)
    section = md[md.index("## T1: warunek rozwiązania umowy") : md.index("## Ścieżki dowodowe")]
    assert "check_termination_condition(" in section and "get_supplier_otif(" in section
    assert "94,6" in section
    assert "↳" in section or "    " in section, "składowy odczyt wcięty pod rodzicem"


def test_course_save_traces_shape(e4_live_report, tmp_path: Path):
    from kovalo.agents.trace import load_traces
    from kovalo.eval.trace.report import write_reports

    out = tmp_path / "mp4"
    write_reports(e4_live_report, out)
    names = {p.name for p in out.iterdir()}
    assert any(n.endswith("-e4.md") for n in names) and any(n.endswith("-e4.json") for n in names)
    assert {"agent_run.replay.json", "agent_run.live.json"} <= names
    for mode in ("replay", "live"):
        run, rows = load_traces(out / f"agent_run.{mode}.json")
        assert len(rows) == 9 and all("score" in row for row in rows)
        assert {"run_id", "date", "provider", "model", "architecture", "budget"} <= set(run)
    full = json.loads(next(out.glob("*-e4.json")).read_text(encoding="utf-8"))
    assert full["replay"]["mode"] == "replay" and full["live"]["mode"] == "live"
    assert full["questions"] == ["P1", "P2", "P3", "T1", "A1", "D1", "S1", "N1", "B1"]
    from kovalo.eval.trace.e4 import E4Report

    assert E4Report.model_validate(full).live is not None


def test_replay_only_writes_no_live_run(e4_replay_report, tmp_path: Path):
    from kovalo.eval.trace.report import write_reports

    write_reports(e4_replay_report, tmp_path)
    names = {p.name for p in tmp_path.iterdir()}
    assert "agent_run.replay.json" in names and "agent_run.live.json" not in names
    md = md_of(e4_replay_report)
    assert md.index("## Testy") < md.index("## Pytanie po pytaniu")
    assert "8/9" in md and "3/3" in md


def test_review_comments_inserted(e4_live_report):
    plain = md_of(e4_live_report)
    review = _review(
        [
            {
                "variants": ["ours-live"],
                "id": "T1",
                "comment": "Komentarz do T1: agent użył check_termination_condition.",
            }
        ],
        intro="Wstęp przeglądu E4.",
    )
    section = md_of(e4_live_report, review)
    section = section[section.index("## Rozbieżności przejrzane ręcznie") :]
    assert "Komentarz do T1: agent użył check_termination_condition." in section
    assert "Wstęp przeglądu E4." in section and "T1" in section
    assert "Komentarz do T1" not in plain
