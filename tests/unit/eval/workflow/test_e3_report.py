"""Raport E3: Markdown, JSON, dialogues.<tryb>.json (kontrakt F3, 10.5; FR-6.4)."""

from __future__ import annotations

import json
from pathlib import Path

SECTIONS = [
    "# E3: workflow decyzyjny (MP3)",
    "## Etapy",
    "## Tura po turze",
    "## Tokeny i koszt",
    "## Rozbieżności przejrzane ręcznie",
]


def md_of(report, review=None) -> str:
    from kovalo.eval.workflow.report import render_markdown

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


def test_markdown_sections_and_polish_numbers(e3_live_report, session_settings):
    md = md_of(e3_live_report)
    positions = [md.index(heading) for heading in SECTIONS]
    assert positions == sorted(positions), "sekcje w kolejności z §10.5"
    etapy = md[md.index("## Etapy") : md.index("## Tura po turze")]
    assert "Referencja kursu" in etapy and "Odtworzenie" in etapy and "Na żywo" in etapy
    for fraction in ("13/13", "11/11", "8/8"):
        assert fraction in etapy, fraction
    assert "cel" in etapy.lower() or "Cel" in etapy
    assert "1,65 centa" in md, "koszt referencji po polsku (przecinek)"
    assert session_settings.reference_llm.model in md, "nazwa modelu referencyjnego z configu"
    assert e3_live_report.replay.run_id in md and e3_live_report.live.run_id in md
    assert "course-v1" in md


def test_per_turn_table_lists_every_turn(e3_live_report, gold_dialogues):
    md = md_of(e3_live_report)
    section = md[md.index("## Tura po turze") : md.index("## Tokeny i koszt")]
    for dialogue in gold_dialogues:
        assert dialogue.id in section
    first_user = gold_dialogues[0].turns[0].user
    assert first_user[:48] in section, "wiadomość skrócona do 48 znaków"
    assert "caching-fake" in section, "kolumna Model (fallback widoczny)"


def test_replay_only_report_has_no_token_section(e3_replay_report):
    md = md_of(e3_replay_report)
    assert "## Tokeny i koszt" not in md
    assert (
        md.index("## Etapy")
        < md.index("## Tura po turze")
        < md.index("## Rozbieżności przejrzane ręcznie")
    )
    assert "13/13" in md and "11/11" in md


def test_course_dialogues_json_shape(e3_live_report, tmp_path: Path):
    from kovalo.eval.workflow.report import write_reports

    out = tmp_path / "mp3"
    write_reports(e3_live_report, out)
    names = {p.name for p in out.iterdir()}
    assert any(n.endswith("-e3.md") for n in names) and any(n.endswith("-e3.json") for n in names)
    assert {"dialogues.replay.json", "dialogues.live.json"} <= names
    for mode in ("replay", "live"):
        data = json.loads((out / f"dialogues.{mode}.json").read_text(encoding="utf-8"))
        assert {
            "run_id",
            "date",
            "provider",
            "model",
            "policy",
            "score",
            "details",
            "transcript",
        } <= set(data)
        assert len(data["transcript"]) == 13 and data["policy"] == "course-v1"
        assert {"user", "turn", "request", "approvers", "pauses", "status"} <= set(
            data["transcript"][0]
        )
    full = json.loads(next(out.glob("*-e3.json")).read_text(encoding="utf-8"))
    assert full["replay"]["mode"] == "replay" and full["live"]["mode"] == "live"
    assert full["reference"]["route"] == [13, 13]


def test_replay_only_writes_no_live_dialogues(e3_replay_report, tmp_path: Path):
    from kovalo.eval.workflow.report import write_reports

    write_reports(e3_replay_report, tmp_path)
    names = {p.name for p in tmp_path.iterdir()}
    assert "dialogues.replay.json" in names and "dialogues.live.json" not in names


def test_review_comments_inserted(e3_live_report):
    plain = md_of(e3_live_report)
    review = _review(
        [
            {
                "variants": ["ours-live"],
                "id": "D2-t2",
                "comment": "Komentarz do D2-t2: model dobrze zrozumiał zmianę.",
            }
        ],
        intro="Wstęp przeglądu E3.",
    )
    section = md_of(e3_live_report, review)
    section = section[section.index("## Rozbieżności przejrzane ręcznie") :]
    assert "Komentarz do D2-t2: model dobrze zrozumiał zmianę." in section
    assert "Wstęp przeglądu E3." in section and "D2-t2" in section
    assert "Komentarz do D2-t2" not in plain
