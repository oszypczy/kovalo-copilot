"""Raport E5 (kontrakt F5, 10.5; FR-6.4): Markdown, JSON, `assistant_run.<tryb>.json`."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

SECTIONS = [
    "# E5: końcowy asystent (MP5)",
    "## Zgodność MCP z wywołaniem bezpośrednim",
    "## Sprawdzenia",
    "## Tura po turze",
    "## Pytania o dane",
    "## Wywołania MCP",
    "## Tokeny i koszt",
    "## Rozbieżności przejrzane ręcznie",
]


def md_of(report, review=None) -> str:
    from kovalo.eval.assistant.report import render_markdown

    return render_markdown(report, review) if review is not None else render_markdown(report)


def make_review(items=(), *, intro="", notes=()):
    from kovalo.eval.review import Review

    return Review.model_validate(
        {
            "version": 1,
            "reviewed_on": "2026-11-04",
            "intro": intro,
            "items": list(items),
            "notes": list(notes),
        }
    )


def section(md: str, heading: str) -> str:
    start = md.index(heading)
    rest = [md.index(h) for h in SECTIONS if md.find(h) > start]
    return md[start : min(rest)] if rest else md[start:]


def test_markdown_sections_and_polish_numbers(e5_live_report, session_settings):
    md = md_of(e5_live_report)
    positions = [md.index(heading) for heading in SECTIONS]
    assert positions == sorted(positions), "sekcje w kolejności z §10.5"
    checks = section(md, "## Sprawdzenia")
    reference_model = session_settings.reference_llm.model
    assert f"Kurs ({reference_model}, MCP)" in checks, "kolumna „referencja kursu”"
    for column in ("Sprawdzenie", "Odtworzenie", "Na żywo", "Cel"):
        assert column in checks, column
    for fraction in ("25/25", "11/11", "22/22", "19/19", "15/15", "4/4", "1/1"):
        assert fraction in checks, fraction
    assert "4,4 centa" in md, "koszt referencji po polsku (przecinek)"
    plain = md.replace("\u00a0", " ")
    assert "41 993" in plain and ("2866" in plain or "2 866" in plain), "tokeny kursu"
    assert e5_live_report.live.run["run_id"] in md and e5_live_report.replay.run["run_id"] in md
    assert "course-v1" in md and "assistant_understand_system" in md
    assert "agent_system" in md and "24" in md and "180" in md, "budżet agenta 24 kroki / 180 s"
    assert "60" in md and "120" in md, "limity MCP 60 / 120 / 10 s"


def test_parity_section_has_20_rows(e5_live_report):
    md = md_of(e5_live_report)
    parity = section(md, "## Zgodność MCP z wywołaniem bezpośrednim")
    rows = [ln for ln in parity.splitlines() if ln.startswith("| ") and "---" not in ln]
    assert len(rows) == 1 + 20, "nagłówek + 20 wierszy"
    for column in ("Narzędzie", "Argumenty", "Wynik", "Różnice"):
        assert column in rows[0], column
    assert "check_termination_condition" in parity and "search_documents" in parity


def test_per_turn_table_lists_every_turn(e5_live_report, conversations):
    md = md_of(e5_live_report)
    table = section(md, "## Tura po turze")
    for column in ("Rozmowa", "Nr", "Oczekiwany kind", "Rozpoznany kind", "Status", "MCP"):
        assert column in table, column
    for dialogue in conversations:
        assert dialogue.id in table
    first_user = conversations[0].turns[0].user
    assert first_user[:44] in table and first_user[:60] not in table, (
        "wiadomość skrócona do 44 znaków"
    )
    assert "document_question" in table


def test_data_questions_show_the_mcp_trace(e5_live_report):
    md = md_of(e5_live_report)
    data = section(md, "## Pytania o dane")
    assert data.count("MCP") >= 4, "dopisek MCP w każdym pytaniu o dane"
    invocation = [e["invocation_id"] for t in e5_live_report.live.turns for e in t["mcp"]]
    assert invocation and all(i in data for i in invocation), "invocation_id w śladzie"
    assert "calculate_otif_penalty" in data


def test_mcp_calls_section(e5_live_report):
    md = md_of(e5_live_report)
    calls = section(md, "## Wywołania MCP")
    assert "7" in calls and "0" in calls


def test_replay_only_report_has_all_sections_and_no_live_cost(e5_replay_report):
    md = md_of(e5_replay_report)
    for heading in SECTIONS:
        assert heading in md, heading
    assert "25/25" in md and "11/11" in md


def test_report_json_and_run_files(e5_live_report, tmp_path: Path):
    from kovalo.eval.assistant.report import write_reports

    out = tmp_path / "mp5"
    paths = write_reports(e5_live_report, out)
    names = {p.name for p in out.iterdir()}
    assert {p.name for p in paths} <= names
    stamp = date.today().isoformat()
    assert any(n.endswith("-e5.md") for n in names) and any(n.endswith("-e5.json") for n in names)
    assert {"assistant_run.replay.json", "assistant_run.live.json"} <= names
    for mode in ("replay", "live"):
        data = json.loads((out / f"assistant_run.{mode}.json").read_text(encoding="utf-8"))
        assert {"run", "turns", "mcp_captures"} <= set(data), (
            "kształt `run_assistant_eval.py` kursu"
        )
        assert len(data["turns"]) == 25
    full = json.loads(next(out.glob("*-e5.json")).read_text(encoding="utf-8"))
    assert full["replay"]["mode"] == "replay" and full["live"]["mode"] == "live"
    assert len(full["parity"]) == 20 and full["transport"] == "memory"
    assert stamp  # data w nazwie pliku: RRRR-MM-DD-e5


def test_replay_only_writes_no_live_run(e5_replay_report, tmp_path: Path):
    from kovalo.eval.assistant.report import write_reports

    write_reports(e5_replay_report, tmp_path)
    names = {p.name for p in tmp_path.iterdir()}
    assert "assistant_run.replay.json" in names and "assistant_run.live.json" not in names


def test_review_comments_inserted(e5_live_report):
    plain = md_of(e5_live_report)
    review = make_review(
        [
            {
                "variants": ["ours-live"],
                "id": "C2-t2",
                "comment": "Komentarz do C2-t2: ponowne przesłanie bez nowej pauzy.",
            }
        ],
        intro="Wstęp przeglądu E5.",
    )
    reviewed = md_of(e5_live_report, review)
    tail = reviewed[reviewed.index("## Rozbieżności przejrzane ręcznie") :]
    assert "Komentarz do C2-t2: ponowne przesłanie bez nowej pauzy." in tail
    assert "Wstęp przeglądu E5." in tail and "C2-t2" in tail
    assert "Komentarz do C2-t2" not in plain
