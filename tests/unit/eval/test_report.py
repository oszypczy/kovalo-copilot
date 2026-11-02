"""Raporty E1 (FR-6.4; kontrakt F1, 7.4)."""

from __future__ import annotations

import json
import re


def test_markdown_sections_and_polish_numbers(e1_report):
    from kovalo.eval.retrieval.report import render_markdown

    md = render_markdown(e1_report)
    headings = [
        "# E1: wyszukiwanie i koszty (MP1)",
        "## Wyszukiwanie (top-3)",
        "## Wyniki pytanie po pytaniu (MP1)",
        "## Wyniki pytanie po pytaniu (MP2, M01–M10)",  # noqa: RUF001
        "## Koszty miesięczne",
        "## Rozbieżności przejrzane ręcznie",
    ]
    positions = [md.find(h) for h in headings]
    assert all(p >= 0 for p in positions), dict(zip(headings, positions, strict=True))
    assert positions == sorted(positions)
    assert "## Wywołanie LLM-a" not in md  # bez --with-llm
    assert "## Zgodność z Postgresem" not in md  # bez --backend pgvector

    assert e1_report.run_id in md
    assert "Chunki: 47 (średnio 603,7, maks. 798)" in md
    assert "18 dokumentów / 8762 tokeny" in md
    assert "referencja kursu (text-embedding-3-small)" in md
    assert "722,7 / 8988,5" in md
    assert "$17,78 / $69,86 (x3,9)" in md
    assert "$15,58 / $67,65" in md
    assert "0,9" in md
    # brak kropek dziesiętnych w kwotach i w trafieniach
    assert not re.search(r"\$\d+\.\d", md)
    assert not re.search(r"\b0\.\d\b", md)


def test_markdown_marks_hits_in_per_question_tables(e1_report):
    from kovalo.eval.retrieval.report import render_markdown

    md = render_markdown(e1_report)
    assert "✓" in md and "✗" in md  # referencja: Q10 i M03 to pudła
    assert "Komentarz:" in md


def test_course_report_shape(e1_report, session_settings):
    from kovalo.eval.retrieval.report import course_report

    data = course_report(e1_report)
    assert {"corpus", "chunks", "embeddings", "retrieval", "cost"} <= set(data)
    assert data["corpus"]["n_documents"] == 18
    assert data["chunks"]["n_chunks"] == 47
    emb = data["embeddings"]
    assert {"provider", "model", "dimensions", "chunks_not_cached"} <= set(emb)
    assert emb["model"] == session_settings.embeddings.local.model
    assert emb["chunks_not_cached"] == 0
    retrieval = data["retrieval"]
    assert retrieval["k"] == 3
    assert {"results", "hit_rate_at_1", "hit_rate_at_3"} <= set(retrieval)
    first = retrieval["results"][0]
    assert {"query_id", "top_ids", "top_scores", "hit_at_1", "hit_at_3"} <= set(first)
    assert "cost_ours" in data or "cost_ours" in data["cost"]
    json.dumps(data)  # serializowalny


def test_write_reports_files(e1_report, chunks, tmp_path):
    from kovalo.eval.retrieval.e1 import E1Report, LlmCallRecord
    from kovalo.eval.retrieval.report import write_reports

    record = LlmCallRecord(
        date="2026-11-03", provider="google_genai", model="m", answered_by="m",
        question="Pytanie?", context_chunks=["POL-001#000"], answer="Odpowiedź.",
        input_tokens=10, output_tokens=5, estimated_input_tokens=9,
        estimate_tokenizer="o200k_base", assumed_output_tokens=350, seconds=1.5,
    )  # fmt: skip
    report = e1_report.model_copy(update={"llm_call": record})
    out = tmp_path / "nowy" / "katalog"
    paths = write_reports(report, out, chunks=chunks)
    names = {p.name for p in out.iterdir()}
    assert names == {
        "2026-11-03-e1.md",
        "2026-11-03-e1.json",
        "report.json",
        "chunks.jsonl",
        "llm_call.json",
    }
    assert {p.name for p in paths} == names
    assert E1Report.model_validate_json((out / "2026-11-03-e1.json").read_text("utf-8")) == report
    assert len((out / "chunks.jsonl").read_text("utf-8").strip().splitlines()) == 47
    assert "## Wywołanie LLM-a" in (out / "2026-11-03-e1.md").read_text("utf-8")
    assert json.loads((out / "llm_call.json").read_text("utf-8"))["answer"] == "Odpowiedź."


def test_write_reports_without_llm_has_no_llm_file(e1_report, chunks, tmp_path):
    from kovalo.eval.retrieval.report import write_reports

    write_reports(e1_report, tmp_path, chunks=chunks)
    assert not (tmp_path / "llm_call.json").exists()
    assert (tmp_path / "2026-11-03-e1.md").is_file()


def _md_with_reasoning(e1_report, chunks, tmp_path, reasoning):
    import datetime

    from kovalo.eval.retrieval.e1 import LlmCallRecord
    from kovalo.eval.retrieval.report import write_reports

    rec = LlmCallRecord(
        date=datetime.date(2026, 11, 3), provider="google_genai", model="m", answered_by="m",
        question="Q?", context_chunks=["POL-001#000"], answer="A", input_tokens=9,
        output_tokens=677, estimated_input_tokens=8, estimate_tokenizer="o200k_base",
        assumed_output_tokens=350, seconds=1.0, reasoning_tokens=reasoning,
    )  # fmt: skip
    write_reports(e1_report.model_copy(update={"llm_call": rec}), tmp_path, chunks=chunks)
    return (tmp_path / "2026-11-03-e1.md").read_text(encoding="utf-8")


def test_report_llm_section_shows_reasoning_tokens(e1_report, chunks, tmp_path):
    # P3 (kontrakt 12.3): „ta odpowiedź zużyła 677 (w tym myślenie N)”
    md = _md_with_reasoning(e1_report, chunks, tmp_path, 300)
    assert "ta odpowiedź zużyła 677 (w tym myślenie 300)" in md


def test_report_llm_section_without_reasoning_has_no_note(e1_report, chunks, tmp_path):
    md = _md_with_reasoning(e1_report, chunks, tmp_path, None)
    assert "ta odpowiedź zużyła 677" in md
    assert "w tym myślenie" not in md
