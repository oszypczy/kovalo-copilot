"""Raporty E2 (FR-6.4; kontrakt F2, 8.8)."""

from __future__ import annotations

import json
import re

import pytest

from tests.unit.eval.rag.conftest import FIXED_NOW

HEADINGS_OFFLINE = [
    "# E2: odpowiedzi ze źródłami (MP2)",
    "## Wyszukiwanie i dowody (poziomy 1–2, bez LLM-a)",  # noqa: RUF001
    "## Wyszukiwanie pytanie po pytaniu",
    "## Tokeny i koszt",
    "## Rekomendacja trybu wyszukiwania",
    "## Rozbieżności przejrzane ręcznie",
]
HEADINGS_LLM = [
    "# E2: odpowiedzi ze źródłami (MP2)",
    "## Wyszukiwanie i dowody (poziomy 1–2, bez LLM-a)",  # noqa: RUF001
    "## Wyszukiwanie pytanie po pytaniu",
    "## Odpowiedzi (poziomy 3–4, z LLM-em)",  # noqa: RUF001
    "## Odpowiedzi pytanie po pytaniu",
    "## Tokeny i koszt",
    "## Rekomendacja trybu wyszukiwania",
    "## Rozbieżności przejrzane ręcznie",
]


def _in_order(md: str, headings: list[str]) -> None:
    positions = [md.find(h) for h in headings]
    assert all(p >= 0 for p in positions), dict(zip(headings, positions, strict=True))
    assert positions == sorted(positions)


def test_markdown_sections_and_polish_numbers(e2_offline_report, session_settings):
    from kovalo.eval.rag.report import render_markdown

    md = render_markdown(e2_offline_report)
    _in_order(md, HEADINGS_OFFLINE)
    assert "## Odpowiedzi (poziomy 3–4, z LLM-em)" not in md  # bez --with-llm  # noqa: RUF001
    assert e2_offline_report.run_id in md
    assert "610,4" in md and "850,3" in md and "631,0" in md
    assert "610.4" not in md and "850.3" not in md and "631.0" not in md
    assert not re.search(r"\$\d+\.\d", md)  # kwoty z przecinkiem
    assert session_settings.reference_llm.model in md  # nazwa modelu referencyjnego z configu
    assert "Komentarz:" in md
    assert "✓" in md and "✗" in md
    assert "zgodne" in md  # referencja kursu odtworzona: "zgodne: tak"
    assert "47" in md and "87" in md and "385,6" in md


def test_markdown_with_llm_section(e2_llm_report, session_settings):
    from kovalo.eval.rag.report import render_markdown

    md = render_markdown(e2_llm_report)
    _in_order(md, HEADINGS_LLM)
    assert f"Kurs ({session_settings.reference_llm.model}" in md
    assert "Wywołania LLM-a" in md
    assert "fake-model" in md  # model, który odpowiedział
    # Sędzia i odpowiedzi to faktycznie ten sam model (atrapa), choć config podaje inny
    assert "Sędzia to ten sam model co odpowiedzi (fake-model)" in md
    assert session_settings.llm.model != "fake-model"
    assert "ocenił: fake-model 10" in md
    assert "| Model sędziego |" in md
    for level in ("sędzia", "odmowy", "odwołania"):
        assert level in md.lower()


def test_course_report_shapes(e2_offline_report, e2_llm_report):
    from kovalo.eval.rag.reference import COURSE_RETRIEVAL
    from kovalo.eval.rag.report import course_live_report, course_retrieval_report

    retrieval = course_retrieval_report(e2_offline_report, "reference")
    assert set(retrieval) == {"k", "n_answerable", "summary", "results"}
    assert retrieval["k"] == 3 and retrieval["n_answerable"] == 10
    assert set(retrieval["summary"]) == {"mp1", "contextual"}  # hybryda poza plikiem kursu
    assert retrieval["summary"]["mp1"] == COURSE_RETRIEVAL["mp1"]
    assert retrieval["summary"]["contextual"] == COURSE_RETRIEVAL["ctx"]
    first = retrieval["results"][0]
    assert {"id", "question", "relevant_docs", "requires_all", "mp1", "contextual"} <= set(first)
    assert set(first["contextual"]) == {
        "top_chunk_ids",
        "hit_at_1",
        "hit_at_3",
        "all_required_found",
        "evidence_phrase_coverage",
        "prompt_tokens",
    }

    result = e2_llm_report.llm[0]
    live = course_live_report(e2_llm_report, result)
    assert {"run_id", "date", "provider", "model", "summary", "results"} <= set(live)
    assert live["run_id"] == e2_llm_report.run_id
    assert live["date"] == FIXED_NOW.date().isoformat()
    assert live["summary"]["questions"] == 12
    row = live["results"][0]
    assert {
        "id",
        "question",
        "answerable",
        "retrieval_hit",
        "answer",
        "citations",
        "error",
        "score",
        "verdict",
    } <= set(row)
    assert set(row["verdict"]) == {"correct", "grounded", "reason", "error"}
    assert "record" not in row  # pełny rekord tylko w answers.*.json
    json.dumps(live)  # serializowalne


def test_save_and_load_run_roundtrip(tmp_path):
    from kovalo.eval.rag.report import load_run, save_run
    from tests.fakes import toy_record

    run = {"run_id": "r1", "date": "2026-11-03", "provider": "p", "model": "m"}
    records = [
        {"id": "M01", "record": toy_record("CFO.", ["TST-001"])},
        {"id": "M02", "record": toy_record(error="ValueError: x")},
    ]
    path = tmp_path / "answers.json"
    save_run(path, run, records)
    loaded_run, loaded = load_run(path)
    assert loaded_run == run
    assert [r["id"] for r in loaded] == ["M01", "M02"]
    assert loaded[0]["record"] == records[0]["record"]
    assert loaded[1]["record"].error == "ValueError: x"
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {"run", "records"}


def test_write_reports_files(e2_llm_report, ctx_chunks, tmp_path):
    from kovalo.eval.rag.report import write_reports

    paths = write_reports(e2_llm_report, tmp_path / "out", ctx_chunks=ctx_chunks)
    names = {p.name for p in paths}
    assert names == {p.name for p in (tmp_path / "out").iterdir()}
    assert {
        "2026-11-03-e2.md",
        "2026-11-03-e2.json",
        "retrieval_report.json",
        "chunks_ctx.jsonl",
        "answers.ours-ctx.json",
        "answers.ours-hybrid.json",
        "live_report.ours-ctx.json",
        "live_report.ours-hybrid.json",
    } <= names
    assert (
        len((tmp_path / "out" / "chunks_ctx.jsonl").read_text(encoding="utf-8").splitlines()) == 87
    )
    assert (
        json.loads((tmp_path / "out" / "2026-11-03-e2.json").read_text(encoding="utf-8"))["run_id"]
        == e2_llm_report.run_id
    )


def test_offline_write_has_no_llm_files(e2_offline_report, ctx_chunks, tmp_path):
    from kovalo.eval.rag.report import write_reports

    write_reports(e2_offline_report, tmp_path, ctx_chunks=ctx_chunks)
    names = {p.name for p in tmp_path.iterdir()}
    assert not [n for n in names if n.startswith(("answers.", "live_report."))]
    data = json.loads((tmp_path / "retrieval_report.json").read_text(encoding="utf-8"))
    assert data["summary"]["contextual"]["n_chunks"] == 87  # strona ours; wartości zależą od modelu


def test_failed_run_replaces_previous(
    session_settings, fake_embeddings_factory, fake_answerer, ctx_chunks, tmp_path
):
    from kovalo.eval.rag.e2 import E2AllErrorsError, run_e2
    from kovalo.eval.rag.report import load_run, write_reports
    from kovalo.rag.records import Answer, Verdict
    from tests.fakes import FakeStructuredChatModel

    common = {
        "variants": ("ctx",),
        "with_llm": True,
        "embeddings_factory": fake_embeddings_factory,
        "clock": lambda: FIXED_NOW,
    }
    first = run_e2(
        session_settings,
        answer_model=fake_answerer.with_structured_output(Answer, include_raw=True),
        judge_model=fake_answerer.with_structured_output(Verdict, include_raw=True),
        **common,
    )
    out = tmp_path / "out"
    write_reports(first, out, ctx_chunks=ctx_chunks)
    _, ok_records = load_run(out / "answers.ours-ctx.json")
    assert not any(r["record"].error for r in ok_records)

    broken = FakeStructuredChatModel(error=ValueError("sztuczny błąd schematu"))
    with pytest.raises(E2AllErrorsError) as info:
        run_e2(
            session_settings,
            answer_model=broken.with_structured_output(Answer, include_raw=True),
            judge_model=broken.with_structured_output(Verdict, include_raw=True),
            **common,
        )
    write_reports(info.value.report, out, ctx_chunks=ctx_chunks)  # zapis także przy samych błędach

    run_info, records = load_run(out / "answers.ours-ctx.json")
    live = json.loads((out / "live_report.ours-ctx.json").read_text(encoding="utf-8"))
    assert live["summary"]["errors"] == len(records) == 12
    assert live["run_id"] == run_info["run_id"]
    assert live["run_id"] != first.run_id  # ta sama sekunda, ale inny przebieg


def _judge_fallback_report(session_settings, fake_embeddings_factory, answer_reply):
    """Odpowiedzi z modelu głównego, wszystkie oceny sędziego z fallbacku (atrapy)."""
    from kovalo.eval.rag.e2 import run_e2
    from kovalo.rag.records import Answer, Verdict
    from tests.fakes import FakeStructuredChatModel

    primary = session_settings.llm.model
    fallback = session_settings.llm.fallbacks[0].model
    answerer = FakeStructuredChatModel(
        results={Answer: answer_reply}, model_name_in_metadata=primary
    )
    judge = FakeStructuredChatModel(
        results={Verdict: Verdict(correct=True, grounded=True, reason="atrapa")},
        model_name_in_metadata=fallback,
    )
    report = run_e2(
        session_settings,
        variants=("ctx",),
        with_llm=True,
        answer_model=answerer.with_structured_output(Answer, include_raw=True),
        judge_model=judge.with_structured_output(Verdict, include_raw=True),
        embeddings_factory=fake_embeddings_factory,
        clock=lambda: FIXED_NOW,
    )
    return report, primary, fallback


def test_report_shows_judge_fallback_from_data(
    session_settings, fake_embeddings_factory, answer_reply
):
    from kovalo.eval.rag.report import render_markdown

    report, primary, fallback = _judge_fallback_report(
        session_settings, fake_embeddings_factory, answer_reply
    )
    assert report.judge_model == primary  # config: sędzia = model główny
    result = report.llm[0]
    assert result.answered_by == {primary: 12}
    assert result.judged_by == {fallback: 10}

    md = render_markdown(report)
    assert "Sędzia to ten sam model co odpowiedzi" not in md  # dane przeczą configowi
    assert f"ocenił: {fallback} 10" in md
    assert f"część ocen sędziego z innego modelu niż {primary} ({fallback})" in md
    assert f"Sędzia to inny model niż odpowiedzi (sędzia: {fallback}; odpowiedzi: {primary})" in md
    table = md[md.index("## Odpowiedzi pytanie po pytaniu") :]
    assert "| Model sędziego |" in table
    judged_line = next(line for line in table.splitlines() if line.startswith("| M01 |"))
    assert judged_line.rstrip().split("|")[-3].strip() == fallback  # kolumna przed „Cache”


def test_report_without_judge_has_no_judge_model_claims(
    session_settings, fake_embeddings_factory, fake_answerer
):
    from kovalo.eval.rag.e2 import run_e2
    from kovalo.eval.rag.report import render_markdown
    from kovalo.rag.records import Answer

    report = run_e2(
        session_settings,
        variants=("ctx",),
        with_llm=True,
        judge=False,
        answer_model=fake_answerer.with_structured_output(Answer, include_raw=True),
        embeddings_factory=fake_embeddings_factory,
        clock=lambda: FIXED_NOW,
    )
    assert report.llm[0].judged_by == {}
    md = render_markdown(report)
    assert "Sędzia to ten sam model" not in md and "Sędzia to inny model" not in md


# ---------------------------------------------------------------------------
# Komentarze z przeglądu ręcznego: review.yaml (kontrakt F2, 14.3)
# ---------------------------------------------------------------------------

PLACEHOLDER = "(do uzupełnienia w T2)"


def _review(items=(), *, intro="", notes=()):
    from kovalo.eval.review import Review

    return Review.model_validate(
        {
            "version": 1,
            "reviewed_on": "2026-11-02",
            "intro": intro,
            "items": list(items),
            "notes": list(notes),
        }
    )


def _review_section(md: str) -> str:
    return md[md.index("## Rozbieżności przejrzane ręcznie") :]


def _row(report, side: str, variant: str, qid: str):
    result = next(r for r in report.llm if (r.side, r.variant) == (side, variant))
    return next(r for r in result.rows if r.id == qid)


def test_review_comment_replaces_placeholder(e2_llm_report):
    from kovalo.eval.rag.report import render_markdown
    from kovalo.eval.review import answer_digest

    plain = _review_section(render_markdown(e2_llm_report))
    review = _review(
        [
            {
                "variants": ["ours-ctx"],
                "id": "M04",
                "comment": "Odmowa uczciwa, ocena trafna.\nDruga linia komentarza.",
            }
        ],
        intro="Przegląd T2 na podstawie answers.ours-*.json.",
        notes=["Odmowy M11, M12: poprawne w obu wariantach."],
    )
    section = _review_section(render_markdown(e2_llm_report, review))
    lines = section.splitlines()

    assert "Przegląd T2 na podstawie answers.ours-*.json." in section
    assert "Komentarze: review.yaml, przegląd 2026-11-02" in section
    assert section.count(PLACEHOLDER) == plain.count(PLACEHOLDER) - 1  # tylko ctx M04 zastąpione
    assert "Odmowa uczciwa, ocena trafna." in section
    # wiele wierszy, wciętych pod punktem pozycji
    first = next(i for i, ln in enumerate(lines) if "Odmowa uczciwa, ocena trafna." in ln)
    assert lines[first].startswith(" ") and lines[first + 1].startswith(" ")
    assert lines[first + 1].strip() == "Druga linia komentarza."
    # komentarz stoi pod pozycją ours-ctx M04 (przed następną pozycją)
    item = max(i for i, ln in enumerate(lines[:first]) if ln.startswith("- "))
    assert re.match(r"- ours[- ]ctx, M04", lines[item])
    # dopisek ze skrótem odpowiedzi w wierszu pozycji
    digest = answer_digest(
        _row(e2_llm_report, "ours", "ctx", "M04").answer,
        _row(e2_llm_report, "ours", "ctx", "M04").error,
    )
    assert f"· odpowiedź #{digest}" in lines[item]
    assert "Uwaga: komentarz dotyczy innej odpowiedzi" not in section
    # notes jako punkty na końcu sekcji
    assert lines[-1].startswith("- ") and "Odmowy M11, M12: poprawne" in lines[-1]


def test_review_comment_in_offline_report(e2_offline_report):
    from kovalo.eval.rag.report import render_markdown

    review = _review([{"variants": ["ours-ctx"], "id": "M01", "comment": "Komentarz offline."}])
    plain = _review_section(render_markdown(e2_offline_report))
    section = _review_section(render_markdown(e2_offline_report, review))
    assert "Komentarz offline." in section
    assert section.count(PLACEHOLDER) == plain.count(PLACEHOLDER) - 1
    assert "odpowiedź #" not in section  # bez LLM-a nie ma odpowiedzi


def test_review_item_without_discrepancy_rendered(e2_llm_report):
    from kovalo.eval.rag.report import render_markdown

    plain = _review_section(render_markdown(e2_llm_report))
    assert not re.search(r"ours[- ]ctx, M11", plain)  # poprawna odmowa: brak rozbieżności
    row = _row(e2_llm_report, "ours", "ctx", "M11")
    review = _review([{"variants": ["ours-ctx"], "id": "M11", "comment": "Odmowa poprawna."}])

    section = _review_section(render_markdown(e2_llm_report, review))
    assert "ours-ctx, M11 (dodane w przeglądzie ręcznym)" in section
    assert "Odmowa poprawna." in section
    assert section.count(PLACEHOLDER) == plain.count(PLACEHOLDER)  # automatyczne bez zmian
    block = section[section.index("ours-ctx, M11 (dodane w przeglądzie ręcznym)") :]
    assert "Komentarze bez pozycji w tym przebiegu" not in section  # wiersz w raporcie istnieje
    # bieżące dane wariantu: pokrycie / sędzia z raportu
    # odmów sędzia nie ocenia (reason = None); uzasadnienie pokazujemy, gdy istnieje
    if row.verdict.reason:
        assert row.verdict.reason in block
    assert row.score.outcome in block  # uzasadnienie sędziego
    assert "pokrycie" in block.lower() and "sędzia" in block.lower()


def test_review_orphan_kept(e2_llm_report):
    from kovalo.eval.rag.report import render_markdown

    # strona reference nie jest w raporcie z LLM-em: wiersza nie ma
    review = _review(
        [{"variants": ["reference-ctx"], "id": "M01", "comment": "Komentarz osierocony."}]
    )
    section = _review_section(render_markdown(e2_llm_report, review))
    assert "Komentarze bez pozycji w tym przebiegu" in section
    assert "Komentarz osierocony." in section
    assert "reference-ctx" in section and "M01" in section


def test_review_stale_digest_warning(e2_llm_report):
    from kovalo.eval.rag.report import render_markdown
    from kovalo.eval.review import answer_digest

    row = _row(e2_llm_report, "ours", "ctx", "M01")
    current = answer_digest(row.answer, row.error)
    stale = "0" * 12 if current != "0" * 12 else "f" * 12
    review = _review(
        [
            {
                "variants": ["ours-ctx"],
                "id": "M01",
                "answer_digest": {"ours-ctx": stale},
                "comment": "Komentarz do starej odpowiedzi.",
            }
        ]
    )
    section = _review_section(render_markdown(e2_llm_report, review))
    assert "Komentarz do starej odpowiedzi." in section  # komentarz zostaje
    warning = f"Uwaga: komentarz dotyczy innej odpowiedzi (#{stale}, teraz #{current}); "
    assert warning + "przejrzyj ponownie." in section

    ok = _review(
        [
            {
                "variants": ["ours-ctx"],
                "id": "M01",
                "answer_digest": {"ours-ctx": current},
                "comment": "Komentarz do bieżącej odpowiedzi.",
            }
        ]
    )
    assert "Uwaga: komentarz dotyczy innej odpowiedzi" not in _review_section(
        render_markdown(e2_llm_report, ok)
    )


def test_render_without_review_unchanged(e2_llm_report, e2_offline_report):
    from kovalo.eval.rag.report import render_markdown

    for report in (e2_llm_report, e2_offline_report):
        md = render_markdown(report)
        assert render_markdown(report, None) == md
        assert render_markdown(report, review=None) == md
        assert "Komentarze: review.yaml" not in md
        assert "odpowiedź #" not in md
        assert "dodane w przeglądzie ręcznym" not in md
        assert PLACEHOLDER in md


def test_render_with_review_idempotent(e2_llm_report, ctx_chunks, tmp_path):
    from kovalo.eval.rag.report import write_reports

    review = _review(
        [
            {"variants": ["ours-ctx", "ours-hybrid"], "id": "M09", "comment": "Przesadzone."},
            {"variants": ["ours-ctx"], "id": "M11", "comment": "Poprawna odmowa."},
            {"variants": ["reference-ctx"], "id": "M01", "comment": "Osierocony."},
        ],
        intro="Wstęp.",
        notes=["Uwaga końcowa."],
    )
    md_name = "2026-11-03-e2.md"
    write_reports(e2_llm_report, tmp_path / "a", ctx_chunks=ctx_chunks, review=review)
    first = (tmp_path / "a" / md_name).read_text(encoding="utf-8")
    write_reports(e2_llm_report, tmp_path / "a", ctx_chunks=ctx_chunks, review=review)
    second = (tmp_path / "a" / md_name).read_text(encoding="utf-8")
    assert first == second
    assert "Przesadzone." in first and "Osierocony." in first and "Uwaga końcowa." in first
    # wiersz z dwiema etykietami: komentarz przy obu wariantach
    assert first.count("Przesadzone.") == 2
