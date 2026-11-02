"""Program E2 (kontrakt F2, 8.5-8.6 i 9.1-9.3). Bez sieci: atrapy modeli, cache kursu."""

from __future__ import annotations

import pytest

from tests.fakes import ExplodingModel, FakeStructuredChatModel
from tests.unit.eval.rag.conftest import FIXED_NOW

COURSE_CTX_TOP3 = {
    "M01": ["POL-001#ctx002", "POL-001#ctx004", "POL-001#ctx005"],
    "M02": ["CON-101#ctx003", "CON-101#ctx001", "MTG-403#ctx003"],
    "M03": ["CON-102#ctx004", "CON-102#ctx002", "CON-102#ctx001"],
    "M04": ["PL-601#ctx002", "PL-601#ctx003", "PL-601#ctx004"],
    "M05": ["PRO-202#ctx002", "POL-004#ctx004", "POL-005#ctx000"],
    "M06": ["POL-003#ctx002", "POL-003#ctx000", "POL-003#ctx001"],
    "M07": ["POL-005#ctx000", "MTG-401#ctx004", "MTG-403#ctx003"],
    "M08": ["POL-001#ctx003", "POL-001#ctx004", "POL-001#ctx002"],
    "M09": ["REP-302#ctx002", "MTG-402#ctx002", "REP-302#ctx004"],
    "M10": ["CON-102#ctx004", "CON-102#ctx002", "CON-102#ctx003"],
}


def pick(report, side, variant):
    return next(r for r in report.retrieval if r.side == side and r.variant == variant)


def llm_result(report, variant, side="ours"):
    return next(r for r in report.llm if r.side == side and r.variant == variant)


def answer_judge_models(fake):
    from kovalo.rag.records import Answer, Verdict

    return (
        fake.with_structured_output(Answer, include_raw=True),
        fake.with_structured_output(Verdict, include_raw=True),
    )


def test_reference_reproduces_course_check(e2_offline_report):
    mp1 = pick(e2_offline_report, "reference", "mp1")
    ctx = pick(e2_offline_report, "reference", "ctx")
    assert (mp1.n_chunks, mp1.hit_at_1, mp1.hit_at_3, mp1.evidence_phrases_complete) == (
        47,
        9,
        9,
        9,
    )
    assert mp1.avg_prompt_tokens == 850.3
    assert (ctx.n_chunks, ctx.hit_at_1, ctx.hit_at_3, ctx.evidence_phrases_complete) == (
        87,
        9,
        9,
        8,
    )
    assert ctx.avg_prompt_tokens == 610.4
    assert mp1.embedding_model == ctx.embedding_model

    # wiersze jak w bloku check.py z README kursu: (hit, dowody)
    mp1_rows = {r.id: (r.hit_at_3, r.evidence_phrase_coverage) for r in mp1.rows}
    ctx_rows = {r.id: (r.hit_at_3, r.evidence_phrase_coverage) for r in ctx.rows}
    assert list(mp1_rows) == [f"M{i:02d}" for i in range(1, 11)]
    for qid in mp1_rows:
        assert mp1_rows[qid] == ((False, 0.0) if qid == "M03" else (True, 1.0)), qid
        if qid == "M03":
            assert ctx_rows[qid] == (False, 0.0)
        elif qid == "M10":
            assert ctx_rows[qid] == (True, 0.0)
        else:
            assert ctx_rows[qid] == (True, 1.0), qid

    assert {r.id: r.top_chunk_ids for r in ctx.rows} == COURSE_CTX_TOP3
    assert all(len(r.top_scores) == 3 for r in ctx.rows)
    assert all(s == round(s, 4) for r in ctx.rows for s in r.top_scores)
    assert ctx.all_required_found == (1, 1)  # M02 wymaga CON-101 i MTG-403
    assert e2_offline_report.chunks_not_in_course_cache == {"mp1": 0, "ctx": 0}
    assert e2_offline_report.chunk_stats["mp1"].n_chunks == 47
    stats = e2_offline_report.chunk_stats["ctx"]
    assert (stats.n_chunks, stats.avg_chars, stats.max_chars) == (87, 385.6, 775)
    assert e2_offline_report.k == 3


def test_reference_hybrid_probe_values(e2_offline_report):
    hybrid = pick(e2_offline_report, "reference", "hybrid")
    assert hybrid.n_chunks == 87
    assert (hybrid.hit_at_1, hybrid.hit_at_3, hybrid.evidence_phrases_complete) == (9, 10, 10)
    assert hybrid.avg_prompt_tokens == 631.0
    rows = {r.id: r for r in hybrid.rows}
    assert rows["M03"].top_chunk_ids == ["CON-101#ctx002", "CON-101#ctx003", "CON-101#ctx000"]
    assert rows["M03"].evidence_phrase_coverage == 1.0
    assert "CON-102#ctx000" in rows["M10"].top_chunk_ids
    assert rows["M10"].evidence_phrase_coverage == 1.0


def test_offline_report_covers_three_variants_and_two_sides(e2_offline_report, session_settings):
    keys = {(r.side, r.variant) for r in e2_offline_report.retrieval}
    assert keys == {(s, v) for s in ("reference", "ours") for v in ("mp1", "ctx", "hybrid")}
    ours = pick(e2_offline_report, "ours", "ctx")
    assert ours.embedding_model == session_settings.embeddings.local.model
    assert pick(e2_offline_report, "reference", "ctx").embedding_model == (
        session_settings.embeddings.model
    )
    assert e2_offline_report.configured_mode == "vector"
    assert e2_offline_report.created_at == FIXED_NOW
    assert e2_offline_report.reference_model == session_settings.reference_llm.model
    assert e2_offline_report.provider == session_settings.llm.provider
    assert e2_offline_report.model == session_settings.llm.model


def test_run_id_format(e2_offline_report, session_settings, fake_embeddings_factory):
    import re

    from kovalo.eval.rag.e2 import run_e2

    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", e2_offline_report.run_id)
    again = run_e2(
        session_settings, embeddings_factory=fake_embeddings_factory, clock=lambda: FIXED_NOW
    )
    assert again.run_id != e2_offline_report.run_id  # ta sama sekunda, inny przebieg


def test_openai_cache_mode_skips_ours_side(reference_settings):
    from kovalo.eval.rag.e2 import run_e2

    report = run_e2(
        reference_settings,
        embeddings_factory=lambda spec: pytest.fail("strona ours nie powinna być liczona"),
        clock=lambda: FIXED_NOW,
    )
    assert {r.side for r in report.retrieval} == {"reference"}
    assert len(report.retrieval) == 3


def test_offline_run_calls_no_model(session_settings, fake_embeddings_factory):
    from kovalo.eval.rag.e2 import run_e2

    report = run_e2(
        session_settings,
        with_llm=False,
        answer_model=ExplodingModel(),
        judge_model=ExplodingModel(),
        embeddings_factory=fake_embeddings_factory,
        clock=lambda: FIXED_NOW,
    )
    assert report.llm == []
    assert report.cache is None


def test_with_llm_fake_run(e2_llm_report, session_settings):
    assert {(r.side, r.variant) for r in e2_llm_report.llm} == {("ours", "ctx"), ("ours", "hybrid")}
    for variant in ("ctx", "hybrid"):
        result = llm_result(e2_llm_report, variant)
        assert len(result.rows) == 12
        assert [r.id for r in result.rows] == [f"M{i:02d}" for i in range(1, 13)]
        assert result.calls.answer == 12 and result.calls.judge == 10  # sędzia tylko dla odpowiedzi
        assert result.calls.from_cache == 0 and result.calls.errors == 0
        assert result.summary.refusals_correct == (2, 2)
        assert result.summary.errors == 0 and result.summary.questions == 12
        assert result.answered_by == {"fake-model": 12}
        assert result.judged_by == {"fake-model": 10}  # tylko ocenione odpowiedzi
        judged = [r for r in result.rows if r.verdict.correct is not None]
        assert len(judged) == 10 and all(r.verdict.judged_by == "fake-model" for r in judged)
        assert result.avg_usage.input_tokens == 120
        assert set(result.meets_target) >= {"retrieval_hit", "judge_correct"}
    assert e2_llm_report.llm_calls_planned == 48  # 12 pytań x 2 warianty x 2 (odpowiedź + sędzia)
    assert e2_llm_report.judge_model  # z configu


def test_with_llm_calls_the_expected_models(
    session_settings, fake_embeddings_factory, fake_answerer
):
    from kovalo.eval.rag.e2 import run_e2
    from kovalo.rag.records import Answer, Verdict

    answer_model, judge_model = answer_judge_models(fake_answerer)
    run_e2(
        session_settings,
        variants=("ctx",),
        with_llm=True,
        answer_model=answer_model,
        judge_model=judge_model,
        embeddings_factory=fake_embeddings_factory,
        clock=lambda: FIXED_NOW,
    )
    schemas = [schema for schema, _ in fake_answerer.calls]
    assert schemas.count(Answer) == 12 and schemas.count(Verdict) == 10


def test_no_judge_halves_the_calls(session_settings, fake_embeddings_factory, fake_answerer):
    from kovalo.eval.rag.e2 import run_e2

    answer_model, judge_model = answer_judge_models(fake_answerer)
    report = run_e2(
        session_settings,
        variants=("ctx",),
        with_llm=True,
        judge=False,
        answer_model=answer_model,
        judge_model=judge_model,
        embeddings_factory=fake_embeddings_factory,
        clock=lambda: FIXED_NOW,
    )
    assert report.llm_calls_planned == 12
    result = llm_result(report, "ctx")
    assert result.calls.judge == 0
    assert result.summary.judge_correct == (0, 0)
    assert len(fake_answerer.calls) == 12


def test_llm_side_reference_uses_course_contexts(
    session_settings, fake_embeddings_factory, fake_answerer
):
    from kovalo.eval.rag.e2 import run_e2

    answer_model, judge_model = answer_judge_models(fake_answerer)
    report = run_e2(
        session_settings,
        variants=("ctx",),
        llm_side="reference",
        with_llm=True,
        judge=False,
        ids=["M03", "M10"],
        answer_model=answer_model,
        judge_model=judge_model,
        embeddings_factory=fake_embeddings_factory,
        clock=lambda: FIXED_NOW,
    )
    result = llm_result(report, "ctx", side="reference")
    assert [r.id for r in result.rows] == ["M03", "M10"]
    for row in result.rows:
        assert [c.chunk_id for c in row.record.context] == COURSE_CTX_TOP3[row.id]
        assert row.record.chunks == "ctx" and row.record.retrieval == "vector"
    assert not [r for r in report.llm if r.side == "ours"]


def test_all_errors_raises_with_report(session_settings, fake_embeddings_factory):
    from kovalo.eval.rag.e2 import E2AllErrorsError, run_e2

    failing = FakeStructuredChatModel(error=ValueError("sztuczny błąd"))
    answer_model, judge_model = answer_judge_models(failing)
    with pytest.raises(E2AllErrorsError) as info:
        run_e2(
            session_settings,
            variants=("ctx",),
            with_llm=True,
            answer_model=answer_model,
            judge_model=judge_model,
            embeddings_factory=fake_embeddings_factory,
            clock=lambda: FIXED_NOW,
        )
    report = info.value.report
    result = llm_result(report, "ctx")
    assert result.summary.errors == 12
    assert all(row.record.error.startswith("ValueError") for row in result.rows)
    assert report.retrieval  # poziomy 1-2 policzone mimo błędów LLM-a
    from kovalo.rag.records import Verdict

    assert [schema for schema, _ in failing.calls].count(Verdict) == 0  # błąd nie jest oceniany


def test_cache_only_miss_is_error_record(
    session_settings, fake_embeddings_factory, make_caching_models, tmp_path
):
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.eval.rag.e2 import E2AllErrorsError, run_e2
    from kovalo.rag.records import Answer, Verdict

    strict = JsonFileLlmCache(tmp_path / "llm", mode="strict")
    m_answer, m_judge = make_caching_models(strict)
    with pytest.raises(E2AllErrorsError) as info:
        run_e2(
            session_settings,
            variants=("ctx",),
            with_llm=True,
            cache=strict,
            answer_model=m_answer.with_structured_output(Answer, include_raw=True),
            judge_model=m_judge.with_structured_output(Verdict, include_raw=True),
            embeddings_factory=fake_embeddings_factory,
            clock=lambda: FIXED_NOW,
        )
    rows = llm_result(info.value.report, "ctx").rows
    assert len(rows) == 12 and all(r.record.error.startswith("LlmCacheMiss") for r in rows)
    assert m_answer.sent == [] and m_judge.sent == []  # zero wywołań modelu


def test_second_run_uses_cache(
    session_settings, fake_embeddings_factory, make_caching_models, tmp_path
):
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.eval.rag.e2 import run_e2
    from kovalo.rag.records import Answer, Verdict

    def run(cache):
        m_answer, m_judge = make_caching_models(cache)
        report = run_e2(
            session_settings,
            variants=("ctx",),
            with_llm=True,
            cache=cache,
            answer_model=m_answer.with_structured_output(Answer, include_raw=True),
            judge_model=m_judge.with_structured_output(Verdict, include_raw=True),
            embeddings_factory=fake_embeddings_factory,
            clock=lambda: FIXED_NOW,
        )
        return report, m_answer, m_judge

    first, a1, j1 = run(JsonFileLlmCache(tmp_path / "llm"))
    result = llm_result(first, "ctx")
    assert (result.calls.answer, result.calls.judge, result.calls.from_cache) == (12, 10, 0)
    assert len(a1.sent) == 12 and len(j1.sent) == 10
    assert first.cache == {"hits": 0, "misses": 22}

    second, a2, j2 = run(JsonFileLlmCache(tmp_path / "llm"))  # nowe liczniki, ten sam katalog
    result = llm_result(second, "ctx")
    assert (result.calls.answer, result.calls.judge) == (0, 0)
    assert result.calls.from_cache == 12 + 10
    assert a2.sent == [] and j2.sent == []
    assert second.cache == {"hits": 22, "misses": 0}
    assert result.summary == llm_result(first, "ctx").summary


def test_ids_filter_and_unknown_id(session_settings, fake_embeddings_factory, fake_answerer):
    from kovalo.config import ConfigError
    from kovalo.eval.rag.e2 import run_e2

    answer_model, judge_model = answer_judge_models(fake_answerer)
    kwargs = {
        "variants": ("ctx",),
        "with_llm": True,
        "answer_model": answer_model,
        "judge_model": judge_model,
        "embeddings_factory": fake_embeddings_factory,
        "clock": lambda: FIXED_NOW,
    }
    report = run_e2(session_settings, ids=["M03"], **kwargs)
    assert [r.id for r in llm_result(report, "ctx").rows] == ["M03"]
    assert report.llm_calls_planned == 2
    with pytest.raises(ConfigError, match="M01"):  # komunikat zawiera listę znanych id
        run_e2(session_settings, ids=["X99"], **kwargs)


def test_missing_key_is_config_error_before_any_call(session_settings, fake_embeddings_factory):
    from kovalo.config import ConfigError, load_settings
    from kovalo.eval.rag.e2 import run_e2

    with pytest.raises(ConfigError):
        run_e2(
            load_settings(),
            with_llm=True,
            embeddings_factory=fake_embeddings_factory,
            clock=lambda: FIXED_NOW,
        )


def test_plan_llm_calls():
    from kovalo.eval.rag.e2 import plan_llm_calls

    assert plan_llm_calls(12, 2, judge=True) == 48
    assert plan_llm_calls(12, 2, judge=False) == 24
    assert plan_llm_calls(12, 1, judge=True) == 24
    assert plan_llm_calls(12, 3, judge=True) == 72
    assert plan_llm_calls(2, 2, judge=True) == 8


def _with_ours_metrics(report, ctx, hybrid):
    """Kopia raportu bez LLM-a, z podmienionymi miarami (hit@1, hit@3, dowody) strony ours."""
    updated = []
    for r in report.retrieval:
        if r.side == "ours" and r.variant in ("ctx", "hybrid"):
            h1, h3, ev = ctx if r.variant == "ctx" else hybrid
            r = r.model_copy(
                update={"hit_at_1": h1, "hit_at_3": h3, "evidence_phrases_complete": ev}
            )
        updated.append(r)
    return report.model_copy(update={"retrieval": updated, "llm": []})


@pytest.mark.parametrize(
    ("ctx", "hybrid", "expected"),
    [
        ((10, 10, 9), (10, 10, 9), "vector"),  # remis
        ((9, 9, 8), (9, 10, 8), "hybrid"),  # lepsza w jednej, nie gorsza w żadnej
        ((10, 10, 9), (9, 10, 9), "vector"),  # gorsza w hit@1
        ((9, 9, 8), (10, 9, 7), "vector"),  # lepsza w jednej, gorsza w innej
        ((9, 9, 8), (9, 9, 9), "hybrid"),
    ],
)
def test_recommend_mode_rule(e2_offline_report, ctx, hybrid, expected):
    from kovalo.eval.rag.e2 import recommend_mode

    assert recommend_mode(_with_ours_metrics(e2_offline_report, ctx, hybrid)) == expected


def test_recommend_mode_without_ours_side(e2_offline_report):
    from kovalo.eval.rag.e2 import recommend_mode

    only_reference = e2_offline_report.model_copy(
        update={"retrieval": [r for r in e2_offline_report.retrieval if r.side == "reference"]}
    )
    assert recommend_mode(only_reference) is None


def test_recommend_mode_includes_llm_levels(e2_llm_report):
    from kovalo.eval.rag.e2 import recommend_mode

    base = _with_ours_metrics(e2_llm_report, (9, 9, 8), (9, 10, 8)).model_copy(
        update={"llm": e2_llm_report.llm}
    )
    assert recommend_mode(base) == "hybrid"  # LLM: oba warianty identyczne
    worse = []
    for r in base.llm:
        if r.variant == "hybrid":
            r = r.model_copy(
                update={"summary": r.summary.model_copy(update={"judge_correct": (7, 8)})}
            )
        worse.append(r)
    assert recommend_mode(base.model_copy(update={"llm": worse})) == "vector"


def test_costs_per_variant(e2_offline_report, session_settings):
    costs = {c.variant: c for c in e2_offline_report.costs}
    assert set(costs) == {"mp1", "ctx", "hybrid"}
    assert all(c.ours_actual_monthly_usd is None for c in costs.values())  # bez LLM-a
    assert costs["ctx"].reference_monthly_usd < costs["mp1"].reference_monthly_usd
    assert (
        costs["ctx"].avg_prompt_tokens == pick(e2_offline_report, "ours", "ctx").avg_prompt_tokens
    )
