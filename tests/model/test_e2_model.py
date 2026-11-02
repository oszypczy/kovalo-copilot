"""E2 z prawdziwym modelem fastembed (marker `model`). Ręcznie: make test-model."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.model

# Sonda architekta 2026-11-02 (kontrakt F2, 9.3): hit@1, hit@3, komplet dowodów, tokeny promptu
OURS_PROBE = {
    "mp1": (9, 10, 9, 874.6),
    "ctx": (10, 10, 9, 649.3),
    "hybrid": (9, 10, 9, 638.5),
}


def test_ours_retrieval_probe_values():
    from kovalo.config import load_settings
    from kovalo.eval.rag.e2 import run_e2

    settings = load_settings()
    report = run_e2(settings, with_llm=False)
    ours = {r.variant: r for r in report.retrieval if r.side == "ours"}
    assert set(ours) == set(OURS_PROBE)
    for variant, (hit1, hit3, evidence, tokens) in OURS_PROBE.items():
        r = ours[variant]
        assert (r.hit_at_1, r.hit_at_3, r.evidence_phrases_complete) == (hit1, hit3, evidence), (
            variant
        )
        assert r.avg_prompt_tokens == tokens, variant
    assert report.llm == []


def test_hybrid_finds_sup_114_with_configured_model():
    # FR-2.5: M03, model z configu: CON-101#ctx000, #ctx003, #ctx001
    from kovalo.config import load_settings
    from kovalo.eval.rag.e2 import run_e2

    report = run_e2(load_settings(), with_llm=False)
    row = next(
        r
        for r in next(
            x for x in report.retrieval if x.side == "ours" and x.variant == "hybrid"
        ).rows
        if r.id == "M03"
    )
    assert row.top_chunk_ids == ["CON-101#ctx000", "CON-101#ctx003", "CON-101#ctx001"]
