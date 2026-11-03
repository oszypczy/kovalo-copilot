"""Agent z prawdziwym Gemini (marker live): P1, najwyżej 4 wywołania (kontrakt F4, 12 i 15.2).

Ręcznie: make test-live T=tests/live/test_e4_live.py
Bez cache LLM-a; `CallMeter(max_calls=4)` pilnuje limitu. Wyszukiwanie na lokalnych
embeddingach fastembed (bez kosztu). Brak klucza → skip.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


def test_p1_live(live_google_key: str | None):
    if not live_google_key:
        pytest.skip("brak GOOGLE_API_KEY w środowisku procesu")
    from kovalo.agents.middleware import CallMeter
    from kovalo.config import load_settings
    from kovalo.eval.datasets import load_agent_gold
    from kovalo.eval.trace.scoring import score_trace
    from kovalo.knowledge.retrievers import build_memory_retriever
    from kovalo.llm import get_embeddings
    from kovalo.workflow.factory import build_agent_runner

    settings = load_settings(google_api_key=live_google_key)
    gold = {g.id: g for g in load_agent_gold(settings.paths.mini_project_4_questions)}
    retriever = build_memory_retriever(settings, get_embeddings(settings=settings))
    meter = CallMeter(max_calls=4)
    runner = build_agent_runner(
        settings, retriever=retriever, extra_middleware=[meter]
    )  # bez cache
    trace = runner(gold["P1"].question, question_id="P1")  # wywołania 1-4
    assert trace.status == "answered", trace.error
    penalty = [e for e in trace.events if e.tool == "calculate_otif_penalty" and e.status == "ok"]
    assert penalty and penalty[-1].result["penalty_grosze"] == 729600
    assert score_trace(trace, gold["P1"])["arguments_correct"] is True
    assert meter.network <= 4 and meter.errors == 0
    assert trace.model_calls and trace.model_calls[0].input_tokens > 0
