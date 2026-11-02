"""E1 z prawdziwym wywołaniem Gemini (marker live). Ręcznie: uv run pytest -m live."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


def test_e1_with_llm_live(live_google_key: str | None):
    if not live_google_key:
        pytest.skip("brak GOOGLE_API_KEY w środowisku procesu")
    from kovalo.config import load_settings
    from kovalo.eval.retrieval.e1 import run_e1

    settings = load_settings(google_api_key=live_google_key)
    report = run_e1(settings, specs=[], include_reference=True, with_llm=True)
    assert report.llm_call is not None
    assert report.llm_call.input_tokens > 0
    assert report.llm_call.answer.strip()
