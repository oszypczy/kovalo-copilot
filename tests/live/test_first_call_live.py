"""Test na żywo: prawdziwy Gemini. Uruchamiany ręcznie: `uv run pytest -m live`."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


def test_first_call_live(live_google_key: str | None):
    key = live_google_key
    if not key:
        pytest.skip("brak GOOGLE_API_KEY w środowisku procesu")
    from kovalo.config import load_settings
    from kovalo.llm import get_resilient_chat_model

    settings = load_settings(google_api_key=key)
    model = get_resilient_chat_model("default", settings=settings)
    reply = model.invoke("Przedstaw się jednym zdaniem po polsku.")
    assert reply.text.strip()
    assert reply.usage_metadata is not None
    assert reply.usage_metadata["input_tokens"] > 0
    allowed = [settings.llm.model, *(f.model for f in settings.llm.fallbacks)]
    assert any(reply.response_metadata["model_name"].startswith(m) for m in allowed)
