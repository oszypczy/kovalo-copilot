"""Testy kovalo.llm (kontrakt F0, sekcja 5). Bez sieci: konstruktor Gemini nie łączy się z API."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelRateLimitError,
    OutputParserException,
)
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

from tests.conftest import make_fake


def _settings(tmp_config: Path, **overrides):
    from kovalo.config import load_settings

    return load_settings(tmp_config, google_api_key="test-key", **overrides)


def test_get_chat_model_uses_config_model(settings):
    from kovalo.llm import get_chat_model

    m = get_chat_model(settings=settings)
    assert isinstance(m, ChatGoogleGenerativeAI)
    assert m.model == settings.llm.model
    assert m.max_retries == settings.llm.max_retries
    assert m.timeout == settings.llm.timeout_s


def test_temperature_null_not_passed(tmp_config: Path, patch_config, monkeypatch):
    import kovalo.llm as llm_module
    from kovalo.llm import get_chat_model

    captured: list[dict] = []
    real_init = llm_module.init_chat_model

    def spy(model, **kwargs):
        captured.append(kwargs)
        return real_init(model, **kwargs)

    monkeypatch.setattr(llm_module, "init_chat_model", spy)

    patch_config(tmp_config, "llm.temperature", None)
    get_chat_model(settings=_settings(tmp_config))
    assert "temperature" not in captured[-1]

    patch_config(tmp_config, "llm.temperature", 0.3)
    m = get_chat_model(settings=_settings(tmp_config))
    assert captured[-1]["temperature"] == 0.3
    assert m.temperature == 0.3


def test_params_passed_to_google_model(settings):
    from kovalo.llm import get_chat_model

    m = get_chat_model(settings=settings)
    assert m.thinking_level == settings.llm.params["thinking_level"]


def test_rate_limiter_from_config(settings):
    from kovalo.llm import get_chat_model

    m = get_chat_model(settings=settings)
    assert isinstance(m.rate_limiter, InMemoryRateLimiter)
    assert m.rate_limiter.requests_per_second == settings.llm.rate_limit.requests_per_second


def test_rate_limiter_shared_per_model(settings):
    from kovalo.llm import get_chat_model, get_fallback_models

    a = get_chat_model("default", settings=settings)
    b = get_chat_model("answer", settings=settings)
    assert a.rate_limiter is b.rate_limiter
    fb = get_fallback_models(settings=settings)[0]
    assert fb.rate_limiter is not a.rate_limiter
    assert fb.rate_limiter.requests_per_second == (
        settings.llm.fallbacks[0].rate_limit.requests_per_second
    )


def test_fallback_models_from_config_in_order(tmp_config: Path, patch_config):
    from kovalo.llm import get_fallback_models

    patch_config(
        tmp_config,
        "llm.fallbacks",
        [
            {"provider": "google_genai", "model": "fb-one"},
            {"provider": "google_genai", "model": "fb-two"},
        ],
    )
    models = get_fallback_models(settings=_settings(tmp_config))
    assert [m.model for m in models] == ["fb-one", "fb-two"]


def test_missing_key_raises_missing_config_error():
    from kovalo.config import MissingConfigError, load_settings
    from kovalo.llm import get_chat_model

    with pytest.raises(MissingConfigError, match="GOOGLE_API_KEY"):
        get_chat_model(settings=load_settings())


def test_purpose_override(tmp_config: Path, patch_config):
    from kovalo.llm import get_chat_model

    patch_config(tmp_config, "llm.purposes", {"judge": {"model": "judge-model"}})
    s = _settings(tmp_config)
    assert get_chat_model("judge", settings=s).model == "judge-model"
    assert get_chat_model("default", settings=s).model == s.llm.model
    assert get_chat_model("answer", settings=s).model == s.llm.model


def test_falls_back_on_rate_limit(failing_chat_model):
    from kovalo.llm import with_model_fallbacks

    primary = failing_chat_model(ModelRateLimitError("429"))
    chain = with_model_fallbacks(primary, [make_fake("z fallbacku")])
    assert chain.invoke("x").text == "z fallbacku"
    assert len(primary.calls) == 1


@pytest.mark.parametrize(
    "exc",
    [ModelAPIError("503"), httpx.ReadTimeout("timeout")],
    ids=["server_error", "httpx_timeout"],
)
def test_falls_back_on_server_error_and_timeout(failing_chat_model, exc):
    from kovalo.llm import with_model_fallbacks

    chain = with_model_fallbacks(failing_chat_model(exc), [make_fake("z fallbacku")])
    assert chain.invoke("x").text == "z fallbacku"


@pytest.mark.parametrize(
    "exc",
    [ModelAuthenticationError("401"), ValueError("zle"), OutputParserException("parse")],
    ids=["auth", "value_error", "output_parser"],
)
def test_no_fallback_on_auth_or_validation_error(failing_chat_model, exc):
    from kovalo.llm import with_model_fallbacks

    fallback = failing_chat_model(RuntimeError("nie powinien byc wywolany"))
    chain = with_model_fallbacks(failing_chat_model(exc), [fallback])
    with pytest.raises(type(exc)):
        chain.invoke("x")
    assert fallback.calls == []


def test_empty_fallbacks_returns_primary(fake_chat_model):
    from kovalo.llm import with_model_fallbacks

    assert with_model_fallbacks(fake_chat_model, []) is fake_chat_model


def test_override_chat_model(settings, fake_chat_model):
    from kovalo.llm import (
        get_chat_model,
        get_fallback_models,
        get_resilient_chat_model,
        override_chat_model,
    )

    with override_chat_model(fake_chat_model):
        assert get_chat_model(settings=settings) is fake_chat_model
        assert get_resilient_chat_model(settings=settings) is fake_chat_model
        assert get_fallback_models(settings=settings) == []
    assert isinstance(get_chat_model(settings=settings), ChatGoogleGenerativeAI)
