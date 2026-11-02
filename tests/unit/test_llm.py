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


# ---------------------------------------------------------------------------
# F1: fabryka embeddingów (kontrakt F1, sekcja 6)
# ---------------------------------------------------------------------------


def test_reset_embeddings_exists():
    import kovalo.llm as llm

    assert callable(llm.reset_embeddings)
    assert callable(llm.get_embeddings) and callable(llm.get_reference_embeddings)


def test_get_embeddings_fastembed_wiring(local_settings, fake_text_embedding):
    from langchain_classic.embeddings import CacheBackedEmbeddings

    from kovalo.knowledge.embeddings import (
        CountingEmbeddings,
        FastEmbedEmbeddings,
        cache_namespace,
    )
    from kovalo.llm import get_embeddings

    emb = get_embeddings(settings=local_settings)  # bez klucza API
    assert isinstance(emb, CacheBackedEmbeddings)
    counting = emb.underlying_embeddings
    assert isinstance(counting, CountingEmbeddings)
    assert isinstance(counting.inner, FastEmbedEmbeddings)  # nazwa atrybutu z parametru `inner`
    emb.embed_documents(["alpha"])
    ns = cache_namespace(local_settings.embeddings.local.model)
    cache_dir = local_settings.embeddings.local.cache_dir
    assert len(list((cache_dir / ns / "passage").iterdir())) == 1
    assert counting.documents_embedded == 1
    init = fake_text_embedding.inits[-1]
    assert init["cache_dir"] == str(local_settings.embeddings.local.models_dir)


def test_get_embeddings_without_cache(local_settings, fake_text_embedding):
    from langchain_classic.embeddings import CacheBackedEmbeddings

    from kovalo.llm import get_embeddings

    emb = get_embeddings(settings=local_settings, cache=False)
    assert not isinstance(emb, CacheBackedEmbeddings)


def test_get_embeddings_uses_given_spec(local_settings, fake_text_embedding):
    from kovalo.config import EmbeddingModelSpec
    from kovalo.llm import get_embeddings

    spec = EmbeddingModelSpec(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", dimensions=384
    )
    emb = get_embeddings(settings=local_settings, spec=spec, cache=False)
    assert len(emb.embed_query("x")) == 384


def test_get_embeddings_instances_are_registered(local_settings, fake_text_embedding):
    from kovalo.llm import get_embeddings, reset_embeddings

    get_embeddings(settings=local_settings, cache=False)
    get_embeddings(settings=local_settings, cache=False)
    assert len(fake_text_embedding.inits) == 1  # model ładowany raz na proces
    reset_embeddings()
    get_embeddings(settings=local_settings, cache=False)
    assert len(fake_text_embedding.inits) == 2


def test_get_embeddings_openai_cache(reference_settings):
    from kovalo.config import ConfigError, EmbeddingModelSpec
    from kovalo.llm import get_embeddings

    emb = get_embeddings(settings=reference_settings)
    question = "Kto musi zatwierdzić rabat 15% dla dużej sieci handlowej?"
    assert len(emb.embed_query(question)) == 1536
    spec = EmbeddingModelSpec(model="x/y", dimensions=8)
    with pytest.raises(ConfigError, match="openai_cache"):
        get_embeddings(settings=reference_settings, spec=spec)


def test_get_reference_embeddings_ignores_provider(settings):
    from kovalo.llm import get_reference_embeddings

    assert settings.embeddings.provider == "fastembed"
    emb = get_reference_embeddings(settings=settings)
    question = "Kto musi zatwierdzić rabat 15% dla dużej sieci handlowej?"
    assert len(emb.embed_query(question)) == 1536


def test_override_embeddings(local_settings, fake_embeddings, fake_text_embedding):
    from kovalo.llm import get_embeddings, override_embeddings

    with override_embeddings(fake_embeddings):
        assert get_embeddings(settings=local_settings) is fake_embeddings
    assert get_embeddings(settings=local_settings, cache=False) is not fake_embeddings

    seen: list[object] = []

    def factory(spec):  # type: ignore[no-untyped-def]
        seen.append(spec)
        return fake_embeddings

    with override_embeddings(factory):
        assert get_embeddings(settings=local_settings) is fake_embeddings
    assert seen == [None]


# ---------------------------------------------------------------------------
# F2: structured output z fallbackiem, params per cel, cache (kontrakt F2, 6)
# ---------------------------------------------------------------------------


def test_structured_model_with_fallbacks(settings):
    from langchain_core.runnables.fallbacks import RunnableWithFallbacks

    from kovalo.llm import FALLBACK_EXCEPTIONS, get_structured_model
    from kovalo.rag.records import Answer

    model = get_structured_model(Answer, "answer", settings=settings)  # bez wywołań sieciowych
    assert isinstance(model, RunnableWithFallbacks)
    assert model.exceptions_to_handle == FALLBACK_EXCEPTIONS
    assert len(model.fallbacks) == 1 == len(settings.llm.fallbacks)


def test_structured_model_override():
    from kovalo.llm import get_structured_model, override_chat_model
    from kovalo.rag.records import Answer
    from tests.fakes import FakeStructuredChatModel

    fake = FakeStructuredChatModel(
        results={Answer: Answer(answer="x", citations=[], answerable=True)}
    )
    with override_chat_model(fake):
        model = get_structured_model(Answer, "answer")
        result = model.invoke("pytanie")
    assert set(result) == {"raw", "parsed", "parsing_error"}
    assert result["parsed"] == Answer(answer="x", citations=[], answerable=True)
    assert result["parsing_error"] is None
    assert fake.schemas == [Answer] and fake.include_raw_flags == [True]


def test_structured_model_missing_key():
    from kovalo.config import MissingConfigError, load_settings
    from kovalo.llm import get_structured_model
    from kovalo.rag.records import Answer

    with pytest.raises(MissingConfigError, match="GOOGLE_API_KEY"):
        get_structured_model(Answer, "answer", settings=load_settings())


def test_purpose_params_passed(tmp_config: Path, patch_config, monkeypatch):
    import kovalo.llm as llm_module
    from kovalo.llm import get_chat_model

    captured: list[dict] = []
    real_init = llm_module.init_chat_model

    def spy(model, **kwargs):
        captured.append(kwargs)
        return real_init(model, **kwargs)

    monkeypatch.setattr(llm_module, "init_chat_model", spy)
    patch_config(tmp_config, "llm.purposes", {"answer": {"params": {"thinking_level": "minimal"}}})
    s = _settings(tmp_config)

    answer = get_chat_model("answer", settings=s)
    assert captured[-1]["thinking_level"] == "minimal"
    assert answer.thinking_level == "minimal"
    default = get_chat_model("default", settings=s)
    assert captured[-1]["thinking_level"] == s.llm.params["thinking_level"] == "low"
    assert default.thinking_level == "low"


def test_purpose_params_passed_to_fallbacks(tmp_config: Path, patch_config):
    from kovalo.llm import get_fallback_models

    patch_config(tmp_config, "llm.purposes", {"answer": {"params": {"thinking_level": "minimal"}}})
    s = _settings(tmp_config)
    models = get_fallback_models("answer", settings=s)
    assert models and all(getattr(m, "thinking_level", None) == "minimal" for m in models)


def test_cache_passed_to_models(settings):
    from langchain_core.caches import InMemoryCache

    from kovalo.llm import get_chat_model, get_fallback_models

    cache = InMemoryCache()
    assert get_chat_model(settings=settings, cache=cache).cache is cache
    fallbacks = get_fallback_models(settings=settings, cache=cache)
    assert fallbacks and all(m.cache is cache for m in fallbacks)
    assert get_chat_model(settings=settings).cache is None  # domyślnie bez cache


def test_llm_string_stable_and_secret_free(tmp_config: Path):
    from kovalo.config import load_settings
    from kovalo.llm import get_chat_model

    secret = "AIza-super-tajny-klucz-123"
    a = get_chat_model(settings=load_settings(tmp_config, google_api_key=secret))
    b = get_chat_model(settings=load_settings(tmp_config, google_api_key=secret))
    assert a._get_llm_string() == b._get_llm_string()
    assert secret not in a._get_llm_string()
