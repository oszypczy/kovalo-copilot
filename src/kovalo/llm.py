"""Fabryka modeli czatu: init_chat_model z configu, rate limiter na model, fallback.

Kontrakt: docs/contracts/f0.md, sekcja 5; decyzje: ADR-0002, ADR-0006.
Jedyny moduł, który importuje integracje dostawców i init_chat_model.
Nazwy modeli pochodzą wyłącznie z config.yaml.
"""

from __future__ import annotations

import contextlib
import importlib.util
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelNotFoundError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable

from kovalo.config import (
    ConfigError,
    MissingConfigError,
    ModelProvider,
    ModelPurpose,
    ModelSpec,
    RateLimitConfig,
    Settings,
    api_key_env_name,
    get_settings,
)

__all__ = [
    "FALLBACK_EXCEPTIONS",
    "get_chat_model",
    "get_fallback_models",
    "get_resilient_chat_model",
    "override_chat_model",
    "reset_rate_limiters",
    "with_model_fallbacks",
]

# Tylko błędy dostępności modelu. 400/401/403, przepełnienie kontekstu i błędy parsowania
# idą wyżej: ten sam klucz albo to samo żądanie zawiodłoby także w fallbacku.
FALLBACK_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ModelRateLimitError,  # 429, wyczerpany limit
    ModelAPIError,  # 5xx
    ModelNotFoundError,  # 404: model wycofany albo zła nazwa
    ModelConnectionError,
    ModelTimeoutError,
    httpx.TimeoutException,
    httpx.TransportError,
)

_API_KEY_HELP: dict[str, str] = {
    "google_genai": "darmowy: https://aistudio.google.com/apikey",
    "openai": "https://platform.openai.com/api-keys",
}

# Rejestr limiterów: jeden na (provider, model) w procesie, wspólny dla wszystkich celów.
_limiters: dict[tuple[str, str], InMemoryRateLimiter] = {}
_limiters_lock = threading.Lock()

ModelOverride = BaseChatModel | Callable[[ModelPurpose], BaseChatModel]
_override: ModelOverride | None = None


def reset_rate_limiters() -> None:
    """Czyści rejestr limiterów (tylko testy)."""
    with _limiters_lock:
        _limiters.clear()


def _limiter_for(provider: str, model: str, rate_limit: RateLimitConfig) -> InMemoryRateLimiter:
    key = (provider, model)
    with _limiters_lock:
        limiter = _limiters.get(key)
        if limiter is None:
            limiter = InMemoryRateLimiter(
                requests_per_second=rate_limit.requests_per_second,
                check_every_n_seconds=rate_limit.check_every_n_seconds,
                max_bucket_size=rate_limit.max_bucket_size,
            )
            _limiters[key] = limiter
        return limiter


def _overridden(purpose: ModelPurpose) -> BaseChatModel | None:
    override = _override
    if override is None:
        return None
    if isinstance(override, BaseChatModel):
        return override
    return override(purpose)


def _require_api_key(settings: Settings, provider: ModelProvider) -> str:
    key = settings.api_key_for(provider)
    if key is None or not key.get_secret_value():
        env_name = api_key_env_name(provider)
        raise MissingConfigError(
            f"Brak {env_name}. Ustaw klucz w .env ({_API_KEY_HELP[provider]})."
        )
    return key.get_secret_value()


def _build_model(
    settings: Settings,
    spec: ModelSpec,
    *,
    temperature: float | None,
) -> BaseChatModel:
    llm = settings.llm
    if spec.provider == "openai" and importlib.util.find_spec("langchain_openai") is None:
        raise ConfigError("Dostawca openai wymaga: uv sync --extra openai")
    api_key = _require_api_key(settings, spec.provider)
    rate_limit = spec.rate_limit or llm.rate_limit
    kwargs: dict[str, Any] = {
        "model_provider": spec.provider,
        "api_key": api_key,
        "max_retries": llm.max_retries,
        "timeout": llm.timeout_s,
        "rate_limiter": _limiter_for(spec.provider, spec.model, rate_limit),
    }
    if temperature is not None:  # null → domyślna temperatura dostawcy
        kwargs["temperature"] = temperature
    if spec.provider == "google_genai":
        kwargs.update(llm.params)
    return init_chat_model(spec.model, **kwargs)


def get_chat_model(
    purpose: ModelPurpose = "default", *, settings: Settings | None = None
) -> BaseChatModel:
    """Model główny z configu (bez fallbacku), z rate limiterem.

    Dla create_agent (F4: fallback przez ModelFallbackMiddleware).
    """
    overridden = _overridden(purpose)
    if overridden is not None:
        return overridden
    settings = settings or get_settings()
    llm = settings.llm
    override = llm.purposes.get(purpose)
    model = override.model if override and override.model else llm.model
    temperature = (
        override.temperature if override and override.temperature is not None else llm.temperature
    )
    spec = ModelSpec(provider=llm.provider, model=model, rate_limit=llm.rate_limit)
    return _build_model(settings, spec, temperature=temperature)


def get_fallback_models(
    purpose: ModelPurpose = "default", *, settings: Settings | None = None
) -> list[BaseChatModel]:
    """Modele z llm.fallbacks w kolejności z configu, każdy z własnym limiterem."""
    if _override is not None:
        return []
    settings = settings or get_settings()
    llm = settings.llm
    override = llm.purposes.get(purpose)
    temperature = (
        override.temperature if override and override.temperature is not None else llm.temperature
    )
    return [_build_model(settings, spec, temperature=temperature) for spec in llm.fallbacks]


def with_model_fallbacks(
    primary: BaseChatModel, fallbacks: Sequence[BaseChatModel]
) -> Runnable[LanguageModelInput, BaseMessage]:
    """primary.with_fallbacks(...) tylko dla błędów dostępności. Pusta lista → primary."""
    if not fallbacks:
        return primary
    return primary.with_fallbacks(list(fallbacks), exceptions_to_handle=FALLBACK_EXCEPTIONS)


def get_resilient_chat_model(
    purpose: ModelPurpose = "default", *, settings: Settings | None = None
) -> Runnable[LanguageModelInput, BaseMessage]:
    """Model główny z fallbackami z configu. Domyślny wybór dla wywołań bez narzędzi."""
    primary = get_chat_model(purpose, settings=settings)
    if _override is not None:
        return primary
    return with_model_fallbacks(primary, get_fallback_models(purpose, settings=settings))


@contextlib.contextmanager
def override_chat_model(model: ModelOverride) -> Iterator[None]:
    """Na czas bloku fabryki zwracają podany model (bez fallbacków). Dla testów i atrap.

    Bez zagnieżdżeń między wątkami: stan jest globalny dla procesu.
    """
    global _override
    previous = _override
    _override = model
    try:
        yield
    finally:
        _override = previous
