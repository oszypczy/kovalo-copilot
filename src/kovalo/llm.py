"""Fabryka modeli: czat (init_chat_model, rate limiter, fallback) i embeddingi (F1).

Kontrakt: docs/contracts/f0.md, sekcja 5, docs/contracts/f1.md, sekcja 6
i docs/contracts/f2.md, sekcja 6 (structured output, cache, params per cel);
decyzje: ADR-0002, ADR-0003, ADR-0006, ADR-0007, ADR-0008.
Jedyny moduł, który importuje integracje dostawców i init_chat_model.
Nazwy modeli pochodzą wyłącznie z config.yaml.
"""

from __future__ import annotations

import contextlib
import importlib.util
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.caches import BaseCache
from langchain_core.embeddings import Embeddings
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
from pydantic import BaseModel

from kovalo.config import (
    ConfigError,
    EmbeddingModelSpec,
    MissingConfigError,
    ModelProvider,
    ModelPurpose,
    ModelSpec,
    RateLimitConfig,
    Settings,
    api_key_env_name,
    get_settings,
)

if TYPE_CHECKING:
    from kovalo.knowledge.embeddings import FastEmbedEmbeddings

__all__ = [
    "FALLBACK_EXCEPTIONS",
    "get_chat_model",
    "get_embeddings",
    "get_fallback_models",
    "get_reference_embeddings",
    "get_resilient_chat_model",
    "get_structured_model",
    "override_chat_model",
    "override_embeddings",
    "reset_embeddings",
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

# Wynik with_structured_output(include_raw=True):
# {"raw": AIMessage, "parsed": ..., "parsing_error": ...}
StructuredResult = dict[str, Any]
StructuredModel = Runnable[LanguageModelInput, StructuredResult]

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
    params: dict[str, Any],
    cache: BaseCache | None = None,
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
    if cache is not None:  # None → bez cache (domyślnie)
        kwargs["cache"] = cache
    if spec.provider == "google_genai":
        kwargs.update(params)
    return init_chat_model(spec.model, **kwargs)


def _purpose_settings(
    settings: Settings, purpose: ModelPurpose
) -> tuple[str, float | None, dict[str, Any]]:
    """(model, temperatura, params) dla celu: llm.* nadpisane przez llm.purposes[purpose]."""
    llm = settings.llm
    override = llm.purposes.get(purpose)
    model = override.model if override and override.model else llm.model
    temperature = (
        override.temperature if override and override.temperature is not None else llm.temperature
    )
    params = {**llm.params, **(override.params if override else {})}
    return model, temperature, params


def get_chat_model(
    purpose: ModelPurpose = "default",
    *,
    settings: Settings | None = None,
    cache: BaseCache | None = None,
) -> BaseChatModel:
    """Model główny z configu (bez fallbacku), z rate limiterem i opcjonalnym cache.

    Dla create_agent (F4: fallback przez ModelFallbackMiddleware).
    """
    overridden = _overridden(purpose)
    if overridden is not None:
        return overridden
    settings = settings or get_settings()
    llm = settings.llm
    model, temperature, params = _purpose_settings(settings, purpose)
    spec = ModelSpec(provider=llm.provider, model=model, rate_limit=llm.rate_limit)
    return _build_model(settings, spec, temperature=temperature, params=params, cache=cache)


def get_fallback_models(
    purpose: ModelPurpose = "default",
    *,
    settings: Settings | None = None,
    cache: BaseCache | None = None,
) -> list[BaseChatModel]:
    """Modele z llm.fallbacks w kolejności z configu, każdy z własnym limiterem."""
    if _override is not None:
        return []
    settings = settings or get_settings()
    _, temperature, params = _purpose_settings(settings, purpose)
    return [
        _build_model(settings, spec, temperature=temperature, params=params, cache=cache)
        for spec in settings.llm.fallbacks
    ]


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


def get_structured_model(
    schema: type[BaseModel],
    purpose: ModelPurpose = "default",
    *,
    settings: Settings | None = None,
    cache: BaseCache | None = None,
) -> Runnable[LanguageModelInput, StructuredResult]:
    """Structured output (include_raw=True) związany z modelem głównym i każdym fallbackiem.

    Fallback tylko dla FALLBACK_EXCEPTIONS (dostępność modelu); błąd parsowania wraca
    w 'parsing_error' i nie przełącza modelu (ADR-0006 pkt 4, ADR-0008 pkt 5).
    Brak klucza → MissingConfigError przy budowie, przed jakimkolwiek wywołaniem.
    """
    overridden = _overridden(purpose)
    if overridden is not None:
        return cast(StructuredModel, overridden.with_structured_output(schema, include_raw=True))
    primary = get_chat_model(purpose, settings=settings, cache=cache)
    fallbacks = get_fallback_models(purpose, settings=settings, cache=cache)
    structured = cast(StructuredModel, primary.with_structured_output(schema, include_raw=True))
    if not fallbacks:
        return structured
    return structured.with_fallbacks(
        [
            cast(StructuredModel, fb.with_structured_output(schema, include_raw=True))
            for fb in fallbacks
        ],
        exceptions_to_handle=FALLBACK_EXCEPTIONS,
    )


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


# ---------------------------------------------------------------------------
# Embeddingi (F1, kontrakt F1 sekcja 6). Bez klucza API: fastembed liczy lokalnie,
# a tryb openai_cache tylko czyta cache kursu.
# ---------------------------------------------------------------------------

EmbeddingsOverride = Embeddings | Callable[[EmbeddingModelSpec | None], Embeddings]
_embeddings_override: EmbeddingsOverride | None = None

# Rejestr modeli fastembed: ładowanie e5-large trwa kilka sekund i zajmuje ok. 2,5 GB RAM,
# więc jedna instancja na (model, katalog modeli, ustawienia onnxruntime) w procesie.
_EmbeddingKey = tuple[EmbeddingModelSpec, Path, int | None, bool]
_fastembed_models: dict[_EmbeddingKey, FastEmbedEmbeddings] = {}
_fastembed_lock = threading.Lock()


def reset_embeddings() -> None:
    """Czyści rejestr modeli fastembed (tylko testy)."""
    with _fastembed_lock:
        _fastembed_models.clear()


def _fastembed_model(settings: Settings, spec: EmbeddingModelSpec) -> FastEmbedEmbeddings:
    from kovalo.knowledge.embeddings import FastEmbedEmbeddings

    local = settings.embeddings.local
    key: _EmbeddingKey = (spec, local.models_dir, local.threads, local.local_files_only)
    with _fastembed_lock:
        model = _fastembed_models.get(key)
        if model is None:
            model = FastEmbedEmbeddings(
                spec,
                models_dir=local.models_dir,
                batch_size=settings.embeddings.batch_size,
                threads=local.threads,
                local_files_only=local.local_files_only,
            )
            _fastembed_models[key] = model
        return model


def get_embeddings(
    *,
    settings: Settings | None = None,
    spec: EmbeddingModelSpec | None = None,
    cache: bool = True,
) -> Embeddings:
    """Model embeddingów z configu (albo kandydat `spec` w E1).

    fastembed: CacheBackedEmbeddings(CountingEmbeddings(FastEmbedEmbeddings)) z cache
    w embeddings.local.cache_dir (cache=False: bez CacheBackedEmbeddings). Każde wywołanie
    daje nowy licznik, model ONNX pochodzi z rejestru procesu.
    openai_cache: cache kursu tylko do odczytu; `spec` → ConfigError.
    """
    override = _embeddings_override
    if override is not None:
        return override if isinstance(override, Embeddings) else override(spec)
    settings = settings or get_settings()
    embeddings = settings.embeddings
    if embeddings.provider == "openai_cache":
        if spec is not None:
            raise ConfigError(
                "Tryb embeddings.provider=openai_cache nie ma kandydatów: porównanie modeli "
                "wymaga provider=fastembed."
            )
        return get_reference_embeddings(settings=settings)

    from kovalo.knowledge.embeddings import CountingEmbeddings, build_cached_embeddings

    spec = spec or embeddings.local.spec()
    counting = CountingEmbeddings(_fastembed_model(settings, spec))
    if not cache:
        return counting
    return build_cached_embeddings(
        counting,
        cache_dir=embeddings.local.cache_dir,
        model=spec.model,
        batch_size=embeddings.batch_size,
    )


def get_reference_embeddings(*, settings: Settings | None = None) -> Embeddings:
    """Zawsze tryb openai_cache (wiersz „referencja kursu” w E1), niezależnie od providera."""
    from kovalo.knowledge.embeddings import build_course_cache_embeddings

    settings = settings or get_settings()
    return build_course_cache_embeddings(
        settings.paths.embedding_cache, model=settings.embeddings.model
    )


@contextlib.contextmanager
def override_embeddings(embeddings: EmbeddingsOverride) -> Iterator[None]:
    """Na czas bloku get_embeddings zwraca atrapę (albo wynik fabryki dla danego `spec`).

    get_reference_embeddings nie jest podmieniany: czyta tylko cache kursu, bez sieci.
    """
    global _embeddings_override
    previous = _embeddings_override
    _embeddings_override = embeddings
    try:
        yield
    finally:
        _embeddings_override = previous
