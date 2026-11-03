"""Middleware agenta: fallback tylko przy błędach dostępności, stabilne usage, licznik wywołań.

Kontrakt: docs/contracts/f4.md, sekcja 7.2 (P1, P4); ADR-0006, ADR-0011 pkt 5.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.caches import BaseCache
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.errors import GraphBubbleUp

__all__ = [
    "AvailabilityFallbackMiddleware",
    "CallBudgetExhausted",
    "CallMeter",
    "Middleware",
    "StableUsageMiddleware",
]

Middleware = (
    AgentMiddleware  # reeksport typu (P4): workflow/ i eval/ nie importują langchain.agents
)

Handler = Callable[[ModelRequest], Any]


class AvailabilityFallbackMiddleware(AgentMiddleware):
    """Jak ModelFallbackMiddleware, ale przełącza model tylko przy wyjątkach z `exceptions`
    (produkcja: llm.FALLBACK_EXCEPTIONS, ADR-0006). Inny wyjątek (400, parsowanie, LlmCacheMiss,
    CallBudgetExhausted) idzie wyżej bez próby fallbacku. GraphBubbleUp zawsze wyżej."""

    def __init__(
        self,
        models: Sequence[BaseChatModel],
        *,
        exceptions: tuple[type[BaseException], ...],
    ) -> None:
        super().__init__()
        self.models = list(models)
        self.exceptions = exceptions

    def wrap_model_call(self, request: ModelRequest, handler: Handler) -> Any:
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except self.exceptions as exc:
            last: BaseException = exc
        for model in self.models:
            try:
                return handler(request.override(model=model))
            except GraphBubbleUp:
                raise
            except self.exceptions as exc:
                last = exc
        raise last

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except self.exceptions as exc:
            last: BaseException = exc
        for model in self.models:
            try:
                return await handler(request.override(model=model))
            except GraphBubbleUp:
                raise
            except self.exceptions as exc:
                last = exc
        raise last


def _stable(message: BaseMessage) -> BaseMessage:
    if not isinstance(message, AIMessage) or not message.usage_metadata:
        return message
    usage = dict(message.usage_metadata)
    if "total_cost" not in usage:
        return message
    usage.pop("total_cost")
    return message.model_copy(update={"usage_metadata": usage or None})


def _stable_response(response: Any) -> Any:
    if isinstance(response, AIMessage):
        return _stable(response)
    if isinstance(response, ModelResponse):
        response.result = [_stable(m) for m in response.result]
    return response


class StableUsageMiddleware(AgentMiddleware):
    """Najbardziej wewnętrzne middleware (P1): z każdej AIMessage zwróconej przez model usuwa
    usage_metadata['total_cost'] (LangChain dopisuje total_cost = 0 przy trafieniu w cache),
    zanim wiadomość trafi do stanu agenta. Historia tury 2+ jest wtedy identyczna w przebiegu
    świeżym i z cache, więc klucz cache też. Tokeny i response_metadata bez zmian."""

    def wrap_model_call(self, request: ModelRequest, handler: Handler) -> Any:
        return _stable_response(handler(request))

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        return _stable_response(await handler(request))


class CallBudgetExhausted(RuntimeError):
    """Limit wywołań sieciowych modelu w przebiegu E4 (--max-calls) wyczerpany."""


class CallMeter(AgentMiddleware):
    """Liczy KAŻDĄ próbę modelu (także fallbacku). Przed próbą: network >= max_calls →
    CallBudgetExhausted. Po udanej próbie: wzrost cache.hits w trakcie próby → from_cache,
    inaczej network; wyjątek → errors. max_calls None = bez limitu. Wątkowo bezpieczny."""

    def __init__(self, *, cache: BaseCache | None = None, max_calls: int | None = None) -> None:
        super().__init__()
        self.cache = cache
        self.max_calls = max_calls
        self.network = 0
        self.from_cache = 0
        self.errors = 0
        self._lock = threading.Lock()

    def _hits(self) -> int:
        return int(getattr(self.cache, "hits", 0) or 0) if self.cache is not None else 0

    def _before(self) -> int:
        with self._lock:
            if self.max_calls is not None and self.network >= self.max_calls:
                raise CallBudgetExhausted(
                    f"wyczerpany limit wywołań sieciowych modelu (--max-calls {self.max_calls})"
                )
        return self._hits()

    def _after(self, hits_before: int) -> None:
        with self._lock:
            if self._hits() > hits_before:
                self.from_cache += 1
            else:
                self.network += 1

    def _failed(self) -> None:
        with self._lock:
            self.errors += 1

    def wrap_model_call(self, request: ModelRequest, handler: Handler) -> Any:
        before = self._before()
        try:
            response = handler(request)
        except GraphBubbleUp:
            raise
        except Exception:
            self._failed()
            raise
        self._after(before)
        return response

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        before = self._before()
        try:
            response = await handler(request)
        except GraphBubbleUp:
            raise
        except Exception:
            self._failed()
            raise
        self._after(before)
        return response
