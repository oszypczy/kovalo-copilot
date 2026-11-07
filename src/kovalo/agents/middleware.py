"""Middleware agenta: fallback tylko przy błędach dostępności, stabilne usage, licznik wywołań.

Kontrakt: docs/contracts/f4.md, sekcja 7.2 (P1, P4); docs/contracts/f5.md, sekcje 6.2-6.3 (R1-6,
P1); ADR-0006, ADR-0011 pkt 5, ADR-0013 pkt 4, ADR-0014 pkt 5.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.caches import BaseCache
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.errors import GraphBubbleUp

__all__ = [
    "AvailabilityFallbackMiddleware",
    "CallBudgetExhausted",
    "CallCounter",
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
    (produkcja: llm.fallback_exceptions(offline), ADR-0006). Inny wyjątek (400, parsowanie,
    CallBudgetExhausted, a na żywo także LlmCacheMiss) idzie wyżej bez próby fallbacku; w trybie
    offline brak w cache przechodzi do fallbacku (ADR-0008, adaptacja F5). GraphBubbleUp zawsze
    wyżej."""

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


def _without_artifacts(request: ModelRequest) -> ModelRequest:
    """Kopie ToolMessage z artifact=None (F5, P1); bez artefaktów żądanie bez zmian."""
    if not any(isinstance(m, ToolMessage) and m.artifact is not None for m in request.messages):
        return request
    messages = [
        m.model_copy(update={"artifact": None})
        if isinstance(m, ToolMessage) and m.artifact is not None
        else m
        for m in request.messages
    ]
    return request.override(messages=messages)


class StableUsageMiddleware(AgentMiddleware):
    """Najbardziej wewnętrzne middleware (P1 F4 i P1 F5).

    1. Z każdej AIMessage zwróconej przez model usuwa usage_metadata['total_cost'] (LangChain
       dopisuje total_cost = 0 przy trafieniu w cache), zanim wiadomość trafi do stanu agenta.
       Historia tury 2+ jest wtedy identyczna w przebiegu świeżym i z cache, więc klucz cache też.
    2. Do modelu wysyła kopie ToolMessage z artifact=None: pochodzenie wywołania MCP (losowy
       invocation_id, ADR-0013 pkt 4) nie wchodzi do klucza cache. Stan agenta (i trace, który
       czyta artefakt) bez zmian; narzędzia bezpośrednie nie mają artefaktu, więc klucze F2-F4
       się nie zmieniają.
    """

    def wrap_model_call(self, request: ModelRequest, handler: Handler) -> Any:
        return _stable_response(handler(_without_artifacts(request)))

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        return _stable_response(await handler(_without_artifacts(request)))


class CallBudgetExhausted(RuntimeError):
    """Limit wywołań modelu w przebiegu ewaluacji (--max-calls) wyczerpany."""


CallOutcome = Literal["network", "cache", "error"]


class CallCounter:
    """Wątkowo bezpieczny licznik prób z limitem (F5, R1-6, P5).

    `admit()` przed próbą (wyczerpany → CallBudgetExhausted); `record(outcome)`: 'network' i
    'error' zjadają limit, 'cache' nie. O tym, czy błąd w ogóle trafia do licznika jako 'error',
    decyduje CallMeter.count_errors. Jeden licznik może być wspólny dla kilku CallMeterów
    (E5: understand + agent). max_calls None = bez limitu.
    """

    def __init__(self, max_calls: int | None = None) -> None:
        self.max_calls = max_calls
        self.network = 0
        self.cache = 0
        self.errors = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        """Próby, które zjadły limit (sieciowe i nieudane liczone)."""
        with self._lock:
            return self.network + self.errors

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self.max_calls is not None and self.network + self.errors >= self.max_calls

    def admit(self) -> None:
        if self.exhausted:
            raise CallBudgetExhausted(
                f"wyczerpany limit wywołań sieciowych modelu (--max-calls {self.max_calls})"
            )

    def record(self, outcome: CallOutcome) -> None:
        with self._lock:
            if outcome == "network":
                self.network += 1
            elif outcome == "cache":
                self.cache += 1
            else:
                self.errors += 1


class CallMeter(AgentMiddleware):
    """Liczy KAŻDĄ próbę modelu (także fallbacku). Przed próbą: wyczerpany licznik →
    CallBudgetExhausted. Po udanej próbie: wzrost cache.hits w trakcie próby → from_cache,
    inaczej network; wyjątek → errors (i, przy count_errors=True, 'error' w liczniku, który
    zjada limit). Limit siedzi w CallCounter: `counter` None → własny CallCounter(max_calls);
    podany counter → max_calls ignorowane (limit wspólny). Wątkowo bezpieczny."""

    def __init__(
        self,
        *,
        cache: BaseCache | None = None,
        max_calls: int | None = None,
        count_errors: bool = False,
        counter: CallCounter | None = None,
    ) -> None:
        super().__init__()
        self.cache = cache
        self.counter = counter if counter is not None else CallCounter(max_calls=max_calls)
        self.max_calls = self.counter.max_calls
        self.count_errors = count_errors
        self.network = 0
        self.from_cache = 0
        self.errors = 0
        self._lock = threading.Lock()

    def _hits(self) -> int:
        return int(getattr(self.cache, "hits", 0) or 0) if self.cache is not None else 0

    def _before(self) -> int:
        self.counter.admit()
        return self._hits()

    def _after(self, hits_before: int) -> None:
        with self._lock:
            from_cache = self._hits() > hits_before
            if from_cache:
                self.from_cache += 1
            else:
                self.network += 1
        self.counter.record("cache" if from_cache else "network")

    def _failed(self, exc: BaseException) -> None:
        if isinstance(exc, CallBudgetExhausted):
            return
        with self._lock:
            self.errors += 1
        if self.count_errors:
            self.counter.record("error")

    def wrap_model_call(self, request: ModelRequest, handler: Handler) -> Any:
        before = self._before()
        try:
            response = handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            self._failed(exc)
            raise
        self._after(before)
        return response

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        before = self._before()
        try:
            response = await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            self._failed(exc)
            raise
        self._after(before)
        return response
