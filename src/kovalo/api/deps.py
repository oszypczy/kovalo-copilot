"""Zależności FastAPI aplikacji api (testy podmieniają je przez app.dependency_overrides)."""

from __future__ import annotations

import threading
from contextlib import ExitStack
from typing import Annotated

from fastapi import Depends, Request

from kovalo.config import ConfigError, Settings
from kovalo.config import get_settings as _global_settings
from kovalo.health import CheckFn, CheckResult, check_http, check_llm_configured, check_postgres
from kovalo.workflow import CheckpointerUnavailableError, CopilotService
from kovalo.workflow.factory import open_copilot_service

__all__ = [
    "CopilotProvider",
    "CopilotUnavailableError",
    "get_copilot_service",
    "get_health_checks",
    "get_settings",
    "mcp_health_url",
]


def get_settings(request: Request) -> Settings:
    """Ustawienia przekazane do create_app albo globalne get_settings()."""
    settings = getattr(request.app.state, "settings", None)
    return settings if settings is not None else _global_settings()


def mcp_health_url(settings: Settings) -> str:
    """{services.mcp_url bez /mcp}/health."""
    base = settings.services.mcp_url.rstrip("/").removesuffix("/mcp")
    return f"{base}/health"


def get_health_checks(
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, CheckFn]:
    """Sprawdzenia api: postgres (krytyczne), mcp, llm (tylko obecność klucza)."""

    async def postgres() -> CheckResult:
        return await check_postgres(settings)

    async def mcp() -> CheckResult:
        return await check_http(
            mcp_health_url(settings), timeout_s=settings.services.health_timeout_s
        )

    async def llm() -> CheckResult:
        return check_llm_configured(settings)

    return {"postgres": postgres, "mcp": mcp, "llm": llm}


class CopilotUnavailableError(RuntimeError):
    """Nie da się zbudować copilota (brak klucza LLM-a, Postgres niedostępny): 503."""


class CopilotProvider:
    """Leniwie (pierwsze żądanie, pod blokadą) wchodzi w open_copilot_service(settings).

    ConfigError (w tym MissingConfigError) i CheckpointerUnavailableError →
    CopilotUnavailableError; kolejne żądanie próbuje ponownie. close() przy zamykaniu
    aplikacji (lifespan). settings None → globalne get_settings() przy pierwszym użyciu.
    """

    def __init__(self, settings: Settings | None) -> None:
        self.settings = settings
        self._service: CopilotService | None = None
        self._stack: ExitStack | None = None
        self._lock = threading.Lock()

    def get(self) -> CopilotService:
        with self._lock:
            if self._service is not None:
                return self._service
            settings = self.settings if self.settings is not None else _global_settings()
            stack = ExitStack()
            try:
                service = stack.enter_context(open_copilot_service(settings))
            except (ConfigError, CheckpointerUnavailableError) as exc:
                stack.close()
                raise CopilotUnavailableError(str(exc)) from exc
            self._service, self._stack = service, stack
            return service

    def close(self) -> None:
        with self._lock:
            stack, self._stack, self._service = self._stack, None, None
        if stack is not None:
            stack.close()


def get_copilot_service(request: Request) -> CopilotService:
    """app.state.copilot_service (testy) albo serwis z providera (CopilotUnavailableError → 503)."""
    service: CopilotService | None = getattr(request.app.state, "copilot_service", None)
    if service is not None:
        return service
    provider: CopilotProvider = request.app.state.copilot_provider
    return provider.get()
