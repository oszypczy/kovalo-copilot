"""Aplikacja FastAPI usługi api: /health (F0) i /threads (F3, kontrakt F3 sekcja 8)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from kovalo import __version__
from kovalo.api.deps import (
    CopilotProvider,
    CopilotUnavailableError,
    get_health_checks,
    get_settings,
)
from kovalo.api.schemas import ApiError
from kovalo.api.threads import ApiProblem
from kovalo.api.threads import router as threads_router
from kovalo.config import Settings
from kovalo.config import get_settings as _global_settings
from kovalo.health import CheckFn, build_report, http_status
from kovalo.workflow import CopilotService

CRITICAL_CHECKS = frozenset({"postgres"})

log = logging.getLogger("kovalo.api")


def _error(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(
        ApiError(error=error, detail=detail).model_dump(mode="json"), status_code=status
    )


def _validation_detail(exc: RequestValidationError) -> str:
    parts = []
    for item in exc.errors():
        loc = ".".join(str(p) for p in item.get("loc", ()) if p != "body") or "żądanie"
        parts.append(f"{loc}: {item.get('msg', 'niepoprawna wartość')}")
    return "Niepoprawne żądanie: " + "; ".join(parts)


def create_app(
    settings: Settings | None = None, *, service: CopilotService | None = None
) -> FastAPI:
    """Fabryka aplikacji; bez ustawień czyta get_settings() przy pierwszym żądaniu.

    `service` podany (testy) → bez providera. Inaczej CopilotProvider buduje serwis przy
    pierwszym żądaniu do /threads/...; brak klucza LLM-a nie blokuje startu ani /health.
    """
    provider = None if service is not None else CopilotProvider(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if provider is not None:
                provider.close()

    app = FastAPI(title="Kovalo Decision Copilot API", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.copilot_service = service
    app.state.copilot_provider = provider

    @app.exception_handler(ApiProblem)
    async def api_problem(request: Request, exc: ApiProblem) -> JSONResponse:
        return JSONResponse(exc.body.model_dump(mode="json"), status_code=exc.status)

    @app.exception_handler(CopilotUnavailableError)
    async def unavailable(request: Request, exc: CopilotUnavailableError) -> JSONResponse:
        return _error(
            503,
            "copilot_unavailable",
            f"Copilot jest niedostępny: {exc} Sprawdź GOOGLE_API_KEY w .env i czy działa "
            "Postgres (make up), potem spróbuj ponownie.",
        )

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(422, "invalid_request", _validation_detail(exc))

    @app.exception_handler(Exception)
    async def internal(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Nieoczekiwany błąd w %s %s", request.method, request.url.path)
        return _error(500, "internal_error", "Nieoczekiwany błąd serwera. Szczegóły są w logu api.")

    @app.get("/health", tags=["health"])
    async def health(
        settings: Annotated[Settings, Depends(get_settings)],
        checks: Annotated[dict[str, CheckFn], Depends(get_health_checks)],
    ) -> JSONResponse:
        report = await build_report(
            "api",
            checks,
            timeout_s=settings.services.health_timeout_s,
            critical=CRITICAL_CHECKS,
        )
        return JSONResponse(report.model_dump(mode="json"), status_code=http_status(report))

    app.include_router(threads_router)
    return app


def run() -> None:
    """Punkt wejścia `kovalo-api`: uvicorn na 0.0.0.0:services.api_port (jeden proces)."""
    settings = _global_settings()
    uvicorn.run(
        "kovalo.api.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=settings.services.api_port,
    )
