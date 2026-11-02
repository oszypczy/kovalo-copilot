"""Aplikacja FastAPI usługi api. W F0 tylko /health (kontrakt F0, sekcja 6)."""

from __future__ import annotations

from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from kovalo import __version__
from kovalo.api.deps import get_health_checks, get_settings
from kovalo.config import Settings
from kovalo.config import get_settings as _global_settings
from kovalo.health import CheckFn, build_report, http_status

CRITICAL_CHECKS = frozenset({"postgres"})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Fabryka aplikacji; bez ustawień czyta get_settings() przy pierwszym żądaniu."""
    app = FastAPI(title="Kovalo Decision Copilot API", version=__version__)
    app.state.settings = settings

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

    return app


def run() -> None:
    """Punkt wejścia `kovalo-api`: uvicorn na 0.0.0.0:services.api_port."""
    settings = _global_settings()
    uvicorn.run(
        "kovalo.api.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=settings.services.api_port,
    )
