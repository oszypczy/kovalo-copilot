"""Zależności FastAPI aplikacji api (testy podmieniają je przez app.dependency_overrides)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from kovalo.config import Settings
from kovalo.config import get_settings as _global_settings
from kovalo.health import CheckFn, CheckResult, check_http, check_llm_configured, check_postgres


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
