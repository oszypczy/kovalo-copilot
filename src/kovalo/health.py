"""Wspólny format /health i sprawdzenia zależności usług.

Kontrakt: docs/contracts/f0.md, sekcja 6.2. Bez LangChaina; zero wywołań LLM-a.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Collection, Mapping
from typing import Literal

import httpx
import psycopg
from pydantic import BaseModel, Field

from kovalo import __version__
from kovalo.config import Settings, api_key_env_name, get_settings

CheckState = Literal["ok", "error", "skipped"]
ServiceState = Literal["ok", "degraded", "error"]
ServiceName = Literal["api", "ui", "mcp-server"]

TIMEOUT_DETAIL = "przekroczono limit czasu"


class CheckResult(BaseModel):
    status: CheckState
    critical: bool = False
    detail: str | None = None  # po polsku, bez sekretów i DSN z hasłem
    latency_ms: float | None = None


class HealthReport(BaseModel):
    status: ServiceState
    service: ServiceName
    version: str
    checks: dict[str, CheckResult] = Field(default_factory=dict)


CheckFn = Callable[[], Awaitable[CheckResult]]


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _short(exc: BaseException, limit: int = 200) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def check_postgres(settings: Settings) -> CheckResult:
    """SELECT 1 przez psycopg AsyncConnection z connect_timeout z configu (krytyczne)."""
    start = time.perf_counter()
    try:
        conn = await psycopg.AsyncConnection.connect(
            settings.postgres_dsn(),
            connect_timeout=max(1, round(settings.db.connect_timeout_s)),
        )
        async with conn:
            await conn.execute("SELECT 1")
    except Exception as exc:
        # Komunikaty psycopg zawierają host i port, ale nie hasło.
        return CheckResult(status="error", critical=True, detail=_short(exc))
    return CheckResult(status="ok", critical=True, latency_ms=_elapsed_ms(start))


async def check_http(url: str, *, timeout_s: float, critical: bool = False) -> CheckResult:
    """GET na adres; odpowiedź 2xx → ok."""
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(url)
    except Exception as exc:
        return CheckResult(
            status="error", critical=critical, detail=f"brak połączenia z {url} ({_short(exc)})"
        )
    latency = _elapsed_ms(start)
    if response.is_success:
        return CheckResult(status="ok", critical=critical, latency_ms=latency)
    return CheckResult(
        status="error",
        critical=critical,
        detail=f"{url} zwrócił HTTP {response.status_code}",
        latency_ms=latency,
    )


def check_llm_configured(settings: Settings) -> CheckResult:
    """Tylko obecność klucza dostawcy z llm.provider; ZERO wywołań LLM-a."""
    provider = settings.llm.provider
    key = settings.api_key_for(provider)
    if key is None or not key.get_secret_value():
        detail = f"brak {api_key_env_name(provider)}"
        return CheckResult(status="error", critical=False, detail=detail)
    return CheckResult(status="ok", critical=False)


async def _run_check(name: str, check: CheckFn, timeout_s: float, critical: bool) -> CheckResult:
    try:
        result = await asyncio.wait_for(check(), timeout=timeout_s)
    except TimeoutError:
        result = CheckResult(status="error", detail=TIMEOUT_DETAIL)
    except Exception as exc:
        result = CheckResult(status="error", detail=f"błąd sprawdzenia: {_short(exc)}")
    if critical and not result.critical:
        result = result.model_copy(update={"critical": True})
    return result


def _aggregate(checks: Mapping[str, CheckResult]) -> ServiceState:
    errors = [c for c in checks.values() if c.status == "error"]
    if any(c.critical for c in errors):
        return "error"
    if errors:
        return "degraded"
    return "ok"


async def build_report(
    service: ServiceName,
    checks: Mapping[str, CheckFn],
    *,
    timeout_s: float | None = None,
    critical: Collection[str] = (),
) -> HealthReport:
    """Uruchamia sprawdzenia równolegle, każde z limitem czasu, i agreguje status.

    `timeout_s` domyślnie z `services.health_timeout_s`; nazwy z `critical` są krytyczne
    także wtedy, gdy sprawdzenie przekroczy limit albo rzuci wyjątek.
    """
    limit = timeout_s if timeout_s is not None else get_settings().services.health_timeout_s
    names = list(checks)
    results = await asyncio.gather(
        *(_run_check(name, checks[name], limit, name in critical) for name in names)
    )
    by_name = dict(zip(names, results, strict=True))
    return HealthReport(
        status=_aggregate(by_name), service=service, version=__version__, checks=by_name
    )


def http_status(report: HealthReport) -> int:
    """error → 503; ok i degraded → 200."""
    return 503 if report.status == "error" else 200
