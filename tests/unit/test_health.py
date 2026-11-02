"""Testy kovalo.health i /health trzech usług (kontrakt F0, sekcja 6)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

pytestmark = pytest.mark.anyio


def _const(status: str, critical: bool = False):
    from kovalo.health import CheckResult

    async def check() -> CheckResult:
        return CheckResult(status=status, critical=critical)  # type: ignore[arg-type]

    return check


@pytest.mark.parametrize(
    ("checks", "expected_status", "expected_http"),
    [
        ({"pg": ("ok", True), "llm": ("ok", False)}, "ok", 200),
        ({"pg": ("ok", True), "llm": ("error", False)}, "degraded", 200),
        ({"pg": ("error", True), "llm": ("ok", False)}, "error", 503),
    ],
    ids=["ok", "degraded", "error"],
)
async def test_report_status_aggregation(checks, expected_status, expected_http):
    from kovalo.health import build_report, http_status

    report = await build_report("api", {k: _const(*v) for k, v in checks.items()})
    assert report.status == expected_status
    assert report.service == "api"
    assert http_status(report) == expected_http


async def test_check_timeout_and_exception_become_error(monkeypatch: pytest.MonkeyPatch):
    from kovalo.config import get_settings
    from kovalo.health import CheckResult, build_report

    async def slow() -> CheckResult:
        await asyncio.sleep(60)
        return CheckResult(status="ok")

    async def boom() -> CheckResult:
        raise RuntimeError("awaria")

    # Interpretacja: build_report(service, checks) nie dostaje settings, więc limit czasu
    # (services.health_timeout_s) czyta z get_settings().
    monkeypatch.setenv("KOVALO_SERVICES__HEALTH_TIMEOUT_S", "0.05")
    get_settings.cache_clear()
    report = await build_report("api", {"slow": slow, "boom": boom})
    assert report.checks["slow"].status == "error"
    assert report.checks["slow"].detail == "przekroczono limit czasu"
    assert report.checks["boom"].status == "error"


def test_llm_check_without_key():
    from kovalo.config import load_settings
    from kovalo.health import check_llm_configured

    result = check_llm_configured(load_settings())
    assert result.status == "error"
    assert result.critical is False
    assert check_llm_configured(load_settings(google_api_key="test-key")).status == "ok"


async def test_api_health_endpoint(settings):
    from kovalo.api.deps import get_health_checks
    from kovalo.api.main import create_app

    app = create_app(settings)
    app.dependency_overrides[get_health_checks] = lambda: {
        "postgres": _const("ok", True),
        "mcp": _const("ok"),
        "llm": _const("error"),
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["service"] == "api"
        assert body["status"] == "degraded"
        assert set(body["checks"]) == {"postgres", "mcp", "llm"}
        assert body["version"]

        app.dependency_overrides[get_health_checks] = lambda: {
            "postgres": _const("error", True),
            "mcp": _const("ok"),
            "llm": _const("ok"),
        }
        r = await client.get("/health")
        assert r.status_code == 503
        assert r.json()["status"] == "error"


async def test_ui_health_endpoint(monkeypatch: pytest.MonkeyPatch):
    # api niedostępne: loopback, port 9 (discard) odrzuca połączenie lokalnie
    monkeypatch.setenv("KOVALO_SERVICES__API_URL", "http://127.0.0.1:9")
    from kovalo.config import get_settings

    get_settings.cache_clear()
    from kovalo.ui.server import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "ui"
    assert body["status"] == "degraded"
    assert body["checks"]["api"]["status"] == "error"


async def test_mcp_health_endpoint(settings):
    from kovalo.mcp_server.server import build_server

    app = build_server(settings).streamable_http_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "mcp-server"
