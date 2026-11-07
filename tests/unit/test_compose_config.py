"""Konfiguracja compose bez Dockera (kontrakt F5, 12.1, R1-10): `stop_grace_period`.

Restart `api` w trakcie tury nie może ucinać trwającej tury: czas łagodnego zatrzymania musi
pokryć `agent.timeout_s` i zamknięcie połączenia MCP (`mcp.cleanup_timeout_s`)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def seconds(value: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|m|h)", str(value).strip())
    assert match, f"nieznany format czasu: {value!r}"
    number, unit = float(match.group(1)), match.group(2)
    return number * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]


def services() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]


def test_api_stop_grace_period_covers_a_running_turn(settings):
    grace = seconds(services()["api"]["stop_grace_period"])
    needed = settings.agent.timeout_s + settings.mcp.cleanup_timeout_s
    assert grace == 200, "wartość z kontraktu (R1-10, U13)"
    assert grace > needed, f"{grace}s ≤ agent.timeout_s + mcp.cleanup_timeout_s = {needed}s"


def test_mcp_server_stops_faster_than_the_apis_grace_period(settings):
    """Serwer MCP musi zdążyć się zatrzymać, a `api` zamyka połączenie dopiero po turze."""
    mcp = seconds(services()["mcp-server"]["stop_grace_period"])
    assert mcp == 15
    assert mcp > settings.mcp.cleanup_timeout_s


def test_api_waits_for_a_healthy_mcp_server_and_mcp_for_postgres():
    svc = services()
    assert svc["api"]["depends_on"]["mcp-server"]["condition"] == "service_healthy"
    assert svc["mcp-server"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert "./data:/app/data:ro" in svc["mcp-server"]["volumes"]
    assert svc["mcp-server"]["ports"] == ["127.0.0.1:8765:8765"], "port tylko na loopbacku"
