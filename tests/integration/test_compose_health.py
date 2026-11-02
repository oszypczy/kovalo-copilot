"""Test integracyjny: wymaga działającego `docker compose up` (marker integration)."""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/health",
        f"http://localhost:{os.environ.get('API_PORT', '8080')}/health",
        "http://localhost:8765/health",
    ],
    ids=["ui", "api", "mcp-server"],
)
def test_services_healthy(url: str):
    r = httpx.get(url, timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] in {"ok", "degraded"}
