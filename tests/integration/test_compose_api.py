"""API wątków na działającym compose (kontrakt F3, 14.2): bez LLM-a. Wymaga `make up`."""

from __future__ import annotations

import os
import uuid

import httpx
import pytest

pytestmark = pytest.mark.integration

API = f"http://localhost:{os.environ.get('API_PORT', '8080')}"


def test_threads_roundtrip_in_compose():
    created = httpx.post(f"{API}/threads", timeout=10)
    assert created.status_code == 201
    thread_id = created.json()["thread_id"]
    assert str(uuid.UUID(thread_id)) == thread_id

    state = httpx.get(f"{API}/threads/{thread_id}", timeout=10)
    assert state.status_code == 200
    body = state.json()
    assert body["status"] is None and body["pending"] is None and body["messages"] == []

    assert httpx.get(f"{API}/threads/nie-uuid", timeout=10).status_code == 422
    assert httpx.post(f"{API}/threads/{thread_id}/resume", json={}, timeout=10).status_code == 422


def test_trace_and_stream_routes_in_compose():
    """F4 (kontrakt 15.2): nowy wątek ma `trace: null`; pusty tekst w strumieniu to 422 JSON (bez LLM-a)."""  # noqa: E501
    thread_id = httpx.post(f"{API}/threads", timeout=10).json()["thread_id"]

    trace = httpx.get(f"{API}/threads/{thread_id}/trace", timeout=10)
    assert trace.status_code == 200
    body = trace.json()
    assert body["thread_id"] == thread_id and body["trace"] is None
    assert httpx.get(f"{API}/threads/nie-uuid/trace", timeout=10).status_code == 422

    empty = httpx.post(f"{API}/threads/{thread_id}/messages/stream", json={"text": ""}, timeout=10)
    assert empty.status_code == 422
    assert empty.headers["content-type"].startswith("application/json")
    paths = httpx.get(f"{API}/openapi.json", timeout=10).json()["paths"]
    assert "post" in paths["/threads/{thread_id}/messages/stream"]
    assert "get" in paths["/threads/{thread_id}/trace"]
