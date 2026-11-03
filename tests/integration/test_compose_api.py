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
