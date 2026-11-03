"""Fixture'y UI: httpx.MockTransport z odpowiedziami TurnResult/ApiError (kontrakt F3, 14.1)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest


class Recorder:
    """Zapisuje żądania (metoda, ścieżka, ciało) i zwraca zaplanowane odpowiedzi po kolei."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status, payload = item
        return httpx.Response(status, json=payload)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


@pytest.fixture
def mock_api() -> Callable[..., Recorder]:
    def make(*responses: Any) -> Recorder:
        return Recorder(list(responses))

    return make


def turn_result(**fields: Any) -> dict[str, Any]:
    """Ciało odpowiedzi 200 w formacie TurnResult (JSON)."""
    base: dict[str, Any] = {
        "thread_id": "T1",
        "status": "answered",
        "replies": ["Odpowiedź."],
        "pending": None,
        "resume_rejected": False,
        "turn": None,
        "request": None,
        "decision": None,
        "approvals": {},
        "sources": [],
        "usage": None,
        "error": None,
    }
    base.update(fields)
    return base


def approval_pending(**fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": "approval",
        "field": None,
        "question": None,
        "role": "CFO",
        "revision": 2,
        "approvers": ["CEO", "CFO"],
        "request": {"discount_pct": 22},
        "summary": "klient: Carrefour Express, rabat: 22%",
        "options": ["approved", "rejected", "cancel"],
    }
    base.update(fields)
    return base


# ---------------------------------------------------------------------------
# F4: MockTransport z ciałem text/event-stream (kontrakt F4, 15.1)
# ---------------------------------------------------------------------------


def sse_body(*events: tuple[str, dict[str, Any]], ping: bool = False) -> bytes:
    """Ciało text/event-stream: ('typ', dane) → `id/event/data` jak api.sse.encode_event."""
    chunks = []
    for n, (name, data) in enumerate(events, start=1):
        if ping and n == 2:
            chunks.append(b": ping\n\n")
        text = json.dumps(data, ensure_ascii=False)
        chunks.append(f"id: {n}\nevent: {name}\ndata: {text}\n\n".encode())
    return b"".join(chunks)


class StreamRecorder:
    """Jak Recorder, ale element ('sse', bytes) zwraca odpowiedź 200 text/event-stream."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, str, Any, dict[str, str]]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body, dict(request.headers)))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status, payload = item
        if status == "sse":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=payload
            )
        return httpx.Response(status, json=payload)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


@pytest.fixture
def mock_stream() -> Callable[..., StreamRecorder]:
    def make(*responses: Any) -> StreamRecorder:
        return StreamRecorder(list(responses))

    return make


def tool_start(
    call_id: str = "c1", tool: str = "get_contract_terms", **fields: Any
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "call_id": call_id,
        "parent_call_id": None,
        "agent": "agent",
        "tool": tool,
        "arguments": {"supplier": "Mleczna Dolina"},
        "origin": "data",
    }
    base.update(fields)
    return base


def tool_end(
    call_id: str = "c1", tool: str = "get_contract_terms", **fields: Any
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "call_id": call_id,
        "parent_call_id": None,
        "tool": tool,
        "status": "ok",
        "brief": 'ok: {"supplier_id": "SUP-114"}',
        "result": {"status": "ok", "supplier_id": "SUP-114"},
        "error": None,
    }
    base.update(fields)
    return base
