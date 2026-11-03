"""API wątków: /threads, /messages, /resume, GET /threads/{id} (kontrakt F3, 8; ADR-0010)."""

from __future__ import annotations

import uuid

import pytest

from tests.workflow_fakes import turn

LONG = "x" * 4001


def new_thread(client) -> str:
    response = client.post("/threads")
    assert response.status_code == 201
    return response.json()["thread_id"]


def pause_ref(pending: dict) -> dict:
    return {
        "type": pending["type"],
        "key": pending.get("role") or pending.get("field"),
        "revision": pending.get("revision"),
    }


def test_create_thread(client):
    response = client.post("/threads")
    assert response.status_code == 201
    thread_id = response.json()["thread_id"]
    assert str(uuid.UUID(thread_id, version=4)) == thread_id
    assert new_thread(client) != thread_id


def test_message_and_resume_d2(d2_client, gold_dialogues):
    """D2 przez HTTP na atrapach: akceptacja 14% i zmiana na 22% (CEO zatwierdza, CFO odrzuca)."""
    client = d2_client
    d2 = next(d for d in gold_dialogues if d.id == "D2")
    thread = new_thread(client)

    r = client.post(f"/threads/{thread}/messages", json={"text": d2.turns[0].user})
    assert r.status_code == 200
    body = r.json()
    assert body["thread_id"] == thread and body["status"] == "pending_approval"
    assert body["pending"]["type"] == "approval" and body["pending"]["role"] == "Dyrektor handlowy"
    assert body["pending"]["options"] == ["approved", "rejected", "cancel"]
    assert body["replies"] and body["usage"]["calls"] == 1

    r = client.post(
        f"/threads/{thread}/resume",
        json={"value": "approved", "pause_ref": pause_ref(body["pending"])},
    )
    assert r.status_code == 200 and r.json()["status"] == "approved" and r.json()["pending"] is None

    r = client.post(f"/threads/{thread}/messages", json={"text": d2.turns[1].user})
    body = r.json()
    assert r.status_code == 200 and body["pending"]["role"] == "CEO"
    assert body["request"]["revision"] == 2 and body["request"]["discount_pct"] == 22
    r = client.post(f"/threads/{thread}/resume", json={"value": "approved"})
    assert r.json()["pending"]["role"] == "CFO"
    r = client.post(f"/threads/{thread}/resume", json={"text": "odrzuć"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "rejected"
    assert body["approvals"] == {"CEO": "approved", "CFO": "rejected"}

    state = client.get(f"/threads/{thread}").json()
    assert state["status"] == "rejected" and state["pending"] is None
    assert state["request"]["revision"] == 2
    assert [m["role"] for m in state["messages"]][:2] == ["user", "assistant"]


def test_invalid_resume_value_is_reasked_not_error(make_client):
    client = make_client(
        [turn(kind="approval_request", request_action="new", discount_pct=14, margin_after_pct=10)]
    )
    thread = new_thread(client)
    client.post(f"/threads/{thread}/messages", json={"text": "Prześlij rabat 14%."})
    r = client.post(f"/threads/{thread}/resume", json={"text": "może"})
    assert r.status_code == 200 and r.json()["resume_rejected"] is True
    assert r.json()["pending"]["role"] == "Dyrektor handlowy"


def test_error_mapping(make_client):
    client = make_client(
        [turn(kind="approval_request", request_action="new", discount_pct=22, margin_after_pct=10)]
    )
    thread = new_thread(client)

    not_paused = client.post(f"/threads/{thread}/resume", json={"value": "approved"})
    assert not_paused.status_code == 409 and not_paused.json()["error"] == "not_paused"
    assert not_paused.json()["detail"]

    first = client.post(f"/threads/{thread}/messages", json={"text": "Prześlij rabat 22%."}).json()
    assert first["pending"]["role"] == "CEO"

    paused = client.post(f"/threads/{thread}/messages", json={"text": "Dzięki!"})
    body = paused.json()
    assert paused.status_code == 409 and body["error"] == "thread_paused"
    assert body["pending"]["role"] == "CEO" and "CEO" in body["detail"]

    stale = client.post(
        f"/threads/{thread}/resume",
        json={"value": "approved", "pause_ref": {"type": "approval", "key": "CFO", "revision": 1}},
    )
    assert stale.status_code == 409 and stale.json()["error"] == "stale_pause"
    assert stale.json()["pending"]["role"] == "CEO"

    # 422: zły UUID, value + text, żadne z nich, pusty tekst, za długi tekst, brak ciała
    assert client.post("/threads/nie-uuid/messages", json={"text": "x"}).status_code == 422
    assert client.get("/threads/nie-uuid").status_code == 422
    assert (
        client.post(
            f"/threads/{thread}/resume", json={"value": "approved", "text": "zatwierdź"}
        ).status_code
        == 422
    )
    assert client.post(f"/threads/{thread}/resume", json={}).status_code == 422
    fresh = new_thread(client)
    assert client.post(f"/threads/{fresh}/messages", json={"text": ""}).status_code == 422
    assert client.post(f"/threads/{fresh}/messages", json={"text": LONG}).status_code == 422
    assert client.post(f"/threads/{fresh}/messages", json={}).status_code == 422
    assert client.post(f"/threads/{thread}/resume", json={"value": "może"}).status_code == 422, (
        "value tylko z listy"
    )


def test_get_unknown_thread_has_null_status(client):
    thread = str(uuid.uuid4())
    r = client.get(f"/threads/{thread}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] is None and body["pending"] is None and body["messages"] == []


class _RaisingService:
    """Serwis, który zgłasza podany wyjątek z każdej metody."""

    def __init__(self, exc):
        self.exc = exc

    def send_message(self, *a, **k):
        raise self.exc

    def resume(self, *a, **k):
        raise self.exc

    def get_thread(self, *a, **k):
        raise self.exc


def _raising_client(settings, exc):
    from fastapi.testclient import TestClient

    from kovalo.api.main import create_app

    return TestClient(
        create_app(settings, service=_RaisingService(exc)), raise_server_exceptions=False
    )


def test_busy_stuck_and_internal_errors(settings):
    from kovalo.workflow.service import ThreadBusyError, ThreadStuckError

    thread = str(uuid.uuid4())
    busy = _raising_client(settings, ThreadBusyError("zajęty"))
    r = busy.post(f"/threads/{thread}/messages", json={"text": "x"})
    assert r.status_code == 409 and r.json()["error"] == "thread_busy"
    r = busy.post(f"/threads/{thread}/resume", json={"value": "approved"})
    assert r.status_code == 409 and r.json()["error"] == "thread_busy"

    stuck = _raising_client(settings, ThreadStuckError("zawieszony"))
    r = stuck.post(f"/threads/{thread}/messages", json={"text": "x"})
    assert r.status_code == 409 and r.json()["error"] == "thread_stuck"

    boom = _raising_client(settings, RuntimeError("tajny szczegół awarii"))
    for r in (
        boom.post(f"/threads/{thread}/messages", json={"text": "x"}),
        boom.get(f"/threads/{thread}"),
    ):
        assert r.status_code == 500 and r.json()["error"] == "internal_error"
        assert "tajny szczegół" not in r.text, "szczegóły awarii tylko w logu"


def test_unavailable_returns_503(monkeypatch):
    """Brak klucza LLM-a: provider zgłasza CopilotUnavailableError (503), a /health nadal działa."""
    from fastapi.testclient import TestClient

    from kovalo.api.main import create_app
    from kovalo.config import load_settings

    monkeypatch.setenv("KOVALO_WORKFLOW__CHECKPOINTER", "memory")
    settings = load_settings()  # bez GOOGLE_API_KEY
    with TestClient(create_app(settings)) as client:
        thread = client.post("/threads").json()["thread_id"]  # bez rejestru: nie potrzebuje serwisu
        for _ in range(2):  # kolejne żądanie próbuje ponownie
            r = client.post(f"/threads/{thread}/messages", json={"text": "Cześć"})
            assert r.status_code == 503 and r.json()["error"] == "copilot_unavailable"
            assert r.json()["detail"]
        assert client.get(f"/threads/{thread}").status_code == 503
        assert (
            client.post(f"/threads/{thread}/resume", json={"value": "approved"}).status_code == 503
        )
        health = client.get("/health")
        assert health.status_code in (200, 503) and health.json()["service"] == "api"


def test_openapi_has_thread_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "post" in paths["/threads"]
    assert "post" in paths["/threads/{thread_id}/messages"]
    assert "post" in paths["/threads/{thread_id}/resume"]
    assert "get" in paths["/threads/{thread_id}"]
    assert "get" in paths["/health"]


@pytest.mark.parametrize("value", [True, False, "1.5"])
def test_resume_value_rejects_bool_and_numeric_strings(value):
    """JSON `true` nie może stać się marżą 1,0% (przegląd R1, R1-1)."""
    from pydantic import ValidationError

    from kovalo.api.schemas import ResumeIn

    with pytest.raises(ValidationError):
        ResumeIn.model_validate({"value": value})


def test_resume_value_accepts_numbers_and_decisions():
    from kovalo.api.schemas import ResumeIn

    assert ResumeIn.model_validate({"value": 10}).value == 10
    assert ResumeIn.model_validate({"value": 10.5}).value == 10.5
    assert ResumeIn.model_validate({"value": "approved"}).value == "approved"
