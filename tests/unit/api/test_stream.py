"""SSE (`/messages/stream`, `/resume/stream`), `GET /threads/{id}/trace`, 503 przy utracie
Postgresa (kontrakt F4, 9; ADR-0012)."""

from __future__ import annotations

import uuid

from tests.agent_fakes import july_events, make_trace
from tests.workflow_fakes import turn

DOC = turn(kind="document_question", standalone="Ile wynosi kara za lipiec 2026 r.?")


def agent_trace():
    from kovalo.agents.trace import ModelCallInfo

    trace = make_trace("Q", "Kara wynosi 7296,00 PLN.", july_events())
    trace.model_calls = [ModelCallInfo(model="m", input_tokens=10, output_tokens=2)]
    return trace


def new_thread(client) -> str:
    return client.post("/threads").json()["thread_id"]


def test_messages_stream_sse(make_agent_client, sse_events):
    from kovalo.workflow.service import TurnResult

    client = make_agent_client([DOC], agent_trace())
    thread = new_thread(client)
    response = client.post(f"/threads/{thread}/messages/stream", json={"text": "Ile wynosi kara?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "no-cache" in response.headers["cache-control"]
    assert response.headers["x-accel-buffering"] == "no"
    events = sse_events(response)
    assert [e for _, e, _ in events] == ["tool_start", "tool_end"] * 4 + ["final"]
    ids = [i for i, _, _ in events]
    assert ids == sorted(set(ids)), "id zdarzeń rosnące"
    assert events[0][2]["tool"] == "get_contract_terms" and events[0][2]["origin"] == "data"
    assert events[1][2]["status"] == "ok" and events[1][2]["brief"]
    final = TurnResult.model_validate(events[-1][2])
    assert final.thread_id == thread and final.status == "answered"
    assert final.replies == ["Kara wynosi 7296,00 PLN."]
    assert final.trace["status"] == "answered" and final.evidence_path
    raw = response.text
    assert "7296,00" in raw and "\\u" not in raw, "ensure_ascii=False: polskie znaki wprost"
    assert all("\n" not in line for line in raw.splitlines()), "data w jednej linii"


def test_final_event_is_the_json_turn_result(make_agent_client, sse_events):
    """Wynik strumienia = wynik `/messages` dla tego samego skryptu (poza id wątku)."""
    trace = agent_trace()
    a = make_agent_client([DOC], trace)
    b = make_agent_client([DOC], trace)
    thread = new_thread(a)
    final = sse_events(a.post(f"/threads/{thread}/messages/stream", json={"text": "Pytanie?"}))[-1][
        2
    ]
    direct = b.post(f"/threads/{thread}/messages", json={"text": "Pytanie?"}).json()
    assert final == direct


def test_resume_stream_returns_final_only(make_client, sse_events):
    client = make_client(
        [turn(kind="approval_request", request_action="new", discount_pct=14, margin_after_pct=10)]
    )
    thread = new_thread(client)
    first = client.post(f"/threads/{thread}/messages", json={"text": "Prześlij rabat 14%."}).json()
    ref = {
        "type": "approval",
        "key": first["pending"]["role"],
        "revision": first["pending"]["revision"],
    }
    response = client.post(
        f"/threads/{thread}/resume/stream", json={"value": "approved", "pause_ref": ref}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response)
    assert [e for _, e, _ in events] == ["final"]
    assert events[0][2]["status"] == "approved" and events[0][2]["pending"] is None


def test_stream_errors_before_body(make_client, make_agent_client, settings):
    """409 thread_paused, 422, 503: zwykły JSON ApiError przed strumieniem."""
    client = make_client(
        [turn(kind="approval_request", request_action="new", discount_pct=22, margin_after_pct=10)]
    )
    thread = new_thread(client)
    client.post(f"/threads/{thread}/messages", json={"text": "Prześlij rabat 22%."})
    paused = client.post(f"/threads/{thread}/messages/stream", json={"text": "Dzięki!"})
    assert paused.status_code == 409 and paused.headers["content-type"].startswith(
        "application/json"
    )
    body = paused.json()
    assert body["error"] == "thread_paused" and body["pending"]["role"] == "CEO"
    not_paused = new_thread(client)
    nope = client.post(f"/threads/{not_paused}/resume/stream", json={"value": "approved"})
    assert nope.status_code == 409 and nope.json()["error"] == "not_paused"
    stale = client.post(
        f"/threads/{thread}/resume/stream",
        json={"value": "approved", "pause_ref": {"type": "approval", "key": "CFO", "revision": 1}},
    )
    assert stale.status_code == 409 and stale.json()["error"] == "stale_pause"

    assert client.post("/threads/nie-uuid/messages/stream", json={"text": "x"}).status_code == 422
    fresh = new_thread(client)
    for bad in ({"text": ""}, {"text": "x" * 4001}, {}):
        r = client.post(f"/threads/{fresh}/messages/stream", json=bad)
        assert r.status_code == 422, bad
        assert r.headers["content-type"].startswith("application/json")
    assert client.post(f"/threads/{fresh}/resume/stream", json={}).status_code == 422


class _RaisingStreamService:
    def __init__(self, exc):
        self.exc = exc

    def start_message(self, *a, **k):
        raise self.exc

    def start_resume(self, *a, **k):
        raise self.exc

    def send_message(self, *a, **k):
        raise self.exc

    def resume(self, *a, **k):
        raise self.exc

    def get_thread(self, *a, **k):
        raise self.exc

    def get_trace(self, *a, **k):
        raise self.exc


def _raising_client(settings, exc):
    from fastapi.testclient import TestClient

    from kovalo.api.main import create_app

    return TestClient(
        create_app(settings, service=_RaisingStreamService(exc)), raise_server_exceptions=False
    )


def test_stream_busy_and_unavailable_before_body(settings):
    from kovalo.workflow import CheckpointerUnavailableError
    from kovalo.workflow.service import ThreadBusyError

    thread = str(uuid.uuid4())
    busy = _raising_client(settings, ThreadBusyError("zajęty"))
    r = busy.post(f"/threads/{thread}/messages/stream", json={"text": "x"})
    assert r.status_code == 409 and r.json()["error"] == "thread_busy"
    down = _raising_client(settings, CheckpointerUnavailableError("Postgres niedostępny"))
    for r in (
        down.post(f"/threads/{thread}/messages/stream", json={"text": "x"}),
        down.post(f"/threads/{thread}/resume/stream", json={"value": "approved"}),
    ):
        assert r.status_code == 503 and r.json()["error"] == "copilot_unavailable"


def test_trace_endpoint(make_agent_client):
    client = make_agent_client([DOC], agent_trace())
    thread = new_thread(client)
    empty = client.get(f"/threads/{thread}/trace")
    assert empty.status_code == 200
    body = empty.json()
    assert body["thread_id"] == thread and body["trace"] is None and body["evidence_path"] is None
    client.post(f"/threads/{thread}/messages", json={"text": "Ile wynosi kara?"})
    r = client.get(f"/threads/{thread}/trace")
    assert r.status_code == 200
    body = r.json()
    assert body["trace"]["status"] == "answered" and len(body["trace"]["events"]) == 4
    assert "calculate_otif_penalty" in body["evidence_path"]
    assert body["trace_id"] and body["created_at"] and body["question"]
    assert client.get("/threads/nie-uuid/trace").status_code == 422


def test_postgres_lost_returns_503(settings):
    """R1-2: serwis zgłasza CheckpointerUnavailableError → 503 copilot_unavailable zamiast 500."""
    from kovalo.workflow import CheckpointerUnavailableError

    client = _raising_client(settings, CheckpointerUnavailableError("Postgres niedostępny"))
    thread = str(uuid.uuid4())
    responses = [
        client.post(f"/threads/{thread}/messages", json={"text": "Cześć"}),
        client.post(f"/threads/{thread}/resume", json={"value": "approved"}),
        client.get(f"/threads/{thread}"),
        client.get(f"/threads/{thread}/trace"),
    ]
    for r in responses:
        assert r.status_code == 503, r.text
        body = r.json()
        assert body["error"] == "copilot_unavailable" and body["detail"]
        assert "postgresql://" not in r.text


def test_error_event_in_stream_hides_internal_details(settings, sse_events):
    """Błąd po starcie strumienia → zdarzenie `error`; szczegóły awarii tylko w logu."""
    from fastapi.testclient import TestClient

    from kovalo.api.main import create_app
    from kovalo.workflow.service import TurnEvent

    class _Stream:
        # §9: sse_body czyta przez next_event(timeout); None = PING, error kończy strumień
        thread_id = "t"

        def __init__(self):
            self._events = iter(self)

        def next_event(self, timeout):
            return next(self._events, None)

        def close(self):
            pass

        def __iter__(self):
            yield TurnEvent(
                type="error", data={"error": "internal_error", "detail": "Błąd wewnętrzny."}
            )

    class _Service:
        def start_message(self, *a, **k):
            return _Stream()

    client = TestClient(create_app(settings, service=_Service()))
    r = client.post(f"/threads/{uuid.uuid4()}/messages/stream", json={"text": "x"})
    assert r.status_code == 200
    events = sse_events(r)
    assert [e for _, e, _ in events] == ["error"] and events[0][2]["error"] == "internal_error"


def test_openapi_has_f4_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "post" in paths["/threads/{thread_id}/messages/stream"]
    assert "post" in paths["/threads/{thread_id}/resume/stream"]
    assert "get" in paths["/threads/{thread_id}/trace"]
    for old in ("/threads", "/threads/{thread_id}/messages", "/threads/{thread_id}/resume"):
        assert "post" in paths[old]
    assert "get" in paths["/threads/{thread_id}"] and "get" in paths["/health"]


def test_sse_encoding_and_ping():
    from kovalo.api.sse import PING, encode_event, sse_body
    from kovalo.workflow.service import TurnEvent

    event = TurnEvent(type="tool_end", data={"tool": "x", "brief": "94,6% ąę"})
    raw = encode_event(event, event_id=7)
    expected = 'id: 7\nevent: tool_end\ndata: {"tool": "x", "brief": "94,6% ąę"}\n\n'
    assert raw.decode("utf-8").replace('":"', '": "').replace('","', '", "') == expected
    assert raw.endswith(b"\n\n") and raw.count(b"\n") == 4, "jedna linia danych"
    assert "ąę".encode() in raw, "ensure_ascii=False"
    assert PING == b": ping\n\n"

    class Slow:
        """Strumień: dwa razy cisza, potem final; zapisuje close()."""

        def __init__(self):
            self.items = [None, None, TurnEvent(type="final", data={"status": "answered"})]
            self.closed = False

        def next_event(self, timeout):
            return self.items.pop(0)

        def close(self):
            self.closed = True

    stream = Slow()
    chunks = list(sse_body(stream, ping_s=0.01))
    assert chunks[0] == PING and chunks[1] == PING
    assert chunks[2].startswith(b"id: ") and b"event: final" in chunks[2]
    assert len(chunks) == 3 and stream.closed, "po final koniec, finally: close()"


def test_sse_body_closes_stream_when_client_disconnects():
    from kovalo.api.sse import sse_body
    from kovalo.workflow.service import TurnEvent

    class Endless:
        closed = False

        def next_event(self, timeout):
            return TurnEvent(type="tool_start", data={"tool": "x"})

        def close(self):
            self.closed = True

    stream = Endless()
    body = sse_body(stream, ping_s=1)
    next(body)
    body.close()  # rozłączenie klienta (GeneratorExit)
    assert stream.closed
