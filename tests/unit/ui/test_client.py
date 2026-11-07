"""CopilotApiClient: żądania HTTP i mapowanie błędów (kontrakt F3, 9.1)."""

from __future__ import annotations

import httpx
import pytest

from tests.unit.ui.conftest import approval_pending, turn_result

pytestmark = pytest.mark.anyio

BASE = "http://api.test"


def make_client(recorder, timeout_s: float = 5):
    from kovalo.ui.client import CopilotApiClient

    return CopilotApiClient(BASE, timeout_s=timeout_s, transport=recorder.transport)


async def test_client_calls_and_errors(mock_api):
    from kovalo.ui.client import ApiCallError

    pending = approval_pending()
    recorder = mock_api(
        (201, {"thread_id": "T1"}),
        (200, turn_result(status="pending_approval", pending=pending, replies=["Opis"])),
        (200, turn_result(status="approved", replies=["Zatwierdzone"])),
        (200, {"thread_id": "T1", "status": "approved", "pending": None, "request": None,
               "decision": None, "approvals": {}, "messages": []}),
        (409, {"error": "thread_paused", "detail": "Wątek czeka: CFO.", "pending": pending}),
        (503, {"error": "copilot_unavailable", "detail": "Brak klucza", "pending": None}),
    )  # fmt: skip
    client = make_client(recorder)

    assert await client.create_thread() == "T1"
    view = await client.send_message("T1", "Prześlij rabat 22%.")
    assert view.status == "pending_approval" and view.replies == ["Opis"]
    assert view.pending.role == "CFO" and view.pending.options == ["approved", "rejected", "cancel"]
    ref = {"type": "approval", "key": "CFO", "revision": 2}
    done = await client.resume("T1", value="approved", pause_ref=ref)
    assert done.status == "approved"
    state = await client.get_thread("T1")
    assert state.status == "approved" and state.replies == []

    with pytest.raises(ApiCallError) as paused:
        await client.send_message("T1", "Dzięki!")
    assert paused.value.status == 409 and paused.value.error == "thread_paused"
    assert paused.value.pending["role"] == "CFO" and "CFO" in paused.value.detail
    with pytest.raises(ApiCallError) as unavailable:
        await client.resume("T1", text="zatwierdź")
    assert unavailable.value.status == 503 and unavailable.value.error == "copilot_unavailable"

    methods_paths = [(m, p) for m, p, _ in recorder.requests]
    assert methods_paths == [
        ("POST", "/threads"),
        ("POST", "/threads/T1/messages"),
        ("POST", "/threads/T1/resume"),
        ("GET", "/threads/T1"),
        ("POST", "/threads/T1/messages"),
        ("POST", "/threads/T1/resume"),
    ]
    bodies = [b for _, _, b in recorder.requests]
    assert bodies[1] == {"text": "Prześlij rabat 22%."}
    assert bodies[2]["value"] == "approved" and bodies[2]["pause_ref"] == ref
    assert bodies[2].get("text") is None
    assert bodies[5]["text"] == "zatwierdź" and bodies[5].get("value") is None


@pytest.mark.parametrize(
    "exc",
    [httpx.ReadTimeout("limit"), httpx.ConnectError("odmowa połączenia")],
    ids=["timeout", "connect_error"],
)
async def test_unreachable_api_is_api_unreachable(mock_api, exc):
    from kovalo.ui.client import ApiCallError

    client = make_client(mock_api(exc, exc, exc))
    for call in (
        lambda: client.create_thread(),
        lambda: client.send_message("T1", "x"),
        lambda: client.resume("T1", value="cancel"),
    ):
        with pytest.raises(ApiCallError) as info:
            await call()
        assert info.value.status == 0 and info.value.error == "api_unreachable"


async def test_turn_view_accepts_turn_result():
    """Kontrakt UI ↔ API: TurnResult(...).model_dump(mode='json') waliduje się w TurnView."""
    from kovalo.ui.client import TurnView
    from kovalo.workflow.service import PendingPause, Source, TurnResult, TurnUsage

    pending = PendingPause(
        type="approval",
        role="CFO",
        revision=2,
        approvers=["CEO", "CFO"],
        request={"discount_pct": 22.0},
        summary="klient: X",
        options=["approved", "rejected", "cancel"],
    )
    result = TurnResult(
        thread_id="T1",
        status="pending_approval",
        replies=["Opis"],
        pending=pending,
        sources=[Source(doc_id="MTG-402", chunk_id="MTG-402#ctx000", text="Komitet…", score=0.7)],
        usage=TurnUsage(calls=1, input_tokens=10, output_tokens=5, models=["m"]),
    )
    view = TurnView.model_validate(result.model_dump(mode="json"))
    assert view.thread_id == "T1" and view.status == "pending_approval" and view.replies == ["Opis"]
    assert view.pending.role == "CFO" and view.pending.options == ["approved", "rejected", "cancel"]
    assert view.sources[0].chunk_id == "MTG-402#ctx000" and view.resume_rejected is False

    clarification = TurnResult(
        thread_id="T1",
        status="pending_clarification",
        replies=[],
        pending=PendingPause(
            type="clarification",
            field="margin_after_pct",
            question="Jaka marża?",
            options=["cancel"],
        ),
    )
    view = TurnView.model_validate(clarification.model_dump(mode="json"))
    assert view.pending.field == "margin_after_pct" and view.pending.question == "Jaka marża?"


# ---------------------------------------------------------------------------
# F4: strumień tury i /trace (kontrakt F4, 10.1; ADR-0012)
# ---------------------------------------------------------------------------

from tests.unit.ui.conftest import sse_body, tool_end, tool_start  # noqa: E402


async def collect(client, text="Ile wynosi kara?"):
    return [item async for item in client.stream_message("T1", text)]


async def test_stream_message_parses_events(mock_stream):
    from kovalo.ui.client import ToolEventView, TurnView

    final = turn_result(
        status="answered",
        replies=["Kara wynosi 7296,00 PLN."],
        trace={"status": "answered", "events": []},
        evidence_path="P1  Ile wynosi kara?\nstatus: answered",
    )
    body = sse_body(
        ("tool_start", tool_start()),
        ("tool_end", tool_end()),
        (
            "tool_start",
            tool_start("c2#get_supplier_otif", "get_supplier_otif", parent_call_id="c2"),
        ),
        ("final", final),
        ping=True,
    )
    recorder = mock_stream(("sse", body))
    items = await collect(make_client(recorder))
    assert [kind for kind, _ in items] == ["tool_start", "tool_end", "tool_start", "final"]
    start, end, derived, last = (item for _, item in items)
    assert isinstance(start, ToolEventView) and start.tool == "get_contract_terms"
    assert start.arguments == {"supplier": "Mleczna Dolina"} and start.origin == "data"
    assert start.parent_call_id is None and start.agent == "agent"
    assert end.status == "ok" and end.brief.startswith("ok:") and end.error is None
    assert derived.parent_call_id == "c2"
    assert isinstance(last, TurnView) and last.replies == ["Kara wynosi 7296,00 PLN."]
    assert last.trace["status"] == "answered" and last.evidence_path.startswith("P1")
    method, path, sent, headers = recorder.requests[0]
    assert (method, path) == ("POST", "/threads/T1/messages/stream")
    assert sent == {"text": "Ile wynosi kara?"}
    assert "text/event-stream" in headers.get("accept", "")


async def test_stream_error_event_is_api_call_error(mock_stream):
    from kovalo.ui.client import ApiCallError

    body = sse_body(
        ("tool_start", tool_start()),
        ("error", {"error": "copilot_unavailable", "detail": "Postgres niedostępny"}),
    )
    client = make_client(mock_stream(("sse", body)))
    seen = []
    with pytest.raises(ApiCallError) as info:
        async for item in client.stream_message("T1", "x"):
            seen.append(item[0])
    assert seen == ["tool_start"]
    assert (info.value.status, info.value.error) == (200, "copilot_unavailable")
    assert "Postgres" in info.value.detail


async def test_truncated_stream_is_interrupted(mock_stream):
    from kovalo.ui.client import ApiCallError

    body = sse_body(("tool_start", tool_start()), ("tool_end", tool_end()))  # bez final
    client = make_client(mock_stream(("sse", body)))
    with pytest.raises(ApiCallError) as info:
        await collect(client)
    assert info.value.status == 0 and info.value.error == "stream_interrupted"


async def test_stream_http_errors_before_the_stream(mock_stream):
    from kovalo.ui.client import ApiCallError

    pending = approval_pending()
    recorder = mock_stream(
        (409, {"error": "thread_paused", "detail": "Wątek czeka: CFO.", "pending": pending}),
        (503, {"error": "copilot_unavailable", "detail": "Brak klucza", "pending": None}),
        (422, {"error": "invalid_request", "detail": "Wiadomość jest pusta.", "pending": None}),
        httpx.ConnectError("odmowa połączenia"),
    )
    client = make_client(recorder)
    with pytest.raises(ApiCallError) as paused:
        await collect(client)
    assert paused.value.status == 409 and paused.value.error == "thread_paused"
    assert paused.value.pending["role"] == "CFO"
    with pytest.raises(ApiCallError) as unavailable:
        await collect(client)
    assert unavailable.value.status == 503 and unavailable.value.error == "copilot_unavailable"
    with pytest.raises(ApiCallError) as invalid:
        await collect(client)
    assert invalid.value.status == 422
    with pytest.raises(ApiCallError) as unreachable:
        await collect(client)
    assert unreachable.value.status == 0 and unreachable.value.error == "api_unreachable"


async def test_stream_without_events_is_interrupted(mock_stream):
    from kovalo.ui.client import ApiCallError

    client = make_client(mock_stream(("sse", b"")))
    with pytest.raises(ApiCallError) as info:
        await collect(client)
    assert info.value.error == "stream_interrupted"


async def test_get_trace(mock_stream):
    from kovalo.ui.client import TraceView

    recorder = mock_stream(
        (
            200,
            {
                "thread_id": "T1",
                "trace_id": "abc",
                "trace": {"status": "answered"},
                "evidence_path": "P1 ...",
                "created_at": None,
                "question": "Q",
            },
        ),
        (200, {"thread_id": "T2", "trace": None, "evidence_path": None}),
    )
    client = make_client(recorder)
    view = await client.get_trace("T1")
    assert isinstance(view, TraceView) and view.thread_id == "T1"
    assert view.trace["status"] == "answered" and view.evidence_path == "P1 ..."
    assert (await client.get_trace("T2")).trace is None
    assert [(m, p) for m, p, *_ in recorder.requests] == [
        ("GET", "/threads/T1/trace"),
        ("GET", "/threads/T2/trace"),
    ]


async def test_turn_view_accepts_f4_turn_result():
    """Kontrakt UI ↔ API: TurnResult z `trace` i `evidence_path` waliduje się w TurnView."""
    from kovalo.ui.client import TurnView
    from kovalo.workflow.service import TurnResult

    result = TurnResult(
        thread_id="T1",
        status="answered",
        replies=["Kara wynosi 7296,00 PLN."],
        pending=None,
        trace={"status": "limit", "events": [], "answer": None},
        evidence_path="P1  Ile wynosi kara?",
    )
    view = TurnView.model_validate(result.model_dump(mode="json"))
    assert view.evidence_path == "P1  Ile wynosi kara?" and view.trace["status"] == "limit"
    plain = TurnView.model_validate(
        TurnResult(thread_id="T1", status="answered", replies=[], pending=None).model_dump(
            mode="json"
        )
    )
    assert plain.trace is None and plain.evidence_path is None


async def test_tool_end_transport_fields_are_optional(mock_stream):
    """F5: `ToolEventView.transport` i `.invocation_id` opcjonalne (stare zdarzenia bez pól)."""
    final = turn_result(
        status="answered", replies=["ok"], trace={"status": "answered", "events": []}
    )
    body = sse_body(
        ("tool_start", tool_start()),
        ("tool_end", tool_end(transport="mcp", invocation_id="inv-1")),
        ("tool_start", tool_start("c2", "get_invoice_value")),
        ("tool_end", tool_end("c2", "get_invoice_value")),
        ("final", final),
    )
    items = await collect(make_client(mock_stream(("sse", body))))
    ends = [item for kind, item in items if kind == "tool_end"]
    assert (ends[0].transport, ends[0].invocation_id) == ("mcp", "inv-1")
    assert (ends[1].transport, ends[1].invocation_id) == (None, None)
