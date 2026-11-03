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
