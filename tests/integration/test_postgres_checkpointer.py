"""PostgresSaver na prawdziwym Postgresie (kontrakt F3, 6.6 i 14.2). Wymaga docker compose (db)."""

from __future__ import annotations

import pytest

from tests.workflow_fakes import turn

pytestmark = pytest.mark.integration


def _service_on(settings, saver, turns):
    from kovalo.workflow.graph import build_copilot_graph
    from kovalo.workflow.service import CopilotService
    from tests.workflow_fakes import make_deps

    deps, _ = make_deps(settings, turns)
    graph = build_copilot_graph(deps, checkpointer=saver)
    return CopilotService(graph, max_message_chars=settings.workflow.max_message_chars)


def test_d2_replay_on_postgres(pg_settings, gold_dialogues):
    """Odtworzenie D2 na postgres_checkpointer (atrapy modeli) = wynik w pamięci."""
    from kovalo.eval.workflow.runner import run_turn
    from kovalo.workflow.checkpoint import memory_checkpointer, postgres_checkpointer
    from tests.workflow_fakes import gold_turns

    def play(saver, thread):
        service = _service_on(pg_settings, saver, gold_turns(gold_dialogues, "D2"))
        d2 = next(d for d in gold_dialogues if d.id == "D2")
        entries = []
        for t in d2.turns:
            scripted = iter(t.resume)
            entries.append(run_turn(service, thread, t.user, lambda p, s=scripted: next(s)))
        return [(e["status"], e["pauses"], e["approvals"]) for e in entries]

    import uuid

    in_memory = play(memory_checkpointer(), f"mem-{uuid.uuid4().hex}")
    with postgres_checkpointer(pg_settings) as saver:
        on_postgres = play(saver, f"pg-{uuid.uuid4().hex}")
    assert on_postgres == in_memory
    assert on_postgres[-1][0] == "rejected"


def test_pause_survives_new_saver(pg_settings):
    """Pauza w jednym checkpointerze, zamknięcie puli, nowy saver i nowy graf: ten sam thread_id."""
    import uuid

    from kovalo.workflow.checkpoint import postgres_checkpointer

    thread = f"pause-{uuid.uuid4().hex}"
    request = turn(
        kind="approval_request", request_action="new", discount_pct=22, margin_after_pct=10
    )
    with postgres_checkpointer(pg_settings) as saver:
        service = _service_on(pg_settings, saver, [request])
        paused = service.send_message(thread, "Prześlij rabat 22%.")
        assert paused.pending.role == "CEO"

    with postgres_checkpointer(pg_settings) as saver:  # nowa pula, nowy graf (jak po restarcie api)
        service = _service_on(pg_settings, saver, [])  # understand nie może być wywołane
        state = service.get_thread(thread)
        assert state.status == "pending_approval" and state.pending.role == "CEO"
        assert state.request["discount_pct"] == 22
        step = service.resume(thread, value="approved")
        assert step.pending.role == "CFO"
        done = service.resume(thread, value="approved")
        assert done.status == "approved" and done.approvals == {
            "CEO": "approved",
            "CFO": "approved",
        }


def test_setup_idempotent(pg_settings):
    from kovalo.workflow.checkpoint import postgres_checkpointer

    with postgres_checkpointer(pg_settings) as saver:
        saver.setup()
        saver.setup()
    with postgres_checkpointer(pg_settings, setup=True):
        pass
    with postgres_checkpointer(pg_settings, setup=False):
        pass


def test_unreachable_postgres_is_checkpointer_unavailable(monkeypatch):
    from kovalo.config import load_settings
    from kovalo.workflow.checkpoint import CheckpointerUnavailableError, postgres_checkpointer

    monkeypatch.setenv("KOVALO_DB__PORT", "1")
    monkeypatch.setenv("KOVALO_DB__CONNECT_TIMEOUT_S", "1")
    with (
        pytest.raises(CheckpointerUnavailableError),
        postgres_checkpointer(load_settings(google_api_key="test-key")),
    ):
        pass
