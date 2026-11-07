"""Zamknięcie przerwanej tury na PostgresSaver (kontrakt F6, 7 i 12). Wymaga docker compose (db).

Skryptowany graf (atrapy modeli), prawdziwy checkpointer: 0 wywołań LLM-a.
"""

from __future__ import annotations

import uuid

import pytest

from tests.workflow_fakes import make_deps, turn

pytestmark = pytest.mark.integration


class _Abort(BaseException):
    """Awaria spoza Exception: proces ginie w trakcie węzła, tura zostaje przerwana."""


def _service_on(settings, saver, turns, *, answerer=None):
    from kovalo.workflow.graph import build_copilot_graph
    from kovalo.workflow.service import CopilotService

    deps, _ = make_deps(settings, turns, answerer=answerer)
    graph = build_copilot_graph(deps, checkpointer=saver)
    return CopilotService(graph, max_message_chars=settings.workflow.max_message_chars)


def test_interrupted_turn_closed_on_postgres(pg_settings):
    from kovalo.workflow.checkpoint import postgres_checkpointer
    from kovalo.workflow.replies import INTERRUPTED_TURN_REPLY

    thread = f"interrupted-{uuid.uuid4().hex}"
    cfg = {"configurable": {"thread_id": thread}}

    def dying(question):
        raise _Abort

    with postgres_checkpointer(pg_settings) as saver:
        first = _service_on(
            pg_settings,
            saver,
            [turn(kind="document_question", standalone="P?")],
            answerer=dying,
        )
        with pytest.raises(_Abort):
            first.send_message(thread, "Pytanie, przy którym proces ginie")
        assert first.graph.get_state(cfg).next, "przerwana tura jest w checkpoincie"

    # restart: nowa pula połączeń, nowy graf i nowy serwis, ten sam thread_id
    with postgres_checkpointer(pg_settings) as saver:
        second = _service_on(
            pg_settings, saver, [turn(kind="policy_question", discount_pct=3)]
        )  # NoAnswerer: nic nie może wrócić do przerwanego węzła
        assert second.graph.get_state(cfg).next, "po restarcie wątek nadal wygląda na przerwany"
        result = second.send_message(thread, "Kto zatwierdza 3%?")  # bez 409 thread_stuck
        assert result.status == "answered" and len(result.replies) == 2
        assert result.replies[0] == INTERRUPTED_TURN_REPLY
        assert "3%" in result.replies[1]
        snapshot = second.graph.get_state(cfg)
        assert snapshot.next == () and not snapshot.interrupts

        history = list(second.graph.get_state_history(cfg))
        closing = [h for h in history if h.metadata.get("source") == "update"]
        # `metadata["writes"]` nie jest zapisywane przez langgraph 1.2.x dla update_state
        # (None na Postgresie), więc as_node="reply_other" widać po `next == ()` (krawędź do END).
        assert closing, "zamknięcie zapisane przez update_state (source == update)"
        assert closing[0].next == ()
        assert closing[0].values.get("error") == "turn_interrupted"
        assert closing[0].values.get("status") == "error"
