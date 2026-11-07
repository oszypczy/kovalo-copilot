"""Agent w grafie na PostgresSaver (kontrakt F4, 15.2). Wymaga docker compose (db), bez LLM-a:
atrapa modelu ze skryptem P1 na prawdziwych danych firmy."""

from __future__ import annotations

import time
import uuid

import pytest

from tests.workflow_fakes import turn

pytestmark = pytest.mark.integration

DOC = turn(kind="document_question", standalone="Ile wynosi kara za lipiec 2026 r.?")


def _service_on(settings, saver, agent):
    from kovalo.workflow.graph import build_copilot_graph
    from kovalo.workflow.service import CopilotService
    from tests.workflow_fakes import make_deps

    deps, _ = make_deps(settings, [DOC] if agent is not None else [], agent=agent)
    graph = build_copilot_graph(deps, checkpointer=saver)
    return CopilotService(graph, max_message_chars=settings.workflow.max_message_chars)


def _p1_runner(settings, gate=None):
    from kovalo.company.data import load_company_data
    from tests.agent_fakes import CountingRetriever, agent_runner, p1_script

    data = load_company_data(settings.paths.company_data_dir)
    runner, _ = agent_runner(settings, data, CountingRetriever(), p1_script(), variant="ours")
    return runner


def _wait_idle(service, timeout=15.0):
    deadline = time.monotonic() + timeout
    while service._locks and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not service._locks


def test_agent_turn_trace_survives_new_saver(pg_settings):
    """Tura z AgentRunner (skrypt P1) na postgres_checkpointer; nowy saver i nowy graf (jak po
    restarcie api): get_trace zwraca ten sam trace."""
    from kovalo.workflow.checkpoint import postgres_checkpointer

    thread = f"agent-{uuid.uuid4().hex}"
    with postgres_checkpointer(pg_settings) as saver:
        service = _service_on(pg_settings, saver, _p1_runner(pg_settings))
        result = service.send_message(thread, "Ile wynosi kara za lipiec?")
        assert result.status == "answered" and result.trace["status"] == "answered"
        before = service.get_trace(thread)
    penalty = next(e for e in before.trace["events"] if e["tool"] == "calculate_otif_penalty")
    assert penalty["result"]["penalty_grosze"] == 729600

    with postgres_checkpointer(pg_settings) as saver:  # nowa pula, nowy graf, bez agenta
        service = _service_on(pg_settings, saver, None)
        after = service.get_trace(thread)
        state = service.get_thread(thread)
    assert after.trace == before.trace and after.trace_id == before.trace_id
    assert after.evidence_path == before.evidence_path and after.created_at is not None
    assert state.status == "answered" and "7296" in state.messages[-1].content


def test_stream_in_process_with_postgres(pg_settings):
    """start_message + close() po pierwszym zdarzeniu: tura dokończona w Postgresie."""
    from kovalo.workflow.checkpoint import postgres_checkpointer

    thread = f"stream-{uuid.uuid4().hex}"
    with postgres_checkpointer(pg_settings) as saver:
        service = _service_on(pg_settings, saver, _p1_runner(pg_settings))
        stream = service.start_message(thread, "Ile wynosi kara za lipiec?")
        first = stream.next_event(timeout=30)
        assert first is not None and first.type == "tool_start"
        stream.close()
        _wait_idle(service)
        state = service.get_thread(thread)
        assert state.status == "answered"
        assert service.get_trace(thread).trace["status"] == "answered"
        service.close()
    with postgres_checkpointer(pg_settings) as saver:
        state = _service_on(pg_settings, saver, None).get_thread(thread)
    assert state.status == "answered" and "7296" in state.messages[-1].content


def test_mcp_agent_turn_trace_survives_new_saver(pg_settings):
    """F5 (kontrakt 13.2): tura z AgentRunner (skrypt P1) z narzędziami MCP (serwer w pamięci,
    prawdziwe dane firmy) na postgres_checkpointer; nowy saver -> `get_trace` ma `transport 'mcp'`
    i `invocation_id` zapisane w checkpoincie."""
    from kovalo.agents.agent import AgentRunner, build_agent
    from kovalo.tools.specs import OUR_TOOLS
    from kovalo.workflow.checkpoint import postgres_checkpointer
    from tests.agent_fakes import ScriptedModel, p1_script
    from tests.mcp_fakes import connect, mcp_server

    thread = f"mcp-agent-{uuid.uuid4().hex}"
    with connect(mcp_server(pg_settings, "ours")) as conn:
        agent = build_agent(
            ScriptedModel(script=p1_script()),
            conn.tools(names=OUR_TOOLS),
            system_prompt=pg_settings.prompts.agent_system,
        )
        runner = AgentRunner(agent, pg_settings.agent)
        with postgres_checkpointer(pg_settings) as saver:
            service = _service_on(pg_settings, saver, runner)
            result = service.send_message(thread, "Ile wynosi kara za lipiec?")
            assert result.status == "answered" and result.trace["status"] == "answered"
            before = service.get_trace(thread)
        invocation_ids = [c["invocation_id"] for c in conn.captures]
    top = [e for e in before.trace["events"] if e["parent_call_id"] is None]
    assert len(top) == 4 and all(e["transport"] == "mcp" for e in top)
    assert [e["invocation_id"] for e in top] == invocation_ids
    penalty = next(e for e in top if e["tool"] == "calculate_otif_penalty")
    assert penalty["result"]["penalty_grosze"] == 729600

    with postgres_checkpointer(pg_settings) as saver:  # nowa pula, nowy graf, bez agenta
        after = _service_on(pg_settings, saver, None).get_trace(thread)
    assert after.trace == before.trace
    assert all(
        e["transport"] == "mcp" for e in after.trace["events"] if e["parent_call_id"] is None
    )
