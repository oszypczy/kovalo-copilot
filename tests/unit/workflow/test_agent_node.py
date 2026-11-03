"""Węzeł answer_documents z agentem (kontrakt F4, 8.1; port `answer_with_tools` kroku 3 MP5)."""

from __future__ import annotations

import warnings

import pytest

from tests.agent_fakes import agent_answerer, july_events, make_trace
from tests.workflow_fakes import ReplayLLM, graph_with, service_with, turn

DOC = turn(kind="document_question", standalone="Ile wynosi kara za lipiec 2026 r.?")


def doc_trace(answer="Kara wynosi 7296,00 PLN.", status=None, error=None, events=None):
    from kovalo.agents.trace import ModelCallInfo

    trace = make_trace("Q", answer, july_events() if events is None else events, status=status)
    trace.error = error
    trace.model_calls = [
        ModelCallInfo(model="model-a", input_tokens=100, output_tokens=10),
        ModelCallInfo(model="model-b", input_tokens=50, output_tokens=5, reasoning_tokens=3),
    ]
    return trace


def test_agent_answer_and_envelope(settings):
    from kovalo.agents.trace import trace_from_record

    asked: list[str] = []
    service = service_with(settings, [DOC], agent=agent_answerer(doc_trace(), calls=asked))
    result = service.send_message("t", "Ile wynosi kara?")
    assert result.status == "answered" and result.replies == ["Kara wynosi 7296,00 PLN."]
    assert asked == ["Ile wynosi kara za lipiec 2026 r.?"], "agent dostaje standalone z understand"
    values = service.get_values("t")
    record = values["record"]
    assert (record["kind"], record["version"]) == ("agent_trace", 1)
    stored = trace_from_record(record)
    assert stored.answer == "Kara wynosi 7296,00 PLN." and len(stored.events) == 4
    assert record["trace_id"] and record["created_at"]
    agent_calls = [c for c in values["calls"] if c["purpose"] == "agent"]
    assert [c["model"] for c in agent_calls] == ["model-a", "model-b"]
    assert [c["usage"]["input_tokens"] for c in agent_calls] == [100, 50]
    assert agent_calls[1]["usage"]["reasoning_tokens"] == 3
    assert next(c["purpose"] for c in values["calls"]) == "understand"
    assert result.usage.calls == 3 and result.usage.input_tokens >= 150


def test_agent_error_and_limit_replies(settings):
    from kovalo.workflow.replies import AGENT_ERROR_REPLY, AGENT_LIMIT_REPLY

    errored = doc_trace(answer=None, status="error", error="RuntimeError: boom")
    service = service_with(settings, [DOC], agent=agent_answerer(errored))
    result = service.send_message("t", "Ile wynosi kara?")
    assert result.replies == [AGENT_ERROR_REPLY] and result.status == "error"
    assert result.error == "RuntimeError: boom"

    limited = doc_trace(
        answer=None, status="limit", error="osiągnięto limit kroków (24 kroków na to pytanie)"
    )
    service = service_with(settings, [DOC], agent=agent_answerer(limited))
    result = service.send_message("t", "Ile wynosi kara?")
    assert result.replies == [AGENT_LIMIT_REPLY] and result.status == "error"

    late = doc_trace(
        answer="Spóźniona odpowiedź: 7296,00 PLN.", status="limit", error="limit czasu"
    )
    service = service_with(settings, [DOC], agent=agent_answerer(late))
    result = service.send_message("t", "Ile wynosi kara?")
    assert result.replies == ["Spóźniona odpowiedź: 7296,00 PLN."], (
        "odpowiedź zostaje także przy limicie"
    )
    assert result.status == "error"
    assert service.get_values("t")["record"]["trace"]["status"] == "limit", (
        "koperta zachowuje status limit"
    )


def test_reply_texts():
    from kovalo.workflow import replies

    assert replies.AGENT_ERROR_REPLY == (
        "Nie udało się odpowiedzieć na to pytanie. Spróbuj ponownie za chwilę albo zadaj je inaczej."  # noqa: E501
    )
    assert replies.AGENT_LIMIT_REPLY == (
        "Przerwałem pracę: pytanie wymagało więcej kroków albo czasu, niż pozwala limit. "
        "Zadaj węższe pytanie."
    )


def test_agent_exception_becomes_error_trace(settings):
    """Wyjątek agenta (nie powinien wystąpić) → Trace(status='error'), nie awaria tury."""

    def exploding(question, *, on_event=None):
        raise RuntimeError("agent padł")

    service = service_with(settings, [DOC], agent=exploding)
    result = service.send_message("t", "Ile wynosi kara?")
    assert result.status == "error" and "agent padł" in (result.error or "")
    assert result.replies, "użytkownik dostaje komunikat, nie 500"


def test_agent_node_never_touches_request(settings):
    """Wniosek i akceptacje z wcześniejszej tury bez zmian; węzeł zwraca tylko swoje klucze."""
    graph, _ = graph_with(
        settings,
        [
            turn(
                kind="approval_request", request_action="new", discount_pct=14, margin_after_pct=10
            ),
            DOC,
        ],
        agent=agent_answerer(doc_trace()),
    )
    from kovalo.workflow.service import CopilotService

    service = CopilotService(graph)
    service.send_message("t", "Prześlij rabat 14%.")
    done = service.resume("t", value="approved")
    assert done.status == "approved"
    before = service.get_thread("t")
    result = service.send_message("t", "Ile wynosi kara?")
    after = service.get_thread("t")
    assert result.status == "answered"
    assert after.request == before.request and after.approvals == before.approvals
    assert after.approvals == {"Dyrektor handlowy": "approved"}

    cfg = {"configurable": {"thread_id": "u"}}
    from langchain_core.messages import HumanMessage

    graph2, _ = graph_with(settings, [DOC], agent=agent_answerer(doc_trace()))
    updates = [
        chunk["answer_documents"]
        for chunk in graph2.stream(
            {"messages": [HumanMessage("Pytanie?")]}, cfg, stream_mode="updates"
        )
        if "answer_documents" in chunk
    ]
    assert len(updates) == 1
    assert set(updates[0]) <= {"status", "error", "record", "calls", "messages"}
    assert not {"request", "approvals", "decision"} & set(updates[0])


def test_deps_exactly_one_answerer(settings):
    from kovalo.workflow.graph import CopilotDeps
    from tests.workflow_fakes import NoAnswerer

    common = {
        "understand_model": ReplayLLM([]).with_structured_output(object),
        "policy": settings.policy,
        "understand_prompt": "p",
    }
    with pytest.raises(ValueError):
        CopilotDeps(**common)
    with pytest.raises(ValueError):
        CopilotDeps(**common, document_answerer=NoAnswerer(), agent=agent_answerer(doc_trace()))
    assert CopilotDeps(**common, agent=agent_answerer(doc_trace())).agent is not None
    assert CopilotDeps(**common, document_answerer=NoAnswerer()).document_answerer is not None


def test_events_streamed_via_custom_mode(settings):
    from langchain_core.messages import HumanMessage

    graph, _ = graph_with(settings, [DOC], agent=agent_answerer(doc_trace()))
    parts = list(
        graph.stream(
            {"messages": [HumanMessage("Ile wynosi kara?")]},
            {"configurable": {"thread_id": "t"}},
            stream_mode=["custom", "updates"],
            version="v2",
        )
    )
    custom = [p["data"] for p in parts if p["type"] == "custom"]
    assert all(d["kind"] == "agent_event" for d in custom)
    assert [(d["phase"], d["event"]["tool"]) for d in custom] == [
        (phase, tool)
        for tool in (
            "get_contract_terms",
            "get_supplier_otif",
            "get_invoice_value",
            "calculate_otif_penalty",
        )
        for phase in ("start", "end")
    ]
    assert custom[0]["event"]["call_id"] and custom[0]["event"]["arguments"] == {
        "supplier": "Mleczna Dolina"
    }
    order = [p["type"] for p in parts]
    assert order.index("custom") < max(i for i, t in enumerate(order) if t == "updates")


def test_real_agent_inside_graph(settings, agent_for, company_data):
    """AgentRunner z ScriptedModel w węźle grafu na memory_checkpointer: budżet działa, trace
    w stanie, brak ostrzeżeń serializera."""
    from kovalo.workflow.checkpoint import memory_checkpointer
    from kovalo.workflow.graph import CopilotDeps, build_copilot_graph
    from kovalo.workflow.schemas import Turn
    from kovalo.workflow.service import CopilotService
    from tests.agent_fakes import p1_script

    runner, model = agent_for(p1_script(), data=company_data)
    llm = ReplayLLM([DOC])
    deps = CopilotDeps(
        understand_model=llm.with_structured_output(Turn),
        policy=settings.policy,
        understand_prompt=settings.prompts.understand_system,
        agent=runner,
    )
    service = CopilotService(build_copilot_graph(deps, checkpointer=memory_checkpointer()))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = service.send_message("t", "Ile wynosi kara za lipiec?")
        state = service.get_values("t")
    serializer_warnings = [
        str(w.message)
        for w in caught
        if any(k in str(w.message).lower() for k in ("msgpack", "serializ", "unregistered"))
    ]
    assert serializer_warnings == []
    assert result.status == "answered" and "7296,00" in result.replies[0]
    trace = result.trace
    assert trace["status"] == "answered" and trace["steps"] == 9
    penalty = next(e for e in trace["events"] if e["tool"] == "calculate_otif_penalty")
    assert penalty["result"]["penalty_grosze"] == 729600, (
        "prawdziwe narzędzia na prawdziwych danych"
    )
    assert state["record"]["kind"] == "agent_trace"
    assert model.calls == 5 and len(trace["model_calls"]) == 5
    assert result.evidence_path and "calculate_otif_penalty" in result.evidence_path
