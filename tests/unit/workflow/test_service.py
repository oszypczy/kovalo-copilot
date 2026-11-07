"""CopilotService: jedyny punkt wejścia do grafu (kontrakt F3, 6.7)."""

from __future__ import annotations

import threading

import pytest

from tests.fakes import TOY_CHUNKS
from tests.workflow_fakes import (
    ReplayLLM,
    answers,
    graph_with,
    service_and_llm,
    service_with,
    toy_answerer,
    turn,
)


def new_request(**fields):
    return turn(kind="approval_request", request_action="new", **fields)


def test_paused_thread_refuses_new_message(settings):
    """Port `test_a_paused_thread_refuses_a_new_message` kursu."""
    from kovalo.eval.workflow.runner import run_turn
    from kovalo.workflow.service import ThreadPausedError

    service = service_with(settings, [new_request(discount_pct=14, margin_after_pct=10)])
    first = service.send_message("t", "Prześlij rabat 14%.")
    assert first.status == "pending_approval" and first.pending.role == "Dyrektor handlowy"
    with pytest.raises(ThreadPausedError) as info:
        service.send_message("t", "Dzięki!")
    assert info.value.pending.role == "Dyrektor handlowy"
    with pytest.raises(RuntimeError, match="pauza"):
        run_turn(service, "t", "Dzięki!", lambda p: None)


def test_unexpected_extra_pause_cancels(settings):
    """Port `test_an_unexpected_extra_pause_keeps_the_evidence_and_cancels` przez run_turn."""
    from kovalo.eval.workflow.runner import run_turn

    service = service_with(
        settings, [new_request(discount_pct=14, margin_after_pct=7), turn(kind="other")]
    )
    t = run_turn(service, "t", "Prześlij rabat 14%, marża 7%.", answers("approved"))
    assert t["turn"]["kind"] == "approval_request", "faktyczne zrozumienie wiadomości zostaje"
    assert t["pauses"] == [["approval", "Dyrektor handlowy"], ["approval", "CFO"]]
    assert t["status"] == "error" and "CFO" in t["error"] and "przy pauzie" in t["error"]
    later = run_turn(service, "t", "Dzięki!", lambda p: None)
    assert later["status"] == "answered", (
        "zawieszoną pauzę anulowano, więc wątek przyjmuje wiadomości"
    )


def test_resume_requires_pause_and_one_input(settings):
    from kovalo.workflow.service import NotPausedError

    service = service_with(settings, [new_request(discount_pct=14, margin_after_pct=10)])
    with pytest.raises(NotPausedError):
        service.resume("t", value="approved")
    service.send_message("t", "Prześlij rabat 14%.")
    with pytest.raises(ValueError):
        service.resume("t", value="approved", text="zatwierdź")
    with pytest.raises(ValueError):
        service.resume("t")
    assert service.get_thread("t").pending is not None, "błędne wejście nie rusza pauzy"
    done = service.resume("t", text="zatwierdź")
    assert done.status == "approved"
    with pytest.raises(NotPausedError):
        service.resume("t", value="approved")


def test_stale_pause_ref(settings):
    from kovalo.workflow.service import PauseRef, StalePauseError

    service = service_with(settings, [new_request(discount_pct=22, margin_after_pct=10)])
    paused = service.send_message("t", "Prześlij rabat 22%.")
    assert paused.pending.ref == PauseRef(type="approval", key="CEO", revision=1)
    with pytest.raises(StalePauseError) as info:
        service.resume(
            "t", value="approved", pause_ref=PauseRef(type="approval", key="CFO", revision=1)
        )
    assert info.value.pending.role == "CEO"
    with pytest.raises(StalePauseError):
        service.resume(
            "t", value="approved", pause_ref=PauseRef(type="approval", key="CEO", revision=2)
        )
    state = service.get_thread("t")
    assert state.pending.role == "CEO" and state.approvals == {}, "stan bez zmian"
    next_pause = service.resume("t", value="approved", pause_ref=paused.pending.ref)
    assert next_pause.pending.role == "CFO" and next_pause.pending.ref.key == "CFO"


def test_thread_busy(settings):
    from kovalo.workflow.service import ThreadBusyError

    entered, release = threading.Event(), threading.Event()
    llm = ReplayLLM(
        [
            turn(kind="policy_question", discount_pct=22),
            turn(kind="policy_question", discount_pct=3),
        ],
        gate=(entered, release),
    )
    service = service_with(settings, llm=llm)
    outcome: dict[str, object] = {}

    def slow_turn():
        outcome["result"] = service.send_message("busy", "Kto zatwierdza 22%?")

    worker = threading.Thread(target=slow_turn)
    worker.start()
    try:
        assert entered.wait(timeout=10), "tura nie weszła do modelu"
        with pytest.raises(ThreadBusyError):
            service.send_message("busy", "Kto zatwierdza 3%?")
        other = service.send_message("other", "Kto zatwierdza 3%?")  # inny wątek działa
        assert other.status == "answered"
    finally:
        release.set()
        worker.join(timeout=10)
    assert outcome["result"].status == "answered"


def test_turn_result_fields(settings):
    """replies tylko z tej tury; pending.summary i options; sources tylko cytowane chunki."""
    from kovalo.rag.records import Answer

    llm = ReplayLLM(
        [
            turn(kind="document_question", standalone="Kto zatwierdza wyjazdy zagraniczne?"),
            new_request(discount_pct=22, margin_after_pct=10, customer="Carrefour Express"),
        ],
        answer=Answer(answer="Zgoda CFO.", citations=["TST-001"], answerable=True),
    )
    service = service_with(
        settings, llm=llm, answerer=toy_answerer(settings, llm, chunks=TOY_CHUNKS, k=2)
    )
    docs = service.send_message("t", "Kto zatwierdza wyjazdy?")
    assert docs.status == "answered" and docs.pending is None and len(docs.replies) == 1
    assert "Zgoda CFO." in docs.replies[0] and "TST-001" in docs.replies[0]
    assert [s.chunk_id for s in docs.sources] == ["TST-001#ctx000"], "tylko cytowane chunki"
    assert docs.sources[0].doc_id == "TST-001" and docs.sources[0].text
    assert docs.usage.calls == 2 and docs.turn["kind"] == "document_question"
    assert docs.thread_id == "t" and docs.error is None and docs.resume_rejected is False

    approval = service.send_message("t", "Prześlij rabat 22% dla Carrefour Express.")
    assert approval.status == "pending_approval" and len(approval.replies) == 1
    assert "Wymagana akceptacja" in approval.replies[0]
    assert approval.sources == [] and approval.usage.calls == 1
    pending = approval.pending
    assert (pending.type, pending.role, pending.revision) == ("approval", "CEO", 1)
    assert pending.approvers == ["CEO", "CFO"] and pending.options == [
        "approved",
        "rejected",
        "cancel",
    ]
    assert "22%" in pending.summary and "Carrefour Express" in pending.summary
    assert pending.request["discount_pct"] == 22
    assert approval.request["revision"] == 1 and approval.decision["approvers"] == ["CEO", "CFO"]
    assert approval.approvals == {}


def test_clarification_pending_shape(settings):
    service = service_with(settings, [new_request(discount_pct=8)])
    pending = service.send_message("t", "Prześlij rabat 8%.").pending
    assert (pending.type, pending.field) == ("clarification", "margin_after_pct")
    assert pending.options == ["cancel"] and "marża" in pending.question
    assert pending.ref.type == "clarification" and pending.ref.key == "margin_after_pct"


class _RawBinder:
    """toy_answerer woła `with_structured_output(Answer)`; tu z include_raw (tokeny w RunRecord)."""

    def __init__(self, model):
        self.model = model

    def with_structured_output(self, schema, **kwargs):
        return self.model.with_structured_output(schema, include_raw=True)


def test_turn_usage_sums_provider_tokens(settings):
    from kovalo.llm import override_chat_model
    from kovalo.rag.records import Answer
    from kovalo.workflow.checkpoint import memory_checkpointer
    from kovalo.workflow.factory import build_copilot
    from kovalo.workflow.schemas import Turn
    from kovalo.workflow.service import CopilotService
    from tests.fakes import FakeStructuredChatModel

    fake = FakeStructuredChatModel(
        results={
            Turn: turn(kind="document_question", standalone="Kto zatwierdza wyjazdy zagraniczne?"),
            Answer: Answer(answer="CFO.", citations=["TST-001"], answerable=True),
        }
    )
    answerer = toy_answerer(settings, _RawBinder(fake), chunks=TOY_CHUNKS, k=2)
    with override_chat_model(fake):
        graph = build_copilot(
            settings, checkpointer=memory_checkpointer(), document_answerer=answerer
        )
        result = CopilotService(graph).send_message("t", "Kto zatwierdza wyjazdy?")
    usage = result.usage
    assert usage.calls == 2
    assert (usage.input_tokens, usage.output_tokens, usage.reasoning_tokens) == (240, 80, 50)
    assert set(usage.models) == {"fake-model"}


def test_get_thread_unknown_and_paused(settings):
    service = service_with(settings, [new_request(discount_pct=14, margin_after_pct=10)])
    unknown = service.get_thread("nigdy-nie-uzywany")
    assert unknown.thread_id == "nigdy-nie-uzywany" and unknown.status is None
    assert unknown.pending is None and unknown.messages == [] and unknown.approvals == {}
    assert unknown.request is None and unknown.decision is None

    service.send_message("t", "Prześlij rabat 14%.")
    state = service.get_thread("t")
    assert state.status == "pending_approval" and state.pending.role == "Dyrektor handlowy"
    assert [m.role for m in state.messages] == ["user", "assistant"]
    assert state.messages[0].content == "Prześlij rabat 14%."
    assert state.request["discount_pct"] == 14
    assert state.decision["approvers"] == ["Dyrektor handlowy"]


def test_get_thread_does_not_call_the_graph(settings):
    service, llm = service_and_llm(settings, [new_request(discount_pct=14, margin_after_pct=10)])
    service.send_message("t", "Prześlij rabat 14%.")
    before = llm.calls["Turn"]
    service.get_thread("t")
    service.get_thread("t")
    assert llm.calls["Turn"] == before


def test_message_validation(settings):
    service = service_with(settings, [turn(kind="other")])
    for bad in ("", "   ", "\n\t", "x" * (settings.workflow.max_message_chars + 1)):
        with pytest.raises(ValueError):
            service.send_message("t", bad)
    assert service.get_thread("t").status is None, "odrzucona wiadomość nie dotyka wątku"
    ok = service.send_message("t", "x" * settings.workflow.max_message_chars)
    assert ok.status == "answered"


class _Abort(BaseException):
    """Awaria spoza Exception: węzeł jej nie łapie, więc wychodzi z grafu."""


# F6: zamiast `thread_stuck` przerwana tura jest zamykana przy następnej wiadomości (test
# `test_thread_stuck_after_failure_in_a_node` z F5 zastąpiły `test_interrupted_turn_*` niżej).


def test_error_classes_hierarchy():
    from kovalo.workflow import service as s

    for name in (
        "ThreadPausedError",
        "NotPausedError",
        "StalePauseError",
        "ThreadBusyError",
        "ThreadStuckError",
    ):
        assert issubclass(getattr(s, name), RuntimeError), name


# ---------------------------------------------------------------------------
# F4: strumień tury, trace, R1-2 i R1-3 (kontrakt F4, 8.3-8.5)
# ---------------------------------------------------------------------------

import time  # noqa: E402

from tests.agent_fakes import (  # noqa: E402
    agent_answerer,
    july_events,
    make_trace,
)

DOC_TURN = turn(kind="document_question", standalone="Ile wynosi kara za lipiec 2026 r.?")


def _passage(doc_id, n, text):
    return {"doc_id": doc_id, "chunk_id": f"{doc_id}#ctx{n:03d}", "text": text}


def search_event(*passages):
    from tests.agent_fakes import event

    return event(
        "search_documents",
        {"status": "ok", "query": "q", "passages": list(passages)},
        question="q",
        call_id="s1",
    )


def agent_trace(answer="Kara wynosi 7296,00 PLN (CON-101).", events=None):
    from kovalo.agents.trace import ModelCallInfo

    trace = make_trace("Q", answer, events if events is not None else july_events())
    trace.model_calls = [ModelCallInfo(model="m", input_tokens=10, output_tokens=2)]
    return trace


def wait_until_idle(service, timeout=10.0):
    deadline = time.monotonic() + timeout
    while service._locks and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not service._locks, "tura nie dobiegła końca"


def test_stream_events_then_final(settings):
    from kovalo.agents.show import brief
    from kovalo.workflow.service import TurnResult

    events = [
        search_event(_passage("CON-101", 4, "mniej niż 94%"), _passage("CON-103", 3, "98% palet")),
        search_event(_passage("CON-101", 4, "mniej niż 94%")),
        *july_events()[:2],
    ]
    events[1].call_id = "s2"
    trace = agent_trace(events=events)
    service = service_with(settings, [DOC_TURN], agent=agent_answerer(trace))
    stream = service.start_message("t", "Ile wynosi kara?")
    assert stream.thread_id == "t"
    items = list(stream)
    types = [i.type for i in items]
    assert types == ["tool_start", "tool_end"] * 4 + ["final"]
    start, end = items[0], items[1]
    assert set(start.data) >= {"call_id", "parent_call_id", "agent", "tool", "arguments", "origin"}
    assert start.data["tool"] == "search_documents" and start.data["origin"] == "data"
    assert set(end.data) >= {
        "call_id",
        "parent_call_id",
        "tool",
        "status",
        "brief",
        "result",
        "error",
    }
    assert end.data["brief"] == brief(end.data["result"]) and end.data["status"] == "ok"
    final = TurnResult.model_validate(items[-1].data)
    assert final.status == "answered" and final.thread_id == "t"
    assert final.trace["status"] == "answered" and len(final.trace["events"]) == 4
    assert final.evidence_path and "search_documents" in final.evidence_path
    assert [(s.chunk_id, s.score) for s in final.sources] == [
        ("CON-101#ctx004", None),
        ("CON-103#ctx003", None),
    ], "fragmenty z udanych wyszukiwań, w kolejności wywołań, bez powtórzeń chunk_id"
    assert final.usage.calls == 2 and stream.result() == final


def test_stream_ends_with_final_and_result_after_iteration(settings):
    service = service_with(settings, [DOC_TURN], agent=agent_answerer(agent_trace()))
    stream = service.start_message("t", "Ile wynosi kara?")
    assert stream.next_event(timeout=10) is not None
    list(stream)
    assert stream.result().status == "answered"


def test_stream_errors_are_eager(settings):
    """ThreadPausedError, ThreadBusyError, ValueError i inne błędy sprawdzalne przed turą lecą z
    start_message, zanim powstanie strumień."""
    from kovalo.workflow.service import ThreadBusyError, ThreadPausedError

    service = service_with(
        settings,
        [
            new_request(discount_pct=14, margin_after_pct=10),
            DOC_TURN,
        ],
        agent=agent_answerer(agent_trace()),
    )
    service.send_message("p", "Prześlij rabat 14%.")
    with pytest.raises(ThreadPausedError):
        service.start_message("p", "Dzięki!")
    with pytest.raises(ValueError):
        service.start_message("e", "   ")
    with pytest.raises(ValueError):
        service.start_message("e", "x" * (settings.workflow.max_message_chars + 1))

    entered, release = threading.Event(), threading.Event()
    inner = agent_answerer(agent_trace())

    def gated(question, *, on_event=None, callbacks=None):
        entered.set()
        release.wait(timeout=10)
        return inner(question, on_event=on_event)

    busy_service = service_with(settings, [DOC_TURN], agent=gated)
    stream = busy_service.start_message("b", "Ile wynosi kara?")
    try:
        assert entered.wait(timeout=10)
        with pytest.raises(ThreadBusyError):
            busy_service.start_message("b", "Jeszcze raz")
        with pytest.raises(ThreadBusyError):
            busy_service.start_resume("b", value="approved")
    finally:
        release.set()
    assert stream.result().status == "answered"


def test_start_resume_streams_final(settings):
    service = service_with(settings, [new_request(discount_pct=14, margin_after_pct=10)])
    service.send_message("t", "Prześlij rabat 14%.")
    stream = service.start_resume("t", value="approved")
    items = list(stream)
    assert [i.type for i in items] == ["final"]
    assert items[0].data["status"] == "approved" and items[0].data["usage"] is None
    from kovalo.workflow.service import NotPausedError

    with pytest.raises(NotPausedError):
        service.start_resume("t", value="approved")


def test_closing_stream_does_not_stop_turn(settings):
    """ADR-0012 pkt 3: rozłączony klient nie przerywa tury; odpowiedź jest w checkpoincie."""
    release = threading.Event()
    inner = agent_answerer(agent_trace(answer="Odpowiedź dla nieobecnego klienta."))

    def slow(question, *, on_event=None, callbacks=None):
        first = inner(question, on_event=on_event)
        release.wait(timeout=10)
        return first

    service = service_with(settings, [DOC_TURN, turn(kind="other")], agent=slow)
    stream = service.start_message("t", "Ile wynosi kara?")
    first = stream.next_event(timeout=10)
    assert first is not None and first.type == "tool_start"
    stream.close()
    release.set()
    wait_until_idle(service)
    state = service.get_thread("t")
    assert state.status == "answered"
    assert state.messages[-1].content == "Odpowiedź dla nieobecnego klienta."
    again = service.send_message("t", "Dzięki!")  # brak ThreadStuckError
    assert again.status == "answered"


def test_send_message_equals_stream_final(settings):
    trace = agent_trace()
    a = service_with(settings, [DOC_TURN], agent=agent_answerer(trace))
    b = service_with(settings, [DOC_TURN], agent=agent_answerer(trace))
    direct = a.send_message("a", "Ile wynosi kara?")
    streamed = b.start_message("a", "Ile wynosi kara?").result()
    assert direct.model_dump(mode="json") == streamed.model_dump(mode="json")


def test_get_trace_last_agent_turn(settings):
    service = service_with(
        settings,
        [DOC_TURN, turn(kind="policy_question", discount_pct=22)],
        agent=agent_answerer(agent_trace()),
    )
    assert service.get_trace("nowy").trace is None
    service.send_message("t", "Ile wynosi kara?")
    first = service.get_trace("t")
    assert first.thread_id == "t" and first.trace["status"] == "answered"
    assert first.trace_id and first.created_at is not None
    assert first.question == "Ile wynosi kara za lipiec 2026 r.?"
    assert first.evidence_path and "calculate_otif_penalty" in first.evidence_path
    service.send_message("t", "Kto zatwierdza 22%?")
    after = service.get_thread("t")
    assert after.status == "answered"
    last = service.get_trace("t")
    assert last.trace_id == first.trace_id, "tura o politykę nie kasuje ostatniego trace agenta"
    assert last.trace == first.trace


def test_get_trace_without_agent_turn(settings):
    service = service_with(settings, [turn(kind="policy_question", discount_pct=22)])
    service.send_message("t", "Kto zatwierdza 22%?")
    view = service.get_trace("t")
    assert view.thread_id == "t" and view.trace is None and view.trace_id is None
    assert view.evidence_path is None and view.question is None and view.created_at is None


def test_get_trace_ignores_rag_records(settings):
    """record RAG (nie koperta agent_trace) nie jest trace."""
    from kovalo.rag.records import Answer

    llm = ReplayLLM(
        [turn(kind="document_question", standalone="Kto zatwierdza wyjazdy zagraniczne?")],
        answer=Answer(answer="Zgoda CFO.", citations=["TST-001"], answerable=True),
    )
    service = service_with(
        settings, llm=llm, answerer=toy_answerer(settings, llm, chunks=TOY_CHUNKS, k=2)
    )
    service.send_message("t", "Kto zatwierdza wyjazdy?")
    assert service.get_trace("t").trace is None


def test_close_waits_for_turns(settings):
    release = threading.Event()
    inner = agent_answerer(agent_trace())

    def slow(question, *, on_event=None, callbacks=None):
        release.wait(timeout=10)
        return inner(question, on_event=on_event)

    service = service_with(settings, [DOC_TURN], agent=slow)
    stream = service.start_message("t", "Ile wynosi kara?")
    timer = threading.Timer(0.2, release.set)
    timer.start()
    service.close()  # executor.shutdown(wait=True)
    assert stream.result().status == "answered"
    assert not service._locks


# --- R1-2: błędy checkpointera w trakcie pracy --------------------------------------------------


def _broken_graph(settings, exc, *, on="get_tuple"):
    from langgraph.checkpoint.memory import InMemorySaver

    from kovalo.workflow.checkpoint import serializer
    from tests.workflow_fakes import graph_with

    class BrokenSaver(InMemorySaver):
        broken = True

        def get_tuple(self, config):
            if on == "get_tuple" and self.broken:
                raise exc
            return super().get_tuple(config)

        def list(self, config, **kwargs):
            if on in ("get_tuple", "list") and self.broken:
                raise exc
            return super().list(config, **kwargs)

        def put(self, config, checkpoint, metadata, new_versions):
            if on == "put" and self.broken:
                raise exc
            return super().put(config, checkpoint, metadata, new_versions)

    saver = BrokenSaver(serde=serializer())
    graph, _ = graph_with(
        settings,
        [turn(kind="other")],
        checkpointer=saver,
    )
    return graph, saver


def _lost_connection_errors():
    import psycopg
    import psycopg_pool

    return [
        psycopg.OperationalError("connection to server at postgres:5432 failed"),
        psycopg.InterfaceError("connection is closed"),
        psycopg_pool.PoolTimeout("couldn't get a connection after 30 sec"),
        ConnectionRefusedError("odmowa"),
        TimeoutError("timeout"),
    ]


@pytest.mark.parametrize("index", range(5))
def test_checkpointer_errors_become_unavailable(settings, index):
    """R1-2: saver zgłaszający utratę połączenia → CheckpointerUnavailableError w get_state,
    get_state_history i start_message; komunikat bez DSN."""
    from kovalo.workflow.checkpoint import CheckpointerUnavailableError
    from kovalo.workflow.service import CopilotService

    exc = _lost_connection_errors()[index]
    graph, _ = _broken_graph(settings, exc)
    service = CopilotService(graph, max_message_chars=settings.workflow.max_message_chars)
    for call in (
        lambda: service.get_thread("t"),
        lambda: service.get_trace("t"),
        lambda: service.send_message("t", "Cześć"),
        lambda: service.start_message("t", "Cześć"),
        lambda: service.resume("t", value="approved"),
    ):
        with pytest.raises(CheckpointerUnavailableError) as info:
            call()
        assert "postgresql://" not in str(info.value) and "password" not in str(info.value).lower()
    assert not service._locks, "błąd checkpointera nie zostawia blokady"


def test_connection_errors_constant():
    import psycopg
    import psycopg_pool

    from kovalo.workflow.checkpoint import CONNECTION_ERRORS

    assert set(CONNECTION_ERRORS) == {
        psycopg.OperationalError,
        psycopg.InterfaceError,
        psycopg_pool.PoolTimeout,
        OSError,
        TimeoutError,
    }


def test_connection_lost_inside_the_turn_is_an_error_event(settings):
    """Utrata połączenia w wątku roboczym (zapis checkpointu) → zdarzenie error copilot_unavailable."""  # noqa: E501
    import psycopg

    from kovalo.workflow.checkpoint import CheckpointerUnavailableError
    from kovalo.workflow.service import CopilotService

    graph, _ = _broken_graph(
        settings, psycopg.OperationalError("server closed the connection"), on="put"
    )
    service = CopilotService(graph)
    stream = service.start_message("t", "Cześć")
    items = list(stream)
    assert items[-1].type == "error" and items[-1].data["error"] == "copilot_unavailable"
    assert "postgresql://" not in str(items[-1].data)
    with pytest.raises(CheckpointerUnavailableError):
        stream.result()
    wait_until_idle(service)


def test_internal_error_event_hides_details(settings):
    from kovalo.workflow.service import CopilotService
    from tests.workflow_fakes import graph_with

    class Boom(BaseException):
        pass

    def exploding(question, *, on_event=None, callbacks=None):
        raise Boom

    # wyjątek spoza Exception z węzła: tura kończy się zdarzeniem error 'internal_error'
    graph, _ = graph_with(settings, [DOC_TURN], agent=exploding)
    service = CopilotService(graph)
    items = list(service.start_message("t", "Ile wynosi kara?"))
    assert items[-1].type == "error" and items[-1].data["error"] == "internal_error"
    wait_until_idle(service)


# --- R1-3: blokady wątków bez wycieku -----------------------------------------------------------


def test_locks_are_released(settings):
    """100 tur w różnych wątkach: `_locks` puste; ThreadBusyError nadal działa."""
    from concurrent.futures import ThreadPoolExecutor

    from kovalo.workflow.service import ThreadBusyError, ThreadPausedError

    service = service_with(settings, [turn(kind="other")] * 100)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: service.send_message(f"t{i}", "Dzięki!"), range(100)))
    assert all(r.status == "answered" for r in results)
    assert len(service._locks) == 0

    for bad in ("", "   "):
        with pytest.raises(ValueError):
            service.send_message("x", bad)
    assert len(service._locks) == 0

    paused = service_with(settings, [new_request(discount_pct=14, margin_after_pct=10)])
    paused.send_message("p", "Prześlij rabat 14%.")
    with pytest.raises(ThreadPausedError):
        paused.send_message("p", "Dzięki!")
    assert len(paused._locks) == 0, "odrzucona wiadomość zwalnia blokadę"

    entered, release = threading.Event(), threading.Event()
    llm = ReplayLLM([turn(kind="other"), turn(kind="other")], gate=(entered, release))
    busy = service_with(settings, llm=llm)
    worker = threading.Thread(target=lambda: busy.send_message("b", "Dzięki!"))
    worker.start()
    try:
        assert entered.wait(timeout=10)
        assert len(busy._locks) == 1
        with pytest.raises(ThreadBusyError):
            busy.send_message("b", "Dzięki!")
        assert len(busy._locks) == 1, "nieudany acquire nie usuwa cudzej blokady"
    finally:
        release.set()
        worker.join(timeout=10)
    assert len(busy._locks) == 0


def test_tool_end_carries_transport(settings):
    """F5 (kontrakt 9.2): `tool_end.data` ma `transport` i `invocation_id` ze zdarzenia trace;
    `tool_start` bez zmian (transport znany dopiero po wywołaniu); wywołania bezpośrednie: None."""
    events = july_events()
    events[0].transport, events[0].invocation_id = "mcp", "inv-0001"
    trace = agent_trace(events=events)
    service = service_with(settings, [DOC_TURN], agent=agent_answerer(trace))
    items = list(service.start_message("t", "Ile wynosi kara?"))
    starts = [i for i in items if i.type == "tool_start"]
    ends = [i for i in items if i.type == "tool_end"]
    assert len(ends) == 4
    assert ends[0].data["transport"] == "mcp" and ends[0].data["invocation_id"] == "inv-0001"
    for end in ends[1:]:
        assert end.data["transport"] is None and end.data["invocation_id"] is None
    assert all("transport" not in s.data for s in starts), "tool_start bez zmian"
    final = items[-1]
    stored = final.data["trace"]["events"][0]
    assert (stored["transport"], stored["invocation_id"]) == ("mcp", "inv-0001")


# ---------------------------------------------------------------------------
# F6: koszt tury (kontrakt F6, 6) i zamknięcie przerwanej tury (kontrakt F6, 7)
# ---------------------------------------------------------------------------

import logging  # noqa: E402

from langchain_core.runnables import RunnableLambda  # noqa: E402


def _two_model_trace():
    """Agent: tura na modelu głównym i tura na fallbacku (dwa modele w jednej turze)."""
    from kovalo.agents.trace import ModelCallInfo

    trace = agent_trace()
    trace.model_calls = [
        ModelCallInfo(model="gemini-3.5-flash-lite", input_tokens=1000, output_tokens=100),
        ModelCallInfo(
            model="gemini-3.6-flash", input_tokens=2000, output_tokens=200, reasoning_tokens=30
        ),
    ]
    return trace


def _priced_service(settings, trace, *, pricing):
    from kovalo.workflow.service import CopilotService

    graph, _ = graph_with(settings, [DOC_TURN], agent=agent_answerer(trace))
    return CopilotService(graph, pricing=pricing)


def test_usage_has_by_model_and_cost(settings):
    from kovalo.cost.usage import price_usage

    service = _priced_service(settings, _two_model_trace(), pricing=settings.cost)
    usage = service.send_message("t", "Ile wynosi kara?").usage
    named = [m for m in usage.by_model if m.model is not None]
    assert [m.model for m in named] == ["gemini-3.5-flash-lite", "gemini-3.6-flash"]
    lite, flash = named
    assert (lite.input_tokens, lite.output_tokens) == (1000, 100)
    assert (flash.input_tokens, flash.output_tokens, flash.reasoning_tokens) == (2000, 200, 30)
    assert sum(m.calls for m in usage.by_model) == usage.calls == 3, "understand + 2 tury agenta"
    assert usage.cost is not None
    assert usage.cost == price_usage(usage.by_model, settings.cost)
    assert usage.cost.ours_usd is not None and usage.cost.unpriced_models == []
    assert usage.cost.reference_usd > usage.cost.ours_usd > 0
    assert (usage.input_tokens, usage.output_tokens) == (3000, 300), "jak w F5"


def test_usage_cost_none_without_pricing(settings):
    """Serwis bez cennika (E3, testy F3-F5): cost None, reszta usage jak w F5."""
    service = _priced_service(settings, _two_model_trace(), pricing=None)
    usage = service.send_message("t", "Ile wynosi kara?").usage
    assert usage.cost is None
    assert usage.calls == 3 and (usage.input_tokens, usage.output_tokens) == (3000, 300)
    assert usage.reasoning_tokens == 30
    assert set(usage.models) == {"gemini-3.5-flash-lite", "gemini-3.6-flash"}
    assert usage.by_model, "podział na modele jest zawsze, cena tylko z cennikiem"


def test_usage_serialises_by_model_and_cost(settings):
    service = _priced_service(settings, _two_model_trace(), pricing=settings.cost)
    dumped = service.send_message("t", "Ile wynosi kara?").model_dump(mode="json")["usage"]
    assert {"by_model", "cost"} <= set(dumped)
    assert set(dumped["cost"]) >= {
        "ours_usd",
        "reference_usd",
        "unpriced_models",
        "our_prices_checked_on",
    }
    assert dumped["cost"]["our_prices_checked_on"] == str(settings.cost.our_prices_checked_on)


def test_usage_unpriced_model_is_reported(settings):
    from kovalo.agents.trace import ModelCallInfo

    trace = agent_trace()
    trace.model_calls = [
        ModelCallInfo(model="model-spoza-cennika", input_tokens=10, output_tokens=2)
    ]
    service = _priced_service(settings, trace, pricing=settings.cost)
    cost = service.send_message("t", "Ile wynosi kara?").usage.cost
    assert cost.ours_usd is None and cost.unpriced_models == ["model-spoza-cennika"]
    assert cost.reference_usd > 0


def test_resume_has_no_usage(settings):
    """Wznowienie pauzy nie woła LLM-a: usage None także z cennikiem."""
    from kovalo.workflow.service import CopilotService

    graph, _ = graph_with(settings, [new_request(discount_pct=14, margin_after_pct=10)])
    service = CopilotService(graph, pricing=settings.cost)
    first = service.send_message("t", "Prześlij rabat 14%.")
    assert first.usage is not None and first.usage.cost is not None
    resumed = service.resume("t", value="approved")
    assert resumed.status == "approved" and resumed.usage is None


# --- zamknięcie przerwanej tury ---

NODES_WITH_INTERRUPTION = ["understand", "answer_documents", "update_request", "decide", "approve"]
FIRST_MESSAGE = "Pierwsza wiadomość."
SECOND_MESSAGE = "Kto zatwierdza 3%?"
SECOND_TURN = turn(kind="policy_question", discount_pct=3)


def _service_interrupted_in(settings, node, monkeypatch):
    """Serwis, w którym pierwsza tura pada (BaseException) w `node` przed pauzą/odpowiedzią,
    a druga wiadomość (pytanie o politykę) przechodzi normalnie. Zwraca (service, thread_id)."""
    from kovalo.workflow import graph as graph_module
    from kovalo.workflow.checkpoint import memory_checkpointer
    from kovalo.workflow.graph import build_copilot_graph
    from kovalo.workflow.schemas import Turn
    from kovalo.workflow.service import CopilotService

    first_turns = {
        "answer_documents": [turn(kind="document_question", standalone="P?")],
        "update_request": [new_request(discount_pct=14, margin_after_pct=10)],
        "decide": [new_request(discount_pct=14, margin_after_pct=10)],
        "approve": [new_request(discount_pct=14, margin_after_pct=10)],
    }.get(node, [])
    llm = ReplayLLM([*first_turns, SECOND_TURN])
    understand_model = None
    answerer = None
    if node == "understand":
        inner = llm.with_structured_output(Turn)
        state = {"failed": False}

        def understand(messages):
            if not state["failed"]:
                state["failed"] = True
                raise _Abort
            return inner.invoke(messages)

        understand_model = RunnableLambda(understand)
    elif node == "answer_documents":

        def answerer(question):
            raise _Abort

    else:
        # funkcje pomocnicze grafu: n-te wywołanie pada raz (druga wiadomość nie jest dotknięta)
        name, target = {
            "update_request": ("required_approvers", 1),
            "decide": ("_request", 1),
            "approve": ("_request", 2),
        }[node]
        real, counter = getattr(graph_module, name), {"n": 0}

        def flaky(*args, **kwargs):
            counter["n"] += 1
            if counter["n"] == target:
                raise _Abort
            return real(*args, **kwargs)

        monkeypatch.setattr(graph_module, name, flaky)

    deps, _ = make_deps(settings, llm=llm, answerer=answerer, understand_model=understand_model)
    graph = build_copilot_graph(deps, checkpointer=memory_checkpointer())
    service = CopilotService(graph, max_message_chars=settings.workflow.max_message_chars)
    thread = f"t-{node}"
    with pytest.raises(_Abort):
        service.send_message(thread, FIRST_MESSAGE)
    snapshot = graph.get_state({"configurable": {"thread_id": thread}})
    assert snapshot.next and not snapshot.interrupts, "punkt wyjścia: przerwana tura bez pauzy"
    return service, thread


from tests.workflow_fakes import make_deps  # noqa: E402


def test_interrupted_turn_is_closed_on_next_message(settings, monkeypatch, caplog):
    from kovalo.workflow.replies import INTERRUPTED_TURN_REPLY

    service, thread = _service_interrupted_in(settings, "answer_documents", monkeypatch)
    with caplog.at_level(logging.WARNING):
        result = service.send_message(thread, SECOND_MESSAGE)  # bez ThreadStuckError
    assert result.status == "answered"
    assert len(result.replies) == 2
    assert result.replies[0] == INTERRUPTED_TURN_REPLY
    assert "3%" in result.replies[1], "odpowiedź na nową wiadomość (pytanie o politykę)"
    assert result.usage is not None and result.usage.calls == 1, "usage tylko nowej tury"
    warned = [
        r for r in caplog.records if r.levelno >= logging.WARNING and thread in r.getMessage()
    ]
    assert warned and any("answer_documents" in r.getMessage() for r in warned), (
        "WARNING z thread_id i przerwanym krokiem"
    )
    state = service.get_thread(thread)
    assert state.status == "answered" and state.pending is None
    assistant = [m.content for m in state.messages if m.role == "assistant"]
    assert assistant[0] == INTERRUPTED_TURN_REPLY
    assert service.graph.get_state({"configurable": {"thread_id": thread}}).next == ()


@pytest.mark.parametrize("node", NODES_WITH_INTERRUPTION)
def test_interrupted_turn_in_each_node(settings, monkeypatch, node):
    """Spike z kontraktu (7): `update_state(as_node="reply_other")` czyści `next` bez względu
    na to, w którym węźle wątek przerwano (przed `interrupt`)."""
    from kovalo.workflow.replies import INTERRUPTED_TURN_REPLY

    service, thread = _service_interrupted_in(settings, node, monkeypatch)
    assert node in service.graph.get_state({"configurable": {"thread_id": thread}}).next
    result = service.send_message(thread, SECOND_MESSAGE)
    assert result.replies[0] == INTERRUPTED_TURN_REPLY, node
    assert result.status == "answered" and len(result.replies) == 2, node
    snapshot = service.graph.get_state({"configurable": {"thread_id": thread}})
    assert snapshot.next == () and not snapshot.interrupts, node
    if node in ("decide", "approve"):
        # stan wniosku zostaje (kontrakt 7): zamknięcie nie czyści `request`
        assert service.get_thread(thread).request["discount_pct"] == 14


def test_pause_is_not_an_interrupted_turn(settings):
    from kovalo.workflow.service import ThreadPausedError

    service = service_with(settings, [new_request(discount_pct=14, margin_after_pct=10)])
    service.send_message("t", "Prześlij rabat 14%.")
    cfg = {"configurable": {"thread_id": "t"}}
    before = service.graph.get_state(cfg).config["configurable"]["checkpoint_id"]
    with pytest.raises(ThreadPausedError):
        service.send_message("t", "Dzięki!")
    after = service.graph.get_state(cfg).config["configurable"]["checkpoint_id"]
    assert after == before, "nic nie zamknięto ani nie zapisano"
    assert service.get_thread("t").pending.role == "Dyrektor handlowy"
    assert service.resume("t", value="approved").status == "approved", "pauza działa jak dotąd"


def test_resume_after_interrupted_turn_is_not_paused(settings, monkeypatch):
    from kovalo.workflow.replies import INTERRUPTED_TURN_REPLY
    from kovalo.workflow.service import NotPausedError

    service, thread = _service_interrupted_in(settings, "answer_documents", monkeypatch)
    with pytest.raises(NotPausedError, match="przerwana"):
        service.start_resume(thread, value="approved")
    cfg = {"configurable": {"thread_id": thread}}
    assert service.graph.get_state(cfg).next == (), "zamknięcie nastąpiło mimo odmowy wznowienia"
    after = service.send_message(thread, SECOND_MESSAGE)
    assert after.replies and INTERRUPTED_TURN_REPLY not in after.replies, "zamknięta tylko raz"
    assert len(after.replies) == 1


def test_close_failure_is_thread_stuck(settings, monkeypatch):
    from langgraph.errors import InvalidUpdateError

    from kovalo.workflow.service import ThreadStuckError

    service, thread = _service_interrupted_in(settings, "answer_documents", monkeypatch)

    def refuse(*args, **kwargs):
        raise InvalidUpdateError("sztuczna odmowa")

    monkeypatch.setattr(service.graph, "update_state", refuse)
    with pytest.raises(ThreadStuckError):
        service.send_message(thread, SECOND_MESSAGE)
    with pytest.raises(ThreadStuckError):  # blokada zwolniona (nie ThreadBusyError)
        service.send_message(thread, SECOND_MESSAGE)


def test_close_connection_error_is_checkpointer_unavailable(settings, monkeypatch):
    from kovalo.workflow.checkpoint import CONNECTION_ERRORS, CheckpointerUnavailableError

    service, thread = _service_interrupted_in(settings, "answer_documents", monkeypatch)

    def down(*args, **kwargs):
        raise CONNECTION_ERRORS[0]("brak połączenia")

    monkeypatch.setattr(service.graph, "update_state", down)
    with pytest.raises(CheckpointerUnavailableError):
        service.send_message(thread, SECOND_MESSAGE)


def test_busy_thread_is_not_closed(settings, monkeypatch):
    """Trwająca tura (blokada wątku): ThreadBusyError, stan nietknięty."""
    from kovalo.workflow.service import ThreadBusyError

    entered, release = threading.Event(), threading.Event()
    llm = ReplayLLM([turn(kind="policy_question", discount_pct=22)], gate=(entered, release))
    service = service_with(settings, llm=llm)
    updates = []
    real_update = service.graph.update_state
    monkeypatch.setattr(
        service.graph, "update_state", lambda *a, **k: updates.append(1) or real_update(*a, **k)
    )
    worker = threading.Thread(target=lambda: service.send_message("busy", "Kto zatwierdza 22%?"))
    worker.start()
    try:
        assert entered.wait(timeout=10)
        with pytest.raises(ThreadBusyError):
            service.send_message("busy", "Jeszcze raz")
        with pytest.raises(ThreadBusyError):
            service.start_resume("busy", value="approved")
        assert updates == [], "stan nie został zmieniony pod trwającą turą"
    finally:
        release.set()
        worker.join(timeout=10)
    assert service.get_thread("busy").status == "answered"


def test_get_thread_and_trace_do_not_close_interrupted_turn(settings, monkeypatch):
    """Odczyt nie zmienia stanu (kontrakt 7): zamyka dopiero start_message / start_resume."""
    service, thread = _service_interrupted_in(settings, "answer_documents", monkeypatch)
    cfg = {"configurable": {"thread_id": thread}}
    before = service.graph.get_state(cfg).config["configurable"]["checkpoint_id"]
    service.get_thread(thread)
    service.get_trace(thread)
    after = service.graph.get_state(cfg).config["configurable"]["checkpoint_id"]
    assert after == before and service.graph.get_state(cfg).next
