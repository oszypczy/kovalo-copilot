"""CopilotService: jedyny punkt wejścia do grafu (kontrakt F3, 6.7)."""

from __future__ import annotations

import threading

import pytest

from tests.fakes import TOY_CHUNKS
from tests.workflow_fakes import (
    ReplayLLM,
    answers,
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


def test_thread_stuck_after_failure_in_a_node(settings):
    """Wyjątek w węźle zostawia snapshot.next bez pauzy: kolejna wiadomość to ThreadStuckError."""
    from kovalo.workflow.service import ThreadStuckError

    def aborting(question):
        raise _Abort

    service = service_with(
        settings, [turn(kind="document_question", standalone="P?")], answerer=aborting
    )
    with pytest.raises(_Abort):
        service.send_message("t", "P?")
    with pytest.raises(ThreadStuckError):
        service.send_message("t", "Jeszcze raz")


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
