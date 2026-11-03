"""Atrapy ewaluacji E3: ScriptedUnderstanding, ScriptedAnswerModel, run_turn (kontrakt F3, 10.1)."""

from __future__ import annotations

import pytest

from tests.workflow_fakes import service_with, turn


def test_scripted_understanding_returns_turns_in_order_and_counts():
    from kovalo.eval.workflow.runner import ScriptedUnderstanding

    first, second = turn(kind="other"), turn(kind="policy_question", discount_pct=22)
    scripted = ScriptedUnderstanding([first, second])
    runnable = scripted.as_runnable()
    assert scripted.calls == 0
    assert runnable.invoke([]) == first
    assert runnable.invoke([]) == second
    assert scripted.calls == 2


def test_scripted_understanding_exhausted_is_assertion_error():
    from kovalo.eval.workflow.runner import ScriptedUnderstanding

    scripted = ScriptedUnderstanding([turn(kind="other")])
    runnable = scripted.as_runnable()
    runnable.invoke([])
    with pytest.raises(AssertionError):
        runnable.invoke([])


def test_scripted_answer_model_default_answer():
    from kovalo.eval.workflow.runner import ScriptedAnswerModel
    from kovalo.rag.answer import unpack_structured
    from kovalo.rag.records import Answer

    model = ScriptedAnswerModel()
    outcome = unpack_structured(model.invoke("dowolny prompt"), Answer)
    assert outcome.error is None
    assert outcome.parsed.answerable is True and outcome.parsed.citations == ["MTG-402"]


def test_run_turn_collects_pauses_and_answers(settings):
    from kovalo.eval.workflow.runner import run_turn

    service = service_with(
        settings, [turn(kind="approval_request", request_action="new", discount_pct=8)]
    )
    seen = []

    def answer(payload):
        seen.append(payload["type"])
        return {"clarification": 7, "approval": "approved"}[payload["type"]]

    entry = run_turn(service, "t", "Prześlij rabat 8%.", answer)
    assert seen == ["clarification", "approval", "approval"]
    assert entry["pauses"] == [
        ["clarification", "margin_after_pct"],
        ["approval", "Regionalny kierownik sprzedaży"],
        ["approval", "CFO"],
    ]
    assert entry["status"] == "approved" and entry["error"] is None
    assert entry["approvers"] == ["Regionalny kierownik sprzedaży", "CFO"]
    assert "zatwierdzony" in entry["reply"]


def test_answer_pause_exception_cancels_and_keeps_evidence(settings):
    from kovalo.eval.workflow.runner import run_turn

    service = service_with(
        settings,
        [turn(kind="approval_request", request_action="new", discount_pct=22, margin_after_pct=10)],
    )

    def broken(payload):
        raise KeyError("brak odpowiedzi")

    entry = run_turn(service, "t", "Prześlij rabat 22%.", broken)
    assert entry["status"] == "error" and "przy pauzie approval CEO" in entry["error"]
    assert entry["turn"]["kind"] == "approval_request" and entry["pauses"] == [["approval", "CEO"]]
    assert service.get_thread("t").pending is None, "pauza anulowana, wątek nie wisi"


def test_too_many_pauses_cancels(settings):
    from kovalo.eval.workflow.runner import run_turn

    service = service_with(
        settings,
        [turn(kind="approval_request", request_action="new", discount_pct=22, margin_after_pct=10)],
    )
    entry = run_turn(
        service, "t", "Prześlij rabat 22%.", lambda p: "może", max_pauses=3
    )  # „może” to nieprawidłowa odpowiedź: ta sama rola pytana w kółko
    assert entry["status"] == "error" and "przy pauzie" in entry["error"]
    assert len(entry["pauses"]) == 4 and service.get_thread("t").pending is None
