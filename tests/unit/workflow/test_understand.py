"""understand_turn i update_request (kontrakt F3, 6.2; port kroku 2 MP3 kursu, 8 testów)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tests.workflow_fakes import ReplayLLM, turn

COURSE_WORKFLOW = (
    Path(__file__).resolve().parents[3] / "reference" / "course" / "helpers" / "workflow.py"
)


def understand(history, message, llm, settings):
    from kovalo.workflow.schemas import Turn
    from kovalo.workflow.understand import understand_turn

    return understand_turn(
        history,
        message,
        llm.with_structured_output(Turn),
        system_prompt=settings.prompts.understand_system,
    )


def test_messages_are_system_history_message(settings):
    llm = ReplayLLM([turn(kind="policy_question", discount_pct=22)])
    history = [HumanMessage("Kto zatwierdza rabat 15%?"), AIMessage("Dyrektor handlowy.")]
    result = understand(history, "A rabat 22%?", llm, settings)
    assert result.turn.kind == "policy_question"
    assert result.call.purpose == "understand" and result.call.error is None
    sent = llm.prompts[0]
    assert (
        isinstance(sent[0], SystemMessage) and sent[0].content == settings.prompts.understand_system
    )
    assert sent[1:3] == history, (
        "historia bez zmian, między wiadomością systemową a nową wiadomością"
    )
    assert isinstance(sent[-1], HumanMessage) and sent[-1].content == "A rabat 22%?"


def test_model_failure_gives_an_error_turn(settings):
    result = understand([], "Prześlij rabat 10%.", ReplayLLM(fail=True), settings)
    assert result.turn.kind == "error" and result.turn.request_action == "none"
    assert result.turn.standalone == "Prześlij rabat 10%."
    assert result.call.error and "ValueError" in result.call.error


def test_new_request_resets_flags():
    from kovalo.policy.models import DiscountRequest
    from kovalo.workflow.understand import update_request

    old = DiscountRequest(revision=3, discount_pct=30, margin_after_pct=4, clearance=True)
    new = update_request(
        old,
        turn(kind="approval_request", request_action="new", discount_pct=30, margin_after_pct=11),
    )
    assert new.revision == 1 and new.clearance is False and new.margin_after_pct == 11, (
        "nowy wniosek zaczyna się od zera: nic nie przechodzi ze starego"
    )


def test_update_patches_only_given_fields():
    from kovalo.policy.models import DiscountRequest
    from kovalo.workflow.understand import update_request

    old = DiscountRequest(revision=1, discount_pct=14, margin_after_pct=10, strategic_account=True)
    new = update_request(
        old, turn(kind="approval_request", request_action="update", discount_pct=18)
    )
    assert (new.revision, new.discount_pct, new.margin_after_pct, new.strategic_account) == (
        2,
        18,
        10,
        True,
    )
    reset = update_request(
        new, turn(kind="approval_request", request_action="update", strategic_account=False)
    )
    assert reset.strategic_account is False and reset.revision == 3, "jawne False zeruje flagę"


def test_unchanged_update_keeps_the_revision():
    from kovalo.policy.models import DiscountRequest
    from kovalo.workflow.understand import update_request

    old = DiscountRequest(revision=2, discount_pct=18, margin_after_pct=10)
    same = update_request(
        old, turn(kind="approval_request", request_action="update", discount_pct=18)
    )
    assert same.revision == 2


def test_only_approval_requests_change_the_request():
    from kovalo.policy.models import DiscountRequest
    from kovalo.workflow.understand import update_request

    old = DiscountRequest(revision=1, discount_pct=14)
    assert update_request(old, turn(kind="policy_question", discount_pct=22)) is old
    assert update_request(None, turn(kind="other")) is None


def test_an_approval_request_must_say_new_or_update():
    from kovalo.policy.models import DiscountRequest
    from kovalo.workflow.understand import update_request

    with pytest.raises(ValueError, match="nowy wniosek"):
        update_request(
            DiscountRequest(revision=1, discount_pct=8),
            turn(kind="approval_request", request_action="none", discount_pct=22),
        )
    with pytest.raises(ValueError):
        update_request(None, turn(kind="approval_request", request_action="none", discount_pct=22))


def test_update_without_a_current_request_starts_one():
    from kovalo.workflow.understand import update_request

    new = update_request(
        None,
        turn(kind="approval_request", request_action="update", discount_pct=8, margin_after_pct=10),
    )
    assert new.revision == 1 and new.discount_pct == 8


def test_new_request_takes_fields_and_nulls_become_false():
    from kovalo.workflow.understand import update_request

    new = update_request(
        None,
        turn(
            kind="approval_request",
            request_action="new",
            discount_pct=22,
            strategic_account=True,
            customer="Carrefour Express",
        ),
    )
    assert (new.clearance, new.strategic_account, new.competitor_offer) == (False, True, False)
    assert new.customer == "Carrefour Express" and new.margin_after_pct is None


def test_turn_schema_matches_course():
    """Opisy pól Turn są częścią schematu wysyłanego do modelu: identyczne z klasą kursu."""
    from kovalo.workflow.schemas import Turn

    spec = importlib.util.spec_from_file_location("course_workflow_helpers", COURSE_WORKFLOW)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass w module z `from __future__ import annotations`
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    assert Turn.model_json_schema() == module.Turn.model_json_schema()
    assert list(Turn.model_fields) == list(module.Turn.model_fields)


def test_include_raw_fills_call(settings):
    from kovalo.workflow.schemas import Turn
    from kovalo.workflow.understand import understand_turn
    from tests.fakes import FakeStructuredChatModel

    fake = FakeStructuredChatModel(
        results={Turn: turn(kind="policy_question", discount_pct=15)},
    )
    result = understand_turn(
        [],
        "Kto zatwierdza 15%?",
        fake.with_structured_output(Turn, include_raw=True),
        system_prompt=settings.prompts.understand_system,
    )
    assert result.turn.kind == "policy_question"
    assert result.call.model == "fake-model" and result.call.error is None
    assert result.call.usage.input_tokens == 120 and result.call.usage.output_tokens == 40
    assert result.call.usage.reasoning_tokens == 25
    assert result.call.seconds is not None and result.call.seconds >= 0


def test_parsing_error_becomes_error_turn(settings):
    from kovalo.workflow.schemas import Turn
    from kovalo.workflow.understand import understand_turn
    from tests.fakes import FakeStructuredChatModel

    fake = FakeStructuredChatModel(parsing_error=True)
    result = understand_turn(
        [],
        "???",
        fake.with_structured_output(Turn, include_raw=True),
        system_prompt=settings.prompts.understand_system,
    )
    assert result.turn.kind == "error" and result.turn.standalone == "???"
    assert result.call.error and "zły JSON" in result.call.error
    assert result.call.usage is not None, "tokeny zostają także przy błędzie parsowania"


def test_missing_object_becomes_error_turn(settings):
    from langchain_core.runnables import RunnableLambda

    from kovalo.workflow.schemas import Turn
    from kovalo.workflow.understand import understand_turn

    result = understand_turn(
        [],
        "x",
        RunnableLambda(lambda _: None),
        system_prompt=settings.prompts.understand_system,
    )
    assert result.turn.kind == "error"
    assert Turn.__name__ in result.call.error


def test_turn_schema_types():
    from kovalo.workflow.schemas import TERMINAL_APPROVAL_STATUSES, LlmCall

    assert frozenset({"approved", "rejected", "cancelled"}) == TERMINAL_APPROVAL_STATUSES
    call = LlmCall(purpose="answer")
    assert call.model is None and call.usage is None and call.seconds is None and call.error is None
