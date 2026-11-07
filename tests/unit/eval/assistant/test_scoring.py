"""Ocena asystenta według ścieżki oczekiwanej we wzorcu (kontrakt F5, 10.3; port
`test_provided_assistant_eval.py` kursu): błąd nigdy nie usuwa sprawdzenia."""

from __future__ import annotations

import copy

import pytest
import yaml


def gold_turn(conversations, dialogue_id, number):
    dialogue = next(d for d in conversations if d.id == dialogue_id)
    return dialogue.turns[number - 1]


def make_entry(kind, *, trace=None, status="answered", **extra):
    from kovalo.agents.trace import trace_envelope

    turn_fields = extra.pop("turn_fields", {})
    return {
        "user": "q",
        "turn": {"kind": kind, **turn_fields},
        "status": status,
        "record": trace_envelope(trace) if trace else None,
        "pauses": [],
        "approvers": None,
        "request": None,
        **extra,
    }


def score(entry, expect, claim_phrases):
    from kovalo.eval.assistant.scoring import score_assistant_turn
    from kovalo.eval.trace.scoring import score_trace

    return score_assistant_turn(entry, expect, claim_phrases, score_trace)


def test_dimensions_are_the_twelve_of_the_course():
    from kovalo.eval.assistant.scoring import DIMENSIONS

    assert DIMENSIONS == [
        "route",
        "status",
        "fields",
        "pauses",
        "approvers",
        "request",
        "agent_trace",
        "tool_coverage",
        "arguments_correct",
        "answer_facts",
        "numbers_backed",
        "no_approval_claim",
    ]


def test_d7_has_mp5_agent_gold_and_mp3s_file_is_unchanged(conversations, session_settings):
    assert gold_turn(conversations, "D7", 1).expect["agent"]["needs"][0]["doc_id"] == "MTG-402"
    mp3 = yaml.safe_load(
        session_settings.paths.mini_project_3_dialogues.read_text(encoding="utf-8")
    )
    assert "agent" not in mp3["dialogues"][6]["turns"][0]["expect"]


def test_a_fabricated_answer_with_no_tool_calls_fails_every_agent_check(
    conversations, claim_phrases
):
    # kontrprzykład z przeglądu: trace ze statusem answered, bez wywołań, z wymyśloną odpowiedzią
    from kovalo.agents.trace import Trace

    t = gold_turn(conversations, "D7", 1)
    fake = Trace(
        question=t.user,
        answer="Zatwierdziłem rabat 99%. Ceny wody butelkowanej wzrosną o 777%.",
        status="answered",
    )
    result = score(make_entry("document_question", trace=fake), t.expect, claim_phrases)
    assert result["route"] and result["agent_trace"] and result["status"]
    assert {
        k: result[k]
        for k in ("tool_coverage", "answer_facts", "numbers_backed", "no_approval_claim")
    } == {
        "tool_coverage": False,
        "answer_facts": False,
        "numbers_backed": False,
        "no_approval_claim": False,
    }


def test_a_missing_trace_fails_every_expected_agent_check(conversations, claim_phrases):
    from kovalo.eval.assistant.scoring import applicable

    t = gold_turn(conversations, "C1", 1)  # kara za lipiec: needs, obliczenie, kluczowe fakty
    result = score(make_entry("document_question"), t.expect, claim_phrases)
    assert list(result) == applicable(t.expect), (
        "sprawdzenia wynikają ze wzorca, nie z tego, co się wydarzyło"
    )
    for dim in (
        "agent_trace",
        "tool_coverage",
        "arguments_correct",
        "answer_facts",
        "numbers_backed",
        "no_approval_claim",
    ):
        assert result[dim] is False, dim


def test_an_expected_calculation_that_never_happened_is_a_failure(conversations, claim_phrases):
    from kovalo.agents.trace import Trace

    t = gold_turn(conversations, "C1", 1)
    no_calc = Trace(question=t.user, answer="Kara wynosi 7296 PLN.", status="answered")
    result = score(make_entry("document_question", trace=no_calc), t.expect, claim_phrases)
    assert result["arguments_correct"] is False, "MP4 zwraca None; wzorzec oczekiwał obliczenia"


def test_a_misrouted_data_turn_keeps_all_its_checks(conversations, claim_phrases):
    from kovalo.eval.assistant.scoring import applicable

    t = gold_turn(conversations, "C1", 1)
    entry = make_entry("policy_question", turn_fields={"discount_pct": 14})
    result = score(entry, t.expect, claim_phrases)
    assert result["route"] is False and list(result) == applicable(t.expect)
    assert all(result[d] is False for d in ("tool_coverage", "arguments_correct", "answer_facts"))


def test_mp3s_extracted_fields_are_checked(conversations, claim_phrases):
    t = gold_turn(conversations, "D1", 1)  # fields: {discount_pct: 15}
    good = make_entry(
        "policy_question", turn_fields={"discount_pct": 15.004}, approvers=["Dyrektor handlowy"]
    )
    bad = make_entry(
        "policy_question", turn_fields={"discount_pct": None}, approvers=["Dyrektor handlowy"]
    )
    assert score(good, t.expect, claim_phrases)["fields"] is True, (
        "liczby równe z dokładnością do 0,01"
    )
    assert score(bad, t.expect, claim_phrases)["fields"] is False


def test_a_trace_on_a_policy_turn_is_a_failure(conversations, claim_phrases):
    from kovalo.agents.trace import Trace

    t = gold_turn(conversations, "C2", 1)
    stray = Trace(question="q", answer="x", status="answered")
    entry = make_entry("policy_question", trace=stray, approvers=["Dyrektor handlowy", "CFO"])
    assert score(entry, t.expect, claim_phrases)["agent_trace"] is False


@pytest.mark.parametrize(
    ("dialogue_id", "number"), [("C1", 1), ("C5", 1), ("D7", 1), ("C2", 1), ("D2", 2)]
)
def test_applicable_checks_depend_on_the_gold_only(
    conversations, claim_phrases, dialogue_id, number
):
    from kovalo.eval.assistant.scoring import applicable

    expect = gold_turn(conversations, dialogue_id, number).expect
    assert applicable(copy.deepcopy(expect)) == applicable(expect)
    nothing = score({"user": "q"}, expect, claim_phrases)
    assert list(nothing) == applicable(expect), (
        "pusty wpis jest oceniany w każdym sprawdzeniu, które definiuje wzorzec"
    )
    assert nothing["route"] is False and nothing["status"] is False


def test_the_teaching_set_has_four_data_turns_with_agent_gold(conversations, claim_phrases):
    from kovalo.eval.assistant.scoring import summarise

    data_turns = [
        t
        for d in conversations
        for t in d.turns
        if t.expect["kind"] == "document_question" and "agent" in t.expect
    ]
    assert len(data_turns) == 4
    totals = summarise([score({"user": "q"}, t.expect, claim_phrases) for t in data_turns])
    assert tuple(totals["numbers_backed"]) == (0, 4)
    assert tuple(totals["no_approval_claim"]) == (0, 4)


def test_numbers_are_compared_with_tolerance_0_01(conversations, claim_phrases):
    """Port reguły z `values_equal` E3: 14 vs 14.004 równe, 14 vs 14.02 różne."""
    t = gold_turn(conversations, "D1", 1)
    near = make_entry(
        "policy_question", turn_fields={"discount_pct": 15.009}, approvers=["Dyrektor handlowy"]
    )
    far = make_entry(
        "policy_question", turn_fields={"discount_pct": 15.02}, approvers=["Dyrektor handlowy"]
    )
    assert score(near, t.expect, claim_phrases)["fields"] is True
    assert score(far, t.expect, claim_phrases)["fields"] is False
