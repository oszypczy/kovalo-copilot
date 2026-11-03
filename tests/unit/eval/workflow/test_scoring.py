"""score_transcript (kontrakt F3, 10.2; port kroku 5 MP3, 3 testy)."""

from __future__ import annotations

import pytest


def entry(
    kind="approval_request",
    request=None,
    approvers=None,
    pauses=(),
    status="approved",
    **turn_fields,
):
    return {
        "user": "u",
        "turn": {"kind": kind, **turn_fields},
        "request": request,
        "approvers": approvers,
        "pauses": [list(p) for p in pauses],
        "status": status,
    }


def test_each_stage_is_scored_separately():
    from kovalo.eval.workflow.scoring import score_transcript

    turns = [
        entry(
            request={"revision": 1, "discount_pct": 14.0},
            approvers=["Dyrektor handlowy"],
            pauses=[["approval", "Dyrektor handlowy"]],
            discount_pct=14.0,
        ),
        entry(
            kind="policy_question", approvers=["CEO", "CFO"], status="answered", discount_pct=22.0
        ),
    ]
    gold = [
        {
            "kind": "approval_request",
            "fields": {"discount_pct": 14},
            "request": {"revision": 1, "discount_pct": 14},
            "approvers": ["Dyrektor handlowy"],
            "pauses": [["approval", "Dyrektor handlowy"]],
            "status": "approved",
        },
        {
            "kind": "approval_request",
            "fields": {"discount_pct": 22.004},
            "approvers": ["CEO", "CFO"],
            "pauses": [],
            "status": "answered",
        },
    ]
    s = score_transcript(turns, gold)
    assert s["route"] == [1, 2], "druga tura trafiła do pytań o politykę zamiast do wniosków"
    assert s["fields"] == [2, 2], "22.0 i 22.004 są równe (różnica < 0.01)"
    assert s["request"] == [1, 1], "ocenia się tylko tury, których wzorzec zawiera 'request'"
    assert s["approvers"] == [2, 2] and s["pauses"] == [2, 2] and s["status"] == [2, 2]
    assert s["details"][1]["wrong"] == ["route"]
    assert s["details"][0]["wrong"] == [] and s["details"][0]["user"] == "u"
    assert s["errors"] == 0


def test_failures_stay_in_the_denominators():
    from kovalo.eval.workflow.scoring import score_transcript

    turns = [entry(kind="error", request=None, approvers=None, pauses=(), status="error")]
    gold = [
        {
            "kind": "approval_request",
            "request": {"revision": 1},
            "approvers": ["CFO"],
            "pauses": [["approval", "CFO"]],
            "status": "approved",
        }
    ]
    s = score_transcript(turns, gold)
    assert s["route"] == [0, 1] and s["request"] == [0, 1] and s["approvers"] == [0, 1]
    assert s["errors"] == 1


def test_lengths_must_match():
    from kovalo.eval.workflow.scoring import score_transcript

    with pytest.raises(ValueError):
        score_transcript([entry()], [])


def test_numbers_equal_below_one_hundredth_and_units_not_mixed():
    from kovalo.eval.workflow.scoring import score_transcript

    turns = [entry(request={"revision": 1, "discount_pct": 14.0}, discount_pct=14.0)]
    close = [
        {
            "kind": "approval_request",
            "fields": {"discount_pct": 14.009},
            "request": {"revision": 1, "discount_pct": 14},
        }
    ]
    far = [{"kind": "approval_request", "fields": {"discount_pct": 14.02}}]
    assert score_transcript(turns, close)["fields"] == [1, 1]
    assert score_transcript(turns, far)["fields"] == [0, 1]


def test_dimension_scored_only_when_gold_has_it():
    from kovalo.eval.workflow.scoring import score_transcript

    s = score_transcript([entry()], [{"kind": "approval_request", "status": "approved"}])
    assert s["fields"] == [0, 0] and s["request"] == [0, 0] and s["approvers"] == [0, 0]
    assert s["route"] == [1, 1] and s["status"] == [1, 1]
