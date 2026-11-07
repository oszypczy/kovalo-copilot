"""Ocena końcowego asystenta tura po turze według OCZEKIWANEJ drogi każdej tury (E5).

Port `helpers/assistant_eval.py` kursu 1:1 (kontrakt: docs/contracts/f5.md, sekcja 10.3). O tym,
które sprawdzenia dotyczą tury, decyduje wyłącznie jej WZORZEC (`applicable`), zanim ktokolwiek
spojrzy na wynik. Działanie asystenta może dać negatywny wynik sprawdzenia, ale nigdy go nie usuwa:

  każda tura         route (rozpoznany rodzaj); status, fields (liczby równe z dokładnością 0,01,
                     jak `values_equal` E3), request, approvers, pauses: tam, gdzie podaje je
                     wzorzec; agent_trace (jeden w turze o dane, żadnego w pozostałych)
  document_question  ze wzorcem agenta: tool_coverage (needs), arguments_correct (calculations),
                     answer_facts (key_facts), numbers_backed, no_approval_claim. Brak zapisu
                     przebiegu albo brak oczekiwanego obliczenia to negatywny wynik.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from kovalo.agents.trace import Trace, trace_from_record
from kovalo.eval.datasets import AgentGold, Need
from kovalo.eval.workflow.scoring import values_equal

__all__ = ["AGENT_DIMENSIONS", "DIMENSIONS", "applicable", "score_assistant_turn", "summarise"]

DIMENSIONS: list[str] = [
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
AGENT_DIMENSIONS = (
    "tool_coverage",
    "arguments_correct",
    "answer_facts",
    "numbers_backed",
    "no_approval_claim",
)

ScoreTrace = Callable[[Trace, AgentGold], dict[str, Any]]


def applicable(expect: Mapping[str, Any]) -> list[str]:
    """Sprawdzenia zdefiniowane przez wzorzec tej tury, w kolejności DIMENSIONS."""
    dims = {"route", "agent_trace"} | {
        k for k in ("status", "fields", "pauses", "approvers", "request") if k in expect
    }
    agent = expect.get("agent") if expect["kind"] == "document_question" else None
    if agent is not None:
        dims |= {"numbers_backed", "no_approval_claim"}
        if agent.get("needs"):
            dims.add("tool_coverage")
        if agent.get("calculations"):
            dims.add("arguments_correct")
        if agent.get("key_facts"):
            dims.add("answer_facts")
    return [d for d in DIMENSIONS if d in dims]


def _fields_match(expected: Mapping[str, Any], actual: Mapping[str, Any] | None) -> bool:
    return actual is not None and all(values_equal(actual.get(k), v) for k, v in expected.items())


def score_assistant_turn(
    entry: Mapping[str, Any],
    expect: Mapping[str, Any],
    claim_phrases: Sequence[str],
    score_trace: ScoreTrace,
) -> dict[str, Any]:
    """{wymiar: True / False} dla dokładnie tych wymiarów, które wymienia applicable(expect),
    oraz "agent_score" (słownik E4), gdy tura o dane ma zapis przebiegu."""
    turn = entry.get("turn") or {}
    trace = trace_from_record(entry.get("record"))
    dims = applicable(expect)
    result: dict[str, Any] = {}
    for dim in dims:
        if dim == "route":
            result[dim] = turn.get("kind") == expect["kind"]
        elif dim == "status":
            result[dim] = entry.get("status") == expect["status"]
        elif dim == "fields":
            result[dim] = bool(turn) and _fields_match(expect["fields"], turn)
        elif dim == "request":
            result[dim] = _fields_match(expect["request"], entry.get("request"))
        elif dim == "approvers":
            result[dim] = entry.get("approvers") == expect["approvers"]
        elif dim == "pauses":
            got = [list(p) for p in entry.get("pauses", [])]
            result[dim] = got == [list(p) for p in expect["pauses"]]
        elif dim == "agent_trace":
            result[dim] = (trace is not None) == (expect["kind"] == "document_question")
    agent = expect.get("agent") if expect["kind"] == "document_question" else None
    if agent is None:
        return result
    agent_dims = [d for d in dims if d in AGENT_DIMENSIONS]
    if trace is None:  # brak zapisu przebiegu: brakuje całej oczekiwanej pracy agenta
        result.update(dict.fromkeys(agent_dims, False))
        return result
    gold = AgentGold(
        id=str(entry.get("user", ""))[:40],
        question=trace.question,
        kind="mp5",
        needs=[Need.model_validate(n) for n in agent.get("needs", [])],
        calculations=agent.get("calculations", []),
        key_facts=agent.get("key_facts", []),
        approval_claim_phrases=list(claim_phrases),
    )
    score = score_trace(trace, gold)
    result["agent_score"] = score
    checks = {
        "tool_coverage": score["tool_coverage"] == 1,
        # None = brak obliczenia tam, gdzie się go oczekiwało
        "arguments_correct": score["arguments_correct"] is True,
        "answer_facts": score["answer_fact_coverage"] == 1,
        "numbers_backed": bool(trace.answer) and not score["uncovered_numbers"],
        "no_approval_claim": bool(trace.answer) and not score["approval_claim"],
    }
    result.update({d: checks[d] for d in agent_dims})
    return result


def summarise(scores: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    """{wymiar: [zaliczone, ocenione]} po turach, które go definiują (listy, jak JSON)."""
    return {
        dim: [
            sum(s[dim] is True for s in scores if dim in s),
            sum(1 for s in scores if dim in s),
        ]
        for dim in DIMENSIONS
    }
