"""Ocena zapisu przebiegu agenta: pięć testów, każdy mierzy jedną rzecz (krok 4 MP4, E4).

Kontrakt: docs/contracts/f4.md, sekcja 11.2. Port WYMAGAŃ kroku 4 1:1: dowodem są tylko udane
wywołania (status "ok", wynik typu dict); potrzeby sumują osobne wywołania; worker_summary nigdy
nie jest źródłem liczb; evidence_values z agents.trace (z naszym wpisem dla
check_termination_condition). Zdarzenia składowe są zwykłymi zdarzeniami `data`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from kovalo.agents.trace import SOURCE_ORIGINS, Trace, TraceEvent, evidence_values
from kovalo.eval.datasets import AgentGold, GoldCalculation, Need
from kovalo.quantities import mentions, numbers_in, same_quantity

__all__ = ["score_trace"]

_CALC_ARGS = ("otif_pct", "target_pct", "rate_per_full_point_pct", "cap_pct", "invoice_grosze")


def _successful(trace: Trace) -> list[TraceEvent]:
    return [e for e in trace.events if e.status == "ok" and isinstance(e.result, dict)]


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if not isinstance(a, int | float) or not isinstance(b, int | float):
        return False
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9)


def _matching(events: Iterable[TraceEvent], tool: str | None, supplier_id: str | None) -> list:
    return [
        e
        for e in events
        if e.tool == tool and (supplier_id is None or e.result.get("supplier_id") == supplier_id)
    ]


def _months_with_data(event: TraceEvent) -> set[str]:
    result = event.result
    months = {o.get("month") for o in result.get("observations") or [] if isinstance(o, dict)}
    if result.get("month") and result.get("invoiced_grosze") is not None:
        months.add(result["month"])
    return {m for m in months if m}


def _need_met(need: Need, events: list[TraceEvent]) -> bool:
    if need.any_of:
        return any(_need_met(alternative, events) for alternative in need.any_of)
    matching = _matching(events, need.tool, need.supplier_id)
    if not matching:
        return False
    with_data: set[str] = set()
    reported_missing: set[str] = set()
    docs: set[str] = set()
    for e in matching:
        with_data |= _months_with_data(e)
        reported_missing |= set(e.result.get("missing_months") or [])
        docs |= {
            str(p["doc_id"])
            for p in e.result.get("passages") or []
            if isinstance(p, dict) and p.get("doc_id")
        }
    if any(m not in with_data for m in need.months):
        return False
    if any(m not in reported_missing for m in need.missing):
        return False
    return need.doc_id is None or need.doc_id in docs


def _same_calculation(arguments: dict[str, Any], gold: GoldCalculation) -> bool:
    return all(
        key in arguments and _close(arguments[key], getattr(gold, key)) for key in _CALC_ARGS
    )


def _supported(gold: GoldCalculation, events: list[TraceEvent]) -> bool:
    otif = any(
        o.get("month") == gold.month and _close(o.get("otif_pct"), gold.otif_pct)
        for e in _matching(events, "get_supplier_otif", gold.supplier_id)
        for o in e.result.get("observations") or []
        if isinstance(o, dict)
    )
    terms = any(
        _close(e.result.get("otif_target_pct"), gold.target_pct)
        and _close(e.result.get("rate_per_full_point_pct"), gold.rate_per_full_point_pct)
        and _close(e.result.get("cap_pct"), gold.cap_pct)
        for e in _matching(events, "get_contract_terms", gold.supplier_id)
    )
    invoice = any(
        e.result.get("month") == gold.month
        and _close(e.result.get("invoiced_grosze"), gold.invoice_grosze)
        for e in _matching(events, "get_invoice_value", gold.supplier_id)
    )
    return otif and terms and invoice


def _arguments_correct(gold: AgentGold, events: list[TraceEvent]) -> bool | None:
    calculations = [
        e for e in events if e.tool == "calculate_otif_penalty" and e.result.get("status") == "ok"
    ]
    if not calculations:
        return None
    for e in calculations:
        matches = [g for g in gold.calculations if _same_calculation(e.arguments, g)]
        if not any(_supported(g, events) for g in matches):
            return False
    return True


def _claims_approval(answer: str | None, gold: AgentGold) -> bool:
    if not answer:
        return False
    return any(mentions(answer, phrase) for phrase in gold.approval_claim_phrases)


def score_trace(trace: Trace, gold: AgentGold) -> dict[str, Any]:
    """outcome, tool_coverage, arguments_correct, answer_fact_coverage, numeric_source_coverage,
    uncovered_numbers, user_stated_numbers, approval_claim, tool_calls, failed_calls."""
    events = _successful(trace)
    answer = trace.answer

    tool_coverage = (
        sum(_need_met(n, events) for n in gold.needs) / len(gold.needs) if gold.needs else None
    )

    if not gold.key_facts:
        facts: float | None = None
    elif answer is None:
        facts = 0.0
    else:
        found = sum(any(mentions(answer, v) for v in variants) for variants in gold.key_facts)
        facts = found / len(gold.key_facts)

    sources = [
        q for e in events if e.origin in SOURCE_ORIGINS for q in evidence_values(e.tool, e.result)
    ]
    asked = numbers_in(trace.question)
    covered, uncovered, user_stated = 0, [], []
    for claim in numbers_in(answer or ""):
        if any(same_quantity(claim, source) for source in sources):
            covered += 1
        elif any(same_quantity(claim, q) for q in asked):
            user_stated.append(claim.text)
        else:
            uncovered.append(claim.text)
    assessed = covered + len(uncovered)

    return {
        "outcome": trace.status,
        "tool_coverage": tool_coverage,
        "arguments_correct": _arguments_correct(gold, events),
        "answer_fact_coverage": facts,
        "numeric_source_coverage": covered / assessed if assessed else None,
        "uncovered_numbers": uncovered,
        "user_stated_numbers": user_stated,
        "approval_claim": _claims_approval(answer, gold),
        "tool_calls": len(trace.events),
        "failed_calls": sum(1 for e in trace.events if e.status == "error"),
    }
