"""Ocena zapisu rozmowy etap po etapie (kontrakt F3, 10.2; port kroku 5 MP3 kursu).

Porównanie z wzorcem sprawdzonym ręcznie (dialogi MP3), nigdy z własnym silnikiem polityki.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["STAGES", "score_transcript", "values_equal"]

STAGES = ("route", "fields", "request", "approvers", "pauses", "status")
_TOLERANCE = 0.01


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def values_equal(actual: Any, expected: Any) -> bool:
    """Liczby równe przy |a - b| < 0,01 (bool nie jest liczbą); inne wartości: ==."""
    if _is_number(actual) and _is_number(expected):
        return math.isfinite(actual) and abs(float(actual) - float(expected)) < _TOLERANCE
    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    return bool(actual == expected)


def _fields_match(source: Mapping[str, Any] | None, expected: Mapping[str, Any]) -> bool:
    if source is None:
        return False
    return all(
        name in source and values_equal(source[name], value) for name, value in expected.items()
    )


def _pairs(value: Any) -> list[list[Any]] | None:
    if value is None:
        return None
    return [list(p) for p in value]


def score_transcript(
    turns: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """{'route','fields','request','approvers','pauses','status': [trafne, ocenione],
    'errors': int, 'details': [{'user', 'wrong': [...]}]}.

    Wymiar oceniany tylko w turach, których wzorzec go zawiera; porażki zostają
    w mianowniku. Różne długości → ValueError. errors: status 'error' albo kind 'error'.
    """
    if len(turns) != len(gold):
        raise ValueError(
            f"Zapis rozmowy ma {len(turns)} tur, a wzorzec {len(gold)}: "
            "porównanie byłoby przesunięte."
        )
    score: dict[str, Any] = {stage: [0, 0] for stage in STAGES}
    errors = 0
    details: list[dict[str, Any]] = []
    for entry, expect in zip(turns, gold, strict=True):
        turn = entry.get("turn") or None
        checks: dict[str, bool] = {}
        if "kind" in expect:
            checks["route"] = turn is not None and turn.get("kind") == expect["kind"]
        if "fields" in expect:
            checks["fields"] = _fields_match(turn, expect["fields"])
        if "request" in expect:
            checks["request"] = _fields_match(entry.get("request"), expect["request"])
        if "approvers" in expect:
            approvers = entry.get("approvers")
            checks["approvers"] = approvers is not None and list(approvers) == list(
                expect["approvers"]
            )
        if "pauses" in expect:
            checks["pauses"] = _pairs(entry.get("pauses")) == _pairs(expect["pauses"])
        if "status" in expect:
            checks["status"] = entry.get("status") == expect["status"]
        wrong = []
        for stage in STAGES:
            if stage not in checks:
                continue
            score[stage][1] += 1
            if checks[stage]:
                score[stage][0] += 1
            else:
                wrong.append(stage)
        if entry.get("status") == "error" or (turn is not None and turn.get("kind") == "error"):
            errors += 1
        details.append({"user": entry.get("user"), "wrong": wrong})
    score["errors"] = errors
    score["details"] = details
    return score
