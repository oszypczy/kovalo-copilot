"""Liczby referencyjne kursu dla E5 i E4 przez MCP (README MP5; kontrakt F5, 10.5, P6).

Przebiegi kursu: asystent 20261006T213833-2ffa23b6 (`run_assistant_eval.py`) i agent przez MCP
20261006T213648-aa9ccf1b (`run_agent_mcp.py`). Bez nazwy modelu: raport bierze ją z
`reference_llm.model` (nazwy modeli tylko w configu).
"""

from __future__ import annotations

from typing import Any

__all__ = ["COURSE_E4_MCP", "COURSE_E4_MCP_RUN", "COURSE_E5", "COURSE_E5_RUN"]

COURSE_E5_RUN = "20261006T213833-2ffa23b6"
COURSE_E4_MCP_RUN = "20261006T213648-aa9ccf1b"

# Klucze = scoring.DIMENSIONS, wartości [zaliczone, ocenione] (listy, jak summary E4).
COURSE_E5: dict[str, Any] = {
    "route": [25, 25],
    "status": [25, 25],
    "fields": [11, 11],
    "pauses": [22, 22],
    "approvers": [19, 19],
    "request": [15, 15],
    "agent_trace": [25, 25],
    "tool_coverage": [4, 4],
    "arguments_correct": [1, 1],
    "answer_facts": [4, 4],
    "numbers_backed": [4, 4],
    "no_approval_claim": [4, 4],
    "mcp_calls": 7,
    "mcp_failures": 0,
    "tokens_in": 41993,
    "tokens_out": 2866,
    "cost_cents": 4.4,
}

# Kształt E4Run.summary (kontrakt F4, 11.4) + cost_cents.
COURSE_E4_MCP: dict[str, Any] = {
    "tools": [9, 9],
    "args": [3, 3],
    "facts": [8, 9],
    "approval": [0, 9],
    "uncovered": 0,
    "user_stated": 1,
    "outcomes": {"answered": 9, "error": 0, "limit": 0},
    "tool_calls": 21,
    "failed": 0,
    "tokens_in": 40931,
    "tokens_out": 1867,
    "cost_cents": 3.9,
}
