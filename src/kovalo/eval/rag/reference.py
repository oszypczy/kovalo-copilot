"""Wyniki referencyjne kursu MP2 (stałe; źródło: docs/course/mp2_README.md).

Poziomy 1-2 (check.py, offline) E2 przelicza i porównuje z COURSE_RETRIEVAL. Poziomów 3-4
nie da się odtworzyć offline (kurs nie dołącza zapisanych odpowiedzi), więc kolumna
referencji LLM-a to podsumowanie przebiegu COURSE_LIVE_RUN_ID (chunki z kontekstem,
LLM referencyjny z `reference_llm` w configu).
"""

from __future__ import annotations

from typing import Any

__all__ = ["COURSE_LIVE_RUN_ID", "COURSE_LIVE_SUMMARY", "COURSE_RETRIEVAL"]

COURSE_RETRIEVAL: dict[str, dict[str, Any]] = {
    "mp1": {
        "n_chunks": 47,
        "hit_at_1": 9,
        "hit_at_3": 9,
        "evidence_phrases_complete": 9,
        "avg_prompt_tokens": 850.3,
    },
    "ctx": {
        "n_chunks": 87,
        "hit_at_1": 9,
        "hit_at_3": 9,
        "evidence_phrases_complete": 8,
        "avg_prompt_tokens": 610.4,
    },
}

COURSE_LIVE_RUN_ID = "20261006T212509-3ab56303"

COURSE_LIVE_SUMMARY: dict[str, Any] = {
    "questions": 12,
    "errors": 0,
    "judge_errors": 0,
    "retrieval_hit": (9, 10),
    "evidence_phrases_complete": (8, 10),
    "answered": (8, 10),
    "key_facts_complete": (8, 8),
    "judge_correct": (8, 8),
    "judge_grounded": (8, 8),
    "citations_valid": (8, 8),
    "cites_relevant": (8, 8),
    "refusals_correct": (2, 2),
    "refusals_on_answerable": {"phrases_missing": 2, "phrases_present": 0},
}
