"""Podsumowanie czterech poziomów E2 (kontrakt F2, 8.3; port summarise z answer_questions.py).

Klucze i mianowniki 1:1 z kursem: każdy poziom ma własny mianownik.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel

from kovalo.eval.rag.judge import JudgeResult
from kovalo.eval.rag.scoring import RecordScore
from kovalo.rag.records import RunRecord

__all__ = ["E2Summary", "QuestionRow", "summarise"]


class QuestionRow(BaseModel):
    """Wiersz 'results' z live_report.json kursu + pełny rekord (NASZE)."""

    id: str
    question: str
    answerable: bool
    retrieval_hit: bool
    answer: str | None
    citations: list[str] | None
    error: str | None
    score: RecordScore
    verdict: JudgeResult
    record: RunRecord


class E2Summary(BaseModel):
    """Sekcja 'summary' z live_report.json kursu."""

    questions: int
    errors: int
    judge_errors: int
    retrieval_hit: tuple[int, int]
    evidence_phrases_complete: tuple[int, int]
    answered: tuple[int, int]
    key_facts_complete: tuple[int, int]
    judge_correct: tuple[int, int]
    judge_grounded: tuple[int, int]
    citations_valid: tuple[int, int]
    cites_relevant: tuple[int, int]
    refusals_correct: tuple[int, int]
    refusals_on_answerable: dict[str, int]


def _count(pred: Callable[[QuestionRow], object], rows: Sequence[QuestionRow]) -> int:
    return sum(1 for r in rows if pred(r))


def summarise(rows: Sequence[QuestionRow]) -> E2Summary:
    """Cztery poziomy, każdy z własnym mianownikiem (jak summarise() kursu)."""
    answerable = [r for r in rows if r.answerable]
    unanswerable = [r for r in rows if not r.answerable]
    answered = [r for r in rows if r.score.outcome == "answered"]
    answered_answerable = [r for r in answered if r.answerable]
    judged = [r for r in answered if r.verdict.correct is not None]
    judged_answerable = [r for r in judged if r.answerable]
    return E2Summary(
        questions=len(rows),
        errors=_count(lambda r: r.score.outcome == "error", rows),
        judge_errors=_count(lambda r: r.verdict.error is not None, rows),
        retrieval_hit=(_count(lambda r: r.retrieval_hit, answerable), len(answerable)),
        evidence_phrases_complete=(
            _count(lambda r: r.score.evidence_phrase_coverage == 1, answerable),
            len(answerable),
        ),
        answered=(len(answered_answerable), len(answerable)),
        key_facts_complete=(
            _count(lambda r: r.score.answer_coverage == 1, answered_answerable),
            len(answered_answerable),
        ),
        judge_correct=(
            _count(lambda r: r.verdict.correct, judged_answerable),
            len(judged_answerable),
        ),
        judge_grounded=(_count(lambda r: r.verdict.grounded, judged), len(judged)),
        citations_valid=(_count(lambda r: r.score.citations_valid, answered), len(answered)),
        cites_relevant=(
            _count(lambda r: r.score.cites_relevant, answered_answerable),
            len(answered_answerable),
        ),
        refusals_correct=(
            _count(lambda r: r.score.refusal == "correct", unanswerable),
            len(unanswerable),
        ),
        refusals_on_answerable={
            kind: _count(lambda r, k=kind: r.score.refusal == k, answerable)  # type: ignore[misc]
            for kind in ("phrases_missing", "phrases_present")
        },
    )
