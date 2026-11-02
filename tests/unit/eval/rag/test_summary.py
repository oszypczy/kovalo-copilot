"""Podsumowanie czterech poziomów (kontrakt F2, 8.3; port summarise z answer_questions.py kursu)."""

from __future__ import annotations

from tests.fakes import toy_record


def row(
    id_: str,
    *,
    answerable: bool = True,
    hit: bool = True,
    outcome: str = "answered",
    evidence: float | None = 1.0,
    coverage: float | None = 1.0,
    valid: bool | None = True,
    relevant: bool | None = True,
    refusal: str | None = None,
    verdict: tuple[bool, bool] | None = (True, True),
    judge_error: str | None = None,
):
    from kovalo.eval.rag.judge import JudgeResult
    from kovalo.eval.rag.scoring import RecordScore
    from kovalo.eval.rag.summary import QuestionRow

    if outcome == "error":
        coverage = valid = relevant = refusal = None
        verdict = None
    result = JudgeResult(error=judge_error)
    if verdict is not None and judge_error is None:
        result = JudgeResult(correct=verdict[0], grounded=verdict[1], reason="atrapa")
    score = RecordScore(
        outcome=outcome,
        evidence_phrase_coverage=evidence,
        answer_coverage=coverage,
        citations_present=None if valid is None else True,
        citations_valid=valid,
        cites_relevant=relevant,
        refusal=refusal,
    )
    rec = toy_record("x", ["TST-001"]) if outcome != "error" else toy_record(error="ValueError: x")
    return QuestionRow(
        id=id_,
        question="Pytanie?",
        answerable=answerable,
        retrieval_hit=hit,
        answer=None if outcome == "error" else "x",
        citations=None if outcome == "error" else ["TST-001"],
        error=rec.error,
        score=score,
        verdict=result,
        record=rec,
    )


def test_summarise_denominators():
    from kovalo.eval.rag.summary import summarise

    rows = [
        row("r1"),
        row("r2", evidence=0.5, coverage=0.5, valid=False, relevant=False, verdict=(False, True)),
        row(
            "r3",
            hit=False,
            outcome="refused",
            coverage=None,
            valid=None,
            relevant=None,
            refusal="phrases_present",
            verdict=None,
        ),
        row("r4", outcome="error"),
        row(
            "r5",
            answerable=False,
            hit=False,
            outcome="refused",
            evidence=None,
            coverage=None,
            valid=None,
            relevant=None,
            refusal="correct",
            verdict=None,
        ),
        row(
            "r6",
            answerable=False,
            hit=False,
            evidence=None,
            coverage=None,
            relevant=False,
            refusal="missed",
            verdict=(False, False),
        ),
        row("r7", answerable=False, hit=False, outcome="error", evidence=None),
        row("r8", verdict=None, judge_error="ValueError: bad JSON"),
    ]
    s = summarise(rows)
    assert s.questions == 8 and s.errors == 2 and s.judge_errors == 1
    assert s.retrieval_hit == (4, 5)
    assert s.evidence_phrases_complete == (4, 5)
    assert s.answered == (3, 5)
    assert s.key_facts_complete == (2, 3)
    assert s.judge_correct == (1, 2)  # pytanie spoza korpusu nie liczy się do poprawności
    assert s.judge_grounded == (2, 3)  # ale liczy się do oparcia w kontekście
    assert s.citations_valid == (3, 4)
    assert s.cites_relevant == (2, 3)
    assert s.refusals_correct == (1, 3)  # błąd nie jest trafną odmową
    assert s.refusals_on_answerable == {"phrases_missing": 0, "phrases_present": 1}


def test_summary_json_shape():
    from kovalo.eval.rag.summary import summarise

    dumped = summarise([row("r1")]).model_dump(mode="json")
    assert set(dumped) == {
        "questions",
        "errors",
        "judge_errors",
        "retrieval_hit",
        "evidence_phrases_complete",
        "answered",
        "key_facts_complete",
        "judge_correct",
        "judge_grounded",
        "citations_valid",
        "cites_relevant",
        "refusals_correct",
        "refusals_on_answerable",
    }
    assert dumped["retrieval_hit"] == [1, 1]


def test_summarise_reproduces_course_shape():
    from kovalo.eval.rag.reference import COURSE_LIVE_SUMMARY
    from kovalo.eval.rag.summary import summarise

    rows = [row(f"M{n:02d}") for n in (1, 2, 4, 5, 6, 7, 8, 9)]
    rows.append(
        row(
            "M03",
            hit=False,
            outcome="refused",
            evidence=0.0,
            coverage=None,
            valid=None,
            relevant=None,
            refusal="phrases_missing",
            verdict=None,
        )
    )
    rows.append(
        row(
            "M10",
            outcome="refused",
            evidence=0.0,
            coverage=None,
            valid=None,
            relevant=None,
            refusal="phrases_missing",
            verdict=None,
        )
    )
    for n in (11, 12):
        rows.append(
            row(
                f"M{n}",
                answerable=False,
                hit=False,
                outcome="refused",
                evidence=None,
                coverage=None,
                valid=None,
                relevant=None,
                refusal="correct",
                verdict=None,
            )
        )
    got = summarise(rows).model_dump(mode="json")
    expected = {
        "questions": 12,
        "errors": 0,
        "judge_errors": 0,
        "retrieval_hit": [9, 10],
        "evidence_phrases_complete": [8, 10],
        "answered": [8, 10],
        "key_facts_complete": [8, 8],
        "judge_correct": [8, 8],
        "judge_grounded": [8, 8],
        "citations_valid": [8, 8],
        "cites_relevant": [8, 8],
        "refusals_correct": [2, 2],
        "refusals_on_answerable": {"phrases_missing": 2, "phrases_present": 0},
    }
    assert got == expected
    assert {
        k: list(v) if isinstance(v, tuple) else v for k, v in COURSE_LIVE_SUMMARY.items()
    } == expected
