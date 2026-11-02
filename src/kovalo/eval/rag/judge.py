"""Sędzia LLM (LLM-as-judge) dla rekordu przebiegu (FR-2.4; kontrakt F2, 8.2; krok 4 MP2).

Rubryka to prompt NASZ `prompts.judge_system` (ADR-0008 pkt 7). Sędzia widzi ZAPISANY kontekst
rekordu, nigdy nowe wyszukiwanie. Ocenia tylko udzielone odpowiedzi; błąd sędziego trafia
do pola `error` (nigdy wyjątek).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from kovalo.eval.datasets import GoldQuestion
from kovalo.rag.answer import error_text, unpack_structured
from kovalo.rag.records import RunRecord, TokenUsage, Verdict

__all__ = ["JudgeResult", "build_judge_prompt", "judge_record"]


class JudgeResult(BaseModel):
    correct: bool | None = None
    grounded: bool | None = None
    reason: str | None = None
    error: str | None = None
    # NASZE (poza słownikiem kursu)
    judged_by: str | None = None  # model, który faktycznie ocenił (fallback widoczny)
    usage: TokenUsage | None = None
    seconds: float | None = None

    @property
    def answered_by(self) -> str | None:
        """Dawna nazwa `judged_by` (kontrakt F2, 8.2), dla zgodności."""
        return self.judged_by

    def course_dict(self) -> dict[str, Any]:
        """Słownik kroku 4 kursu: correct, grounded, reason, error."""
        return {
            "correct": self.correct,
            "grounded": self.grounded,
            "reason": self.reason,
            "error": self.error,
        }


def build_judge_prompt(system: str, record: RunRecord, gold: GoldQuestion) -> str:
    """Rubryka, pytanie, zapisany kontekst, oceniana odpowiedź z odwołaniami, wzorzec."""
    block = "\n\n---\n\n".join(f"[{c.doc_id}] {c.chunk_id}\n{c.text}" for c in record.context)
    output = record.output
    answer = output.answer if output is not None else ""
    citations = ", ".join(output.citations) if output is not None else ""
    return (
        f"{system.strip()}\n\n# Pytanie\n\n{record.question}\n\n# Kontekst\n\n{block or '(brak)'}"
        f"\n\n# Oceniana odpowiedź\n\n{answer}\n\nOdwołania: {citations or '-'}\n\n"
        f"# Odpowiedź wzorcowa\n\n{gold.reference.strip()}"
    )


def judge_record(
    record: RunRecord,
    gold: GoldQuestion,
    model: Runnable[LanguageModelInput, Any],
    *,
    system_prompt: str,
    clock: Callable[[], float] = time.perf_counter,
) -> JudgeResult:
    """Werdykt dla udzielonej odpowiedzi; odmowa albo błąd → JudgeResult() bez wywołania."""
    output = record.output
    if record.error is not None or output is None or not output.answerable:
        return JudgeResult()
    prompt = build_judge_prompt(system_prompt, record, gold)
    start = clock()
    try:
        result = model.invoke(prompt)
    except Exception as exc:  # błąd dostawcy, LlmCacheMiss, błąd atrapy
        return JudgeResult(error=error_text(exc), seconds=clock() - start)
    seconds = clock() - start
    outcome = unpack_structured(result, Verdict)
    common: dict[str, Any] = {
        "judged_by": outcome.answered_by,
        "usage": outcome.usage,
        "seconds": seconds,
    }
    if outcome.error is not None:
        return JudgeResult(error=outcome.error, **common)
    verdict: Verdict = outcome.parsed
    return JudgeResult(
        correct=verdict.correct, grounded=verdict.grounded, reason=verdict.reason, **common
    )
