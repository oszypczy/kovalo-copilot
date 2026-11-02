r"""Ocena rekordu przebiegu bez LLM-a (FR-2.3; kontrakt F2, 8.1; krok 3 MP2 kursu).

Dopasowanie faktu: bez wielkości liter, białe znaki jak jedna spacja, wariant tylko jako
cała fraza (nie wewnątrz słowa ani liczby: „9%” nie pasuje do „19%”, „19,9%”, „1.9%”;
„500 000 PLN” nie pasuje do „1 500 000 PLN”). To przybliżone sygnały, nie werdykty.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from kovalo.eval.datasets import GoldQuestion
from kovalo.rag.records import RunRecord

__all__ = [
    "Outcome",
    "RecordScore",
    "Refusal",
    "all_required_found",
    "fact_coverage",
    "fact_found",
    "normalize_text",
    "phrase_pattern",
    "retrieval_hit",
    "score_record",
]

Outcome = Literal["answered", "refused", "error"]
Refusal = Literal["correct", "missed", "phrases_missing", "phrases_present"]


def normalize_text(text: str) -> str:
    """Każdy ciąg białych znaków → jedna spacja, casefold; zerowe ułamki („,00”) pomijane."""
    return re.sub(r"(?<=\d)[.,]00(?!\d)", "", re.sub(r"\s+", " ", text)).casefold()


@functools.lru_cache(maxsize=1024)
def phrase_pattern(variant: str) -> re.Pattern[str]:
    """Wzorzec całej frazy z asercjami, które nie pozwalają wejść w słowo ani liczbę."""
    v = normalize_text(variant).strip()
    before = r"(?<!\w)(?<!\d[.,])"
    after = r"(?!\w)(?![.,]\d)"
    if v[:1].isdigit():
        before += r"(?<!\d )"
    if v[-1:].isdigit():
        after += r"(?! \d)"
    return re.compile(before + re.escape(v) + after)


def fact_found(variants: Sequence[str], text: str) -> bool:
    """Czy któryś z akceptowanych wariantów faktu występuje w tekście."""
    normalized = normalize_text(text)
    return any(phrase_pattern(v).search(normalized) for v in variants)


def fact_coverage(facts: Sequence[Sequence[str]], text: str) -> float | None:
    """Udział znalezionych faktów; brak faktów → None."""
    if not facts:
        return None
    return sum(fact_found(variants, text) for variants in facts) / len(facts)


class RecordScore(BaseModel):
    """model_dump() == słownik kroku 3 kursu (te same klucze, ta sama kolejność)."""

    outcome: Outcome
    evidence_phrase_coverage: float | None
    answer_coverage: float | None
    citations_present: bool | None
    citations_valid: bool | None
    cites_relevant: bool | None
    refusal: Refusal | None


def score_record(record: RunRecord, gold: GoldQuestion) -> RecordScore:
    """Reguły 1:1 z docstringu kroku 3 kursu (zob. kontrakt F2, 8.1)."""
    evidence = (
        fact_coverage(gold.evidence_facts, "\n".join(c.text for c in record.context))
        if gold.answerable
        else None
    )
    output = record.output
    if record.error is not None or output is None:
        return RecordScore(
            outcome="error",
            evidence_phrase_coverage=evidence,
            answer_coverage=None,
            citations_present=None,
            citations_valid=None,
            cites_relevant=None,
            refusal=None,
        )
    answered = output.answerable
    refusal: Refusal | None = None
    if not gold.answerable:
        refusal = "missed" if answered else "correct"
    elif not answered:
        refusal = "phrases_missing" if evidence is not None and evidence < 1 else "phrases_present"
    if not answered:
        return RecordScore(
            outcome="refused",
            evidence_phrase_coverage=evidence,
            answer_coverage=None,
            citations_present=None,
            citations_valid=None,
            cites_relevant=None,
            refusal=refusal,
        )
    citations = output.citations
    context_ids = {c.doc_id for c in record.context}
    present = bool(citations)
    return RecordScore(
        outcome="answered",
        evidence_phrase_coverage=evidence,
        answer_coverage=fact_coverage(gold.key_facts, output.answer),
        citations_present=present,
        citations_valid=present and all(c in context_ids for c in citations),
        cites_relevant=any(c in gold.relevant_docs for c in citations),
        refusal=refusal,
    )


def retrieval_hit(doc_ids: Sequence[str], gold: GoldQuestion) -> bool:
    """Czy wśród znalezionych dokumentów jest trafny."""
    return any(d in gold.relevant_docs for d in doc_ids)


def all_required_found(doc_ids: Sequence[str], gold: GoldQuestion) -> bool | None:
    """Czy znaleziono wszystkie trafne dokumenty; None, gdy pytanie tego nie wymaga."""
    if not gold.requires_all:
        return None
    return all(d in doc_ids for d in gold.relevant_docs)
