"""Pakiet `rag` (MP2, FR-2.2): odpowiedź ze źródłami, rekord przebiegu, normalizacja odwołań.

Kontrakt: docs/contracts/f2.md, sekcja 7. Moduły poza factory.py nie importują kovalo.llm
(model przychodzi przez parametr). Import pakietu nie ładuje kovalo.llm: fabrykę importuj
z `kovalo.rag.factory`.
"""

from kovalo.rag.answer import DocumentAnswerer, answer_question
from kovalo.rag.citations import normalize_citation, normalize_citations
from kovalo.rag.records import Answer, ContextItem, RunRecord, TokenUsage, Verdict

__all__ = [
    "Answer",
    "ContextItem",
    "DocumentAnswerer",
    "RunRecord",
    "TokenUsage",
    "Verdict",
    "answer_question",
    "normalize_citation",
    "normalize_citations",
]
