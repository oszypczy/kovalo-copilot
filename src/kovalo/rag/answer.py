"""Odpowiedź na pytanie z odwołaniami do źródeł (FR-2.2; kontrakt F2, 7.3; krok 2 MP2 kursu).

Model przychodzi przez parametr jako gotowy Runnable ze structured output (testy podają atrapę,
produkcja: rag.factory). Moduł nie importuje kovalo.llm. Błąd modelu nigdy nie jest wyjątkiem:
kończy się rekordem z polem `error`. Błąd wyszukiwania (np. Postgres niedostępny) idzie wyżej.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from kovalo.knowledge.prompts import build_prompt
from kovalo.knowledge.retrievers import Retriever
from kovalo.rag.citations import normalize_citations
from kovalo.rag.records import Answer, ContextItem, RunRecord, TokenUsage

__all__ = [
    "MAX_ERROR_CHARS",
    "DocumentAnswerer",
    "StructuredOutcome",
    "answer_question",
    "error_text",
    "unpack_structured",
]

DocumentAnswerer = Callable[[str], RunRecord]
MAX_ERROR_CHARS = 300


def error_text(exc: BaseException) -> str:
    """'<Klasa>: <komunikat>' obcięte do MAX_ERROR_CHARS znaków."""
    return f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]


@dataclass(frozen=True)
class StructuredOutcome:
    """Wynik wywołania modelu ze structured output po rozpakowaniu."""

    parsed: Any = None  # instancja schematu albo None
    error: str | None = None
    usage: TokenUsage | None = None
    answered_by: str | None = None


def unpack_structured[SchemaT: BaseModel](result: Any, schema: type[SchemaT]) -> StructuredOutcome:
    """Rozpakowanie wyniku: słownik include_raw, gotowa instancja schematu albo błąd.

    - {'raw', 'parsed', 'parsing_error'}: parsing_error → error; parsed None → error;
      raw → usage i answered_by (response_metadata['model_name']);
    - instancja schematu (atrapy kursu) → parsed, bez usage;
    - None albo inny typ → error 'Model nie zwrócił obiektu <schemat>'.
    """
    missing = f"Model nie zwrócił obiektu {schema.__name__}"
    if isinstance(result, dict) and "parsed" in result:
        raw = result.get("raw")
        usage = TokenUsage.from_message(raw) if isinstance(raw, BaseMessage) else None
        metadata = getattr(raw, "response_metadata", None) or {}
        answered_by = metadata.get("model_name") if isinstance(metadata, dict) else None
        common: dict[str, Any] = {
            "usage": usage,
            "answered_by": str(answered_by) if answered_by else None,
        }
        parsing_error = result.get("parsing_error")
        if parsing_error is not None:
            return StructuredOutcome(error=error_text(parsing_error), **common)
        parsed = result.get("parsed")
        if not isinstance(parsed, schema):
            return StructuredOutcome(error=missing, **common)
        return StructuredOutcome(parsed=parsed, **common)
    if isinstance(result, schema):
        return StructuredOutcome(parsed=result)
    return StructuredOutcome(error=missing)


def answer_question(
    question: str,
    retriever: Retriever,
    model: Runnable[LanguageModelInput, Any],
    *,
    system_prompt: str,
    k: int,
    chunks: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RunRecord:
    """Wyszukiwanie top-k → prompt kursu → structured output → RunRecord.

    Kontekst rekordu to dokładnie teksty z promptu, w kolejności wyszukiwania. Odwołania są
    normalizowane (normalize_citations). Nigdy nie zgłasza wyjątku z powodu modelu.
    """
    hits = retriever.retrieve(question, k)  # wyjątek wyszukiwania to nie błąd modelu
    prompt = build_prompt(system_prompt, [doc for doc, _ in hits], question)
    context = [
        ContextItem(
            doc_id=str(doc.metadata["doc_id"]),
            chunk_id=str(doc.metadata["chunk_id"]),
            text=doc.page_content,
            score=float(score),
        )
        for doc, score in hits
    ]
    record = RunRecord(
        question=question,
        context=context,
        chunks=chunks,
        retrieval=retriever.name,
        prompt=prompt,
    )
    start = clock()
    try:
        result = model.invoke(prompt)
    except Exception as exc:  # błąd dostawcy, LlmCacheMiss, błąd atrapy
        return record.model_copy(update={"error": error_text(exc), "seconds": clock() - start})
    seconds = clock() - start
    outcome = unpack_structured(result, Answer)
    update: dict[str, Any] = {
        "seconds": seconds,
        "usage": outcome.usage,
        "answered_by": outcome.answered_by,
    }
    if outcome.error is not None:
        update["error"] = outcome.error
    else:
        parsed: Answer = outcome.parsed
        update["output"] = parsed.model_copy(
            update={"citations": normalize_citations(parsed.citations)}
        )
    return record.model_copy(update=update)
