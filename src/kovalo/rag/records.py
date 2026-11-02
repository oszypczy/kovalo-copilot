"""Schematy odpowiedzi i rekord przebiegu (FR-2.2; kontrakt F2, 7.1).

`Answer` i `Verdict` są kopią 1:1 z reference/course/helpers/records.py (docstringi i opisy pól
trafiają do schematu wysyłanego do modelu, więc traktujemy je jak prompt kursu: zmiana = ADR).
`ContextItem` i `RunRecord` mają pola kursu i nasze pola z wartościami domyślnymi
(ADR-0008 pkt 10), więc load_run kursu czyta nasze pliki.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

__all__ = ["Answer", "ContextItem", "RunRecord", "TokenUsage", "Verdict"]


class Answer(BaseModel):
    """Odpowiedź oparta na dokumentach z kontekstu."""

    answer: str = Field(description="odpowiedź oparta wyłącznie na faktach z dokumentów kontekstu")
    citations: list[str] = Field(
        description="identyfikatory dokumentów kontekstu, z których korzysta odpowiedź, np. POL-001"
    )
    answerable: bool = Field(
        description="false, jeśli dokumenty kontekstu nie zawierają odpowiedzi"
    )


class Verdict(BaseModel):
    """Werdykt sędziego (LLM-as-judge) dla jednej odpowiedzi."""

    correct: bool = Field(
        description="true, jeśli odpowiedź podaje fakty z odpowiedzi wzorcowej i im nie przeczy"
    )
    grounded: bool = Field(
        description="true, jeśli każde twierdzenie odpowiedzi jest poparte kontekstem"
    )
    reason: str = Field(description="jedno lub dwa zdania uzasadnienia werdyktu")


class ContextItem(BaseModel):
    """Jeden znaleziony chunk, dokładnie w takiej postaci, w jakiej zobaczył go model."""

    doc_id: str
    chunk_id: str
    text: str
    score: float | None = None  # NASZE: cosine (vector) albo wynik RRF (hybrid)


class TokenUsage(BaseModel):
    """Tokeny policzone przez dostawcę (usage_metadata surowej odpowiedzi)."""

    input_tokens: int
    output_tokens: int  # langchain-google-genai: razem z tokenami myślenia (ADR-0006)
    reasoning_tokens: int | None = None  # output_token_details['reasoning']

    @classmethod
    def from_usage(cls, usage: Mapping[str, Any] | None) -> TokenUsage | None:
        if not usage:
            return None
        reasoning = (usage.get("output_token_details") or {}).get("reasoning")
        return cls(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            reasoning_tokens=None if reasoning is None else int(reasoning),
        )

    @classmethod
    def from_message(cls, message: BaseMessage | None) -> TokenUsage | None:
        """Brak wiadomości albo usage_metadata → None."""
        if message is None:
            return None
        return cls.from_usage(getattr(message, "usage_metadata", None))


class RunRecord(BaseModel):
    """Jedno pytanie z odpowiedzią: dane wejściowe, kontekst oraz Answer albo błąd."""

    question: str
    context: list[ContextItem]
    output: Answer | None = None
    error: str | None = None
    # NASZE (wszystkie z wartością domyślną)
    chunks: str | None = None  # "mp1" | "ctx"
    retrieval: str | None = None  # Retriever.name
    prompt: str | None = None  # pełny prompt wysłany do modelu
    answered_by: str | None = None  # raw.response_metadata["model_name"]
    usage: TokenUsage | None = None
    seconds: float | None = None  # czas wywołania modelu (z cache ≈ 0)
