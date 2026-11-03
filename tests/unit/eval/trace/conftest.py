"""Fixture'y E4 (kontrakt F4, 15.1). Bez sieci: atrapa modelu z idealnymi skryptami, embeddingi
`DeterministicFakeEmbedding` (override), zapisane przebiegi kursu."""

from __future__ import annotations

from typing import Any

import pytest


def fraction(value: Any) -> tuple[int, int]:
    """Wynik testu w `E4Run.summary`: dokładnie `[trafione, ocenione]` (P3, kontrakt F4 17)."""
    assert isinstance(value, (list, tuple)) and len(value) == 2, value
    assert all(type(item) is int for item in value), value
    return value[0], value[1]


def count(value: Any) -> int:
    """Liczba liczb bez źródła w `E4Run.summary`: goły `int` (P3)."""
    assert type(value) is int, value
    return value


@pytest.fixture(scope="session")
def e4_live_report(session_settings, ctx_chunks):  # type: ignore[no-untyped-def]
    """Raport z przebiegiem „na żywo” na idealnych skryptach (9 pytań, bez limitu wywołań)."""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from kovalo.eval.trace.e4 import run_e4
    from kovalo.llm import override_embeddings
    from tests.agent_fakes import ideal_e4_model

    model = ideal_e4_model(ctx_chunks)
    with override_embeddings(DeterministicFakeEmbedding(size=16)):
        return run_e4(session_settings, live=True, model=model, max_calls=None)


@pytest.fixture(scope="session")
def e4_replay_report(session_settings):  # type: ignore[no-untyped-def]
    """Przeliczenie + odtworzenie (zero LLM-a); wyszukiwanie na atrapie embeddingów."""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from kovalo.eval.trace.e4 import run_e4
    from kovalo.llm import override_embeddings

    with override_embeddings(DeterministicFakeEmbedding(size=16)):
        return run_e4(session_settings)
