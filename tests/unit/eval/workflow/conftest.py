"""Fixture'y E3 (kontrakt F3, 14.1)."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def e3_replay_report(session_settings):
    """Odtworzenie wszystkich dialogów (zero LLM-a, embeddingi referencyjne z cache kursu)."""
    from kovalo.eval.workflow.e3 import run_e3

    return run_e3(session_settings)


@pytest.fixture
def ideal_model(session_settings, gold_dialogues):
    """BaseChatModel zwracający idealne Turn i stałą odpowiedź RAG (fabryka z opcjonalnym cache)."""
    from tests.workflow_fakes import ideal_chat_model

    def make(**kwargs):
        return ideal_chat_model(gold_dialogues, session_settings, **kwargs)

    return make


@pytest.fixture
def e3_live_report(session_settings, ideal_model, fake_embeddings):
    """Raport z przebiegiem „na żywo” na atrapie (14 wywołań)."""
    from kovalo.eval.workflow.e3 import run_e3
    from kovalo.llm import override_chat_model, override_embeddings

    with override_chat_model(ideal_model()), override_embeddings(fake_embeddings):
        return run_e3(session_settings, live=True)
