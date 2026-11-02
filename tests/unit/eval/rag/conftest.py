"""Fixture'y ewaluacji E2 (kontrakt F2, 11.1)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

FIXED_NOW = datetime(2026, 11, 3, 10, 15, 0, tzinfo=UTC)


@pytest.fixture(scope="session")
def fake_embeddings_factory():
    fake = DeterministicFakeEmbedding(size=16)
    return lambda spec: fake


@pytest.fixture(scope="session")
def e2_offline_report(session_settings, fake_embeddings_factory):
    """Przebieg E2 bez LLM-a: strona reference z prawdziwego cache kursu, strona ours na atrapie."""
    from kovalo.eval.rag.e2 import run_e2

    return run_e2(
        session_settings, embeddings_factory=fake_embeddings_factory, clock=lambda: FIXED_NOW
    )


@pytest.fixture(scope="session")
def unanswerable_questions(gold) -> list[str]:
    return [q.question for q in gold if not q.answerable]


@pytest.fixture
def answer_reply(unanswerable_questions):
    """Funkcja prompt -> Answer: odmowa tylko dla pytań spoza korpusu."""

    def reply(prompt: str):
        from kovalo.rag.records import Answer

        refused = any(q in prompt for q in unanswerable_questions)
        return Answer(
            answer="Brak tej informacji w dokumentach." if refused else "Odpowiedź atrapy.",
            citations=[] if refused else ["POL-001"],
            answerable=not refused,
        )

    return reply


@pytest.fixture
def fake_answerer(answer_reply):
    from kovalo.rag.records import Answer, Verdict
    from tests.fakes import FakeStructuredChatModel

    return FakeStructuredChatModel(
        results={
            Answer: answer_reply,
            Verdict: Verdict(correct=True, grounded=True, reason="atrapa"),
        }
    )


@pytest.fixture
def make_caching_models(unanswerable_questions):
    """Fabryka (model odpowiedzi, model sędziego): prawdziwe BaseChatModel z licznikiem wywołań."""
    from tests.fakes import CachingFakeChatModel

    def answer(prompt: str) -> str:
        refused = any(q in prompt for q in unanswerable_questions)
        return json.dumps(
            {"answer": "x", "citations": [] if refused else ["POL-001"], "answerable": not refused}
        )

    def judge(prompt: str) -> str:
        return json.dumps({"correct": True, "grounded": True, "reason": "atrapa"})

    def make(cache=None):
        kwargs = {} if cache is None else {"cache": cache}
        return (
            CachingFakeChatModel(reply=answer, sent=[], **kwargs),
            CachingFakeChatModel(reply=judge, sent=[], **kwargs),
        )

    return make


@pytest.fixture(scope="session")
def e2_llm_report(session_settings, fake_embeddings_factory, unanswerable_questions):
    """Przebieg E2 z LLM-em na atrapie (warianty ctx i hybrid, strona ours)."""
    from kovalo.eval.rag.e2 import run_e2
    from kovalo.rag.records import Answer, Verdict
    from tests.fakes import FakeStructuredChatModel

    def reply(prompt: str):
        refused = any(q in prompt for q in unanswerable_questions)
        return Answer(
            answer="Brak." if refused else "Odpowiedź atrapy.",
            citations=[] if refused else ["POL-001"],
            answerable=not refused,
        )

    fake = FakeStructuredChatModel(
        results={Answer: reply, Verdict: Verdict(correct=True, grounded=True, reason="atrapa")}
    )
    return run_e2(
        session_settings,
        variants=("ctx", "hybrid"),
        with_llm=True,
        answer_model=fake.with_structured_output(Answer, include_raw=True),
        judge_model=fake.with_structured_output(Verdict, include_raw=True),
        embeddings_factory=fake_embeddings_factory,
        clock=lambda: FIXED_NOW,
    )
