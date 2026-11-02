"""build_document_answerer (FR-2.2; kontrakt F2, 7.4)."""

from __future__ import annotations

from tests.fakes import TOY_QUESTION, FakeStructuredChatModel


def test_document_answerer_uses_config(settings, toy_retriever):
    from kovalo.llm import override_chat_model
    from kovalo.rag.factory import build_document_answerer
    from kovalo.rag.records import Answer, RunRecord

    fake = FakeStructuredChatModel(
        results={Answer: Answer(answer="CFO.", citations=["TST-001"], answerable=True)}
    )
    with override_chat_model(fake):
        answerer = build_document_answerer(settings, retriever=toy_retriever)
        record = answerer(TOY_QUESTION)
    assert isinstance(record, RunRecord) and record.error is None
    assert record.output.citations == ["TST-001"]
    prompt = fake.prompts[0]
    assert settings.prompts.answer_system.strip()[:40] in prompt
    assert len(record.context) == min(
        settings.retrieval.top_k, 2
    )  # k = top_k, korpus zabawkowy ma 2 chunki
    assert fake.schemas == [Answer] and fake.include_raw_flags == [True]
    assert record.usage is not None and record.answered_by == "fake-model"


def test_document_answerer_without_key_fails_at_build(settings, toy_retriever):
    import pytest

    from kovalo.config import MissingConfigError, load_settings
    from kovalo.rag.factory import build_document_answerer

    with pytest.raises(MissingConfigError):
        build_document_answerer(load_settings(), retriever=toy_retriever)
