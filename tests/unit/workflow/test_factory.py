"""workflow/factory.py: jedyne miejsce z modelem, RAG i indeksem (kontrakt F3, 6.8)."""

from __future__ import annotations

import pytest

from tests.workflow_fakes import NoAnswerer, turn


def test_build_copilot_uses_config(settings):
    from kovalo.llm import override_chat_model
    from kovalo.workflow.checkpoint import memory_checkpointer
    from kovalo.workflow.factory import build_copilot
    from kovalo.workflow.schemas import Turn
    from kovalo.workflow.service import CopilotService
    from tests.fakes import FakeStructuredChatModel

    fake = FakeStructuredChatModel(results={Turn: turn(kind="policy_question", discount_pct=22)})
    with override_chat_model(fake):
        graph = build_copilot(
            settings, checkpointer=memory_checkpointer(), document_answerer=NoAnswerer()
        )
        result = CopilotService(graph).send_message("t", "Kto zatwierdza 22%?")
    first_line = settings.prompts.understand_system.splitlines()[0]
    assert first_line in fake.prompts[0], "prompt understand_system w pierwszej wiadomości"
    assert result.decision["approvers"] == ["CEO", "CFO"], "polityka z configu"
    assert fake.schemas == [Turn] and fake.include_raw_flags == [True]
    assert result.usage.calls == 1 and result.usage.input_tokens == 120


def test_missing_key_raises_on_build():
    from kovalo.config import MissingConfigError, load_settings
    from kovalo.workflow.checkpoint import memory_checkpointer
    from kovalo.workflow.factory import build_copilot

    with pytest.raises(MissingConfigError):
        build_copilot(
            load_settings(), checkpointer=memory_checkpointer(), document_answerer=NoAnswerer()
        )


def test_offline_understand_model_needs_cache_not_key(llm_cache):
    from kovalo.config import load_settings
    from kovalo.workflow.factory import build_understand_model

    settings = load_settings()  # bez GOOGLE_API_KEY
    assert build_understand_model(settings, cache=llm_cache, offline=True) is not None
    with pytest.raises(ValueError):
        build_understand_model(settings, cache=None, offline=True)


def test_lazy_answerer_errors_are_records(settings, monkeypatch, fake_embeddings, toy_retriever):
    from kovalo.knowledge.pgvector import IndexMismatchError
    from kovalo.llm import override_chat_model, override_embeddings
    from kovalo.rag.records import Answer, RunRecord
    from kovalo.workflow.factory import LazyDocumentAnswerer
    from tests.fakes import FakeStructuredChatModel

    calls: list[int] = []

    def fake_open_retriever(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise IndexMismatchError("Indeks ma inną liczbę chunków")
        return toy_retriever

    # open_retriever może być importowany leniwie albo na poziomie modułu: podmieniamy wszędzie
    monkeypatch.setattr("kovalo.knowledge.retrievers.open_retriever", fake_open_retriever)
    monkeypatch.setattr(
        "kovalo.workflow.factory.open_retriever", fake_open_retriever, raising=False
    )
    monkeypatch.setattr("kovalo.knowledge.open_retriever", fake_open_retriever, raising=False)

    fake = FakeStructuredChatModel(
        results={Answer: Answer(answer="CFO.", citations=["TST-001"], answerable=True)}
    )
    answerer = LazyDocumentAnswerer(settings)
    with override_embeddings(fake_embeddings), override_chat_model(fake):
        first = answerer("Kto zatwierdza wyjazdy?")
        assert isinstance(first, RunRecord) and first.context == [] and first.output is None
        assert first.error and "IndexMismatchError" in first.error
        second = answerer("Kto zatwierdza wyjazdy?")
        third = answerer("Kto zatwierdza wyjazdy?")
    assert second.error is None and second.output.citations == ["TST-001"]
    assert third.error is None
    assert len(calls) == 2, "po błędzie budowa jest ponawiana, po sukcesie nie"


def test_open_copilot_service_memory(monkeypatch):
    from kovalo.config import load_settings
    from kovalo.llm import override_chat_model
    from kovalo.workflow.factory import open_copilot_service
    from kovalo.workflow.schemas import Turn
    from tests.fakes import FakeStructuredChatModel

    monkeypatch.setenv("KOVALO_WORKFLOW__CHECKPOINTER", "memory")
    settings = load_settings(google_api_key="test-key")
    fake = FakeStructuredChatModel(results={Turn: turn(kind="other")})
    with override_chat_model(fake), open_copilot_service(settings) as service:
        result = service.send_message("t", "Dzięki!")
        with pytest.raises(ValueError):
            service.send_message("t", "x" * (settings.workflow.max_message_chars + 1))
    assert result.status == "answered"


def test_open_copilot_service_without_key_raises(monkeypatch):
    from kovalo.config import MissingConfigError, load_settings
    from kovalo.workflow.factory import open_copilot_service

    monkeypatch.setenv("KOVALO_WORKFLOW__CHECKPOINTER", "memory")
    with pytest.raises(MissingConfigError), open_copilot_service(load_settings()):
        pass
