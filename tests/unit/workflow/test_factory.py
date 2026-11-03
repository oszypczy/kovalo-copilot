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


# ---------------------------------------------------------------------------
# F4: agent i prompt z configu, leniwy retriever (kontrakt F4, 8.2)
# ---------------------------------------------------------------------------

DISTINCT_ASSISTANT_LINE = "OTIF dostawców, faktury, kary"


class _StubAnswerer:
    """Zastępuje LazyAgentRunner / LazyDocumentAnswerer: zapisuje, że został zbudowany i wywołany."""  # noqa: E501

    built: list[str] = []  # noqa: RUF012

    def __init__(self, settings, **kwargs):
        self.kind = type(self).__name__
        _StubAnswerer.built.append(self.kind)

    def __call__(self, question, *, on_event=None, **kwargs):
        from tests.agent_fakes import july_events, make_trace

        return make_trace(question, "Odpowiedź agenta.", july_events())


def _doc_service(settings, monkeypatch, *, answerer: str):
    from kovalo.llm import override_chat_model
    from kovalo.workflow.checkpoint import memory_checkpointer
    from kovalo.workflow.factory import build_copilot
    from kovalo.workflow.schemas import Turn
    from kovalo.workflow.service import CopilotService
    from tests.fakes import FakeStructuredChatModel

    class StubAgent(_StubAnswerer):
        pass

    class StubRag(_StubAnswerer):
        def __call__(self, question, **kwargs):
            from kovalo.rag.records import Answer, ContextItem, RunRecord

            return RunRecord(
                question=question,
                context=[ContextItem(doc_id="TST-001", chunk_id="TST-001#ctx000", text="t")],
                output=Answer(answer="Odpowiedź RAG.", citations=["TST-001"], answerable=True),
            )

    _StubAnswerer.built = []
    monkeypatch.setattr("kovalo.workflow.factory.LazyAgentRunner", StubAgent)
    monkeypatch.setattr("kovalo.workflow.factory.LazyDocumentAnswerer", StubRag)
    monkeypatch.setenv("KOVALO_WORKFLOW__ANSWERER", answerer)
    from kovalo.config import load_settings

    configured = load_settings(google_api_key="test-key")
    fake = FakeStructuredChatModel(
        results={Turn: turn(kind="document_question", standalone="Ile wynosi kara?")}
    )
    with override_chat_model(fake):
        graph = build_copilot(configured, checkpointer=memory_checkpointer())
        result = CopilotService(graph).send_message("t", "Ile wynosi kara?")
    return result, fake, configured


def test_answerer_and_prompt_from_config(settings, monkeypatch):
    """`answerer: agent` → LazyAgentRunner i prompt assistant_understand_system; `rag` → F3."""
    result, fake, configured = _doc_service(settings, monkeypatch, answerer="agent")
    assert configured.workflow.answerer == "agent"
    assert _StubAnswerer.built == ["StubAgent"]
    assert result.replies == ["Odpowiedź agenta."] and result.trace is not None
    assert DISTINCT_ASSISTANT_LINE in fake.prompts[0], "prompt assistant_understand_system"

    result, fake, configured = _doc_service(settings, monkeypatch, answerer="rag")
    assert _StubAnswerer.built == ["StubRag"]
    assert "Odpowiedź RAG." in result.replies[0] and result.trace is None
    assert DISTINCT_ASSISTANT_LINE in fake.prompts[0], (
        "prompt zależy od workflow.understand_prompt, nie od answerer"
    )


def test_understand_prompt_from_config(settings, monkeypatch):
    from kovalo.config import load_settings
    from kovalo.llm import override_chat_model
    from kovalo.workflow.checkpoint import memory_checkpointer
    from kovalo.workflow.factory import build_copilot
    from kovalo.workflow.schemas import Turn
    from kovalo.workflow.service import CopilotService
    from tests.fakes import FakeStructuredChatModel

    monkeypatch.setenv("KOVALO_WORKFLOW__UNDERSTAND_PROMPT", "understand_system")
    configured = load_settings(google_api_key="test-key")
    fake = FakeStructuredChatModel(results={Turn: turn(kind="other")})
    with override_chat_model(fake):
        graph = build_copilot(
            configured, checkpointer=memory_checkpointer(), document_answerer=NoAnswerer()
        )
        CopilotService(graph).send_message("t", "Dzięki!")
    assert DISTINCT_ASSISTANT_LINE not in fake.prompts[0]
    assert configured.prompts.understand_system.splitlines()[-1] in fake.prompts[0]


def test_explicit_understand_prompt_wins(settings):
    from kovalo.llm import override_chat_model
    from kovalo.workflow.checkpoint import memory_checkpointer
    from kovalo.workflow.factory import build_copilot
    from kovalo.workflow.schemas import Turn
    from kovalo.workflow.service import CopilotService
    from tests.fakes import FakeStructuredChatModel

    fake = FakeStructuredChatModel(results={Turn: turn(kind="other")})
    with override_chat_model(fake):
        graph = build_copilot(
            settings,
            checkpointer=memory_checkpointer(),
            document_answerer=NoAnswerer(),
            understand_prompt="PROMPT-TESTOWY-XYZ",
        )
        CopilotService(graph).send_message("t", "Dzięki!")
    assert "PROMPT-TESTOWY-XYZ" in fake.prompts[0]


def test_lazy_agent_errors_are_traces(settings, monkeypatch, tmp_path):
    """Uszkodzone dane firmy → Trace(status='error'), drugie pytanie próbuje ponownie."""
    import shutil

    from kovalo.config import load_settings
    from kovalo.llm import override_chat_model
    from kovalo.workflow.factory import LazyAgentRunner
    from tests.agent_fakes import ScriptedModel, answer

    folder = tmp_path / "company"
    shutil.copytree(settings.paths.company_data_dir, folder)
    otif = folder / "supplier_otif.csv"
    original = otif.read_text(encoding="utf-8")
    otif.write_text(original.replace("93.8", "n/a", 1), encoding="utf-8")
    monkeypatch.setenv("KOVALO_PATHS__COMPANY_DATA_DIR", str(folder))
    configured = load_settings(google_api_key="test-key")

    runner = LazyAgentRunner(configured)
    with override_chat_model(ScriptedModel(script=[answer("Teraz działa.")])):
        first = runner("Ile wynosi kara?")
        assert first.status == "error" and "CompanyDataError" in first.error
        assert first.answer is None and first.question == "Ile wynosi kara?"
        otif.write_text(original, encoding="utf-8")  # naprawione dane
        second = runner("Ile wynosi kara?")
    assert second.status == "answered" and second.answer == "Teraz działa."


def test_lazy_agent_without_key_is_an_error_trace(settings):
    from kovalo.config import load_settings
    from kovalo.workflow.factory import LazyAgentRunner

    trace = LazyAgentRunner(load_settings())("Ile wynosi kara?")  # bez GOOGLE_API_KEY
    assert trace.status == "error" and "GOOGLE_API_KEY" in trace.error


def test_lazy_retriever_error_is_tool_error(settings, monkeypatch, fake_embeddings, toy_retriever):
    """open_retriever zgłasza IndexUnavailableError → zdarzenie search_documents z status 'error';
    kolejne retrieve() próbuje ponownie; agent odpowiada dalej."""
    from kovalo.knowledge.pgvector import IndexUnavailableError
    from kovalo.llm import override_chat_model, override_embeddings
    from kovalo.workflow.factory import LazyRetriever, build_agent_runner
    from tests.agent_fakes import ScriptedModel, answer, call

    attempts: list[int] = []

    def fake_open_retriever(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise IndexUnavailableError("Postgres niedostępny")
        return toy_retriever

    monkeypatch.setattr("kovalo.knowledge.retrievers.open_retriever", fake_open_retriever)
    monkeypatch.setattr(
        "kovalo.workflow.factory.open_retriever", fake_open_retriever, raising=False
    )
    monkeypatch.setattr("kovalo.knowledge.open_retriever", fake_open_retriever, raising=False)

    model = ScriptedModel(
        script=[
            call("search_documents", question="wyjazdy zagraniczne"),
            answer("Nie mogę teraz przeszukać dokumentów."),
            call("search_documents", question="wyjazdy zagraniczne"),
            answer("Wyjazdy zagraniczne wymagają zgody CFO."),
        ]
    )
    with override_chat_model(model), override_embeddings(fake_embeddings):
        runner = build_agent_runner(settings, retriever=LazyRetriever(settings))
        first = runner("Kto zatwierdza wyjazdy?")
        second = runner("Kto zatwierdza wyjazdy?")
    assert first.status == "answered"
    assert [(e.tool, e.status) for e in first.events] == [("search_documents", "error")]
    assert "IndexUnavailableError" in first.events[0].error
    assert [(e.tool, e.status) for e in second.events] == [("search_documents", "ok")]
    assert len(attempts) == 2, "po błędzie otwarcie jest ponawiane, po sukcesie nie"


def test_build_tool_context(settings):
    from kovalo.workflow.factory import build_tool_context

    ctx = build_tool_context(settings, retriever=None)
    assert ctx.top_k == settings.retrieval.top_k == 3
    assert ctx.policy == settings.policy and ctx.company == settings.company
    assert ctx.retriever is None and ctx.data.terms_version
    assert set(ctx.data.suppliers) >= {"SUP-114", "SUP-207"}


def test_build_agent_runner_variants_and_missing_key(settings, monkeypatch, retriever):
    from kovalo.config import MissingConfigError, load_settings
    from kovalo.llm import override_chat_model
    from kovalo.workflow.factory import build_agent_runner
    from tests.agent_fakes import ScriptedModel, answer

    for variant, expected in (("course", 6), ("ours", 7), (None, 7)):
        model = ScriptedModel(script=[answer("ok")])
        with override_chat_model(model):
            runner = build_agent_runner(settings, retriever=retriever, variant=variant)
            assert runner("Dzień dobry").status == "answered"
        assert len(model.bound) == expected, variant
    with pytest.raises(MissingConfigError):
        build_agent_runner(load_settings(), retriever=retriever)


def test_open_copilot_service_uses_config_answerer(monkeypatch):
    from kovalo.config import load_settings
    from kovalo.llm import override_chat_model
    from kovalo.workflow.factory import open_copilot_service
    from kovalo.workflow.schemas import Turn
    from tests.fakes import FakeStructuredChatModel

    monkeypatch.setenv("KOVALO_WORKFLOW__CHECKPOINTER", "memory")
    monkeypatch.setattr("kovalo.workflow.factory.LazyAgentRunner", _StubAnswerer)
    _StubAnswerer.built = []
    settings = load_settings(google_api_key="test-key")
    fake = FakeStructuredChatModel(
        results={Turn: turn(kind="document_question", standalone="Ile wynosi kara?")}
    )
    with override_chat_model(fake), open_copilot_service(settings) as service:
        result = service.send_message("t", "Ile wynosi kara?")
    assert result.trace is not None and result.replies == ["Odpowiedź agenta."]
    assert _StubAnswerer.built == ["_StubAnswerer"]
