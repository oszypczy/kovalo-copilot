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


def test_lazy_agent_errors_are_traces(settings, monkeypatch, tmp_path, direct_tools):
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


def test_lazy_agent_without_key_is_an_error_trace(settings, direct_tools):
    from kovalo.config import load_settings
    from kovalo.workflow.factory import LazyAgentRunner

    trace = LazyAgentRunner(load_settings())("Ile wynosi kara?")  # bez GOOGLE_API_KEY
    assert trace.status == "error" and "GOOGLE_API_KEY" in trace.error


def test_lazy_retriever_error_is_tool_error(settings, monkeypatch, fake_embeddings, toy_retriever):
    """open_retriever zgłasza IndexUnavailableError → zdarzenie search_documents z status 'error';
    kolejne retrieve() próbuje ponownie; agent odpowiada dalej."""
    from kovalo.knowledge.pgvector import IndexUnavailableError
    from kovalo.knowledge.retrievers import LazyRetriever  # F5: przeniesiony z workflow/factory.py
    from kovalo.llm import get_embeddings, override_chat_model, override_embeddings
    from kovalo.workflow.factory import build_agent_runner
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
        lazy = LazyRetriever(
            lambda: fake_open_retriever(settings, get_embeddings(settings=settings))
        )
        runner = build_agent_runner(settings, retriever=lazy)
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


# ---------------------------------------------------------------------------
# F5: źródło narzędzi agenta (kontrakt F5, 9.1 i 13.2)
# ---------------------------------------------------------------------------


class FakeSource:
    """Atrapa `McpToolSource`: narzędzia bezpośrednie pod interfejsem źródła; liczy zamknięcia."""

    def __init__(self, tools, *, fail_first=False):
        self._tools = tools
        self.fail_first = fail_first
        self.tools_calls = 0
        self.closed = 0

    def tools(self):
        from kovalo.mcp_client.connection import McpStartupError

        self.tools_calls += 1
        if self.fail_first:
            raise McpStartupError("serwer MCP się nie uruchomił (atrapa)")
        return self._tools

    def close(self):
        self.closed += 1


def direct_tool_list(settings, small_data, retriever):
    from kovalo.tools.langchain import langchain_tools
    from tests.agent_fakes import tool_context

    return langchain_tools(tool_context(settings, small_data, retriever), "ours")


def configured(monkeypatch, **env):
    from kovalo.config import load_settings

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return load_settings(google_api_key="test-key")


def test_mcp_source_from_config(settings, monkeypatch, small_data, retriever):
    """`tools.source: mcp` → LazyAgentRunner woła `open_tool_source` z nazwami wariantu i buduje
    agenta na `source.tools()`; źródło otwarte raz; `direct` → F4 (bez MCP)."""
    from kovalo.llm import override_chat_model
    from kovalo.tools.specs import OUR_TOOLS
    from kovalo.workflow.factory import LazyAgentRunner
    from tests.agent_fakes import ScriptedModel, answer, call

    opened: list[dict] = []
    sources: list[FakeSource] = []

    def fake_open(settings_arg, *, names, **kwargs):
        opened.append({"names": set(names), **kwargs})
        sources.append(FakeSource(direct_tool_list(settings, small_data, retriever)))
        return sources[-1]

    monkeypatch.setattr("kovalo.workflow.factory.open_tool_source", fake_open)
    mcp_settings = configured(monkeypatch, KOVALO_TOOLS__SOURCE="mcp")
    assert (mcp_settings.tools.source, mcp_settings.tools.variant) == ("mcp", "ours")
    runner = LazyAgentRunner(mcp_settings)
    model = ScriptedModel(
        script=[
            call("get_contract_terms", supplier="SUP-114"),
            answer("Cel OTIF 97%."),
            answer("Druga odpowiedź."),
        ]
    )
    with override_chat_model(model):
        first = runner("Warunki?")
        second = runner("Jeszcze raz?")
    assert first.status == "answered" and [e.tool for e in first.events] == ["get_contract_terms"]
    assert second.status == "answered"
    assert len(opened) == 1 and opened[0]["names"] == set(OUR_TOOLS)
    assert sources[0].tools_calls == 1, "źródło budowane raz, kolejne pytania je współdzielą"
    runner.close()
    runner.close()
    assert sources[0].closed >= 1


def test_direct_source_does_not_touch_mcp(settings, monkeypatch):
    from kovalo.llm import override_chat_model
    from kovalo.workflow.factory import LazyAgentRunner
    from tests.agent_fakes import ScriptedModel, answer, call

    def boom(*args, **kwargs):
        raise AssertionError("tools.source: direct nie używa MCP")

    monkeypatch.setattr("kovalo.workflow.factory.open_tool_source", boom)
    direct_settings = configured(monkeypatch, KOVALO_TOOLS__SOURCE="direct")
    assert direct_settings.tools.source == "direct"
    runner = LazyAgentRunner(direct_settings)
    model = ScriptedModel(
        script=[
            call("get_invoice_value", supplier="SUP-114", month="2026-07"),
            answer("182 400 PLN"),
        ]
    )
    with override_chat_model(model):
        trace = runner("Faktura?")
    assert trace.status == "answered" and trace.events[0].result["invoiced_grosze"] == 18240000
    assert trace.events[0].transport is None


def test_mcp_startup_error_is_a_trace(settings, monkeypatch, small_data, retriever):
    """McpStartupError przy budowie → Trace(status='error'), stare źródło zamknięte; następne
    pytanie próbuje ponownie i działa (bez restartu procesu)."""
    from kovalo.llm import override_chat_model
    from kovalo.workflow.factory import LazyAgentRunner
    from tests.agent_fakes import ScriptedModel, answer

    sources: list[FakeSource] = []

    def fake_open(settings_arg, *, names, **kwargs):
        broken = not sources
        sources.append(
            FakeSource(direct_tool_list(settings, small_data, retriever), fail_first=broken)
        )
        return sources[-1]

    monkeypatch.setattr("kovalo.workflow.factory.open_tool_source", fake_open)
    runner = LazyAgentRunner(configured(monkeypatch, KOVALO_TOOLS__SOURCE="mcp"))
    with override_chat_model(ScriptedModel(script=[answer("Teraz działa.")])):
        first = runner("Warunki?")
        assert first.status == "error" and "McpStartupError" in first.error
        assert first.answer is None and first.question == "Warunki?"
        assert sources[0].closed >= 1, "stare źródło zamknięte"
        second = runner("Warunki?")
    assert second.status == "answered" and second.answer == "Teraz działa."
    assert len(sources) == 2


def test_service_close_closes_source(settings, monkeypatch, small_data, retriever):
    """`open_copilot_service` na wyjściu zamyka serwis, potem źródło narzędzi agenta."""
    from langchain_core.messages import AIMessage

    from kovalo.llm import override_chat_model
    from kovalo.workflow.factory import open_copilot_service
    from kovalo.workflow.schemas import Turn
    from tests.agent_fakes import ScriptedModel, answer, call

    sources: list[FakeSource] = []

    def fake_open(settings_arg, *, names, **kwargs):
        sources.append(FakeSource(direct_tool_list(settings, small_data, retriever)))
        return sources[-1]

    monkeypatch.setattr("kovalo.workflow.factory.open_tool_source", fake_open)
    mcp_settings = configured(
        monkeypatch, KOVALO_TOOLS__SOURCE="mcp", KOVALO_WORKFLOW__CHECKPOINTER="memory"
    )
    understand = turn(kind="document_question", standalone="Jakie są warunki?")
    model = ScriptedModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "Turn", "args": understand.model_dump(mode="json"), "id": "u1"}
                ],
            ),
            call("get_contract_terms", supplier="SUP-114"),
            answer("Cel OTIF 97%."),
        ]
    )
    with override_chat_model(model), open_copilot_service(mcp_settings) as service:
        result = service.send_message("t", "Jakie są warunki?")
        assert result.status == "answered" and Turn is not None
        assert len(sources) == 1 and sources[0].closed == 0
    assert sources[0].closed >= 1


def test_build_agent_runner_with_given_tools_needs_no_company_data(settings, monkeypatch):
    """`tools=` podane → używa ich i nie wczytuje danych firmy (dane ma serwer MCP)."""
    from langchain_core.tools import tool

    from kovalo.llm import override_chat_model
    from kovalo.workflow.factory import build_agent_runner
    from tests.agent_fakes import ScriptedModel, answer, call

    @tool
    def get_contract_terms(supplier: str) -> dict:
        """Warunki (dubler)."""
        return {"status": "ok", "supplier_id": "SUP-114"}

    monkeypatch.setenv("KOVALO_PATHS__COMPANY_DATA_DIR", "/nie/ma/takiego/katalogu")
    from kovalo.config import load_settings

    no_data = load_settings(google_api_key="test-key")
    model = ScriptedModel(script=[call("get_contract_terms", supplier="x"), answer("ok")])
    with override_chat_model(model):
        runner = build_agent_runner(no_data, tools=[get_contract_terms])
        trace = runner("Warunki?")
    assert model.bound == ["get_contract_terms"] and trace.status == "answered"


def test_agent_replays_a_fallback_answer_from_strict_cache(monkeypatch, tmp_path):
    """F5 (po A7): agent na żywo odpowiedział fallbackiem (model główny 429), więc w cache jest
    tylko wpis fallbacku. `--cache-only` (offline, strict): AvailabilityFallbackMiddleware
    przechodzi na fallback po braku trafienia u modelu głównego; bez wpisu → błąd LlmCacheMiss."""
    from langchain_core.tools import tool

    from kovalo.config import load_settings
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.workflow.factory import build_agent_runner
    from tests.unit.test_llm import cache_fakes

    @tool
    def get_contract_terms(supplier: str) -> dict:
        """Warunki (dubler)."""
        return {"status": "ok"}

    settings = load_settings(google_api_key="k")
    primary, fallback = settings.llm.model, settings.llm.fallbacks[0].model
    agent_model = settings.llm.purposes["agent"].model if "agent" in settings.llm.purposes else None
    primary = agent_model or primary

    calls = cache_fakes(monkeypatch, failing=primary)
    live = build_agent_runner(
        settings, tools=[get_contract_terms], cache=JsonFileLlmCache(tmp_path / "llm")
    )
    trace = live("Warunki?")
    assert trace.status == "answered" and trace.answer == f"od {fallback}"
    assert calls == {primary: 1, fallback: 1}

    calls = cache_fakes(monkeypatch, failing=None)
    strict = JsonFileLlmCache(tmp_path / "llm", mode="strict")
    replay = build_agent_runner(
        load_settings(), tools=[get_contract_terms], cache=strict, offline=True
    )
    again = replay("Warunki?")
    assert again.status == "answered" and again.answer == f"od {fallback}"
    assert calls == {} and strict.hits == 1

    empty = JsonFileLlmCache(tmp_path / "pusty", mode="strict")
    missing = build_agent_runner(
        load_settings(), tools=[get_contract_terms], cache=empty, offline=True
    )
    failed = missing("Warunki?")
    assert failed.status == "error" and "LlmCacheMiss" in failed.error
    assert calls == {}
