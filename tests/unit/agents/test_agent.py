"""Agent z narzędziami: pętla, trace, fallback, licznik wywołań.

Porty testów kroku 3 MP4; kontrakt F4, 7.2-7.3.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, SystemMessage

from tests.agent_fakes import ScriptedModel, answer, call

COURSE = {
    "search_documents",
    "get_supplier_otif",
    "get_invoice_value",
    "get_contract_terms",
    "calculate_otif_penalty",
    "check_discount_policy",
}


def run(agent, question, steps=20, seconds=60):
    from kovalo.agents.trace import Budget, run_question

    return run_question(agent, question, Budget(steps, seconds))


def test_the_agent_calls_tools_and_the_trace_records_them(agent_graph, settings):
    agent, model = agent_graph(
        [
            call("get_contract_terms", supplier="SUP-114"),
            call("get_supplier_otif", supplier="SUP-114", from_month="2026-06", to_month="2026-06"),
            answer("Czerwiec: 95,2%."),
        ]
    )
    trace = run(agent, "OTIF w czerwcu?")
    assert trace.status == "answered" and trace.answer == "Czerwiec: 95,2%."
    assert [e.tool for e in trace.events] == ["get_contract_terms", "get_supplier_otif"]
    assert all(e.status == "ok" and e.origin == "data" for e in trace.events)
    assert trace.events[1].result["observations"] == [{"month": "2026-06", "otif_pct": 95.2}]
    assert set(model.bound) == COURSE, "agent musi dostać langchain_tools(ctx, 'course')"
    first = model.seen[0]
    assert isinstance(first[0], SystemMessage)
    assert first[0].content == settings.prompts.agent_system


def test_every_question_starts_fresh(agent_for):
    runner, model = agent_for([answer("pierwsza"), answer("druga")])
    first = runner("pierwsze pytanie")
    second = runner("drugie pytanie")
    texts = [m.content for m in model.seen[1]]
    assert "pierwsze pytanie" not in texts, "bez pamięci między pytaniami"
    assert first.status == second.status == "answered"
    assert first.steps == second.steps == 1, "każde pytanie ma własny Budget"


def test_runner_passes_question_id_and_uses_budget_from_settings(agent_for, settings):
    runner, _ = agent_for([answer("ok")])
    trace = runner("Dzień dobry", question_id="D1")
    assert trace.question_id == "D1" and trace.question == "Dzień dobry"
    assert settings.agent.max_steps == 24 and settings.agent.timeout_s == 180


def test_invalid_arguments_and_unknown_tools_are_recorded(agent_graph):
    agent, _ = agent_graph(
        [
            call("get_supplier_otif", supplier="SUP-114"),  # brak to_month
            call("approve_discount", discount_pct=14),  # nie ma takiego narzędzia
            answer("Nie udało mi się pobrać danych."),
        ]
    )
    trace = run(agent, "OTIF?")
    assert trace.status == "answered"
    assert [(e.tool, e.status) for e in trace.events] == [
        ("get_supplier_otif", "error"),
        ("approve_discount", "error"),
    ]
    assert trace.events[1].origin == "unknown"


def test_a_model_failure_after_a_call_keeps_the_call(agent_graph):
    agent, _ = agent_graph(
        [call("get_contract_terms", supplier="SUP-207"), RuntimeError("provider unavailable")]
    )
    trace = run(agent, "Warunki Aqua?")
    assert trace.status == "error" and "provider unavailable" in trace.error
    assert [(e.tool, e.status) for e in trace.events] == [("get_contract_terms", "ok")]
    assert trace.events[0].result["penalty_clause"] is False


def test_a_repeating_tool_hits_the_step_limit_and_keeps_its_events(agent_graph):
    agent, _ = agent_graph(
        [
            lambda messages: call(
                "get_supplier_otif", supplier="SUP-114", from_month="2026-06", to_month="2026-06"
            )
        ]
        * 50
    )
    trace = run(agent, "OTIF?", steps=6)
    assert trace.status == "limit" and "limit kroków" in trace.error
    assert len(trace.events) >= 2 and trace.answer is None


def test_an_exception_inside_a_valid_call_is_an_error_the_model_can_read(agent_graph, small_data):
    from tests.agent_fakes import BrokenRetriever

    agent, model = agent_graph(
        [
            call("search_documents", question="TransBałt palety"),
            answer("Nie mogę teraz przeszukać dokumentów."),
        ],
        retriever=BrokenRetriever(),
    )
    trace = run(agent, "Palety TransBałtu?")
    assert trace.status == "answered", "przebieg trwa dalej: model dostaje błąd i odpowiada"
    assert [(e.tool, e.status) for e in trace.events] == [("search_documents", "error")]
    assert "RuntimeError" in trace.events[0].error
    assert "embedding service unavailable" in trace.events[0].error
    assert model.calls == 2


def test_build_agent_wires_ours_variant(agent_graph):
    agent, model = agent_graph(
        [
            call("check_termination_condition", supplier="SUP-114", as_of_month="2026-09"),
            answer("ok"),
        ],
        variant="ours",
    )
    trace = run(agent, "Czy można rozwiązać umowę?")
    assert "check_termination_condition" in model.bound and len(model.bound) == 7
    assert trace.events[0].tool == "check_termination_condition"
    assert trace.events[0].origin == "calculation"


# --- fallback tylko przy błędach dostępności (ADR-0006, U10) -----------------------------------


def test_fallback_only_on_availability_errors(agent_for):
    from langchain_core.exceptions import ModelRateLimitError

    primary = ScriptedModel(script=[ModelRateLimitError("429 limit")], label="primary")
    backup = ScriptedModel(script=[answer("odpowiada zapasowy", model="backup")], label="backup")
    runner, _ = agent_for([], model=primary, fallbacks=[backup])
    trace = runner("Dzień dobry")
    assert trace.status == "answered" and trace.answer == "odpowiada zapasowy"
    assert (primary.calls, backup.calls) == (1, 1)

    bad = ScriptedModel(script=[ValueError("400 bad request")], label="primary")
    unused = ScriptedModel(script=[answer("nie powinien odpowiadać")], label="backup")
    runner, _ = agent_for([], model=bad, fallbacks=[unused])
    trace = runner("Dzień dobry")
    assert trace.status == "error" and "ValueError" in trace.error
    assert unused.calls == 0, "błąd 400 nie przełącza modelu"


def _availability_errors():
    from kovalo.llm import FALLBACK_EXCEPTIONS

    return list(FALLBACK_EXCEPTIONS)


@pytest.mark.parametrize("exc_type", _availability_errors(), ids=lambda t: t.__name__)
def test_every_availability_error_triggers_fallback(agent_for, exc_type):
    """Każdy wyjątek z llm.FALLBACK_EXCEPTIONS przełącza na zapasowy model (ADR-0006)."""
    primary = ScriptedModel(script=[exc_type("niedostępny")], label="primary")
    backup = ScriptedModel(script=[answer("zapasowy odpowiada", model="backup")], label="backup")
    runner, _ = agent_for([], model=primary, fallbacks=[backup])
    trace = runner("Dzień dobry")
    assert trace.status == "answered" and trace.answer == "zapasowy odpowiada"
    assert (primary.calls, backup.calls) == (1, 1)


def test_fallback_that_also_fails_is_an_error_trace(agent_for):
    from langchain_core.exceptions import ModelRateLimitError

    primary = ScriptedModel(script=[ModelRateLimitError("429 główny")], label="primary")
    backup = ScriptedModel(script=[ModelRateLimitError("429 zapasowy")], label="backup")
    runner, _ = agent_for([], model=primary, fallbacks=[backup])
    trace = runner("Dzień dobry")
    assert trace.status == "error" and "429 zapasowy" in trace.error
    assert (primary.calls, backup.calls) == (1, 1), "po jednej próbie na model, bez pętli"


def test_a_non_availability_error_in_fallback_is_not_retried(agent_for):
    from langchain_core.exceptions import ModelRateLimitError

    primary = ScriptedModel(script=[ModelRateLimitError("429")], label="primary")
    backup = ScriptedModel(script=[ValueError("400 w zapasowym")], label="backup")
    runner, _ = agent_for([], model=primary, fallbacks=[backup])
    trace = runner("Dzień dobry")
    assert trace.status == "error" and "ValueError" in trace.error
    assert backup.calls == 1


@pytest.mark.parametrize("kind", ["cache_miss", "budget"])
def test_cache_miss_and_exhausted_budget_do_not_fall_back(agent_for, kind):
    from kovalo.agents.middleware import CallBudgetExhausted
    from kovalo.eval.llm_cache import LlmCacheMiss

    exc = LlmCacheMiss("brak w cache") if kind == "cache_miss" else CallBudgetExhausted("limit")
    primary = ScriptedModel(script=[exc], label="primary")
    backup = ScriptedModel(script=[answer("zapasowy")], label="backup")
    runner, _ = agent_for([], model=primary, fallbacks=[backup])
    trace = runner("Dzień dobry")
    assert trace.status == "error" and type(exc).__name__ in trace.error
    assert backup.calls == 0


def test_fallback_is_visible_in_model_calls(agent_for):
    from langchain_core.exceptions import ModelAPIError

    primary = ScriptedModel(script=[ModelAPIError("503")], label="primary")
    backup = ScriptedModel(script=[answer("ok", model="backup-model", usage=(10, 3))])
    runner, _ = agent_for([], model=primary, fallbacks=[backup])
    trace = runner("Dzień dobry")
    assert [c.model for c in trace.model_calls] == ["backup-model"]


# --- CallMeter (--max-calls) -------------------------------------------------------------------


def test_call_meter_counts_and_stops(agent_for):
    from kovalo.agents.middleware import CallMeter

    meter = CallMeter(max_calls=2)
    runner, model = agent_for(
        [
            call("get_contract_terms", supplier="SUP-114"),
            call("get_contract_terms", supplier="SUP-207"),
            answer("trzecia tura nie powinna się zacząć"),
        ],
        extra_middleware=[meter],
    )
    trace = runner("Warunki?")
    assert trace.status == "error" and "CallBudgetExhausted" in trace.error
    assert (meter.network, meter.from_cache, meter.errors) == (2, 0, 0)
    assert model.calls == 2, "trzecie wywołanie sieciowe nie doszło do modelu"
    assert [e.tool for e in trace.events] == ["get_contract_terms", "get_contract_terms"]


def test_call_meter_counts_cache_hits_apart(agent_for, llm_cache):
    from kovalo.agents.middleware import CallMeter

    def cached(message):
        def respond(messages):
            llm_cache.hits += 1  # trafienie w cache w trakcie próby
            return message

        return respond

    meter = CallMeter(cache=llm_cache, max_calls=1)
    runner, _ = agent_for(
        [
            cached(call("get_contract_terms", supplier="SUP-114")),
            cached(call("get_contract_terms", supplier="SUP-207")),
            answer("sieciowa odpowiedź"),
        ],
        extra_middleware=[meter],
    )
    trace = runner("Warunki?")
    assert trace.status == "answered"
    assert (meter.from_cache, meter.network, meter.errors) == (2, 1, 0)


def test_call_meter_counts_errors_and_has_no_default_limit(agent_for):
    from kovalo.agents.middleware import CallMeter

    meter = CallMeter()
    runner, _ = agent_for([RuntimeError("provider unavailable")], extra_middleware=[meter])
    trace = runner("Dzień dobry")
    assert trace.status == "error"
    assert meter.errors == 1 and meter.network == 0
    unlimited = CallMeter(max_calls=None)
    runner, _ = agent_for(
        [call("get_contract_terms", supplier="SUP-114")] * 5 + [answer("koniec")],
        extra_middleware=[unlimited],
    )
    assert runner("Warunki?").status == "answered" and unlimited.network == 6


def test_call_meter_counts_every_fallback_attempt(agent_for):
    from langchain_core.exceptions import ModelRateLimitError

    from kovalo.agents.middleware import CallMeter

    meter = CallMeter()
    primary = ScriptedModel(script=[ModelRateLimitError("429")], label="primary")
    backup = ScriptedModel(script=[answer("ok")], label="backup")
    runner, _ = agent_for([], model=primary, fallbacks=[backup], extra_middleware=[meter])
    assert runner("Dzień dobry").status == "answered"
    assert meter.errors == 1 and meter.network == 1, (
        "CallMeter jest wewnątrz fallbacku: liczy każdą próbę"
    )


# --- trace.model_calls -------------------------------------------------------------------------


def test_model_calls_recorded(agent_for):
    first = AIMessage(
        content="",
        tool_calls=[{"name": "get_contract_terms", "args": {"supplier": "SUP-114"}, "id": "c1"}],
        response_metadata={"model_name": "model-a"},
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
            "output_token_details": {"reasoning": 7},
        },
    )
    runner, _ = agent_for([first, answer("Gotowe.", model="model-b", usage=(150, 20))])
    trace = runner("Warunki?")
    assert [c.model for c in trace.model_calls] == ["model-a", "model-b"]
    assert [(c.input_tokens, c.output_tokens) for c in trace.model_calls] == [(100, 10), (150, 20)]
    assert [c.reasoning_tokens for c in trace.model_calls] == [7, None]
    assert (trace.input_tokens, trace.output_tokens) == (250, 30)
    assert len(trace.model_calls) == 2, "jedna pozycja na każdy AIMessage"


# --- klucz cache LLM-a musi być stabilny między przebiegami (kontrakt F4, 12) ---------------------


def test_cache_key_is_stable_across_runs(agent_for, tmp_path):
    """Powtórka tego samego pytania z tym samym cache to same trafienia, także w trybie strict.

    LangChain dopisuje `usage_metadata.total_cost = 0` do wiadomości z trafienia w cache, więc
    historia wiadomości tury 2+ różni się między przebiegiem „świeżym” a „z cache”. Bez
    normalizacji (np. wycięcia `total_cost` przed liczeniem klucza) `--live --cache-only`
    nie odtworzy żadnego pytania z narzędziami (LlmCacheMiss od drugiej tury).
    """
    from kovalo.agents.middleware import CallMeter
    from kovalo.eval.llm_cache import JsonFileLlmCache

    messages = [
        call("get_contract_terms", supplier="SUP-114"),
        call("get_contract_terms", supplier="SUP-207"),
        answer("Gotowe: warunki obu umów.", model="scripted-model", usage=(120, 15)),
    ]

    def run(cache):
        model = ScriptedModel(script=list(messages), cache=cache)
        meter = CallMeter(cache=cache)
        runner, _ = agent_for([], model=model, extra_middleware=[meter])
        return runner("Warunki?"), model, meter

    first, _, meter1 = run(JsonFileLlmCache(tmp_path / "llm"))
    assert first.status == "answered" and (meter1.network, meter1.from_cache) == (3, 0)
    second, m2, meter2 = run(JsonFileLlmCache(tmp_path / "llm"))
    assert second.status == "answered" and second.answer == first.answer
    assert m2.calls == 0 and (meter2.network, meter2.from_cache) == (0, 3), (
        "każda tura z cache, także druga i trzecia"
    )
    strict = JsonFileLlmCache(tmp_path / "llm", mode="strict")
    third, m3, _ = run(strict)
    assert third.status == "answered" and m3.calls == 0 and strict.misses == 0


# ---------------------------------------------------------------------------
# F5: A0 (kontrakt F5, 6.1-6.2 i 13.2): callbacki (R1-2), count_errors i wspólny licznik (R1-6)
# ---------------------------------------------------------------------------


def recording_handler():
    from langchain_core.callbacks import BaseCallbackHandler

    class Recording(BaseCallbackHandler):
        def __init__(self):
            self.model_starts = []
            self.chain_starts = []

        def on_chat_model_start(
            self, serialized, messages, *, run_id, parent_run_id=None, **kwargs
        ):
            self.model_starts.append((run_id, parent_run_id))

        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
            self.chain_starts.append((run_id, parent_run_id))

    return Recording()


def test_callbacks_reach_the_agent_model(agent_for):
    handler = recording_handler()
    runner, _ = agent_for([call("get_contract_terms", supplier="SUP-114"), answer("ok")])
    trace = runner("Warunki?", callbacks=[handler])
    assert trace.status == "answered"
    assert len(handler.model_starts) == 2, "handler widzi start każdej tury modelu agenta"


def test_no_callbacks_is_the_f4_behaviour(agent_for):
    runner, _ = agent_for([answer("ok"), answer("ok")])
    assert runner("Cześć").status == "answered"
    assert runner("Cześć", callbacks=None).status == "answered"


def test_call_meter_counts_errors_when_asked(agent_for):
    from kovalo.agents.middleware import CallMeter

    meter = CallMeter(max_calls=2, count_errors=True)
    runner, model = agent_for(
        [RuntimeError("provider down"), RuntimeError("provider down"), answer("trzecia próba")],
        extra_middleware=[meter],
    )
    first = runner("Dzień dobry")
    second = runner("Dzień dobry")
    assert first.status == second.status == "error"
    assert meter.errors == 2 and meter.network == 0
    third = runner("Dzień dobry")
    assert third.status == "error" and "CallBudgetExhausted" in third.error
    assert model.calls == 2, "trzecia próba nie doszła do modelu: błędy liczą się do limitu"


def test_call_meter_default_does_not_count_errors(agent_for):
    """count_errors=False (domyślnie, E4): jak F4, nieudane próby nie zjadają limitu."""
    from kovalo.agents.middleware import CallMeter

    meter = CallMeter(max_calls=2)
    runner, model = agent_for(
        [RuntimeError("x"), RuntimeError("x"), RuntimeError("x"), answer("ok")],
        extra_middleware=[meter],
    )
    for _ in range(3):
        assert runner("Dzień dobry").status == "error"
    assert runner("Dzień dobry").status == "answered"
    assert model.calls == 4 and meter.errors == 3 and meter.network == 1


def test_shared_counter(agent_for):
    """Dwa CallMeter ze wspólnym CallCounter wyczerpują wspólny limit (E5: understand + agent)."""
    from kovalo.agents.middleware import CallBudgetExhausted, CallCounter, CallMeter

    counter = CallCounter(max_calls=3)
    one = CallMeter(counter=counter, count_errors=True)
    two = CallMeter(counter=counter, count_errors=True)
    runner_a, model_a = agent_for([answer("a"), answer("b")], extra_middleware=[one])
    runner_b, model_b = agent_for([answer("c"), answer("d")], extra_middleware=[two])
    assert runner_a("q").status == "answered" and runner_b("q").status == "answered"
    assert runner_a("q").status == "answered"  # trzecia próba w sumie
    exhausted = runner_b("q")
    assert exhausted.status == "error" and "CallBudgetExhausted" in exhausted.error
    assert model_b.calls == 1 and model_a.calls == 2
    with pytest.raises(CallBudgetExhausted):
        counter.admit()


def test_call_counter_records_outcomes():
    """`admit()` przed próbą (wyczerpany → CallBudgetExhausted); sieć zjada limit, trafienie
    w cache nie. (Liczenie `error` zależy od `count_errors`: tylko przez CallMeter, zob. wyżej.)"""
    from kovalo.agents.middleware import CallBudgetExhausted, CallCounter

    counter = CallCounter(max_calls=2)
    counter.admit()
    counter.record("network")
    counter.admit()
    counter.record("cache")  # trafienie w cache nie zjada limitu
    counter.admit()
    counter.record("network")
    with pytest.raises(CallBudgetExhausted):
        counter.admit()


def test_cache_key_does_not_depend_on_the_tool_artifact(agent_for, small_data, tmp_path):
    """ADR-0013 pkt 4 / U2: pochodzenie wywołania MCP (losowy `invocation_id`) idzie do artefaktu
    ToolMessage, więc NIE może zmieniać klucza cache. Sprawdzamy to bezpośrednio: powtórka tego
    samego pytania z NOWYM losowym artefaktem to same trafienia, także z tym samym cache co
    przebieg na narzędziach bezpośrednich (artefakt None), czyli warunek U9 (E4 przez MCP
    `--cache-only` odtwarza wynik F4 bez wywołań). `dumps(messages)` serializuje `artifact`, więc
    potrzebna jest normalizacja (jak `total_cost` w StableUsageMiddleware)."""
    import uuid

    from langchain_core.tools import StructuredTool

    from kovalo.agents.agent import AgentRunner, build_agent
    from kovalo.agents.middleware import CallMeter
    from kovalo.eval.llm_cache import JsonFileLlmCache

    def terms(supplier: str):
        return {"status": "ok", "supplier_id": "SUP-114", "supplier": "Mleczna Dolina"}

    def mcp_tool():
        return StructuredTool.from_function(
            func=lambda supplier: (
                terms(supplier),
                {"transport": "mcp", "invocation_id": uuid.uuid4().hex[:12]},
            ),
            name="get_contract_terms",
            description="dubler narzędzia MCP",
            args_schema={"type": "object", "properties": {"supplier": {"type": "string"}}},
            response_format="content_and_artifact",
        )

    def direct_tool():
        return StructuredTool.from_function(
            func=terms,
            name="get_contract_terms",
            description="dubler narzędzia MCP",
            args_schema={"type": "object", "properties": {"supplier": {"type": "string"}}},
        )

    messages = [
        call("get_contract_terms", supplier="SUP-114"),
        answer("Gotowe.", model="scripted-model", usage=(120, 15)),
    ]

    def run(tool, cache):
        model = ScriptedModel(script=list(messages), cache=cache)
        meter = CallMeter(cache=cache)
        agent = build_agent(model, [tool], system_prompt="prompt", extra_middleware=[meter])
        from kovalo.config import load_settings

        return AgentRunner(agent, load_settings(google_api_key="k").agent)("Warunki?"), model, meter

    cache = JsonFileLlmCache(tmp_path / "llm")
    first, m1, _ = run(direct_tool(), cache)
    assert first.status == "answered" and m1.calls == 2
    second, m2, meter2 = run(mcp_tool(), JsonFileLlmCache(tmp_path / "llm"))
    assert second.status == "answered" and second.events[0].tool == "get_contract_terms"
    assert m2.calls == 0 and (meter2.network, meter2.from_cache) == (0, 2), (
        "artefakt MCP nie zmienia klucza: przebieg z MCP trafia w cache przebiegu bezpośredniego"
    )
    _, m3, meter3 = run(mcp_tool(), JsonFileLlmCache(tmp_path / "llm"))
    assert m3.calls == 0 and meter3.from_cache == 2, "nowy losowy invocation_id: nadal trafienia"


# --- F5 (po A7): fallback offline, liczenie błędów każdej próby, budżet nie jest błędem ---------


def _offline():
    from kovalo.llm import fallback_exceptions

    return fallback_exceptions(offline=True)


def test_offline_cache_miss_falls_back_but_only_to_a_model_that_hits(agent_for):
    """`--cache-only`: brak wpisu u modelu głównego przechodzi do fallbacku (tura odebrana na żywo
    przez fallback ma w cache tylko jego wpis). Na żywo (domyślne wyjątki) to nie zachodzi."""
    from kovalo.eval.llm_cache import LlmCacheMiss

    primary = ScriptedModel(script=[LlmCacheMiss("brak wpisu głównego")], label="primary")
    backup = ScriptedModel(script=[answer("z wpisu fallbacku", model="backup")], label="backup")
    runner, _ = agent_for([], model=primary, fallbacks=[backup], fallback_exceptions=_offline())
    trace = runner("Dzień dobry")
    assert trace.status == "answered" and trace.answer == "z wpisu fallbacku"
    assert (primary.calls, backup.calls) == (1, 1)

    live_primary = ScriptedModel(script=[LlmCacheMiss("brak")], label="primary")
    live_backup = ScriptedModel(script=[answer("nie powinien")], label="backup")
    runner, _ = agent_for([], model=live_primary, fallbacks=[live_backup])
    assert runner("Dzień dobry").status == "error" and live_backup.calls == 0


def test_offline_miss_in_every_model_is_an_error_trace_not_a_loop(agent_for):
    from kovalo.eval.llm_cache import LlmCacheMiss

    primary = ScriptedModel(script=[LlmCacheMiss("brak 1")], label="primary")
    backup = ScriptedModel(script=[LlmCacheMiss("brak 2")], label="backup")
    runner, _ = agent_for([], model=primary, fallbacks=[backup], fallback_exceptions=_offline())
    trace = runner("Dzień dobry")
    assert trace.status == "error" and "LlmCacheMiss" in trace.error
    assert (primary.calls, backup.calls) == (1, 1), "po jednej próbie na model"


def test_offline_exhausted_budget_still_does_not_fall_back(agent_for):
    """Tryb offline poszerza fallback tylko o CacheMissError; CallBudgetExhausted i błędy 400
    nadal idą wyżej bez próby zapasowego modelu."""
    from kovalo.agents.middleware import CallBudgetExhausted

    for exc in (CallBudgetExhausted("limit"), ValueError("400 bad request")):
        primary = ScriptedModel(script=[exc], label="primary")
        backup = ScriptedModel(script=[answer("zapasowy")], label="backup")
        runner, _ = agent_for([], model=primary, fallbacks=[backup], fallback_exceptions=_offline())
        trace = runner("Dzień dobry")
        assert trace.status == "error" and type(exc).__name__ in trace.error
        assert backup.calls == 0, type(exc).__name__


def test_offline_miss_then_fallback_counts_the_miss_as_an_error(agent_for, llm_cache):
    """Licznik przy powtórce offline: chybienie modelu głównego to `error` i zjada limit przy
    count_errors (tu atrapa fallbacku nie czyta cache, więc jego próba to `network`)."""
    from kovalo.agents.middleware import CallCounter, CallMeter
    from kovalo.eval.llm_cache import LlmCacheMiss

    counter = CallCounter(max_calls=5)
    meter = CallMeter(cache=llm_cache, count_errors=True, counter=counter)
    primary = ScriptedModel(script=[LlmCacheMiss("brak")], label="primary")
    backup = ScriptedModel(script=[answer("ok", model="backup")], label="backup")
    runner, _ = agent_for(
        [],
        model=primary,
        fallbacks=[backup],
        fallback_exceptions=_offline(),
        extra_middleware=[meter],
    )
    assert runner("Dzień dobry").status == "answered"
    assert (counter.errors, meter.errors) == (1, 1)
    assert counter.network == 1 and counter.used == 2


def test_budget_exhaustion_is_not_counted_as_an_error(agent_for):
    """Odmowa wejścia przy wyczerpanym limicie nie jest nieudaną próbą: nie powiększa `errors`
    ani licznika (inaczej limit rósłby sam przy każdej pominiętej turze)."""
    from kovalo.agents.middleware import CallCounter, CallMeter

    counter = CallCounter(max_calls=1)
    meter = CallMeter(counter=counter, count_errors=True)
    runner, model = agent_for([answer("pierwsza"), answer("druga")], extra_middleware=[meter])
    assert runner("a").status == "answered"
    for _ in range(3):
        trace = runner("b")
        assert trace.status == "error" and "CallBudgetExhausted" in trace.error
    assert (meter.errors, counter.errors, counter.network) == (0, 0, 1)
    assert model.calls == 1


def test_count_errors_counts_each_failed_fallback_attempt(agent_for):
    """Obie nieudane próby (główny 429, zapasowy 429) zjadają limit, bo CallMeter jest wewnątrz
    fallbacku i liczy każdą próbę (R1-6)."""
    from langchain_core.exceptions import ModelRateLimitError

    from kovalo.agents.middleware import CallCounter, CallMeter

    counter = CallCounter(max_calls=3)
    meter = CallMeter(counter=counter, count_errors=True)
    primary = ScriptedModel(script=[ModelRateLimitError("429 a")] * 2, label="primary")
    backup = ScriptedModel(script=[ModelRateLimitError("429 b")] * 2, label="backup")
    runner, _ = agent_for([], model=primary, fallbacks=[backup], extra_middleware=[meter])
    assert runner("Dzień dobry").status == "error"
    assert counter.errors == 2 and counter.used == 2
    second = runner("Dzień dobry")  # trzecia próba jeszcze wchodzi, czwarta już nie
    assert second.status == "error"
    assert counter.used == 3 and (primary.calls, backup.calls) == (2, 1)
    assert "CallBudgetExhausted" in runner("Dzień dobry").error
