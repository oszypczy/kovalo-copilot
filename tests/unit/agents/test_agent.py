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
