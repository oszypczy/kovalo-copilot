"""Program E5 (kontrakt F5, 10.4 i 13.2): zgodność MCP, odtworzenie 13 rozmów z idealnym
`understand` (port `test_provided_conversation_replay.py`), przebieg na żywo na atrapach, wspólny
limit wywołań, cache. Bez sieci: serwer w pamięci, `DeterministicFakeEmbedding`."""

from __future__ import annotations

import json

import pytest

from tests.mcp_fakes import ScriptedChat, call, e5_interleaved_script, turn_message
from tests.unit.eval.assistant.conftest import turn_ids

DIMS_COURSE = {
    "route": 25,
    "status": 25,
    "fields": 11,
    "pauses": 22,
    "approvers": 19,
    "request": 15,
    "agent_trace": 25,
}
AGENT_DIMS = {
    "tool_coverage": 4,
    "arguments_correct": 1,
    "answer_facts": 4,
    "numbers_backed": 4,
    "no_approval_claim": 4,
}


def skipped(turn):
    entry = turn["entry"]
    return entry.get("status") == "skipped" and "wyczerpany limit --max-calls" in json.dumps(
        entry, ensure_ascii=False
    )


def flatten(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from flatten(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from flatten(v)
    else:
        yield value


# --- odtworzenie i zgodność (zawsze, zero LLM-a) --------------------------------------------------


def test_replay_is_perfect_over_mcp(e5_replay_report, conversations):
    """Port `test_every_conversation_replays_perfectly` kursu: 100% w każdym sprawdzeniu, 25 tur,
    `understand` wywołany dla każdej tury, agent tylko w 4 turach o dane, 0 nieudanych MCP."""
    from kovalo.eval.assistant.scoring import DIMENSIONS

    report = e5_replay_report
    replay = report.replay
    assert replay.mode == "replay" and report.live is None
    assert report.transport == "memory" and report.backend == "memory"
    assert report.conversations == [d.id for d in conversations]
    assert len(replay.turns) == 25
    assert [f"{t['dialogue']}-t{t['n']}" for t in replay.turns] == turn_ids(conversations)
    wrong = [
        (f"{t['dialogue']}-t{t['n']}", {k: v for k, v in t["score"].items() if v is False})
        for t in replay.turns
        if any(v is False for k, v in t["score"].items() if k in DIMENSIONS)
    ]
    assert not wrong, f"przy idealnym rozumieniu każde sprawdzenie musi przejść; błędne: {wrong}"
    totals = {k: tuple(v) for k, v in replay.totals.items()}
    assert all(passed == assessed for passed, assessed in totals.values())
    for dim, n in DIMS_COURSE.items():
        assert totals[dim] == (n, n), dim  # twarde liczby kursu
    for dim in AGENT_DIMS:
        assert totals[dim] == (0, 0), "odpowiedź agenta ze skryptu: tylko na żywo"
    # understand: każda tura rozumiana raz (tura bez rodzaju 'error'); agent: 4 tury o dane
    kinds = [t["entry"]["turn"]["kind"] for t in replay.turns]
    assert "error" not in kinds and len(kinds) == 25
    data = [t for t in replay.turns if t["entry"]["turn"]["kind"] == "document_question"]
    assert len(data) == 4 and all(t["entry"]["record"] for t in data)
    assert replay.calls.errors == 0 and sum(replay.calls.mcp_failures.values()) == 0
    assert (replay.calls.understand_network, replay.calls.agent_network) == (0, 0)


def test_parity_is_20_of_20(e5_replay_report):
    report = e5_replay_report
    assert len(report.parity) == 20
    assert all(row.same and row.error is None for row in report.parity)
    assert [row.tool for row in report.parity[17:]] == ["check_termination_condition"] * 3


def test_replay_matches_course_reference(e5_replay_report, session_settings):
    from kovalo.eval.assistant.reference import COURSE_E4_MCP, COURSE_E5

    ref = e5_replay_report.reference
    assert ref == COURSE_E5
    for dim, n in {**DIMS_COURSE, **AGENT_DIMS}.items():
        assert list(ref[dim]) == [n, n], dim
    values = set(flatten(COURSE_E5))
    assert {41993, 2866, 4.4, 7} <= values, "tokeny, koszt i 7 wywołań MCP kursu"
    e4 = set(flatten(COURSE_E4_MCP))
    assert {21, 40931, 1867, 3.9} <= e4


def test_replay_calls_no_model(session_settings, fake_embeddings):
    from kovalo.eval.assistant.e5 import run_e5
    from kovalo.llm import override_chat_model, override_embeddings
    from tests.workflow_fakes import ExplodingChatModel

    exploding = ExplodingChatModel()
    with override_chat_model(exploding), override_embeddings(fake_embeddings):
        report = run_e5(session_settings, transport="memory")
    assert exploding.calls == [], "odtworzenie nie wywołuje ani nie wiąże modelu"
    assert report.live is None and all(row.same for row in report.parity)


def test_run_info_and_prompt_names(e5_replay_report, session_settings):
    run = e5_replay_report.replay.run
    assert run["understand_prompt"] == "assistant_understand_system"
    assert run["policy_version"] == "course-v1" and run["tools_variant"] == "ours"
    assert (run["transport"], run["backend"]) == ("memory", "memory")
    assert run["run_id"] and run["date"] and run["terms_version"]


def test_report_round_trips_through_json(e5_replay_report):
    from kovalo.eval.assistant.e5 import E5Report

    restored = E5Report.model_validate_json(e5_replay_report.model_dump_json())
    assert restored.replay.totals == e5_replay_report.replay.totals
    assert len(restored.parity) == 20


def test_plan_llm_calls(conversations):
    from kovalo.eval.assistant.e5 import plan_llm_calls

    assert plan_llm_calls(conversations) == {
        "understand": 25,
        "agent_estimate": 9,
        "estimate": 34,
        "upper_bound": 73,
    }
    c5 = [d for d in conversations if d.id == "C5"]
    plan = plan_llm_calls(c5)
    assert (plan["understand"], plan["agent_estimate"], plan["estimate"]) == (2, 2, 4)
    assert plan["upper_bound"] == 2 + 1 * 12
    c1 = [d for d in conversations if d.id == "C1"]
    assert plan_llm_calls(c1)["agent_estimate"] == 3 + 2, "kara (z obliczeniem) 3, OTIF Aqua 2"
    assert plan_llm_calls(c5, max_steps=10)["upper_bound"] == 2 + 1 * 5


def test_ids_filter_and_unknown(session_settings, fake_embeddings):
    from kovalo.config import ConfigError
    from kovalo.eval.assistant.e5 import run_e5
    from kovalo.llm import override_embeddings

    with override_embeddings(fake_embeddings):
        report = run_e5(session_settings, transport="memory", ids=["C1", "D7"])
        assert report.conversations == ["C1", "D7"]
        assert [(t["dialogue"], t["n"]) for t in report.replay.turns] == [
            ("C1", 1), ("C1", 2), ("C1", 3), ("C1", 4), ("D7", 1), ("D7", 2),
        ]  # fmt: skip
        with pytest.raises(ConfigError, match="X9"):
            run_e5(session_settings, transport="memory", ids=["X9"])


# --- na żywo (atrapy) -----------------------------------------------------------------------------


def test_live_fake_run(e5_live_report, e5_replay_report, session_settings):
    """Idealne `understand` i skrypty agenta (P1/A1/N1/D7 przez MCP): cele §10.7 spełnione,
    `understand_network` 25, agent 9 tur modelu, 7 wywołań MCP bez błędów (kurs: 7 (0))."""
    live = e5_live_report.live
    assert live.mode == "live"
    totals = {k: tuple(v) for k, v in live.totals.items()}
    for dim, n in DIMS_COURSE.items():
        assert totals[dim] == (n, n), dim
    for dim, n in AGENT_DIMS.items():
        assert totals[dim] == (n, n), dim
    calls = live.calls
    assert (calls.understand_network, calls.understand_cache) == (25, 0)
    assert (calls.agent_network, calls.agent_cache, calls.errors) == (9, 0, 0)
    assert calls.mcp_calls == 7 and sum(calls.mcp_failures.values()) == 0
    assert (
        live.run["transport"] == "memory" and live.run["provider"] == session_settings.llm.provider
    )
    assert e5_live_report.stopped_by_max_calls is False
    for dim in DIMS_COURSE:
        assert tuple(e5_replay_report.replay.totals[dim]) == tuple(live.totals[dim]), dim


def test_live_data_turns_carry_mcp_provenance(e5_live_report):
    """Każde zdarzenie agenta ma `transport mcp` i `invocation_id` z captures tej tury."""
    from kovalo.agents.trace import trace_from_record

    data = [
        t for t in e5_live_report.live.turns if t["entry"]["turn"]["kind"] == "document_question"
    ]
    assert [f"{t['dialogue']}-t{t['n']}" for t in data] == ["C1-t1", "C1-t3", "C5-t1", "D7-t1"]
    for t in data:
        trace = trace_from_record(t["entry"]["record"])
        top = [e for e in trace.events if e.parent_call_id is None]
        assert top and all(e.transport == "mcp" and e.invocation_id for e in top)
        # równoległe narzędzia: captures wg zakończenia, trace wg kolejności wywołań
        assert sorted(e.invocation_id for e in top) == sorted(c["invocation_id"] for c in t["mcp"])
        assert all(c["outcome"] == "ok" for c in t["mcp"])
        assert all(c["thread_id"] and c["turn"] == t["n"] for c in t["mcp"])
    assert [len(t["mcp"]) for t in data] == [4, 1, 1, 1]
    penalty = trace_from_record(data[0]["entry"]["record"])
    assert any(
        e.tool == "calculate_otif_penalty" and e.result["penalty_grosze"] == 729600
        for e in penalty.events
    )
    others = [t for t in e5_live_report.live.turns if t not in data]
    assert all(t["mcp"] == [] for t in others)


def test_live_tokens_and_answered_by(e5_live_report):
    live = e5_live_report.live
    assert live.tokens_in > 0 and live.tokens_out > 0
    assert live.answered_by == {"scripted-model": 4}


def test_shared_max_calls_skips_rest(session_settings, conversations, ctx_chunks, fake_embeddings):
    """`--max-calls` to JEDEN licznik dla `understand` i agenta: limit 5 = C1-t1 (understand + 3
    tury agenta) i C1-t2 (understand); od C1-t3 wszystkie tury „skipped”, w mianowniku."""
    from kovalo.eval.assistant.e5 import run_e5
    from kovalo.eval.workflow.runner import ScriptedUnderstanding
    from kovalo.llm import override_embeddings
    from kovalo.workflow.schemas import Turn
    from tests.agent_fakes import ScriptedModel
    from tests.mcp_fakes import ideal_e5_agent_messages

    turns = [Turn.model_validate(t.understand) for d in conversations for t in d.turns]
    agent = ScriptedModel(script=ideal_e5_agent_messages(ctx_chunks)["C1-t1"])
    understand = ScriptedUnderstanding(turns)
    with override_embeddings(fake_embeddings):
        report = run_e5(
            session_settings,
            live=True,
            transport="memory",
            max_calls=5,
            understand_model=understand.as_runnable(),
            agent_model=agent,
        )
    live = report.live
    assert report.stopped_by_max_calls is True and report.max_calls == 5
    done = [t for t in live.turns if not skipped(t)]
    assert [(t["dialogue"], t["n"]) for t in done] == [("C1", 1), ("C1", 2)]
    assert all(skipped(t) for t in live.turns[2:]) and len(live.turns) == 25
    calls = live.calls
    assert calls.understand_network + calls.agent_network == 5
    assert (calls.understand_network, calls.agent_network) == (2, 3)
    assert agent.calls == 3 and understand.calls == 2
    passed, assessed = live.totals["route"]
    assert assessed == 25, "pominięte zostają w mianowniku"
    assert passed < 25


def test_all_live_errors_raise_with_report(session_settings, fake_embeddings):
    from langchain_core.runnables import RunnableLambda

    from kovalo.eval.assistant.e5 import E5AllErrorsError, run_e5
    from kovalo.llm import override_embeddings
    from tests.agent_fakes import ScriptedModel

    def boom(messages):
        raise ValueError("sztuczny błąd")

    with override_embeddings(fake_embeddings), pytest.raises(E5AllErrorsError) as info:
        run_e5(
            session_settings,
            live=True,
            ids=["C5"],
            transport="memory",
            max_calls=None,
            understand_model=RunnableLambda(boom),
            agent_model=ScriptedModel(script=[]),
        )
    report = info.value.report
    assert report.live is not None and report.live.calls.errors >= 1
    assert report.replay.totals["route"][0] == report.replay.totals["route"][1], (
        "odtworzenie zapisane mimo błędów na żywo"
    )


def test_failed_attempts_eat_the_shared_max_calls(session_settings, fake_embeddings):
    """R1-6 / U12: w E5 nieudana próba zjada wspólny limit (CLI zawsze liczy błędy), więc awaria
    dostawcy nie wypala darmowego limitu: po 2 błędach `understand` limit 2 jest wyczerpany,
    kolejne tury są pominięte (a nie próbowane w nieskończoność), model nie dostaje 3. wywołania."""
    from langchain_core.runnables import RunnableLambda

    from kovalo.eval.assistant.e5 import E5AllErrorsError, run_e5
    from kovalo.llm import override_embeddings
    from tests.agent_fakes import ScriptedModel

    seen = {"n": 0}

    def boom(messages):
        seen["n"] += 1
        raise ValueError("sztuczny błąd")

    with override_embeddings(fake_embeddings), pytest.raises(E5AllErrorsError) as info:
        run_e5(
            session_settings,
            live=True,
            ids=["C2", "C3", "C4", "C5"],
            transport="memory",
            max_calls=2,
            understand_model=RunnableLambda(boom),
            agent_model=ScriptedModel(script=[]),
        )
    report = info.value.report
    assert seen["n"] == 2, "trzecia próba nie dochodzi do modelu"
    assert report.stopped_by_max_calls is True
    assert report.live.calls.errors == 2 and report.live.calls.understand_network == 0
    assert any(skipped(t) for t in report.live.turns)
    assert len(report.live.turns) == 8


def test_live_without_key_is_config_error():
    from kovalo.config import ConfigError, load_settings
    from kovalo.eval.assistant.e5 import run_e5

    with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
        run_e5(load_settings(), live=True, ids=["C5"], transport="memory")


def test_understand_and_agent_use_the_assistant_prompts(
    session_settings, conversations, ctx_chunks, fake_embeddings
):
    """Prompt `assistant_understand_system` (nie `understand_system`) i `agent_system` (PINNED)."""
    from kovalo.eval.assistant.e5 import run_e5
    from kovalo.llm import override_chat_model, override_embeddings
    from kovalo.workflow.schemas import Turn
    from tests.mcp_fakes import ideal_e5_agent_messages

    c5 = next(d for d in conversations if d.id == "C5")
    n1 = ideal_e5_agent_messages(ctx_chunks)["C5-t1"]
    agent_systems: list[str] = []

    def spy(message):
        def respond(messages):
            agent_systems.append(messages[0].content)
            return message

        return respond

    llm = ScriptedChat(
        turns=[Turn.model_validate(t.understand) for t in c5.turns],
        script=[spy(n1[0]), spy(n1[1])],
    )
    with override_chat_model(llm), override_embeddings(fake_embeddings):
        report = run_e5(session_settings, live=True, ids=["C5"], transport="memory", max_calls=None)
    prompts = session_settings.prompts
    assert llm.seen_systems == [prompts.assistant_understand_system] * 2
    assert prompts.assistant_understand_system != prompts.understand_system
    assert agent_systems == [prompts.agent_system] * 2
    assert report.live.calls.understand_network == 2 and report.live.calls.agent_network == 2


# --- cache LLM-a (jak E3/E4) ---------------------------------------------------------------------


def test_second_run_uses_cache(
    session_settings, conversations, ctx_chunks, tmp_path, fake_embeddings
):
    """Drugi przebieg z tym samym cache = zero wywołań modelu, te same wyniki. Warunek: klucz
    cache nie zależy od losowego `invocation_id` w artefakcie ToolMessage (ADR-0013 pkt 4)."""
    from kovalo.eval.assistant.e5 import run_e5
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.llm import override_chat_model, override_embeddings
    from tests.agent_fakes import ScriptedModel

    script = e5_interleaved_script(conversations, ctx_chunks, ids=["C5"])
    assert len(script) == 4  # understand, 2 x agent, understand

    def run(cache):
        model = ScriptedModel(script=list(script), cache=cache)
        with override_chat_model(model), override_embeddings(fake_embeddings):
            report = run_e5(
                session_settings,
                live=True,
                ids=["C5"],
                transport="memory",
                cache=cache,
                max_calls=None,
            )
        return report, model

    first, m1 = run(JsonFileLlmCache(tmp_path / "llm"))
    calls = first.live.calls
    assert (calls.understand_network, calls.agent_network) == (2, 2)
    assert (calls.understand_cache, calls.agent_cache, calls.errors) == (0, 0, 0)
    assert m1.calls == 4

    second, m2 = run(JsonFileLlmCache(tmp_path / "llm"))  # nowe liczniki, ten sam katalog
    calls = second.live.calls
    assert m2.calls == 0, "drugi przebieg: zero wywołań modelu"
    assert (calls.understand_network, calls.agent_network) == (0, 0)
    assert (calls.understand_cache, calls.agent_cache) == (2, 2)
    assert second.live.totals == first.live.totals


def test_cache_only_without_key(session_settings, tmp_path, fake_embeddings):
    """offline=True, pusty cache strict, bez klucza: tury z błędem LlmCacheMiss, zero sieci."""
    from kovalo.config import load_settings
    from kovalo.eval.assistant.e5 import E5AllErrorsError, run_e5
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.llm import override_embeddings

    settings = load_settings()  # bez klucza (izolowane środowisko)
    strict = JsonFileLlmCache(tmp_path / "llm", mode="strict")
    with override_embeddings(fake_embeddings), pytest.raises(E5AllErrorsError) as info:
        run_e5(
            settings,
            live=True,
            ids=["C5"],
            transport="memory",
            cache=strict,
            offline=True,
            max_calls=None,
        )
    report = info.value.report
    assert report.live.calls.understand_network == 0 and report.live.calls.errors >= 1
    assert "LlmCacheMiss" in json.dumps(report.live.turns, default=str)
    assert all(row.same for row in report.parity), "zgodność i odtworzenie zapisane mimo błędów"


# --- pomocnicze: jedno połączenie MCP na przebieg, kontekst tury ---------------------------------


def test_one_connection_per_run_and_it_is_closed(session_settings, fake_embeddings, monkeypatch):
    """Połączenie MCP jedno na przebieg, zamykane na końcu (kontrakt 10.4)."""
    from kovalo.eval.assistant.e5 import run_e5
    from kovalo.llm import override_embeddings
    from kovalo.mcp_client import connection as module

    opened = []
    real = module.in_memory_connection

    def spy(*args, **kwargs):
        conn = real(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(module, "in_memory_connection", spy)
    monkeypatch.setattr("kovalo.eval.assistant.e5.in_memory_connection", spy, raising=False)
    with override_embeddings(fake_embeddings):
        run_e5(session_settings, transport="memory", ids=["C5"])
    assert opened and all(not c.alive for c in opened), "wszystkie połączenia zamknięte"


def test_turn_message_helper_roundtrip(conversations):
    """Pomocnik testów: wiadomość `Turn` dla structured output parsuje się z powrotem."""
    from kovalo.workflow.schemas import Turn
    from tests.agent_fakes import ScriptedModel

    t = conversations[0].turns[0]
    model = ScriptedModel(script=[turn_message(t.understand, "x")])
    result = model.with_structured_output(Turn, include_raw=True).invoke("hej")
    assert result["parsed"] == Turn.model_validate(t.understand)
    assert call("get_invoice_value", supplier="a", month="b").tool_calls
