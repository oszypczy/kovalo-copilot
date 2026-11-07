"""Program E4 (kontrakt F4, 11.3-11.5, 15.2): przeliczenie przebiegów kursu, odtworzenie ich wywołań
na naszych narzędziach i przebieg na żywo. Bez sieci: atrapa modelu z idealnymi skryptami,
`DeterministicFakeEmbedding`, cache embeddingów niepotrzebny."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.agent_fakes import ScriptedModel
from tests.unit.eval.trace.conftest import count, fraction

COURSE_RUNS = (
    Path(__file__).resolve().parents[4] / "reference/course/mini_project_4/reference_traces"
)
IDS = ["P1", "P2", "P3", "T1", "A1", "D1", "S1", "N1", "B1"]


def qids(run):
    return [r["trace"].question_id for r in run.rows]


def course_traces(name):
    from kovalo.agents.trace import load_traces

    return load_traces(COURSE_RUNS / name)


# --- przeliczenie i odtworzenie (zawsze, zero LLM-a) --------------------------------------------


def test_course_runs_are_rescored_exactly(e4_replay_report):
    """Kryterium 2 F4: wyniki zapisane przez kurs: 8/9, 3/3, 8/9, 0 (0), 0/9 oraz 8/9, 3/3, 8/9, 3 (2), 0/9."""  # noqa: E501
    report = e4_replay_report
    assert (report.course.mode, report.course_supervisor.mode) == ("course", "course_supervisor")
    single, team = report.course.summary, report.course_supervisor.summary
    assert fraction(single["tools"]) == (8, 9) and fraction(team["tools"]) == (8, 9)
    assert fraction(single["args"]) == (3, 3) and fraction(team["args"]) == (3, 3)
    assert fraction(single["facts"]) == (8, 9) and fraction(team["facts"]) == (8, 9)
    assert count(single["uncovered"]) == 0 and count(team["uncovered"]) == 3
    assert fraction(single["approval"])[0] == 0 and fraction(team["approval"])[0] == 0
    assert single["outcomes"] == {"answered": 9, "error": 0, "limit": 0}
    assert (single["tool_calls"], single["failed"]) == (21, 0)
    assert qids(report.course) == IDS and qids(report.course_supervisor) == IDS
    for name, run in (
        ("agent_run.json", report.course),
        ("supervisor_run.json", report.course_supervisor),
    ):
        saved_run, saved_rows = course_traces(name)
        assert run.run["run_id"] == saved_run["run_id"]
        assert [r["score"]["tool_coverage"] for r in run.rows] == [
            r["score"]["tool_coverage"] for r in saved_rows
        ]


def test_rescore_mismatch_is_reported(session_settings, tmp_path, monkeypatch):
    """Przeliczenie ≠ zapis kursu → E4ReferenceMismatch z id pytania i kluczem (kod 4 w CLI)."""
    from kovalo.eval.trace import reference

    saved = json.loads((COURSE_RUNS / "agent_run.json").read_text(encoding="utf-8"))
    saved["traces"][3]["score"]["tool_coverage"] = 1.0  # T1: kurs zapisał 0.667
    broken = tmp_path / "agent_run.json"
    broken.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(reference, "COURSE_RUNS", {**reference.COURSE_RUNS, "single": str(broken)})
    with pytest.raises(reference.E4ReferenceMismatch, match="T1") as info:
        reference.rescore_course(session_settings, "single")
    assert "tool_coverage" in str(info.value)


def test_course_e4_reference(session_settings, e4_replay_report):
    from kovalo.eval.trace.reference import course_e4

    reference = course_e4(session_settings.reference_llm.model)
    assert e4_replay_report.reference == reference
    assert session_settings.reference_llm.model in json.dumps(reference, ensure_ascii=False)
    assert e4_replay_report.reference_cost_model == session_settings.cost.reference_model


def test_replay_reproduces_course_calls(e4_replay_report):
    """Odtworzenie wywołań kursu na NASZYCH narzędziach: 9 razy answered, 21 wywołań, 0 nieudanych,
    argumenty 3/3, kary P1 i P2 = 729 600 / 362 400 groszy (P3 = 0)."""
    replay = e4_replay_report.replay
    assert replay.mode == "replay" and e4_replay_report.live is None
    assert replay.summary["outcomes"] == {"answered": 9, "error": 0, "limit": 0}
    assert (replay.summary["tool_calls"], replay.summary["failed"]) == (21, 0)
    assert fraction(replay.summary["args"]) == (3, 3)
    assert qids(replay) == IDS
    _, course_rows = course_traces("agent_run.json")
    for ours, theirs in zip(replay.rows, course_rows, strict=True):
        qid = theirs["trace"].question_id
        assert [(e.tool, e.arguments) for e in ours["trace"].events] == [
            (e.tool, e.arguments) for e in theirs["trace"].events
        ], f"{qid}: nazwy i argumenty jak w kursie"
        assert all(e.status == "ok" for e in ours["trace"].events), qid
        assert ours["trace"].answer == theirs["trace"].answer, qid
    penalties = {
        r["trace"].question_id: next(
            e.result["penalty_grosze"]
            for e in r["trace"].events
            if e.tool == "calculate_otif_penalty"
        )
        for r in replay.rows
        if any(e.tool == "calculate_otif_penalty" for e in r["trace"].events)
    }
    assert penalties["P1"] == 729600 and penalties["P2"] == 362400
    assert penalties.get("P3", 0) == 0
    assert replay.calls.network == 0 and replay.calls.from_cache == 0 and replay.calls.errors == 0
    assert replay.calls.tool_calls == 21 and replay.calls.model_turns == sum(
        len(r["trace"].model_calls) for r in replay.rows
    )


def test_replay_calls_no_model(session_settings, fake_embeddings):
    from kovalo.eval.trace.e4 import run_e4
    from kovalo.llm import override_chat_model, override_embeddings
    from tests.workflow_fakes import ExplodingChatModel

    exploding = ExplodingChatModel()
    with override_chat_model(exploding), override_embeddings(fake_embeddings):
        report = run_e4(session_settings)
    assert exploding.calls == [], "przeliczenie i odtworzenie nie wywołują ani nie wiążą modelu"
    assert report.live is None and report.replay.summary["outcomes"]["answered"] == 9


def test_scripted_tool_model_and_script_from_trace(agent_gold):
    """replay.py: skrypt z trace kursu = po jednym AIMessage z jednym wywołaniem + odpowiedź."""
    from kovalo.eval.trace.replay import ScriptedToolModel, script_from_trace

    _, rows = course_traces("agent_run.json")
    p1 = next(r["trace"] for r in rows if r["trace"].question_id == "P1")
    script = script_from_trace(p1)
    assert len(script) == len(p1.events) + 1
    assert [m.tool_calls[0]["name"] for m in script[:-1]] == [e.tool for e in p1.events]
    assert [m.tool_calls[0]["id"] for m in script[:-1]] == [e.call_id for e in p1.events]
    assert [m.tool_calls[0]["args"] for m in script[:-1]] == [e.arguments for e in p1.events]
    assert script[-1].content == p1.answer and not script[-1].tool_calls
    model = ScriptedToolModel(script=script)
    assert model.bind_tools([]) is model
    _, team = course_traces("supervisor_run.json")
    with pytest.raises(ValueError, match="jednego agenta"):
        script_from_trace(next(r["trace"] for r in team if r["trace"].question_id == "P1"))


def test_scripted_tool_model_runs_out(session_settings):
    from langchain_core.messages import AIMessage, HumanMessage

    from kovalo.eval.trace.replay import ScriptedToolModel

    model = ScriptedToolModel(script=[AIMessage("jedna")])
    assert model.invoke([HumanMessage("a")]).content == "jedna" and model.calls == 1
    with pytest.raises(AssertionError):
        model.invoke([HumanMessage("b")])


# --- przebieg na żywo (atrapa modelu) -----------------------------------------------------------


def test_live_fake_run(e4_live_report, session_settings, e4_fakes):
    """Idealne skrypty (T1 przez check_termination_condition): cele z 11.5 spełnione."""
    report = e4_live_report
    live = report.live
    assert live.mode == "live" and report.replay.mode == "replay"
    summary = live.summary
    assert fraction(summary["tools"]) == (9, 9), "T1 dzięki FR-4.7"
    assert fraction(summary["args"]) == (3, 3)
    assert fraction(summary["facts"]) == (9, 9)
    assert count(summary["uncovered"]) == 0
    assert fraction(summary["approval"])[0] == 0
    assert summary["outcomes"] == {"answered": 9, "error": 0, "limit": 0}
    assert qids(live) == IDS
    turns = len(e4_fakes().script)
    assert live.calls.network == turns and live.calls.from_cache == 0 and live.calls.errors == 0
    assert live.calls.model_turns == turns
    assert live.calls.tool_calls == sum(len(r["trace"].events) for r in live.rows)
    t1 = next(r["trace"] for r in live.rows if r["trace"].question_id == "T1")
    assert [(e.tool, e.parent_call_id is None) for e in t1.events] == [
        ("search_documents", True),
        ("check_termination_condition", True),
        ("get_supplier_otif", False),
    ], "T1: zdarzenie składowe get_supplier_otif pod check_termination_condition"
    assert live.run["model"] == session_settings.llm.model
    assert live.run["provider"] == session_settings.llm.provider
    assert live.run["tools_variant"] == "ours"
    assert live.run["budget"] == {"max_steps": 24, "timeout_s": 180}
    assert {"run_id", "date", "architecture", "policy_version", "terms_version"} <= set(live.run)
    assert set(live.answered_by) == {"scripted-model"}
    assert report.backend == "memory" and report.embedding_model
    assert report.stopped_by_max_calls is False and report.max_calls is None
    assert summary["tokens_in"] == 900 and summary["tokens_out"] == 180
    assert report.live_cost_cents is not None and set(report.live_cost_cents) == {
        "ours",
        "reference",
    }


def test_live_per_question_scores_are_ours(e4_live_report, agent_gold):
    """Ocena na żywo = score_trace na wzorcu: żadne pytanie nie twierdzi, że coś zatwierdzono."""
    from kovalo.eval.trace.scoring import score_trace

    for row in e4_live_report.live.rows:
        qid = row["trace"].question_id
        assert row["score"] == score_trace(row["trace"], agent_gold[qid]), qid
        assert row["score"]["approval_claim"] is False


def test_max_calls_skips_rest(session_settings, e4_fakes, fake_embeddings):
    from kovalo.eval.trace.e4 import run_e4
    from kovalo.llm import override_embeddings

    model = e4_fakes()
    with override_embeddings(fake_embeddings):
        report = run_e4(session_settings, live=True, model=model, max_calls=3)
    live = report.live
    assert report.stopped_by_max_calls is True and report.max_calls == 3
    by_id = {r["trace"].question_id: r["trace"] for r in live.rows}
    assert by_id["P1"].status == "answered", "pierwsze pytanie mieści się w limicie 3 wywołań"
    for qid in IDS[1:]:
        assert by_id[qid].status == "error" and "pominięte" in by_id[qid].error, qid
        assert "--max-calls" in by_id[qid].error
    assert live.calls.network == 3 and model.calls == 3, "limit dotyczy wywołań sieciowych"
    assert fraction(live.summary["tools"])[1] == 9, "pominięte zostają w mianowniku"


def test_max_calls_inside_first_question_skips_everything(
    session_settings, e4_fakes, fake_embeddings
):
    """Limit 2 przerywa P1 w środku (potrzebuje 3 tur): P1 i reszta „pominięte”, model wywołany 2 razy."""  # noqa: E501
    from kovalo.eval.trace.e4 import run_e4
    from kovalo.llm import override_embeddings

    model = e4_fakes()
    with override_embeddings(fake_embeddings):
        report = run_e4(session_settings, live=True, model=model, max_calls=2)
    live = report.live
    assert report.stopped_by_max_calls is True
    assert [r["trace"].status for r in live.rows] == ["error"] * 9
    assert all(r["trace"].error.startswith("pominięte") for r in live.rows)
    assert live.calls.network == 2 and model.calls == 2


def test_max_calls_run_resumes_from_cache(session_settings, ctx_chunks, tmp_path, fake_embeddings):
    """Scenariusz „jutro”: przebieg przerwany limitem w środku pytania, ponowny przebieg z tym
    samym cache wykonuje tylko brakujące tury; trafienia w cache nie liczą się do limitu."""
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.eval.trace.e4 import run_e4
    from kovalo.llm import override_embeddings
    from tests.agent_fakes import ideal_e4_scripts

    ids = ["P1", "P2", "P3"]
    scripts = ideal_e4_scripts(ctx_chunks)
    messages = [m for i in ids for m in scripts[i]]
    total = len(messages)
    first_limit = len(scripts["P1"]) + 1  # P1 w całości i jedna tura P2

    cache = JsonFileLlmCache(tmp_path / "llm")
    model1 = ScriptedModel(script=list(messages), cache=cache)
    with override_embeddings(fake_embeddings):
        part = run_e4(
            session_settings, live=True, ids=ids, model=model1, cache=cache, max_calls=first_limit
        )
    assert part.stopped_by_max_calls is True and model1.calls == first_limit
    assert [r["trace"].status for r in part.live.rows] == ["answered", "error", "error"]

    cache2 = JsonFileLlmCache(tmp_path / "llm")  # nowy proces: nowe liczniki, ten sam katalog
    model2 = ScriptedModel(script=list(messages[first_limit:]), cache=cache2)
    limit2 = total - first_limit  # dokładnie tyle, ile brakuje: cache nie zjada limitu
    with override_embeddings(fake_embeddings):
        rest = run_e4(
            session_settings, live=True, ids=ids, model=model2, cache=cache2, max_calls=limit2
        )
    assert rest.stopped_by_max_calls is False
    assert [r["trace"].status for r in rest.live.rows] == ["answered"] * 3
    assert model2.calls == limit2
    assert (rest.live.calls.network, rest.live.calls.from_cache) == (limit2, first_limit)


def test_second_run_uses_cache(session_settings, ctx_chunks, tmp_path, fake_embeddings):
    """Jak E3: drugi przebieg z tym samym cache = zero wywołań modelu, te same wyniki."""
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.eval.trace.e4 import run_e4
    from kovalo.llm import override_embeddings
    from tests.agent_fakes import ideal_e4_scripts

    scripts = ideal_e4_scripts(ctx_chunks)
    ids = ["P1", "T1"]
    messages = [m for i in ids for m in scripts[i]]

    def run(cache):
        model = ScriptedModel(script=list(messages), cache=cache)
        with override_embeddings(fake_embeddings):
            report = run_e4(
                session_settings, live=True, ids=ids, model=model, cache=cache, max_calls=None
            )
        return report, model

    first, m1 = run(JsonFileLlmCache(tmp_path / "llm"))
    assert (first.live.calls.network, first.live.calls.from_cache) == (5, 0)
    assert m1.calls == 5

    second, m2 = run(JsonFileLlmCache(tmp_path / "llm"))  # nowe liczniki, ten sam katalog
    assert (second.live.calls.network, second.live.calls.from_cache) == (0, 5)
    assert m2.calls == 0, "drugi przebieg: zero wywołań modelu"
    assert second.live.summary["outcomes"] == first.live.summary["outcomes"]
    assert [r["trace"].answer for r in second.live.rows] == [
        r["trace"].answer for r in first.live.rows
    ]


def test_cache_only_without_key(tmp_path, fake_embeddings):
    """offline=True, pusty cache strict, bez klucza: pytanie kończy się błędem LlmCacheMiss, zero sieci."""  # noqa: E501
    from kovalo.config import load_settings
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.eval.trace.e4 import E4AllErrorsError, run_e4
    from kovalo.llm import override_embeddings

    settings = load_settings()  # bez klucza (izolowane środowisko)
    strict = JsonFileLlmCache(tmp_path / "llm", mode="strict")
    with override_embeddings(fake_embeddings), pytest.raises(E4AllErrorsError) as info:
        run_e4(settings, live=True, ids=["P1"], cache=strict, offline=True)
    report = info.value.report
    live = report.live
    assert len(live.rows) == 1
    trace = live.rows[0]["trace"]
    assert trace.status == "error" and "LlmCacheMiss" in trace.error
    # F5 (ADR-0008, adaptacja): offline brak u modelu głównego przechodzi do fallbacku, który
    # też nie trafia; CallMeter liczy każdą próbę (także fallbacku), więc 2 nieudane próby.
    fallbacks = len(settings.llm.fallbacks)
    assert live.calls.network == 0 and live.calls.errors == 1 + fallbacks
    assert report.replay.summary["outcomes"]["answered"] == 1, (
        "odtworzenie zapisane mimo błędów na żywo"
    )


def test_live_without_key_is_config_error():
    from kovalo.config import ConfigError, load_settings
    from kovalo.eval.trace.e4 import run_e4

    with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
        run_e4(load_settings(), live=True, ids=["P1"])


def test_all_live_errors_raise_with_report(session_settings, fake_embeddings):
    from kovalo.eval.trace.e4 import E4AllErrorsError, run_e4
    from kovalo.llm import override_embeddings

    failing = ScriptedModel(script=[ValueError("sztuczny błąd")] * 9)
    with override_embeddings(fake_embeddings), pytest.raises(E4AllErrorsError) as info:
        run_e4(session_settings, live=True, model=failing, max_calls=None)
    report = info.value.report
    assert all("ValueError: sztuczny błąd" in r["trace"].error for r in report.live.rows)
    assert report.replay.summary["outcomes"]["answered"] == 9


def test_tools_variant_course_has_six_tools(session_settings, e4_fakes, fake_embeddings):
    """--tools course: model dostaje sześć narzędzi kursu (T1 bez FR-4.7)."""
    from kovalo.eval.trace.e4 import run_e4
    from kovalo.llm import override_embeddings

    model = e4_fakes(["P1"])
    with override_embeddings(fake_embeddings):
        report = run_e4(
            session_settings, live=True, ids=["P1"], model=model, variant="course", max_calls=None
        )
    assert len(model.bound) == 6 and "check_termination_condition" not in model.bound
    assert report.live.run["tools_variant"] == "course"


# --- plan i filtr pytań --------------------------------------------------------------------------


def test_plan_llm_calls(agent_gold):
    from kovalo.eval.trace.e4 import plan_llm_calls

    assert plan_llm_calls(list(agent_gold.values())) == {"estimate": 22, "upper_bound": 108}
    assert plan_llm_calls([agent_gold["T1"]]) == {"estimate": 3, "upper_bound": 12}
    assert plan_llm_calls([agent_gold["P1"], agent_gold["D1"]]) == {
        "estimate": 5,
        "upper_bound": 24,
    }


def test_ids_filter_and_unknown(session_settings, fake_embeddings):
    from kovalo.config import ConfigError
    from kovalo.eval.trace.e4 import run_e4
    from kovalo.llm import override_embeddings

    with override_embeddings(fake_embeddings):
        report = run_e4(session_settings, ids=["P1", "T1"])
        assert report.questions == ["P1", "T1"]
        assert qids(report.replay) == ["P1", "T1"]
        assert report.llm_calls_planned == {"estimate": 6, "upper_bound": 24}
        with pytest.raises(ConfigError, match="X9"):
            run_e4(session_settings, ids=["X9"])


# --- F5: E4 przez MCP (kontrakt F5, 10.6 i 13.2; ADR-0014) ---------------------------------------


def _top_events(run):
    return {
        r["trace"].question_id: [e for e in r["trace"].events if e.parent_call_id is None]
        for r in run.rows
    }


def test_e4_live_over_mcp(session_settings, ctx_chunks, e4_live_report, fake_embeddings):
    """`run_e4(live=True, transport='memory', model=atrapa)`: zdarzenia mają `transport 'mcp'`,
    a wyniki (narzędzie, argumenty, rekord) są te same co przy przebiegu `direct`."""
    from kovalo.eval.trace.e4 import run_e4
    from kovalo.llm import override_embeddings
    from tests.agent_fakes import ideal_e4_model

    model = ideal_e4_model(ctx_chunks)
    with override_embeddings(fake_embeddings):
        report = run_e4(
            session_settings, live=True, transport="memory", model=model, max_calls=None
        )
    over_mcp, direct = _top_events(report.live), _top_events(e4_live_report.live)
    assert list(over_mcp) == IDS
    for qid in IDS:
        events = over_mcp[qid]
        assert events and all(e.transport == "mcp" and e.invocation_id for e in events), qid
        assert [(e.tool, e.arguments, e.result) for e in events] == [
            (e.tool, e.arguments, e.result) for e in direct[qid]
        ], f"{qid}: te same wyniki co direct"
        assert all(e.transport is None for e in direct[qid])
    assert report.live.summary["outcomes"] == e4_live_report.live.summary["outcomes"]
    for key in ("tools", "args", "facts", "uncovered", "approval"):
        assert report.live.summary[key] == e4_live_report.live.summary[key], key
    # 21 = liczba tur w skryptach P1…B1, tyle samo co w przebiegu bezpośrednim (test_live_fake_run)
    assert report.live.calls.network == e4_live_report.live.calls.network == 21


def test_e4_default_transport_is_direct(e4_live_report):
    for events in _top_events(e4_live_report.live).values():
        assert all(e.transport is None for e in events)


def test_e4_transport_needs_live_and_known_value(session_settings, fake_embeddings):
    from kovalo.config import ConfigError
    from kovalo.eval.trace.e4 import run_e4
    from kovalo.llm import override_embeddings

    with override_embeddings(fake_embeddings), pytest.raises((ConfigError, ValueError)):
        run_e4(session_settings, live=True, transport="smtp", ids=["P1"])


def test_e4_count_errors_makes_failed_attempts_count(session_settings, fake_embeddings):
    """R1-6: `count_errors=True` — nieudane próby zjadają `max_calls` (E4 domyślnie nie, liczby F4
    zostają porównywalne)."""
    from kovalo.eval.trace.e4 import E4AllErrorsError, run_e4
    from kovalo.llm import override_embeddings

    failing = ScriptedModel(script=[ValueError("sztuczny błąd")] * 9)
    with override_embeddings(fake_embeddings), pytest.raises(E4AllErrorsError) as info:
        run_e4(session_settings, live=True, model=failing, max_calls=2, count_errors=True)
    report = info.value.report
    assert report.stopped_by_max_calls is True and failing.calls == 2
    statuses = [r["trace"].error for r in report.live.rows]
    assert sum("pominięte" in (e or "") for e in statuses) == 7
    assert report.live.calls.errors == 2 and report.live.calls.network == 0

    plain = ScriptedModel(script=[ValueError("sztuczny błąd")] * 9)
    with override_embeddings(fake_embeddings), pytest.raises(E4AllErrorsError) as info:
        run_e4(session_settings, live=True, model=plain, max_calls=2)
    assert info.value.report.stopped_by_max_calls is False and plain.calls == 9, (
        "domyślnie błędy nie liczą się do limitu (jak F4)"
    )
