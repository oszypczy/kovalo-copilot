"""Program E3 (kontrakt F3, 10.3-10.4 i 14.2). Bez sieci: atrapy modeli, cache embeddingów kursu."""

from __future__ import annotations

import pytest

from tests.workflow_fakes import ExplodingChatModel

STAGES = ("route", "fields", "request", "approvers", "pauses", "status")
COURSE = {"route": 13, "fields": 11, "request": 8, "approvers": 11, "pauses": 13, "status": 13}


def stage_text(score: dict) -> dict[str, list[int]]:
    return {k: list(score[k]) for k in STAGES}


def test_every_gold_dialogue_replays_perfectly(e3_replay_report):
    """Port `test_every_gold_dialogue_replays_perfectly` kursu przez run_e3(live=False)."""
    replay = e3_replay_report.replay
    assert replay.mode == "replay" and e3_replay_report.live is None
    wrong = [(d["user"], d["wrong"]) for d in replay.details if d["wrong"]]
    assert not wrong, (
        f"przy idealnym zrozumieniu każdy etap musi zgadzać się ze wzorcem; błędne: {wrong}"
    )
    assert stage_text(replay.score) == {k: [n, n] for k, n in COURSE.items()}  # twarde liczby kursu
    assert replay.score["errors"] == 0
    assert len(replay.transcript) == 13 and e3_replay_report.dialogues == [
        f"D{i}" for i in range(1, 9)
    ]
    assert replay.policy == "course-v1" and replay.provider is None and replay.model is None
    assert replay.calls.understand == 0 and replay.calls.answer == 0


def test_replay_matches_course_reference(e3_replay_report):
    from kovalo.eval.workflow.reference import COURSE_E3

    ref = e3_replay_report.reference
    assert ref == COURSE_E3
    for stage, total in COURSE.items():
        assert ref[stage] == [total, total]
    assert ref["errors"] == 0 and ref["cost_cents"] == 1.65
    assert ref["calls"] == {"understand": 13, "answer": 1}
    assert ref["run_id"] == "20261006T212757-fe6041ef" and ref["provider"] == "openai"


def test_reference_model_comes_from_config(session_settings):
    from kovalo.eval.workflow.reference import COURSE_E3

    assert COURSE_E3["model"] == session_settings.reference_llm.model


def test_replay_transcript_follows_gold(e3_replay_report, gold_dialogues):
    transcript = e3_replay_report.replay.transcript
    gold = [(d.id, t) for d in gold_dialogues for t in d.turns]
    assert [e["dialogue"] for e in transcript] == [d for d, _ in gold]
    assert [e["status"] for e in transcript] == [t.expect["status"] for _, t in gold]
    d2_t2 = transcript[4]
    assert d2_t2["dialogue"] == "D2" and d2_t2["approvals"] == {
        "CEO": "approved",
        "CFO": "rejected",
    }
    d7_t1 = transcript[10]
    assert d7_t1["record"]["question"].startswith("Co komitet cenowy postanowił")
    assert d7_t1["record"]["output"]["citations"] == ["MTG-402"]


def test_replay_calls_no_model(session_settings):
    from kovalo.eval.workflow.e3 import E3Calls, run_e3
    from kovalo.llm import override_chat_model

    exploding = ExplodingChatModel()
    with override_chat_model(exploding):
        report = run_e3(session_settings)
    assert exploding.calls == [], "odtworzenie nie wywołuje ani nie wiąże modelu"
    assert report.replay.calls == E3Calls()
    assert report.replay.score["errors"] == 0


def test_replay_threads_use_run_id_and_dialogue(e3_replay_report):
    run = e3_replay_report.replay
    import re

    assert re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{8}", run.run_id)


def test_live_fake_run(e3_live_report, e3_replay_report, session_settings):
    live = e3_live_report.live
    assert live.mode == "live"
    assert stage_text(live.score) == stage_text(e3_replay_report.replay.score), (
        "idealne Turn „na żywo” = odtworzenie"
    )
    assert live.score["errors"] == 0
    assert (live.calls.understand, live.calls.answer, live.calls.from_cache, live.calls.errors) == (
        13,
        1,
        0,
        0,
    )
    assert live.provider == session_settings.llm.provider
    assert live.model == session_settings.llm.model
    assert live.answered_by == {"caching-fake": 14}
    assert live.usage.input_tokens == 14 * 50 and live.usage.output_tokens == 14 * 10
    assert e3_live_report.replay.mode == "replay"
    assert e3_live_report.embedding_model, "live: model wyszukiwania dla D7"
    assert e3_live_report.backend == "memory"


def test_live_failed_pause_skips_rest(
    session_settings, gold_dialogues, ideal_model, fake_embeddings
):
    """Zły Turn w D2-t1 (22% zamiast 14%) wymaga dodatkowej pauzy: skrypt jej nie ma."""
    from kovalo.eval.workflow.e3 import E3AllErrorsError, run_e3
    from kovalo.llm import override_chat_model, override_embeddings

    d2 = next(d for d in gold_dialogues if d.id == "D2")
    bad = dict(d2.turns[0].understand, discount_pct=22)
    model = ideal_model(overrides={d2.turns[0].user: bad})
    with override_chat_model(model), override_embeddings(fake_embeddings):
        try:
            report = run_e3(session_settings, live=True, ids=["D2"])
        except E3AllErrorsError as exc:  # dwie tury z błędem: wszystkie na żywo zakończone błędem
            report = exc.report
    live = report.live
    assert len(live.transcript) == 2
    first, second = live.transcript
    assert first["status"] == "error" and "przy pauzie" in first["error"]
    assert first["pauses"] == [["approval", "CEO"], ["approval", "CFO"]]
    assert second["status"] == "error", "D2-t2 pominięta, wpis błędu"
    assert live.score["status"][1] == 2 and live.score["route"][1] == 2, (
        "pominięte tury zostają w mianowniku"
    )
    assert live.score["errors"] == 2
    assert len(model.sent) == 1, "druga wiadomość D2 nie została wysłana do modelu"


def test_second_run_uses_cache(session_settings, ideal_model, tmp_path, fake_embeddings):
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.eval.workflow.e3 import run_e3
    from kovalo.llm import override_chat_model, override_embeddings

    def run(cache):
        model = ideal_model(cache=cache)
        with override_chat_model(model), override_embeddings(fake_embeddings):
            report = run_e3(session_settings, live=True, ids=["D2"], cache=cache)
        return report, model

    first, m1 = run(JsonFileLlmCache(tmp_path / "llm"))
    assert (first.live.calls.understand, first.live.calls.from_cache) == (2, 0)
    assert len(m1.sent) == 2

    second, m2 = run(JsonFileLlmCache(tmp_path / "llm"))  # nowe liczniki, ten sam katalog
    assert (second.live.calls.understand, second.live.calls.answer) == (0, 0)
    assert second.live.calls.from_cache == 2
    assert m2.sent == [], "drugi przebieg: zero wywołań modelu"
    assert stage_text(second.live.score) == stage_text(first.live.score)


def test_cache_only_without_key(session_settings, tmp_path, fake_embeddings):
    """offline=True, pusty cache strict, bez klucza: tury error z LlmCacheMiss, zero wywołań."""
    from kovalo.config import load_settings
    from kovalo.eval.llm_cache import JsonFileLlmCache
    from kovalo.eval.workflow.e3 import E3AllErrorsError, run_e3
    from kovalo.llm import override_embeddings

    settings = load_settings()  # bez klucza (izolowane środowisko)
    strict = JsonFileLlmCache(tmp_path / "llm", mode="strict")
    with override_embeddings(fake_embeddings), pytest.raises(E3AllErrorsError) as info:
        run_e3(settings, live=True, ids=["D1"], cache=strict, offline=True)
    live = info.value.report.live
    assert len(live.transcript) == 3
    assert all(e["status"] == "error" and "LlmCacheMiss" in e["error"] for e in live.transcript)
    assert live.calls.understand == 0 and live.calls.errors == 3
    assert info.value.report.replay.score["errors"] == 0, "odtworzenie zapisane mimo błędów na żywo"


def test_plan_llm_calls(gold_dialogues):
    from kovalo.eval.workflow.e3 import plan_llm_calls

    assert plan_llm_calls(gold_dialogues) == {"understand": 13, "answer": 1, "total": 14}
    d7 = [d for d in gold_dialogues if d.id == "D7"]
    assert plan_llm_calls(d7) == {"understand": 2, "answer": 1, "total": 3}
    d2 = [d for d in gold_dialogues if d.id == "D2"]
    assert plan_llm_calls(d2) == {"understand": 2, "answer": 0, "total": 2}


def test_ids_filter_and_unknown(session_settings):
    from kovalo.config import ConfigError
    from kovalo.eval.workflow.e3 import run_e3

    report = run_e3(session_settings, ids=["D2"])
    assert report.dialogues == ["D2"] and len(report.replay.transcript) == 2
    assert [e["dialogue"] for e in report.replay.transcript] == ["D2", "D2"]
    with pytest.raises(ConfigError, match="D9"):
        run_e3(session_settings, ids=["D9"])


def test_live_without_key_is_config_error(session_settings):
    from kovalo.config import ConfigError, load_settings
    from kovalo.eval.workflow.e3 import run_e3

    with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
        run_e3(load_settings(), live=True, ids=["D2"])


def test_all_live_errors_raise_with_report(session_settings, fake_embeddings):
    from kovalo.eval.workflow.e3 import E3AllErrorsError, run_e3
    from kovalo.llm import override_chat_model, override_embeddings
    from tests.fakes import FakeStructuredChatModel

    failing = FakeStructuredChatModel(error=ValueError("sztuczny błąd"))
    with (
        override_chat_model(failing),
        override_embeddings(fake_embeddings),
        pytest.raises(E3AllErrorsError) as info,
    ):
        run_e3(session_settings, live=True, ids=["D1"])
    report = info.value.report
    assert all("ValueError: sztuczny błąd" in e["error"] for e in report.live.transcript)
    assert report.replay.score["route"] == [3, 3]
