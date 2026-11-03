"""Testy `kovalo first-call` (kontrakt F0, sekcja 7)."""

from __future__ import annotations

import pytest
from langchain_core.exceptions import ModelAPIError

from tests.conftest import make_fake


def test_first_call_ok(capsys: pytest.CaptureFixture[str], fake_chat_model, settings):
    from kovalo.cli import main
    from kovalo.llm import override_chat_model

    with override_chat_model(fake_chat_model):
        assert main(["first-call"]) == 0
    out = capsys.readouterr().out
    assert "Odpowiedział: fake" in out
    assert "Cześć" in out
    assert "Tokeny: wejście 9, wyjście 21, razem 30" in out


def test_first_call_missing_key_exit_2(capsys: pytest.CaptureFixture[str]):
    from kovalo.cli import main

    assert main(["first-call"]) == 2
    captured = capsys.readouterr()
    assert "GOOGLE_API_KEY" in captured.err
    assert captured.out == ""


def test_first_call_provider_error_exit_3(
    capsys: pytest.CaptureFixture[str], failing_chat_model, settings
):
    from kovalo.cli import main
    from kovalo.llm import override_chat_model

    with override_chat_model(failing_chat_model(ModelAPIError("503"))):
        assert main(["first-call", "--no-fallback"]) == 3
    assert "Wywołanie LLM-a nie powiodło się" in capsys.readouterr().err


def test_first_call_empty_answer_exit_4(capsys: pytest.CaptureFixture[str], settings):
    from kovalo.cli import main
    from kovalo.llm import override_chat_model

    with override_chat_model(make_fake("")):
        assert main(["first-call"]) == 4
    assert "Model zwrócił pustą odpowiedź." in capsys.readouterr().err


# ---------------------------------------------------------------------------
# F1: `kovalo ingest` i `kovalo eval e1` (kontrakt F1, sekcja 7)
# ---------------------------------------------------------------------------


@pytest.fixture
def counting_cached_factory(tmp_path, fake_embeddings):
    """Fabryka dla override_embeddings: atrapa -> licznik -> cache w tmp_path."""
    from kovalo.knowledge.embeddings import CountingEmbeddings, build_cached_embeddings

    def factory(spec):
        return build_cached_embeddings(
            CountingEmbeddings(fake_embeddings), cache_dir=tmp_path / "emb", model="fake/model"
        )

    return factory


def _patch_writer(monkeypatch, rows=(47, 0), exc=None):
    import kovalo.cli as cli

    written: list[tuple[int, object]] = []

    def writer(chunks, meta):
        if exc is not None:
            raise exc
        written.append((len(list(chunks)), meta))
        return rows

    monkeypatch.setattr(cli, "_index_writer", lambda *a, **k: writer)
    return written


def test_ingest_ok(capsys, monkeypatch, settings, counting_cached_factory):
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    written = _patch_writer(monkeypatch)
    with override_embeddings(counting_cached_factory):
        assert main(["ingest"]) == 0
    out = capsys.readouterr().out
    assert "Korpus: 18 dokumentów, 8762 tokeny (o200k_base)" in out
    assert "Chunki: 47 (średnio 603,7 znaku, maks. 798)" in out
    assert f"Embeddingi: fastembed / {settings.embeddings.local.model} (" in out
    assert "policzone 47, z cache 0" in out
    assert "Indeks: tabela kovalo_mp1, wierszy 47 (wcześniej 0)" in out
    assert "Czas: " in out and out.rstrip().endswith(" s")
    assert written and written[0][0] == 47


def test_ingest_corpus_error_exit_5(capsys, monkeypatch, tmp_path, counting_cached_factory):
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    _patch_writer(monkeypatch)
    monkeypatch.setenv("KOVALO_PATHS__CORPUS_DIR", str(tmp_path / "brak-korpusu"))
    with override_embeddings(counting_cached_factory):
        assert main(["ingest"]) == 5
    captured = capsys.readouterr()
    assert "brak-korpusu" in captured.err
    assert captured.out == ""


def test_ingest_db_unavailable_exit_6(capsys, monkeypatch, counting_cached_factory):
    from kovalo.cli import main
    from kovalo.knowledge.pgvector import IndexUnavailableError
    from kovalo.llm import override_embeddings

    _patch_writer(monkeypatch, exc=IndexUnavailableError("connection refused"))
    with override_embeddings(counting_cached_factory):
        assert main(["ingest"]) == 6
    err = capsys.readouterr().err
    assert "Postgres niedostępny (localhost:5432)" in err
    assert "Uruchom: make up" in err


def test_ingest_config_error_exit_2(capsys, monkeypatch):
    from kovalo.cli import main
    from kovalo.config import ConfigError
    from kovalo.llm import override_embeddings

    _patch_writer(monkeypatch)

    def broken(spec):
        raise ConfigError("Brak modelu: uruchom docker compose build")

    with override_embeddings(broken):
        assert main(["ingest"]) == 2
    assert "Brak modelu" in capsys.readouterr().err


def test_eval_e1_ok_writes_reports(capsys, tmp_path, fake_embeddings):
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    out_dir = tmp_path / "raporty"
    with override_embeddings(fake_embeddings):
        assert main(["eval", "e1", "--out", str(out_dir)]) == 0
    names = {p.name for p in out_dir.iterdir()}
    assert {"report.json", "chunks.jsonl"} <= names
    assert any(n.endswith("-e1.md") for n in names) and any(n.endswith("-e1.json") for n in names)
    assert "llm_call.json" not in names
    md = next(out_dir.glob("*-e1.md")).read_text(encoding="utf-8")
    assert "referencja kursu" in md


def test_eval_unknown_stage_or_model_exit_2(capsys, fake_embeddings, tmp_path):
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    assert main(["eval", "e5"]) == 2
    assert "Etap e5 jeszcze nie istnieje (F5)." in capsys.readouterr().err
    with override_embeddings(fake_embeddings):
        assert main(["eval", "e1", "--model", "nieznany", "--out", str(tmp_path)]) == 2
    assert "nieznany" in capsys.readouterr().err


def test_eval_with_llm_failure_exit_3(capsys, tmp_path, fake_embeddings, failing_chat_model):
    from langchain_core.exceptions import ModelAPIError

    from kovalo.cli import main
    from kovalo.llm import override_chat_model, override_embeddings

    out_dir = tmp_path / "raporty"
    with (
        override_embeddings(fake_embeddings),
        override_chat_model(failing_chat_model(ModelAPIError("503"))),
    ):
        assert main(["eval", "e1", "--with-llm", "--out", str(out_dir)]) == 3
    assert "Wywołanie LLM-a nie powiodło się" in capsys.readouterr().err
    assert not (out_dir / "llm_call.json").exists()
    md = next(out_dir.glob("*-e1.md")).read_text(encoding="utf-8")
    assert "## Wywołanie LLM-a" not in md


def test_eval_with_llm_ok_writes_llm_call(capsys, tmp_path, fake_embeddings, fake_chat_model):
    from kovalo.cli import main
    from kovalo.llm import override_chat_model, override_embeddings

    out_dir = tmp_path / "raporty"
    with override_embeddings(fake_embeddings), override_chat_model(fake_chat_model):
        assert main(["eval", "e1", "--with-llm", "--out", str(out_dir)]) == 0
    assert (out_dir / "llm_call.json").is_file()


def test_eval_pgvector_without_configured_model_exit_2(capsys, fake_embeddings, tmp_path, settings):
    # P4: --backend pgvector bez modelu z configu wśród ocenianych: kod 2, przed połączeniem z DB
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    local = settings.embeddings.local.model
    other = next(c.model for c in settings.embeddings.candidates if c.model != local)
    with override_embeddings(fake_embeddings):
        argv = ["eval", "e1", "--backend", "pgvector", "--model", other, "--out", str(tmp_path)]
        code = main(argv)
    assert code == 2
    assert "pgvector" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# F2: `kovalo ingest --chunks` i `kovalo eval e2` (kontrakt F2, 5.4 i 8.7)
# ---------------------------------------------------------------------------


def _capture_writer(monkeypatch):
    import kovalo.cli as cli

    written: list[tuple[str, int]] = []  # (tabela z meta, liczba chunków)
    tables: list[str] = []  # tabela przekazana do _index_writer

    def factory(settings, embeddings, table):
        tables.append(table)

        def writer(chunks, meta):
            n = len(list(chunks))
            written.append((meta.table, n))
            return n, 0

        return writer

    monkeypatch.setattr(cli, "_index_writer", factory)
    return written, tables


def test_ingest_all_two_tables(capsys, monkeypatch, settings, counting_cached_factory):
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    written, _ = _capture_writer(monkeypatch)
    with override_embeddings(counting_cached_factory):
        assert main(["ingest"]) == 0  # domyślnie --chunks all
    out = capsys.readouterr().out
    assert written == [("kovalo_mp1", 47), ("kovalo_ctx", 87)]  # najpierw mp1, potem ctx
    assert out.count("Korpus: ") == 1
    assert "Chunki: 47 (średnio 603,7 znaku, maks. 798)" in out
    assert "Chunki: 87 (średnio 385,6 znaku, maks. 775)" in out
    assert "Indeks: tabela kovalo_mp1, wierszy 47" in out
    assert "Indeks: tabela kovalo_ctx, wierszy 87" in out
    assert out.index("kovalo_mp1") < out.index("kovalo_ctx")
    assert "mp1" in out and "ctx" in out  # nazwy wariantów przed blokami


def test_ingest_single_variant(capsys, monkeypatch, counting_cached_factory):
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    written, tables = _capture_writer(monkeypatch)
    with override_embeddings(counting_cached_factory):
        assert main(["ingest", "--chunks", "ctx"]) == 0
    assert written == [("kovalo_ctx", 87)] and tables == ["kovalo_ctx"]
    assert "Indeks: tabela kovalo_ctx, wierszy 87" in capsys.readouterr().out


def test_ingest_table_requires_single_variant(capsys, monkeypatch, counting_cached_factory):
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    written, _ = _capture_writer(monkeypatch)
    with override_embeddings(counting_cached_factory):
        assert main(["ingest", "--table", "x"]) == 2
        assert main(["ingest", "--chunks", "all", "--table", "x"]) == 2
    assert "--table wymaga --chunks mp1 albo ctx" in capsys.readouterr().err
    assert written == []

    written, tables = _capture_writer(monkeypatch)
    with override_embeddings(counting_cached_factory):
        assert main(["ingest", "--chunks", "mp1", "--table", "inna_tabela"]) == 0
    assert tables == ["inna_tabela"]


def test_ingest_unknown_chunks_variant_exit_2(capsys, monkeypatch, counting_cached_factory):
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    _capture_writer(monkeypatch)
    with override_embeddings(counting_cached_factory), pytest.raises(SystemExit) as info:
        main(["ingest", "--chunks", "nieznany"])
    assert info.value.code == 2


def test_eval_e2_offline_ok(capsys, tmp_path, fake_embeddings):
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    out_dir = tmp_path / "raporty"
    with override_embeddings(fake_embeddings):
        assert main(["eval", "e2", "--out", str(out_dir)]) == 0
    names = {p.name for p in out_dir.iterdir()}
    assert {"retrieval_report.json", "chunks_ctx.jsonl"} <= names
    assert any(n.endswith("-e2.md") for n in names) and any(n.endswith("-e2.json") for n in names)
    assert not [n for n in names if n.startswith(("answers.", "live_report."))]
    out = capsys.readouterr().out
    assert "reference mp1: hit@1 9/10, hit@3 9/10, dowody 9/10, tokeny 850,3" in out
    assert "reference ctx: hit@1 9/10, hit@3 9/10, dowody 8/10, tokeny 610,4" in out
    assert "reference hybrid: hit@1 9/10, hit@3 10/10, dowody 10/10, tokeny 631,0" in out
    assert str(out_dir) in out  # ścieżki raportów


def test_eval_e2_with_llm_fake(capsys, tmp_path, fake_embeddings):
    from kovalo.cli import main
    from kovalo.llm import override_chat_model, override_embeddings
    from kovalo.rag.records import Answer, Verdict
    from tests.fakes import FakeStructuredChatModel

    fake = FakeStructuredChatModel(
        results={
            Answer: Answer(answer="CFO.", citations=["POL-001"], answerable=True),
            Verdict: Verdict(correct=True, grounded=True, reason="atrapa"),
        }
    )
    out_dir = tmp_path / "raporty"
    argv = ["eval", "e2", "--with-llm", "--no-cache", "--variants", "ctx", "--out", str(out_dir)]
    with override_embeddings(fake_embeddings), override_chat_model(fake):
        assert main(argv) == 0
    assert (out_dir / "answers.ours-ctx.json").is_file()
    assert (out_dir / "live_report.ours-ctx.json").is_file()
    assert not (out_dir / "answers.ours-hybrid.json").exists()
    out = capsys.readouterr().out
    assert "ctx: odpowiedzi " in out and "wywołania " in out


def test_eval_e2_errors_exit_codes(capsys, tmp_path, fake_embeddings):
    from kovalo.cli import main
    from kovalo.llm import override_chat_model, override_embeddings
    from tests.fakes import FakeStructuredChatModel

    out_dir = tmp_path / "raporty"
    base = ["eval", "e2", "--out", str(out_dir)]
    with override_embeddings(fake_embeddings):
        assert main([*base, "--variants", "nieznany"]) == 2
        assert main([*base, "--with-llm", "--ids", "X99"]) == 2
        assert main([*base, "--no-cache", "--cache-only"]) == 2
        assert main(["eval", "e5"]) == 2
        err = capsys.readouterr().err
        assert "nieznany" in err and "X99" in err and "Etap e5 jeszcze nie istnieje (F5)." in err

        failing = FakeStructuredChatModel(error=ValueError("sztuczny błąd"))
        with override_chat_model(failing):
            assert main([*base, "--with-llm", "--no-cache", "--variants", "ctx"]) == 3
        assert "ValueError: sztuczny błąd" in capsys.readouterr().err
    assert (out_dir / "answers.ours-ctx.json").is_file()  # raporty zapisane mimo błędów
    assert any(p.name.endswith("-e2.md") for p in out_dir.iterdir())


def test_eval_e2_with_llm_missing_key_exit_2(capsys, tmp_path, fake_embeddings):
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    out_dir = tmp_path / "raporty"
    with override_embeddings(fake_embeddings):
        assert main(["eval", "e2", "--with-llm", "--no-cache", "--out", str(out_dir)]) == 2
    assert "GOOGLE_API_KEY" in capsys.readouterr().err
    assert any(p.name.endswith("-e2.md") for p in out_dir.iterdir())  # raport bez sekcji LLM-a
    assert not list(out_dir.glob("answers.*"))


def test_eval_e2_plan(capsys, tmp_path, fake_embeddings):
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    out_dir = tmp_path / "raporty"
    with override_embeddings(fake_embeddings):
        assert main(["eval", "e2", "--with-llm", "--plan", "--out", str(out_dir)]) == 0
    assert "48" in capsys.readouterr().out
    assert not out_dir.exists() or not list(out_dir.iterdir())


# ---------------------------------------------------------------------------
# review.yaml (kontrakt F2, 14.3)
# ---------------------------------------------------------------------------


def test_eval_e2_reads_review(capsys, tmp_path, fake_embeddings):
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    out_dir = tmp_path / "raporty"
    out_dir.mkdir()
    (out_dir / "review.yaml").write_text(
        "version: 1\n"
        "reviewed_on: 2026-11-02\n"
        "intro: Wstęp przeglądu CLI.\n"
        "items:\n"
        "  - variants: [ours-ctx]\n"
        "    id: M01\n"
        "    comment: |\n"
        "      Komentarz z pliku review.\n"
        "      Druga linia.\n",
        encoding="utf-8",
    )
    with override_embeddings(fake_embeddings):
        assert main(["eval", "e2", "--out", str(out_dir)]) == 0
    md = next(p for p in out_dir.iterdir() if p.name.endswith("-e2.md")).read_text(encoding="utf-8")
    assert "Wstęp przeglądu CLI." in md
    assert "Komentarz z pliku review." in md and "Druga linia." in md
    assert "(do uzupełnienia w T2)" in md  # pozostałe pozycje bez wpisu
    assert (out_dir / "review.yaml").is_file()  # plik wejściowy nietknięty


def test_eval_e2_bad_review_exit_2(capsys, tmp_path, fake_embeddings):
    from kovalo.cli import main
    from kovalo.llm import override_chat_model, override_embeddings
    from kovalo.rag.records import Answer, Verdict
    from tests.fakes import FakeStructuredChatModel

    fake = FakeStructuredChatModel(
        results={
            Answer: Answer(answer="CFO.", citations=["POL-001"], answerable=True),
            Verdict: Verdict(correct=True, grounded=True, reason="atrapa"),
        }
    )
    out_dir = tmp_path / "raporty"
    out_dir.mkdir()
    bad = {
        "unknown_label": "  - {variants: [ours-nieznany], id: M01, comment: x}\n",
        "unknown_id": "  - {variants: [ours-ctx], id: X99, comment: x}\n",
        "schema": "  - {variants: [ours-ctx], id: M01, comment: ''}\n",
    }
    argv = ["eval", "e2", "--with-llm", "--no-cache", "--variants", "ctx", "--out", str(out_dir)]
    for name, item in bad.items():
        (out_dir / "review.yaml").write_text(
            f"version: 1\nreviewed_on: 2026-11-02\nitems:\n{item}", encoding="utf-8"
        )
        with override_embeddings(fake_embeddings), override_chat_model(fake):
            assert main(argv) == 2, name
        err = capsys.readouterr().err
        assert "review.yaml" in err, name
        assert fake.calls == [], name  # zero wywołań modelu
        assert not list(out_dir.glob("answers.*")), name
    assert not any(p.name.endswith("-e2.md") for p in out_dir.iterdir())  # nic nie zapisano
    assert fake_embeddings.calls == []  # także modele embeddingów nie zbudowane


# ---------------------------------------------------------------------------
# F3: `kovalo eval e3` i N3 (kontrakt F3, 10.6 i 14.2)
# ---------------------------------------------------------------------------

REPLAY_LINE = (
    "odtworzenie: kierowanie 13/13, pola 11/11, wniosek 8/8, zatwierdzający 11/11, "
    "pauzy 13/13, status 13/13, błędy 0"
)


def _e3_names(out_dir) -> set[str]:
    return {p.name for p in out_dir.iterdir()}


def test_eval_e3_replay_ok(capsys, tmp_path):
    from kovalo.cli import main

    out_dir = tmp_path / "mp3"
    assert main(["eval", "e3", "--out", str(out_dir)]) == 0
    names = _e3_names(out_dir)
    assert any(n.endswith("-e3.md") for n in names) and any(n.endswith("-e3.json") for n in names)
    assert "dialogues.replay.json" in names and "dialogues.live.json" not in names
    out = capsys.readouterr().out
    assert REPLAY_LINE in out
    assert str(out_dir) in out  # ścieżki raportów


def test_eval_e3_live_fake(capsys, tmp_path, settings, gold_dialogues, fake_embeddings):
    from kovalo.cli import main
    from kovalo.llm import override_chat_model, override_embeddings
    from tests.workflow_fakes import ideal_chat_model

    out_dir = tmp_path / "mp3"
    model = ideal_chat_model(gold_dialogues, settings)
    with override_chat_model(model), override_embeddings(fake_embeddings):
        assert main(["eval", "e3", "--live", "--no-cache", "--out", str(out_dir)]) == 0
    assert "dialogues.live.json" in _e3_names(out_dir)
    out = capsys.readouterr().out
    assert REPLAY_LINE in out
    assert "wywołania 14, z cache 0" in out
    assert len(model.sent) == 14


def test_eval_e3_exit_codes(capsys, tmp_path, monkeypatch, fake_embeddings):
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    monkeypatch.setenv("KOVALO_PATHS__LLM_CACHE_DIR", str(tmp_path / "llm-cache"))
    out_dir = tmp_path / "mp3"
    base = ["eval", "e3", "--out", str(out_dir)]
    with override_embeddings(fake_embeddings):
        assert main([*base, "--ids", "D9"]) == 2
        assert "D9" in capsys.readouterr().err
        assert main([*base, "--no-cache", "--cache-only"]) == 2
        capsys.readouterr()
        assert not out_dir.exists() or not list(out_dir.iterdir()), "błąd argumentów przed zapisem"

        # brak klucza przy --live (bez --cache-only): kod 2, odtworzenie zapisane
        assert main([*base, "--live", "--no-cache", "--ids", "D2"]) == 2
        assert "GOOGLE_API_KEY" in capsys.readouterr().err
        names = _e3_names(out_dir)
        assert "dialogues.replay.json" in names and "dialogues.live.json" not in names

        # --live --cache-only bez klucza i z pustym cache: wszystkie tury na żywo to błędy -> 3
        assert main([*base, "--live", "--cache-only", "--ids", "D1"]) == 3
        err = capsys.readouterr().err
        assert "GOOGLE_API_KEY" not in err, "--cache-only nie wymaga klucza"
        assert "dialogues.live.json" in _e3_names(out_dir), "raporty zapisane mimo błędów"

        assert main(["eval", "e5"]) == 2
        assert "Etap e5 jeszcze nie istnieje (F5)." in capsys.readouterr().err


def test_eval_e3_plan(capsys, tmp_path):
    from kovalo.cli import main

    out_dir = tmp_path / "mp3"
    assert main(["eval", "e3", "--live", "--plan", "--out", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert "understand 13, odpowiedzi 1, razem 14" in out
    assert not out_dir.exists() or not list(out_dir.iterdir())


def test_eval_e3_reads_review(capsys, tmp_path):
    from kovalo.cli import main

    out_dir = tmp_path / "mp3"
    out_dir.mkdir()
    (out_dir / "review.yaml").write_text(
        "version: 1\nreviewed_on: 2026-11-03\nintro: Wstęp przeglądu E3 z CLI.\nitems: []\n",
        encoding="utf-8",
    )
    assert main(["eval", "e3", "--out", str(out_dir)]) == 0
    md = next(p for p in out_dir.iterdir() if p.name.endswith("-e3.md")).read_text(encoding="utf-8")
    assert "Wstęp przeglądu E3 z CLI." in md
    assert (out_dir / "review.yaml").is_file()


def test_eval_e2_cache_only_without_key(capsys, tmp_path, monkeypatch, fake_embeddings):
    """N3: `eval e2 --with-llm --cache-only` nie sprawdza klucza (pusty cache: kod 3, nie 2)."""
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    monkeypatch.setenv("KOVALO_PATHS__LLM_CACHE_DIR", str(tmp_path / "llm-cache"))
    out_dir = tmp_path / "raporty"
    argv = ["eval", "e2", "--with-llm", "--cache-only", "--variants", "ctx", "--out", str(out_dir)]
    with override_embeddings(fake_embeddings):
        code = main(argv)
    err = capsys.readouterr().err
    assert code != 2 and "GOOGLE_API_KEY" not in err
    assert code == 3, err


# ---------------------------------------------------------------------------
# F4: `kovalo eval e4`, `kovalo ask`, `kovalo show-trace` (kontrakt F4, 11.6 i 15.2)
# ---------------------------------------------------------------------------


def _names(out_dir) -> set[str]:
    return {p.name for p in out_dir.iterdir()}


def test_eval_e4_offline_ok(capsys, tmp_path, fake_embeddings):
    from kovalo.cli import main
    from kovalo.llm import override_chat_model, override_embeddings
    from tests.workflow_fakes import ExplodingChatModel

    out_dir = tmp_path / "mp4"
    exploding = ExplodingChatModel()
    with override_chat_model(exploding), override_embeddings(fake_embeddings):
        assert main(["eval", "e4", "--out", str(out_dir)]) == 0
    assert exploding.calls == [], "zero wywołań LLM-a"
    names = _names(out_dir)
    assert any(n.endswith("-e4.md") for n in names) and any(n.endswith("-e4.json") for n in names)
    assert "agent_run.replay.json" in names and "agent_run.live.json" not in names
    assert str(out_dir) in capsys.readouterr().out  # ścieżki raportów


def test_eval_e4_exit_codes(capsys, tmp_path, monkeypatch, fake_embeddings, e4_fakes):
    from kovalo.cli import main
    from kovalo.llm import override_chat_model, override_embeddings

    monkeypatch.setenv("KOVALO_PATHS__LLM_CACHE_DIR", str(tmp_path / "llm-cache"))
    out_dir = tmp_path / "mp4"
    base = ["eval", "e4", "--out", str(out_dir)]
    with override_embeddings(fake_embeddings):
        assert main([*base, "--ids", "X9"]) == 2
        assert "X9" in capsys.readouterr().err
        assert main([*base, "--no-cache", "--cache-only"]) == 2
        capsys.readouterr()
        assert not out_dir.exists() or not list(out_dir.iterdir()), "błąd argumentów przed zapisem"

        # brak klucza przy --live (bez --cache-only): kod 2, odtworzenie zapisane
        assert main([*base, "--live", "--no-cache", "--ids", "P1"]) == 2
        assert "GOOGLE_API_KEY" in capsys.readouterr().err
        names = _names(out_dir)
        assert "agent_run.replay.json" in names and "agent_run.live.json" not in names

        # --live --cache-only bez klucza i z pustym cache: pytanie kończy się błędem -> 3
        assert main([*base, "--live", "--cache-only", "--ids", "P1"]) == 3
        err = capsys.readouterr().err
        assert "GOOGLE_API_KEY" not in err, "--cache-only nie wymaga klucza"
        assert "agent_run.live.json" in _names(out_dir), "raporty zapisane mimo błędów"

        # wyczerpany --max-calls: kod 7, raport częściowy zapisany
        model = e4_fakes()
        with override_chat_model(model):
            assert main([*base, "--live", "--no-cache", "--max-calls", "3"]) == 7
        assert model.calls == 3
        assert "agent_run.live.json" in _names(out_dir)
        capsys.readouterr()

        assert main(["eval", "e5"]) == 2
        assert "Etap e5 jeszcze nie istnieje (F5)." in capsys.readouterr().err


def test_eval_e4_cache_only_with_key_never_touches_network(
    capsys, tmp_path, monkeypatch, fake_embeddings
):
    """Ochrona limitu: nawet z kluczem `--live --cache-only` z pustym cache kończy się
    LlmCacheMiss (kod 3), a próba połączenia z siecią byłaby błędem testu."""
    import socket

    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    def no_network(*args, **kwargs):
        raise AssertionError("połączenie sieciowe w trybie --cache-only")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setenv("GOOGLE_API_KEY", "klucz-testowy-nieprawdziwy")
    monkeypatch.setenv("KOVALO_PATHS__LLM_CACHE_DIR", str(tmp_path / "llm-cache"))
    out_dir = tmp_path / "mp4"
    with override_embeddings(fake_embeddings):
        code = main(["eval", "e4", "--live", "--cache-only", "--ids", "P1", "--out", str(out_dir)])
    assert code == 3, capsys.readouterr().err
    live = next(p for p in out_dir.iterdir() if p.name.endswith("-e4.json")).read_text("utf-8")
    assert "LlmCacheMiss" in live


def test_eval_e4_live_fake_and_unlimited_calls(capsys, tmp_path, fake_embeddings, e4_fakes):
    from kovalo.cli import main
    from kovalo.llm import override_chat_model, override_embeddings

    out_dir = tmp_path / "mp4"
    model = e4_fakes()
    with override_chat_model(model), override_embeddings(fake_embeddings):
        code = main(
            ["eval", "e4", "--live", "--no-cache", "--max-calls", "0", "--out", str(out_dir)]
        )
    assert code == 0, capsys.readouterr().err
    assert len(model.script) == model.calls, "0 = bez limitu: skrypt wykonany do końca"
    assert "agent_run.live.json" in _names(out_dir)


def test_eval_e4_default_max_calls_is_20(capsys, tmp_path, fake_embeddings, e4_fakes):
    """Bez --max-calls pełny przebieg (22 tury) staje po 20 wywołaniach sieciowych (kod 7)."""
    from kovalo.cli import main
    from kovalo.llm import override_chat_model, override_embeddings

    model = e4_fakes()
    assert len(model.script) > 20
    with override_chat_model(model), override_embeddings(fake_embeddings):
        code = main(["eval", "e4", "--live", "--no-cache", "--out", str(tmp_path / "mp4")])
    assert code == 7, capsys.readouterr().err
    assert model.calls == 20


def test_eval_e4_reference_mismatch_exit_4(capsys, tmp_path, monkeypatch, fake_embeddings):
    """Przeliczenie ≠ kurs to błąd portu (kod 4), nie modelu."""
    from kovalo.cli import main
    from kovalo.eval.trace import reference
    from kovalo.llm import override_embeddings

    def mismatch(*args, **kwargs):
        raise reference.E4ReferenceMismatch("T1: tool_coverage: 1.0 != 0.6667")

    monkeypatch.setattr(reference, "rescore_course", mismatch)
    monkeypatch.setattr("kovalo.eval.trace.e4.rescore_course", mismatch, raising=False)
    with override_embeddings(fake_embeddings):
        assert main(["eval", "e4", "--out", str(tmp_path / "mp4")]) == 4
    assert "T1" in capsys.readouterr().err


def test_eval_e4_plan(capsys, tmp_path):
    from kovalo.cli import main

    out_dir = tmp_path / "mp4"
    assert main(["eval", "e4", "--live", "--plan", "--out", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert "szacunek 22, górna granica 108, limit 20" in out
    assert not out_dir.exists() or not list(out_dir.iterdir())


def test_eval_e4_plan_variants(capsys, tmp_path):
    """`--plan` odzwierciedla --ids i --max-calls, nie wymaga klucza i nie zapisuje;
    bez --live: 0 wywołań; ujemny limit to kod 2."""
    from kovalo.cli import main

    out_dir = tmp_path / "mp4"
    base = ["eval", "e4", "--plan", "--out", str(out_dir)]
    assert main([*base, "--live", "--ids", "T1", "--max-calls", "5"]) == 0
    assert "szacunek 3, górna granica 12, limit 5" in capsys.readouterr().out
    assert main([*base, "--live", "--max-calls", "0"]) == 0
    assert "limit brak" in capsys.readouterr().out
    assert main([*base]) == 0
    assert "0 wywołań" in capsys.readouterr().out
    assert main([*base, "--live", "--max-calls", "-1"]) == 2
    assert main([*base, "--live", "--ids", "X9"]) == 2
    assert not out_dir.exists() or not list(out_dir.iterdir())


def test_eval_e4_reads_review(capsys, tmp_path, fake_embeddings):
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    out_dir = tmp_path / "mp4"
    out_dir.mkdir()
    (out_dir / "review.yaml").write_text(
        "version: 1\nreviewed_on: 2026-11-03\nintro: Wstęp przeglądu E4 z CLI.\nitems: []\n",
        encoding="utf-8",
    )
    with override_embeddings(fake_embeddings):
        assert main(["eval", "e4", "--out", str(out_dir)]) == 0
    md = next(p for p in out_dir.iterdir() if p.name.endswith("-e4.md")).read_text(encoding="utf-8")
    assert "Wstęp przeglądu E4 z CLI." in md
    (out_dir / "review.yaml").write_text("version: 7\nitems: zły\n", encoding="utf-8")
    with override_embeddings(fake_embeddings):
        assert main(["eval", "e4", "--out", str(out_dir)]) == 2, "zły review.yaml"


def test_ask_and_show_trace(capsys, tmp_path, monkeypatch, fake_embeddings, session_settings):
    from kovalo.cli import main
    from kovalo.llm import override_chat_model, override_embeddings
    from tests.agent_fakes import ScriptedModel, p1_script

    monkeypatch.setenv("KOVALO_PATHS__LLM_CACHE_DIR", str(tmp_path / "llm-cache"))
    question = "Ile wynosi kara, którą Mleczna Dolina jest winna Kovalo za lipiec 2026 r.?"
    model = ScriptedModel(script=p1_script())
    with override_chat_model(model), override_embeddings(fake_embeddings):
        assert main(["ask", question]) == 0
    out = capsys.readouterr().out
    assert "729 600" in out or "7296,00" in out, "odpowiedź agenta"
    assert "calculate_otif_penalty(" in out and "odpowiedź:" in out, "ścieżka dowodowa"
    assert "729600" in out, "wynik narzędzia na prawdziwych danych"
    assert len(model.bound) == 7, "domyślnie 7 narzędzi (tools.variant: ours)"

    model = ScriptedModel(script=p1_script())
    with override_chat_model(model), override_embeddings(fake_embeddings):
        assert main(["ask", question, "--tools", "course"]) == 0
    assert len(model.bound) == 6
    capsys.readouterr()

    # show-trace: przebieg kursu bez API
    assert main(["show-trace", "P1", "--run", "course"]) == 0
    shown = capsys.readouterr().out
    assert "get_contract_terms(supplier='Mleczna Dolina')" in shown
    assert "calculate_otif_penalty(otif_pct=94.6" in shown and "penalty_grosze" in shown
    assert main(["show-trace", "P1", "--run", "course-supervisor"]) == 0
    assert "ask_data_analyst" in capsys.readouterr().out
    assert main(["show-trace", "ZZ", "--run", "course"]) == 2
    assert "ZZ" in capsys.readouterr().err
    assert main(["show-trace", "P1", "--run", "live", "--out", str(tmp_path / "pusto")]) == 2
    assert capsys.readouterr().err.strip(), "brak pliku: komunikat, kod 2"


def test_ask_errors(capsys, tmp_path, monkeypatch, fake_embeddings):
    from kovalo.cli import main
    from kovalo.llm import override_chat_model, override_embeddings
    from tests.agent_fakes import ScriptedModel

    monkeypatch.setenv("KOVALO_PATHS__LLM_CACHE_DIR", str(tmp_path / "llm-cache"))
    with override_embeddings(fake_embeddings):
        assert main(["ask", "Ile wynosi kara?"]) == 2, "brak klucza"
        err = capsys.readouterr().err
        assert "GOOGLE_API_KEY" in err and "Traceback" not in err
        assert len(err.strip().splitlines()) == 1, "jedna linia"
        assert main(["ask", "   "]) == 2
        capsys.readouterr()
        assert main(["ask", "x", "--no-cache", "--cache-only"]) == 2
        capsys.readouterr()
        with override_chat_model(ScriptedModel(script=[RuntimeError("provider unavailable")])):
            assert main(["ask", "Ile wynosi kara?"]) == 3, "agent zakończył błędem"
        captured = capsys.readouterr()
        assert "provider unavailable" in captured.err, "P7: komunikat błędu na stderr"
        assert captured.out.strip(), "P7: ścieżka dowodowa także przy kodzie 3, na stdout"


def test_show_trace_reads_replay_report(capsys, tmp_path, fake_embeddings):
    """`show-trace --run replay` czyta agent_run.replay.json z --out (format `save_traces` kursu)."""  # noqa: E501
    from kovalo.cli import main
    from kovalo.llm import override_embeddings

    out_dir = tmp_path / "mp4"
    with override_embeddings(fake_embeddings):
        assert main(["eval", "e4", "--out", str(out_dir)]) == 0
    capsys.readouterr()
    assert main(["show-trace", "T1", "--run", "replay", "--out", str(out_dir)]) == 0
    assert "search_documents(" in capsys.readouterr().out
