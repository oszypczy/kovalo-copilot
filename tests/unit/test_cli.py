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

    assert main(["eval", "e2"]) == 2
    assert "Etap e2 jeszcze nie istnieje (F2)." in capsys.readouterr().err
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
