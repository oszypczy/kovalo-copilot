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
