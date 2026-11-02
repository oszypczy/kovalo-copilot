"""Wspólne fixture'y testów F0 (kontrakt: docs/contracts/f0.md, sekcja 8).

Zasady: zero sieci i zero prawdziwego LLM-a; testy nie czytają prawdziwego `.env`
ani zmiennych środowiskowych użytkownika (autouse `isolated_env`).
"""

from __future__ import annotations

import importlib
import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import Field

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_CONFIG = REPO_ROOT / "config.yaml"
COURSE_CONFIG = REPO_ROOT / "reference" / "course" / "config.yaml"

# Klucz z prawdziwego środowiska procesu (nie z .env), tylko dla testu live.
_LIVE_GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

_SECRET_ENV = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "LANGSMITH_API_KEY")
_OTHER_ENV = (
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
)


def _reset_kovalo_state() -> None:
    """Czyści cache get_settings i rejestr limiterów (jeśli moduły już istnieją)."""
    for mod_name, attrs in (
        ("kovalo.config", ("get_settings",)),
        ("kovalo.llm", ("reset_rate_limiters",)),
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            if exc.name != mod_name:
                raise
            continue  # moduł jeszcze nie istnieje: testy i tak padną na własnym imporcie
        for attr in attrs:
            obj = getattr(mod, attr)
            if attr == "get_settings":
                obj.cache_clear()
            else:
                obj()


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Usuwa klucze i KOVALO_*; cwd w pustym katalogu, żeby `.env` repo nie był czytany."""
    for name in (*_SECRET_ENV, *_OTHER_ENV):
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith("KOVALO_"):
            monkeypatch.delenv(name, raising=False)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)  # `.env` jest czytany z katalogu bieżącego (kontrakt 4.3)
    _reset_kovalo_state()
    yield
    _reset_kovalo_state()


@pytest.fixture
def live_google_key() -> str | None:
    return _LIVE_GOOGLE_KEY


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def settings():  # type: ignore[no-untyped-def]
    from kovalo.config import load_settings

    return load_settings(google_api_key="test-key")


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Kopia repozytoryjnego config.yaml w tmp_path (do modyfikacji)."""
    target = tmp_path / "config.yaml"
    shutil.copy(REPO_CONFIG, target)
    return target


@pytest.fixture
def patch_config() -> Callable[..., None]:
    """patch_config(path, "llm.purposes", {...}) ustawia wartość pod kropkowaną ścieżką."""

    def _patch(path: Path, dotted: str, value: Any) -> None:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        node = data
        *parents, last = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[last] = value
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    return _patch


def make_fake(
    text: str = "Cześć", model_name: str = "fake", *, usage: bool = True
) -> BaseChatModel:
    msg = AIMessage(
        content=text,
        usage_metadata=(
            {"input_tokens": 9, "output_tokens": 21, "total_tokens": 30} if usage else None
        ),
        response_metadata={"model_name": model_name},
    )
    return GenericFakeChatModel(messages=iter([msg]))


@pytest.fixture
def fake_chat_model() -> BaseChatModel:
    return make_fake()


class FailingChatModel(BaseChatModel):
    """Atrapa, której `_generate` zawsze rzuca podany wyjątek (liczy wywołania)."""

    exc: Any
    calls: list[int] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "failing-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(1)
        raise self.exc


@pytest.fixture
def failing_chat_model() -> Callable[[BaseException], FailingChatModel]:
    def _factory(exc: BaseException) -> FailingChatModel:
        return FailingChatModel(exc=exc, calls=[])

    return _factory
