"""Wspólne fixture'y testów F0 (kontrakt: docs/contracts/f0.md, sekcja 8).

Zasady: zero sieci i zero prawdziwego LLM-a; testy nie czytają prawdziwego `.env`
ani zmiennych środowiskowych użytkownika (autouse `isolated_env`).
"""

from __future__ import annotations

import csv
import importlib
import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
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
        ("kovalo.llm", ("reset_rate_limiters", "reset_embeddings")),
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            if exc.name != mod_name:
                raise
            continue  # moduł jeszcze nie istnieje: testy i tak padną na własnym imporcie
        for attr in attrs:
            obj = getattr(mod, attr, None)
            if obj is None:
                continue  # np. reset_embeddings przed A4; jego istnienie pilnuje test_llm
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


# ---------------------------------------------------------------------------
# F1 (kontrakt: docs/contracts/f1.md, sekcja 10.1)
# ---------------------------------------------------------------------------

CATALOG_COLUMNS = [
    "doc_id",
    "file",
    "title",
    "doc_type",
    "department",
    "owner",
    "effective_date",
    "language",
]


@pytest.fixture(scope="session")
def session_settings():  # type: ignore[no-untyped-def]
    """Ustawienia z repozytoryjnego config.yaml (sesyjne: tylko do odczytu)."""
    from kovalo.config import load_settings

    return load_settings(google_api_key="test-key")


@pytest.fixture(scope="session")
def documents(session_settings):  # type: ignore[no-untyped-def]
    """Prawdziwy korpus (18 dokumentów)."""
    from kovalo.knowledge.corpus import load_corpus

    return load_corpus(session_settings)


@pytest.fixture(scope="session")
def chunks(documents, session_settings):  # type: ignore[no-untyped-def]
    """Chunki MP1 prawdziwego korpusu (47)."""
    from kovalo.knowledge.chunking import chunk_documents

    return chunk_documents(documents, session_settings.chunking)


@pytest.fixture
def toy_corpus(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Mały korpus w tmp_path: zwraca (add, corpus_dir, catalog_path)."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    catalog = corpus_dir / "catalog.csv"
    with catalog.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow(CATALOG_COLUMNS)

    def add(
        file: str, text: str = "Przykładowy tekst.", in_catalog: bool = True, **fields: str
    ) -> None:
        (corpus_dir / file).write_text(text, encoding="utf-8")
        if in_catalog:
            row = {
                "doc_id": "TST-001",
                "file": file,
                "title": "Dokument testowy",
                "doc_type": "policy",
                "department": "Testy",
                "owner": "Kierownik testów",
                "effective_date": "2026-01-01",
                "language": "en",
            }
            row.update(fields)
            with catalog.open("a", encoding="utf-8", newline="") as fh:
                csv.writer(fh).writerow([row[c] for c in CATALOG_COLUMNS])

    return add, corpus_dir, catalog


class CountingFakeEmbeddings(DeterministicFakeEmbedding):
    """Ten sam tekst -> ten sam wektor; zapisuje każdy tekst zamieniany na wektor."""

    calls: list[str] = []  # noqa: RUF012

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return super().embed_query(text)


class ForbiddenEmbeddings(Embeddings):
    """Każde wywołanie to AssertionError: dowodzi, że wektor pochodzi z cache."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError(f"{len(texts)} tekstów spoza cache, np. {texts[0][:60]!r}")

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError(f"Pytania nie ma w cache: {text[:60]!r}")


@pytest.fixture
def fake_embeddings() -> CountingFakeEmbeddings:
    return CountingFakeEmbeddings(size=16, calls=[])


@pytest.fixture
def forbidden_embeddings() -> ForbiddenEmbeddings:
    return ForbiddenEmbeddings()


# Modele znane atrapie fastembed: nazwa -> (wymiar, rozmiar GB)
FAKE_FASTEMBED_MODELS: dict[str, tuple[int, float]] = {
    "intfloat/multilingual-e5-large": (1024, 2.24),
    "jinaai/jina-embeddings-v3": (1024, 2.29),
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": (768, 1.11),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": (384, 0.22),
}


class FakeTextEmbeddingState:
    """Co atrapa `fastembed.TextEmbedding` dostała."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []  # ("passage" | "query", tekst)
        self.inits: list[dict[str, Any]] = []


@pytest.fixture
def fake_text_embedding(monkeypatch: pytest.MonkeyPatch) -> FakeTextEmbeddingState:
    """Podmienia fastembed.TextEmbedding na atrapę (bez pobierania modeli i onnxruntime)."""
    import fastembed

    state = FakeTextEmbeddingState()

    def _vector(text: str, dim: int) -> np.ndarray:
        seed = int.from_bytes(text.encode("utf-8")[:8].ljust(8, b"\0"), "little") % (2**32)
        return np.random.default_rng(seed).normal(size=dim).astype(np.float32)

    class FakeTextEmbedding:
        def __init__(
            self,
            model_name: str,
            cache_dir: str | None = None,
            threads: int | None = None,
            local_files_only: bool = False,
            **kwargs: Any,
        ) -> None:
            if model_name not in FAKE_FASTEMBED_MODELS:
                raise ValueError(f"Model {model_name} is not supported in TextEmbedding.")
            self.model_name = model_name
            state.inits.append(
                {
                    "model_name": model_name,
                    "cache_dir": cache_dir,
                    "threads": threads,
                    "local_files_only": local_files_only,
                }
            )

        @classmethod
        def get_embedding_size(cls, model_name: str) -> int:
            if model_name not in FAKE_FASTEMBED_MODELS:
                raise ValueError(f"Model {model_name} is not supported in TextEmbedding.")
            return FAKE_FASTEMBED_MODELS[model_name][0]

        @classmethod
        def list_supported_models(cls) -> list[dict[str, Any]]:
            return [
                {"model": name, "dim": dim, "size_in_GB": size}
                for name, (dim, size) in FAKE_FASTEMBED_MODELS.items()
            ]

        def passage_embed(self, texts: Any, batch_size: int = 256, **kwargs: Any) -> Any:
            dim = FAKE_FASTEMBED_MODELS[self.model_name][0]
            for text in texts:
                state.calls.append(("passage", text))
                yield _vector(text, dim)

        def query_embed(self, query: Any, **kwargs: Any) -> Any:
            dim = FAKE_FASTEMBED_MODELS[self.model_name][0]
            for text in [query] if isinstance(query, str) else query:
                state.calls.append(("query", text))
                yield _vector("q:" + text, dim)

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)
    return state


@pytest.fixture
def local_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    """Ustawienia z cache embeddingów i katalogiem modeli w tmp_path (nic nie ląduje w repo)."""
    from kovalo.config import load_settings

    monkeypatch.setenv("KOVALO_EMBEDDINGS__LOCAL__CACHE_DIR", str(tmp_path / "emb-cache"))
    monkeypatch.setenv("KOVALO_EMBEDDINGS__LOCAL__MODELS_DIR", str(tmp_path / "models"))
    return load_settings(google_api_key="test-key")


@pytest.fixture
def reference_settings(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Tryb openai_cache (cache kursu z data/embeddings)."""
    from kovalo.config import load_settings

    monkeypatch.setenv("KOVALO_EMBEDDINGS__PROVIDER", "openai_cache")
    return load_settings(google_api_key="test-key")


# ---------------------------------------------------------------------------
# F2 (kontrakt: docs/contracts/f2.md, sekcja 11.1). Importy kodu produkcyjnego wewnątrz fixture'ów.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ctx_chunks(documents, session_settings):  # type: ignore[no-untyped-def]
    """Chunki z kontekstem prawdziwego korpusu (87)."""
    from kovalo.knowledge.chunking import build_contextual_chunks

    return build_contextual_chunks(documents, session_settings.chunking)


@pytest.fixture(scope="session")
def gold(session_settings):  # type: ignore[no-untyped-def]
    """12 pytań MP2 z faktami wzorcowymi."""
    from kovalo.eval.datasets import load_gold

    return load_gold(session_settings.paths.mini_project_2_questions)


@pytest.fixture
def toy_store(fake_embeddings):  # type: ignore[no-untyped-def]
    from kovalo.knowledge.retrieval import build_vector_store
    from tests.fakes import TOY_CHUNKS

    return build_vector_store(TOY_CHUNKS, fake_embeddings)


@pytest.fixture
def toy_retriever(toy_store):  # type: ignore[no-untyped-def]
    from kovalo.knowledge.retrievers import VectorRetriever

    return VectorRetriever(toy_store)


@pytest.fixture
def llm_cache(tmp_path: Path):  # type: ignore[no-untyped-def]
    from kovalo.eval.llm_cache import JsonFileLlmCache

    return JsonFileLlmCache(tmp_path / "llm")
