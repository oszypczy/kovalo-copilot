"""Embeddingi: adapter fastembed, cache, tryb kursu (FR-1.4, krok 4 kursu; kontrakt F1, 5.3).

Jedyny moduł, który importuje fastembed, i to leniwie (wewnątrz funkcji), żeby import
`kovalo.knowledge` nie ładował onnxruntime. Decyzje: ADR-0003, ADR-0007.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.embeddings import Embeddings

from kovalo.config import ConfigError, EmbeddingModelSpec

__all__ = [
    "CountingEmbeddings",
    "CourseCacheMissError",
    "CourseCacheOnlyEmbeddings",
    "FastEmbedEmbeddings",
    "build_cached_embeddings",
    "build_course_cache_embeddings",
    "cache_namespace",
    "missing_from_course_cache",
    "model_size_gb",
]

_KEY_ENCODER: Literal["sha256"] = "sha256"  # jak w cache kursu


def _to_floats(vector: Any) -> list[float]:
    """Wektor numpy → lista floatów Pythona (CacheBackedEmbeddings serializuje do JSON-a)."""
    return [float(x) for x in vector]


class FastEmbedEmbeddings(Embeddings):
    """Adapter fastembed.TextEmbedding z prefiksami pytań i dokumentów z configu."""

    def __init__(
        self,
        spec: EmbeddingModelSpec,
        *,
        models_dir: Path,
        batch_size: int = 64,
        threads: int | None = None,
        local_files_only: bool = False,
    ) -> None:
        from fastembed import TextEmbedding

        self.spec = spec
        self.batch_size = batch_size
        try:
            size = TextEmbedding.get_embedding_size(spec.model)
        except ValueError as exc:
            raise ConfigError(
                f"Model embeddingów {spec.model} nie jest obsługiwany przez fastembed "
                "(TextEmbedding). Popraw embeddings.local.model albo embeddings.candidates."
            ) from exc
        if size != spec.dimensions:
            raise ConfigError(
                f"Model {spec.model} zwraca wektory o wymiarze {size}, a config podaje "
                f"{spec.dimensions}. Popraw wymiar w config.yaml."
            )
        try:
            self._model = TextEmbedding(
                spec.model,
                cache_dir=str(models_dir),
                threads=threads,
                local_files_only=local_files_only,
            )
        except ConfigError:
            raise
        except Exception as exc:
            if local_files_only:
                raise ConfigError(
                    f"Brak plików modelu {spec.model} w {models_dir} (local_files_only=true): "
                    "uruchom docker compose build albo wyłącz local_files_only "
                    "(KOVALO_EMBEDDINGS__LOCAL__LOCAL_FILES_ONLY=false)."
                ) from exc
            raise ConfigError(
                f"Nie udało się załadować modelu {spec.model} do {models_dir}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prefixed = [self.spec.passage_prefix + t for t in texts]
        return [
            _to_floats(v) for v in self._model.passage_embed(prefixed, batch_size=self.batch_size)
        ]

    def embed_query(self, text: str) -> list[float]:
        (vector,) = list(self._model.query_embed(self.spec.query_prefix + text))
        return _to_floats(vector)


class CountingEmbeddings(Embeddings):
    """Opakowanie liczące teksty przekazane do modelu (te, których nie było w cache)."""

    def __init__(self, inner: Embeddings) -> None:
        self.inner = inner
        self.documents_embedded = 0
        self.queries_embedded = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.inner.embed_documents(texts)
        self.documents_embedded += len(texts)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vector = self.inner.embed_query(text)
        self.queries_embedded += 1
        return vector


class CourseCacheMissError(ConfigError):
    """Tekstu nie ma w data/embeddings/cache (tryb openai_cache)."""


class CourseCacheOnlyEmbeddings(Embeddings):
    """Odpowiednik helpers.embedding_cache.CacheOnlyEmbeddings: nigdy nie liczy wektora."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        sample = repr(texts[0][:60]) if texts else "''"
        raise CourseCacheMissError(
            f"Liczba tekstów spoza cache embeddingów kursu: {len(texts)}, np. {sample}. "
            "Tryb openai_cache działa tylko na tekstach z cache; brak chunku zwykle oznacza, "
            "że chunki różnią się od ustalonego podziału kursu (krok 3)."
        )

    def embed_query(self, text: str) -> list[float]:
        raise CourseCacheMissError(
            f"Tego pytania nie ma w cache embeddingów kursu: {text[:60]!r}. Tryb openai_cache "
            "obsługuje tylko pytania testowe; dla nowych pytań ustaw embeddings.provider: "
            "fastembed."
        )


def cache_namespace(model: str) -> str:
    """Nazwa modelu jako bezpieczny klucz LocalFileStore: znaki spoza [A-Za-z0-9_.-] → '_'."""
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", model)


def build_cached_embeddings(
    underlying: Embeddings,
    *,
    cache_dir: Path,
    model: str,
    batch_size: int | None = None,
) -> CacheBackedEmbeddings:
    """Cache fastembed: osobne magazyny dokumentów i pytań (<cache_dir>/<ns>/passage|query)."""
    ns = cache_namespace(model)
    root = Path(cache_dir) / ns
    return CacheBackedEmbeddings.from_bytes_store(
        underlying,
        LocalFileStore(root / "passage"),
        namespace=ns,
        key_encoder=_KEY_ENCODER,
        batch_size=batch_size,
        query_embedding_cache=LocalFileStore(root / "query"),
    )


def build_course_cache_embeddings(
    cache_dir: Path, *, model: str, underlying: Embeddings | None = None
) -> CacheBackedEmbeddings:
    """Tryb openai_cache zgodny 1:1 z krokiem 4 kursu (jeden magazyn, namespace = model).

    Domyślnie pod spodem CourseCacheOnlyEmbeddings: brak w cache kończy się wyjątkiem
    przed zapisem, więc data/ nigdy się nie zmienia.
    """
    return CacheBackedEmbeddings.from_bytes_store(
        underlying if underlying is not None else CourseCacheOnlyEmbeddings(),
        LocalFileStore(Path(cache_dir)),
        namespace=model,
        key_encoder=_KEY_ENCODER,
        query_embedding_cache=True,
    )


def missing_from_course_cache(cache_dir: Path, model: str, texts: Sequence[str]) -> list[int]:
    """Pozycje tekstów bez wektora w cache kursu (port helpers missing_from_cache)."""
    store = LocalFileStore(Path(cache_dir))
    keys = [model + hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    return [i for i, vector in enumerate(store.mget(keys)) if vector is None]


def model_size_gb(model: str) -> float | None:
    """size_in_GB z TextEmbedding.list_supported_models(); None dla nieznanego modelu."""
    from fastembed import TextEmbedding

    for description in TextEmbedding.list_supported_models():
        if str(description.get("model", "")).lower() == model.lower():
            size = description.get("size_in_GB")
            return float(size) if size is not None else None
    return None
