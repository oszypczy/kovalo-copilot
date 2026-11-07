"""Cache odpowiedzi LLM-a na dysku dla ewaluacji (ADR-0008 pkt 6; kontrakt F2, 8.4).

`JsonFileLlmCache` to `langchain_core.caches.BaseCache`: model sprawdza go w
`BaseChatModel._generate_with_cache` przed rate limiterem, więc trafienie nie zużywa limitu
i zwraca oryginalne `usage_metadata`. Jeden plik JSON na wpis, zapis atomowy.
Wyjątki dostawcy nie trafiają do cache (BaseChatModel zapisuje tylko udane odpowiedzi).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.caches import BaseCache
from langchain_core.load import Serializable, dumps, loads
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, Generation

from kovalo.llm import CacheMissError

__all__ = ["CacheMode", "JsonFileLlmCache", "LlmCacheMiss"]

CacheMode = Literal["read_write", "read_only", "strict"]

# Klasy, które wolno odtworzyć z pliku cache (bez dowolnych obiektów LangChaina).
_ALLOWED: list[type[Serializable]] = [
    Generation,
    ChatGeneration,
    ChatGenerationChunk,
    AIMessage,
    AIMessageChunk,
]


class LlmCacheMiss(CacheMissError):
    """Tryb strict: brak odpowiedzi w cache.

    Nie jest w FALLBACK_EXCEPTIONS, więc na żywo nie przełącza modelu. W trybie offline
    (`--cache-only`, `llm.fallback_exceptions(offline=True)`) przełącza na następny model,
    który sprawdza swój wpis: tura odebrana na żywo przez fallback (model główny zawiódł
    błędem dostępności) ma w cache tylko wpis fallbacku. Żaden model nie trafi → wyjątek
    idzie wyżej (ADR-0008 pkt 6, adaptacja F5).
    """


class JsonFileLlmCache(BaseCache):
    """Cache (prompt, llm_string) → generacje w plikach <dir>/<key[:2]>/<key>.json.

    Tryby: read_write (domyślny), read_only (tylko odczyt), strict (odczyt, brak → LlmCacheMiss).
    Liczniki `hits` i `misses` służą raportom (wątkowo bezpieczne).
    """

    def __init__(self, directory: Path, *, mode: CacheMode = "read_write") -> None:
        if mode not in ("read_write", "read_only", "strict"):
            raise ValueError(f"Nieznany tryb cache LLM-a: {mode!r}")
        self.directory = Path(directory)
        self.mode: CacheMode = mode
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        return hashlib.sha256(f"{prompt}\x1f{llm_string}".encode()).hexdigest()

    def _path(self, prompt: str, llm_string: str) -> Path:
        key = self._key(prompt, llm_string)
        return self.directory / key[:2] / f"{key}.json"

    def lookup(self, prompt: str, llm_string: str) -> list[Generation] | None:
        path = self._path(prompt, llm_string)
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # `loads` jest oznaczone jako beta
                generations = loads(json.dumps(data["generations"]), allowed_objects=_ALLOWED)
            with self._lock:
                self.hits += 1
            return list(generations)
        with self._lock:
            self.misses += 1
        if self.mode == "strict":
            raise LlmCacheMiss(
                f"Brak odpowiedzi w cache LLM-a ({self.directory}) w trybie --cache-only."
            )
        return None

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        if self.mode != "read_write":
            return
        path = self._path(prompt, llm_string)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "llm_string": llm_string,
            "generations": json.loads(dumps(list(return_val))),
            "created_at": datetime.now(UTC).isoformat(),
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def clear(self, **kwargs: Any) -> None:
        """Usuwa wszystkie pliki cache (testy)."""
        if self.directory.exists():
            shutil.rmtree(self.directory)

    def stats(self) -> dict[str, int]:
        """{'hits', 'misses'} do raportu."""
        with self._lock:
            return {"hits": self.hits, "misses": self.misses}
