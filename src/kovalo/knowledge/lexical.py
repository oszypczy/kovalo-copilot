"""Wyszukiwanie leksykalne BM25 (FR-2.5; kontrakt F2, sekcja 5.2; ADR-0008 pkt 2).

Jedyny moduł, który importuje rank_bm25. Korpus podajemy już ztokenizowany (bez argumentu
`tokenizer`), więc BM25Okapi działa w jednym procesie, bez multiprocessing.Pool.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from kovalo.config import Bm25Config
from kovalo.knowledge.retrieval import ScoredChunk

__all__ = ["TOKEN_PATTERN", "BM25Index", "tokenize"]

# Myślnik wewnątrz słowa zostaje, więc identyfikatory (SUP-114, CON-101) są jednym tokenem.
TOKEN_PATTERN = re.compile(r"\w+(?:-\w+)*")


def tokenize(text: str) -> list[str]:
    """Tokeny po casefold, bez stemmingu i bez listy stop-słów.

    'Co SUP-114 mówi o karach?' → ['co', 'sup-114', 'mówi', 'o', 'karach'].
    """
    return TOKEN_PATTERN.findall(text.casefold())


class BM25Index:
    """Indeks BM25 Okapi nad chunkami (tekst = page_content razem z nagłówkiem kontekstu)."""

    def __init__(self, chunks: Sequence[Document], params: Bm25Config) -> None:
        if not chunks:
            raise ValueError("Indeks BM25 wymaga co najmniej jednego chunku.")
        ids = [str(c.metadata["chunk_id"]) for c in chunks]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"Powtórzone chunk_id w indeksie BM25: {', '.join(duplicates)}")
        self._chunks = list(chunks)
        self._ids = ids
        self._bm25 = BM25Okapi(
            [tokenize(c.page_content) for c in self._chunks],
            k1=params.k1,
            b=params.b,
            epsilon=params.epsilon,
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, question: str, k: int) -> list[ScoredChunk]:
        """Najwyżej k par (chunk, wynik BM25) z wynikiem > 0, od najlepszej.

        Remis → rosnąco po chunk_id. k < 1 → ValueError. Pytanie bez tokenów → [].
        """
        if k < 1:
            raise ValueError(f"k musi być ≥ 1, jest {k}")
        tokens = tokenize(question)
        if not tokens:
            return []
        scores = [float(s) for s in self._bm25.get_scores(tokens)]
        order = sorted(
            (i for i, score in enumerate(scores) if score > 0),
            key=lambda i: (-scores[i], self._ids[i]),
        )
        return [(self._chunks[i], scores[i]) for i in order[:k]]
