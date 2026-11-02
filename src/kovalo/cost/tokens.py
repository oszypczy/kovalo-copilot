"""Liczenie tokenów (FR-1.2, krok 2 kursu). Jedyny moduł, który importuje tiktoken."""

from __future__ import annotations

import functools

import tiktoken

__all__ = ["count_tokens", "get_encoding", "tokens_per_word"]


@functools.cache
def get_encoding(name: str) -> tiktoken.Encoding:
    """Tokenizer o podanej nazwie (pierwsze użycie może pobrać plik do TIKTOKEN_CACHE_DIR)."""
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding: str) -> int:
    """Liczba tokenów tekstu. '' → 0. count_tokens('Witaj, świecie!', 'o200k_base') == 5."""
    if not text:
        return 0
    return len(get_encoding(encoding).encode(text, disallowed_special=()))


def tokens_per_word(text: str, encoding: str) -> float:
    """count_tokens / len(text.split()), bez zaokrąglania. Tekst bez słów → ValueError."""
    words = len(text.split())
    if words == 0:
        raise ValueError("Tekst nie zawiera słów: nie da się policzyć tokenów na słowo.")
    return count_tokens(text, encoding) / words
