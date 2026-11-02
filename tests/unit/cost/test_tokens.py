"""Liczenie tokenów (FR-1.2; port testów kroku 2 kursu). Wartości: kontrakt F1, sekcja 8.1."""

from __future__ import annotations

import pytest

O200K = "o200k_base"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("Hello, world!", 4),
        ("Kovalo Distribution", 3),
        (
            "OTIF = zamówienia dostarczone na czas i w komplecie "
            "/ wszystkie dostarczone zamówienia",
            25,
        ),
        ("Witaj, świecie!", 5),
    ],
)
def test_count_tokens(text: str, expected: int):
    from kovalo.cost.tokens import count_tokens

    assert count_tokens(text, O200K) == expected


def test_count_tokens_uses_given_encoding():
    from kovalo.cost.tokens import count_tokens

    assert count_tokens("Kovalo Distribution", "cl100k_base") == 4
    assert count_tokens("Kovalo Distribution", O200K) == 3


@pytest.mark.parametrize("text", ["", "   "])
def test_tokens_per_word_empty_raises(text: str):
    from kovalo.cost.tokens import tokens_per_word

    with pytest.raises(ValueError):
        tokens_per_word(text, O200K)


def test_tokens_per_word_is_not_rounded():
    from kovalo.cost.tokens import count_tokens, tokens_per_word

    text = "Kovalo Distribution sp. z o.o."
    assert tokens_per_word(text, O200K) == count_tokens(text, O200K) / len(text.split())
