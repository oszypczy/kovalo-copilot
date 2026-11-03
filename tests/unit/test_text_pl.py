"""kovalo.text_pl: liczby i odpowiedzi człowieka (kontrakt F3, 5.1).

Port `mini_project_3/tests/test_provided_polish_input.py` kursu (4 testy bez ścieżki czatu;
ścieżka czatu: tests/unit/workflow/test_copilot_graph.py).
"""

from __future__ import annotations

from decimal import Decimal

import pytest


@pytest.mark.parametrize(
    ("text", "value"),
    [("7,5", 7.5), ("7.5", 7.5), ("7", 7.0), ("-2", -2.0), (" 7,5 % ", 7.5), ("1,500", 1.5)],
)
def test_a_typed_margin_in_either_notation(text, value):
    from kovalo.text_pl import parse_number

    assert parse_number(text) == value


@pytest.mark.parametrize("text", ["siedem", "1,5.0", "7 5", "inf", "nan", ""])
def test_anything_else_is_not_a_number(text):
    from kovalo.text_pl import parse_number

    assert parse_number(text) is None


def test_polish_and_english_commands_map_to_the_workflow_values():
    from kovalo.text_pl import human_answer

    answers = [
        human_answer("approval", t) for t in ("Zatwierdź", "odrzuć", "anuluj", "approved", "cancel")
    ]
    assert answers == ["approved", "rejected", "cancel", "approved", "cancel"]
    assert human_answer("clarification", "anuluj") == "cancel"
    assert human_answer("clarification", " 7,5 ") == 7.5
    assert human_answer("clarification", "siedem") == "siedem"
    assert human_answer("approval", "może") == "może", "nieznana odpowiedź przechodzi bez zmian"
    assert human_answer("approval", "7") == "7", "w akceptacji liczba nie jest decyzją"


def test_numbers_for_people_use_the_polish_notation():
    from kovalo.text_pl import number

    assert [number(x) for x in (5.01, 12.0, 18240000, 7296.0, -3.25)] == [
        "5,01",
        "12",
        "18 240 000",
        "7296",
        "-3,25",
    ]
    assert number(Decimal("5.01")) == "5,01" and number(Decimal(12)) == "12"


def test_approval_words_table():
    from kovalo.text_pl import APPROVAL_WORDS

    assert APPROVAL_WORDS["zatwierdź"] == "approved" and APPROVAL_WORDS["odrzuć"] == "rejected"
    assert APPROVAL_WORDS["anuluj"] == "cancel" and APPROVAL_WORDS["cancel"] == "cancel"
