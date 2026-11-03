"""kovalo.quantities: port `helpers/quantities.py` 1:1 (kontrakt F4, 5.1).

Porty testów liczb z `test_provided_trace_plumbing.py` kursu MP4. Od tych reguł zależy
porównywalność E4 z kursem; zmiana wymaga ADR-u.
"""

from __future__ import annotations

from decimal import Decimal

import pytest


@pytest.fixture
def q():
    """Moduł kovalo.quantities importowany leniwie: brak kodu = czerwone testy, nie błąd zbierania."""  # noqa: E501
    import kovalo.quantities

    return kovalo.quantities


# angielskie zapisy celowo: moduł czyta je obok polskich (polskie przykłady: test_polish_notation)
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("7,296 PLN", [("7296", "PLN")]),
        ("PLN 7,296.00", [("7296.00", "PLN")]),
        ("7 296,00 zł", [("7296.00", "PLN")]),
        ("94.6% and 2 full percentage points", [("94.6", "%"), ("2", "pp")]),
        ("2,5%", [("2.5", "%")]),
        ("SUP-114, CON-101, 2026-07, Q2, July 2026, 20 August", []),
        ("1. pierwszy\n2. drugi", []),
        ("1926 PLN w 1926 r.", [("1926", "PLN")]),
        ("18,240,000 grosze = 182,400 PLN", [("18240000", "grosze"), ("182400", "PLN")]),
    ],
)
def test_numbers_in(q, text, expected):
    assert [(str(q.value), q.unit) for q in q.numbers_in(text)] == expected


def test_source_values_use_the_field_units(q):
    values = {
        (str(q.value), q.unit)
        for q in q.source_values(
            {
                "penalty_grosze": 729600,
                "full_points": 2,
                "rate_per_full_point_pct": 2.0,
                "complete": True,
                "passages": [
                    {
                        "doc_id": "CON-101",
                        "text": "jeśli OTIF przez trzy kolejne miesiące wynosi mniej niż 94%",
                    }
                ],
            }
        )
    }
    assert values == {("7296", "PLN"), ("729600", "grosze"), ("2", "pp"), ("2.0", "%"), ("94", "%")}


def test_same_quantity(q):
    pln = q.numbers_in("7296,00 PLN")[0]
    assert q.same_quantity(pln, q.source_values({"penalty_grosze": 729600})[0])
    grosze = q.source_values({"penalty_grosze": 729600})
    assert not any(q.same_quantity(q.numbers_in("729 600 PLN")[0], s) for s in grosze), (
        "729600 groszy to nie 729 600 PLN"
    )
    assert any(q.same_quantity(q.numbers_in("729 600 groszy")[0], s) for s in grosze)
    assert not any(q.same_quantity(q.numbers_in("729,600 grosze")[0], s) for s in grosze), (
        "niejednoznaczne: po polsku 729,6"
    )
    assert not q.same_quantity(q.numbers_in("2%")[0], q.source_values({"full_points": 2})[0])
    assert q.same_quantity(q.numbers_in("94,6")[0], q.source_values({"otif_pct": 94.6})[0]), (
        "liczba bez jednostki zgadza się z każdą jednostką"
    )


def test_money_helpers(q):
    assert q.grosze_to_pln(729600) == Decimal("7296.00")
    assert [q.round_grosze(Decimal(x)) for x in ("2.5", "2.49", "3.5", "246.4")] == [3, 2, 4, 246]


@pytest.mark.parametrize(
    ("text", "phrase", "found"),
    [
        ("zapisano w notatce", "nota", False),
        ("warunek nie jest spełniony", "nie jest spełniony", True),
        ("27 296 PLN", "7 296", False),
        ("marża 19,9%", "9%", False),
        ("1 500 000 PLN", "500 000", False),
        ("zgoda (CFO).", "CFO", True),
    ],
)
def test_mentions_whole_phrases(q, text, phrase, found):
    assert q.mentions(text, phrase) is found


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("**729 600** PLN", [("729600", "PLN")]),
        ("*7296,00* zł", [("7296.00", "PLN")]),
        ("`94,6`% i __2__ punkty procentowe", [("94.6", "%"), ("2", "pp")]),
        ("-7296 PLN; (−3%)", [("-7296", "PLN"), ("-3", "%")]),  # noqa: RUF001
        ("przedział 5-12%", [("5", None), ("12", "%")]),
        ("**2026** PLN", [("2026", "PLN")]),
    ],
)
def test_numbers_in_formatted_text(q, text, expected):
    assert [(str(q.value), q.unit) for q in q.numbers_in(text)] == expected


def test_mentions_ignores_markdown_and_respects_signs(q):
    assert q.mentions("**7 296** PLN", "7 296")
    assert not q.mentions("-7 296 PLN", "7 296") and not q.mentions("wynosi -7 296", "7 296")
    assert q.mentions("przedział 5-12%", "12%")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Kara: 7296,00 PLN, czyli 2 pełne punkty procentowe poniżej celu; lipiec: 94,6%.",
            [("7296.00", "PLN"), ("2", "pp"), ("94.6", "%")],
        ),
        ("18 240 000 groszy = 182 400 zł", [("18240000", "grosze"), ("182400", "PLN")]),
        ("7 296,00 zł i 12 złotych", [("7296.00", "PLN"), ("12", "PLN")]),
        (
            "2 pkt proc., 3 proc., 4 procent, 2 punkty procentowe",
            [("2", "pp"), ("3", "%"), ("4", "%"), ("2", "pp")],
        ),
        ("Spotkanie 20 sierpnia 2026 r., nota do 5 września.", []),
        ("od 2 do 6°C; 2–6 °C", [("2", None), ("6", None), ("2", None), ("6", None)]),  # noqa: RUF001
    ],
)
def test_polish_notation(q, text, expected):
    assert [(str(q.value), q.unit) for q in q.numbers_in(text)] == expected


def test_an_ambiguous_number_is_reported_not_guessed(q):
    one_comma = q.numbers_in("1,500 PLN")[0]
    assert one_comma.ambiguous, "1,500 to po polsku 1,5, a po angielsku 1500"
    assert not q.same_quantity(one_comma, q.numbers_in("1500 PLN")[0])
    assert not q.same_quantity(one_comma, q.numbers_in("1,5 PLN")[0])
    clear = "1,5 PLN; 1 500 PLN; 1,500.00 PLN; 1,500,000 PLN; 7296,00 PLN"
    assert not any(q.ambiguous for q in q.numbers_in(clear))


def test_polish_numeric_dates_are_not_quantities(q):
    assert q.numbers_in("Umowa obowiązuje od 1.04.2025 do 31.03.2027 r.; kara 2%.") == q.numbers_in(
        "kara 2%."
    )
