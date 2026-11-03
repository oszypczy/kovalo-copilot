"""Liczby i odpowiedzi w polskim tekście (FR-3.7; port reference/course/helpers/text_pl.py).

Moduł obsługuje dwa kierunki wymiany informacji między człowiekiem a programem:
  - number(): liczba do tekstu dla człowieka, w polskim zapisie (5.01 -> "5,01";
    18240000 -> "18 240 000"). Tylko do tekstu: dane (JSON, YAML, rekordy) zachowują liczby;
  - parse_number() i human_answer(): to, co człowiek wpisuje w odpowiedzi po wstrzymaniu
    przepływu pracy, zamienione na wartości, których graf oczekuje: liczbę (marżę),
    "approved", "rejected" albo "cancel". Graf nigdy nie widzi polskich słów ani przecinka
    dziesiętnego.

Kontrakt: docs/contracts/f3.md, sekcja 5.1. Tylko biblioteka standardowa.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal
from typing import Literal

__all__ = ["APPROVAL_WORDS", "PauseType", "human_answer", "number", "parse_number"]

PauseType = Literal["clarification", "approval"]

APPROVAL_WORDS: dict[str, str] = {
    "approved": "approved",
    "approve": "approved",
    "zatwierdź": "approved",
    "zatwierdz": "approved",
    "zatwierdzam": "approved",
    "zatwierdzone": "approved",
    "rejected": "rejected",
    "reject": "rejected",
    "odrzuć": "rejected",
    "odrzuc": "rejected",
    "odrzucam": "rejected",
    "odrzucone": "rejected",
    "cancel": "cancel",
    "anuluj": "cancel",
    "anuluję": "cancel",
    "anuluje": "cancel",
}
_PLAIN_NUMBER = re.compile(r"[-−]?\d+(?:[.,]\d+)?")  # noqa: RUF001 (minus typograficzny)


def number(value: float | int | Decimal) -> str:
    """Liczba do tekstu dla człowieka, w polskim zapisie: przecinek dziesiętny, bez zer na końcu,
    cyfry grupowane spacją od pięciu cyfr wzwyż (jak w polskiej typografii).

        number(5.01) -> '5,01'    number(12.0) -> '12'    number(18240000) -> '18 240 000'
    """
    text = format(Decimal(str(value)).normalize(), "f")
    sign, text = ("-", text[1:]) if text.startswith("-") else ("", text)
    whole, _, frac = text.partition(".")
    if len(whole) >= 5:
        whole = f"{int(whole):,}".replace(",", " ")
    return sign + whole + ("," + frac if frac else "")


def parse_number(text: str) -> float | None:
    """Liczba wpisana przez człowieka: "7,5", "7.5", "7", "-2", "7,5%" -> 7.5 itd. Wszystko inne
    (słowa, "1,5.0", "7 5", nieskończoność) -> None: program pyta jeszcze raz. Przecinek jest tu
    zawsze znakiem dziesiętnym: człowiek odpowiada marżą, nigdy kwotą z grupowaniem tysięcy."""
    cleaned = text.strip().rstrip("%").strip().replace("−", "-")  # noqa: RUF001
    if not _PLAIN_NUMBER.fullmatch(cleaned):
        return None
    value = float(cleaned.replace(",", "."))
    return value if math.isfinite(value) else None


def human_answer(payload_type: PauseType | str, text: str) -> float | str:
    """To, co człowiek wpisał przy pauzie, jako wartość, której oczekuje workflow.

    clarification: liczba (z polskim albo angielskim znakiem dziesiętnym) -> float; "anuluj" /
    "cancel" -> "cancel"; approval: "zatwierdź" / "odrzuć" / "anuluj" (albo angielskie słowa) ->
    "approved" / "rejected" / "cancel". Wszystko inne wraca jako text.strip(), a workflow pyta
    jeszcze raz.
    """
    word = text.strip().casefold()
    if APPROVAL_WORDS.get(word) == "cancel":
        return "cancel"
    if payload_type == "clarification":
        value = parse_number(text)
        return value if value is not None else text.strip()
    return APPROVAL_WORDS.get(word, text.strip())
