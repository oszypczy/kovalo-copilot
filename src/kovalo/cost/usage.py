"""Koszt tury: wywołania LLM-a pogrupowane po modelu i ich wycena (FR-6.5, kontrakt F6, 6.1).

Ta sama funkcja kosztu co raporty E3-E5 (`pricing.tokens_cost`). Kwoty to szacunek w USD
jako float, jak w `cost/pricing.py` (to nie są kwoty biznesowe w groszach).
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from kovalo.config import CostConfig, TokenPrice
from kovalo.cost.pricing import reference_price, tokens_cost

__all__ = [
    "UNKNOWN_MODEL",
    "ModelUsage",
    "TurnCost",
    "group_by_model",
    "price_for_model",
    "price_usage",
]

UNKNOWN_MODEL = "nieznany"
_MODELS_PREFIX = "models/"


class ModelUsage(BaseModel):
    """Wywołania jednego modelu w turze."""

    model: str | None  # faktyczny model z LlmCall.model; None: wywołanie bez odpowiedzi (błąd)
    calls: int
    input_tokens: int
    output_tokens: int  # jak w raportach E2-E5: wyjście razem z tokenami myślenia
    reasoning_tokens: int | None = None


class TurnCost(BaseModel):
    """Koszt tury po cenach płatnych (szacunek; darmowy tier = 0 USD)."""

    ours_usd: float | None  # None, gdy któryś model z tokenami nie ma ceny
    reference_usd: float  # wszystkie tokeny po cenie cost.reference_model (cennik kursu)
    unpriced_models: list[str] = []
    our_prices_checked_on: datetime.date


def group_by_model(calls: Iterable[Mapping[str, Any]]) -> list[ModelUsage]:
    """Słowniki LlmCall → jedna pozycja na model, w kolejności pierwszego wystąpienia.

    Wywołanie bez `usage` liczy się do `calls` z zerowymi tokenami. `reasoning_tokens`: suma
    albo None, gdy żadne wywołanie modelu ich nie podało.
    """
    grouped: dict[str | None, ModelUsage] = {}
    for call in calls:
        model = call.get("model")
        entry = grouped.get(model)
        if entry is None:
            entry = ModelUsage(model=model, calls=0, input_tokens=0, output_tokens=0)
            grouped[model] = entry
        entry.calls += 1
        usage = call.get("usage")
        if not usage:
            continue
        entry.input_tokens += int(usage.get("input_tokens") or 0)
        entry.output_tokens += int(usage.get("output_tokens") or 0)
        reasoning = usage.get("reasoning_tokens")
        if reasoning is not None:
            entry.reasoning_tokens = (entry.reasoning_tokens or 0) + int(reasoning)
    return list(grouped.values())


def price_for_model(model: str, prices: Mapping[str, TokenPrice]) -> TokenPrice | None:
    """Cena modelu: dokładne trafienie albo najdłuższy klucz K, gdy nazwa zaczyna się od K + "-".

    Prefiks "models/" jest usuwany. Bez zgadywania po rodzinie modelu.
    """
    name = model.removeprefix(_MODELS_PREFIX)
    if name in prices:
        return prices[name]
    matches = [key for key in prices if name.startswith(key + "-")]
    if not matches:
        return None
    return prices[max(matches, key=len)]


def price_usage(usage: Sequence[ModelUsage], cost: CostConfig) -> TurnCost:
    """Koszt tury po cenach faktycznych modeli (`cost.our_*`) i po cenniku kursu (referencja).

    Pozycje z model None i zerowymi tokenami są pomijane; model None z tokenami → "nieznany"
    w `unpriced_models`. Kwoty bez zaokrąglania (zaokrągla widok).
    """
    ours = 0.0
    unpriced: set[str] = set()
    total_in = total_out = 0
    for item in usage:
        has_tokens = item.input_tokens > 0 or item.output_tokens > 0
        if not has_tokens:
            continue
        total_in += item.input_tokens
        total_out += item.output_tokens
        price = (
            None
            if item.model is None
            else price_for_model(item.model, cost.our_prices_per_1m_tokens)
        )
        if price is None:
            unpriced.add(item.model or UNKNOWN_MODEL)
            continue
        ours += tokens_cost(item.input_tokens, item.output_tokens, price=price)
    return TurnCost(
        ours_usd=None if unpriced else ours,
        reference_usd=tokens_cost(total_in, total_out, price=reference_price(cost)),
        unpriced_models=sorted(unpriced),
        our_prices_checked_on=cost.our_prices_checked_on,
    )
