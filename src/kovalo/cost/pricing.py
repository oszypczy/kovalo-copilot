"""Koszty miesięczne (FR-1.2, krok 2 kursu i koszt z run_pipeline.py).

Kwoty w USD jako float, jak w kursie (szacunek; kontrakt F1, sekcja 5.4).
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean

from pydantic import BaseModel

from kovalo.config import CostConfig, CostScenario, TokenPrice

__all__ = ["CostEstimate", "estimate_costs", "monthly_cost", "our_price", "reference_price"]

_PER_MILLION = 1_000_000


def monthly_cost(
    input_tokens_per_question: float, *, price: TokenPrice, scenario: CostScenario
) -> float:
    """questions_per_day * working_days_per_month * (wejście * cena + wyjście * cena) / 1e6.

    Bez zaokrąglania. Ujemna liczba tokenów → ValueError.
    """
    if input_tokens_per_question < 0:
        raise ValueError(f"Liczba tokenów nie może być ujemna: {input_tokens_per_question}")
    per_question = (
        input_tokens_per_question * price.input + scenario.output_tokens_per_answer * price.output
    ) / _PER_MILLION
    return scenario.questions_per_day * scenario.working_days_per_month * per_question


def reference_price(cost: CostConfig) -> TokenPrice:
    """Cena modelu referencyjnego kursu (PINNED), niezależna od llm.model."""
    return cost.prices_per_1m_tokens[cost.reference_model]


def our_price(cost: CostConfig) -> TokenPrice:
    """Cena naszego modelu (cost.our_model) z drugiego cennika (ADR-0007)."""
    return cost.our_prices_per_1m_tokens[cost.our_model]


class CostEstimate(BaseModel):
    """Pola jak sekcja "cost" w report.json kursu."""

    llm_model: str
    rag_avg_input_tokens: float
    full_context_avg_input_tokens: float
    rag_monthly_usd: float
    full_context_monthly_usd: float
    full_to_rag_cost_ratio: float


def estimate_costs(
    rag_tokens: Sequence[int],
    full_tokens: Sequence[int],
    *,
    model: str,
    price: TokenPrice,
    scenario: CostScenario,
) -> CostEstimate:
    """Średnie tokeny promptu (retrieval i cały korpus) → koszty miesięczne i ich stosunek.

    Stosunek liczony przed zaokrągleniem kwot, jak w run_pipeline.py kursu.
    Pusta lista → ValueError; ujemne tokeny → ValueError (z monthly_cost).
    """
    if not rag_tokens or not full_tokens:
        raise ValueError("Listy tokenów promptów nie mogą być puste.")
    rag_avg, full_avg = mean(rag_tokens), mean(full_tokens)
    rag = monthly_cost(rag_avg, price=price, scenario=scenario)
    full = monthly_cost(full_avg, price=price, scenario=scenario)
    if rag == 0:
        raise ValueError("Koszt wariantu retrieval wynosi 0: stosunek kosztów nieokreślony.")
    return CostEstimate(
        llm_model=model,
        rag_avg_input_tokens=round(rag_avg, 1),
        full_context_avg_input_tokens=round(full_avg, 1),
        rag_monthly_usd=round(rag, 2),
        full_context_monthly_usd=round(full, 2),
        full_to_rag_cost_ratio=round(full / rag, 1),
    )
