"""Ceny i koszty miesięczne (FR-1.2; kontrakt F1, sekcja 5.4)."""

from __future__ import annotations

import pytest


def test_monthly_cost_example(settings):
    from kovalo.cost.pricing import monthly_cost, reference_price

    got = monthly_cost(1000, price=reference_price(settings.cost), scenario=settings.cost.scenario)
    assert got == pytest.approx(19.53)


def test_monthly_cost_grows_with_input_tokens(settings):
    from kovalo.cost.pricing import monthly_cost, reference_price

    price, scenario = reference_price(settings.cost), settings.cost.scenario
    assert (
        monthly_cost(2000, price=price, scenario=scenario)
        > monthly_cost(1000, price=price, scenario=scenario)
        > monthly_cost(0, price=price, scenario=scenario)
        > 0
    )


def test_monthly_cost_negative_raises(settings):
    from kovalo.cost.pricing import monthly_cost, reference_price

    with pytest.raises(ValueError):
        monthly_cost(-1, price=reference_price(settings.cost), scenario=settings.cost.scenario)


def test_reference_price_independent_of_llm_model(monkeypatch: pytest.MonkeyPatch, settings):
    from kovalo.cost.pricing import reference_price

    from kovalo.config import load_settings

    monkeypatch.setenv("KOVALO_LLM__MODEL", "inny-model")
    other = load_settings(google_api_key="test-key")
    assert other.llm.model == "inny-model"
    assert reference_price(other.cost) == reference_price(settings.cost)
    assert reference_price(settings.cost).input == 0.75
    assert reference_price(settings.cost).output == 4.5


def test_our_price_is_gemini(settings):
    from kovalo.cost.pricing import our_price

    price = our_price(settings.cost)
    assert (price.input, price.output) == (0.75, 3.75)


def test_estimate_costs_rounding_and_ratio(settings):
    from kovalo.cost.pricing import estimate_costs, monthly_cost, reference_price

    price, scenario = reference_price(settings.cost), settings.cost.scenario
    est = estimate_costs([1000], [9000], model="m", price=price, scenario=scenario)
    assert est.llm_model == "m"
    assert est.rag_avg_input_tokens == 1000.0 and est.full_context_avg_input_tokens == 9000.0
    assert est.rag_monthly_usd == 19.53 and est.full_context_monthly_usd == 69.93
    rag = monthly_cost(1000, price=price, scenario=scenario)
    full = monthly_cost(9000, price=price, scenario=scenario)
    assert est.full_to_rag_cost_ratio == round(full / rag, 1)  # liczone przed zaokrągleniem kwot

    # średnie zaokrąglane do 1 miejsca
    assert (
        estimate_costs(
            [1, 1, 2], [3], model="m", price=price, scenario=scenario
        ).rag_avg_input_tokens
        == 1.3
    )


def test_estimate_costs_ratio_uses_unrounded_amounts(settings):
    from kovalo.cost.pricing import estimate_costs, monthly_cost

    from kovalo.config import TokenPrice

    scenario = settings.cost.scenario
    price = TokenPrice(input=0.75, output=4.5)
    rag_tokens, full_tokens = [723], [8989]
    est = estimate_costs(rag_tokens, full_tokens, model="m", price=price, scenario=scenario)
    rag = monthly_cost(723, price=price, scenario=scenario)
    full = monthly_cost(8989, price=price, scenario=scenario)
    assert est.full_to_rag_cost_ratio == round(full / rag, 1)
    assert est.rag_monthly_usd == round(rag, 2)


def test_estimate_costs_invalid_input_raises(settings):
    from kovalo.cost.pricing import estimate_costs, reference_price

    kwargs = {
        "model": "m",
        "price": reference_price(settings.cost),
        "scenario": settings.cost.scenario,
    }
    with pytest.raises(ValueError):
        estimate_costs([], [1], **kwargs)
    with pytest.raises(ValueError):
        estimate_costs([1], [], **kwargs)
    with pytest.raises(ValueError):
        estimate_costs([-5], [1], **kwargs)
