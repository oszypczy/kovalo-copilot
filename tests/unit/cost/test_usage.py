"""Koszt tury: grupowanie wywołań po modelu i wycena (FR-6.5; kontrakt F6, 6.1)."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from kovalo.config import get_settings


def _call(model, input_tokens=None, output_tokens=0, reasoning=None, purpose="agent"):
    usage = (
        None
        if input_tokens is None
        else {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning,
        }
    )
    return {"purpose": purpose, "model": model, "usage": usage, "seconds": 0.1, "error": None}


def test_group_by_model_keeps_order_and_sums():
    from kovalo.cost.usage import group_by_model

    calls = [
        _call("gemini-3.5-flash-lite", 100, 10, 4, purpose="understand"),
        _call("gemini-3.6-flash", 200, 20, None),  # fallback w tej samej turze
        _call("gemini-3.5-flash-lite", 300, 30, 6),
        _call("gemini-3.6-flash", None),  # wywołanie bez usage: liczy się do calls, 0 tokenów
    ]
    grouped = group_by_model(calls)
    assert [g.model for g in grouped] == ["gemini-3.5-flash-lite", "gemini-3.6-flash"], (
        "kolejność pierwszego wystąpienia"
    )
    lite, flash = grouped
    assert (lite.calls, lite.input_tokens, lite.output_tokens) == (2, 400, 40)
    assert lite.reasoning_tokens == 10
    assert (flash.calls, flash.input_tokens, flash.output_tokens) == (2, 200, 20)
    assert flash.reasoning_tokens is None, "żadne wywołanie modelu nie podało tokenów myślenia"


def test_group_by_model_reasoning_none_when_nobody_reported_it():
    from kovalo.cost.usage import group_by_model

    (only,) = group_by_model([_call("m", 10, 1), _call("m", 20, 2)])
    assert only.reasoning_tokens is None and only.calls == 2


def test_group_by_model_call_without_response_has_model_none():
    from kovalo.cost.usage import group_by_model

    grouped = group_by_model([_call(None, None), _call("m", 5, 1)])
    assert [g.model for g in grouped] == [None, "m"]
    assert (grouped[0].calls, grouped[0].input_tokens, grouped[0].output_tokens) == (1, 0, 0)


def test_group_by_model_empty():
    from kovalo.cost.usage import group_by_model

    assert group_by_model([]) == []


def test_price_for_model_matching(settings):
    from kovalo.cost.usage import price_for_model

    prices = settings.cost.our_prices_per_1m_tokens
    lite = prices["gemini-3.5-flash-lite"]
    flash = prices["gemini-3.6-flash"]
    assert price_for_model("gemini-3.5-flash-lite", prices) == lite
    assert price_for_model("models/gemini-3.5-flash-lite", prices) == lite
    assert price_for_model("gemini-3.5-flash-lite-001", prices) == lite, "sufiks po myślniku"
    assert price_for_model("gemini-3.6-flash-preview-05", prices) == flash
    assert price_for_model("gemini-3.5-flash", prices) is None, (
        "gemini-3.5-flash to nie gemini-3.5-flash-lite (bez zgadywania po rodzinie)"
    )
    assert price_for_model("gemini-3.5-flash-liteX", prices) is None, "prefiks bez myślnika"
    assert price_for_model("nieznany-model", prices) is None


def test_price_for_model_longest_key_wins():
    from kovalo.config import TokenPrice
    from kovalo.cost.usage import price_for_model

    short, long = TokenPrice(input=1.0, output=2.0), TokenPrice(input=3.0, output=4.0)
    prices = {"m": short, "m-pro": long}
    assert price_for_model("m-pro-001", prices) == long
    assert price_for_model("m-001", prices) == short


def test_price_usage_values(settings):
    from kovalo.cost.usage import ModelUsage, price_usage

    cost = get_settings().cost
    ours = cost.our_prices_per_1m_tokens["gemini-3.5-flash-lite"]
    ref = cost.prices_per_1m_tokens[cost.reference_model]
    usage = [
        ModelUsage(
            model="gemini-3.5-flash-lite",
            calls=5,
            input_tokens=9120,
            output_tokens=612,
            reasoning_tokens=210,
        )
    ]
    result = price_usage(usage, cost)
    assert result.ours_usd == pytest.approx((9120 * ours.input + 612 * ours.output) / 1e6)
    assert result.reference_usd == pytest.approx((9120 * ref.input + 612 * ref.output) / 1e6)
    assert result.unpriced_models == []
    assert result.our_prices_checked_on == cost.our_prices_checked_on
    assert isinstance(result.our_prices_checked_on, datetime.date)
    # wyjście liczy już tokeny myślenia (usage_metadata Gemini): reasoning nie jest doliczane
    no_reasoning = [usage[0].model_copy(update={"reasoning_tokens": None})]
    assert price_usage(no_reasoning, cost).ours_usd == result.ours_usd


def test_price_usage_example_matches_contract_numbers(settings):
    """Przykład z kontraktu (6.2): 9120 / 612 tokenów → 0,004266 i 0,009594 USD."""
    from kovalo.cost.usage import ModelUsage, price_usage

    cost = settings.cost
    got = price_usage(
        [ModelUsage(model="gemini-3.5-flash-lite", calls=5, input_tokens=9120, output_tokens=612)],
        cost,
    )
    assert got.ours_usd == pytest.approx(0.004266, abs=1e-9)
    assert got.reference_usd == pytest.approx(0.009594, abs=1e-9)


def test_price_usage_sums_models_and_reference_uses_all_tokens(settings):
    from kovalo.cost.usage import ModelUsage, price_usage

    cost = settings.cost
    lite = cost.our_prices_per_1m_tokens["gemini-3.5-flash-lite"]
    flash = cost.our_prices_per_1m_tokens["gemini-3.6-flash"]
    ref = cost.prices_per_1m_tokens[cost.reference_model]
    usage = [
        ModelUsage(model="gemini-3.5-flash-lite", calls=1, input_tokens=1000, output_tokens=100),
        ModelUsage(model="gemini-3.6-flash", calls=1, input_tokens=2000, output_tokens=200),
    ]
    got = price_usage(usage, cost)
    expected = (
        1000 * lite.input + 100 * lite.output + 2000 * flash.input + 200 * flash.output
    ) / 1e6
    assert got.ours_usd == pytest.approx(expected)
    assert got.reference_usd == pytest.approx((3000 * ref.input + 300 * ref.output) / 1e6)


def test_unpriced_model(settings):
    from kovalo.cost.usage import ModelUsage, price_usage

    cost = settings.cost
    ref = cost.prices_per_1m_tokens[cost.reference_model]
    usage = [
        ModelUsage(model="gemini-3.5-flash-lite", calls=1, input_tokens=1000, output_tokens=100),
        ModelUsage(model="model-spoza-cennika", calls=2, input_tokens=500, output_tokens=50),
        ModelUsage(model="model-spoza-cennika", calls=1, input_tokens=10, output_tokens=1),
    ]
    got = price_usage(usage, cost)
    assert got.ours_usd is None, "któryś model z tokenami nie ma ceny"
    assert got.unpriced_models == ["model-spoza-cennika"], "bez powtórzeń"
    assert got.reference_usd == pytest.approx((1510 * ref.input + 151 * ref.output) / 1e6)


def test_unpriced_models_sorted(settings):
    from kovalo.cost.usage import ModelUsage, price_usage

    usage = [
        ModelUsage(model="zzz", calls=1, input_tokens=1, output_tokens=1),
        ModelUsage(model="aaa", calls=1, input_tokens=1, output_tokens=1),
    ]
    assert price_usage(usage, settings.cost).unpriced_models == ["aaa", "zzz"]


def test_model_none_is_skipped_without_tokens_and_unknown_with_tokens(settings):
    from kovalo.cost.usage import ModelUsage, price_usage

    cost = settings.cost
    priced = ModelUsage(
        model="gemini-3.5-flash-lite", calls=1, input_tokens=1000, output_tokens=100
    )
    empty = ModelUsage(model=None, calls=1, input_tokens=0, output_tokens=0)
    got = price_usage([empty, priced], cost)
    assert got.ours_usd is not None and got.unpriced_models == []

    with_tokens = ModelUsage(model=None, calls=1, input_tokens=7, output_tokens=1)
    got = price_usage([with_tokens, priced], cost)
    assert got.ours_usd is None
    assert got.unpriced_models == ["nieznany"]


def test_price_usage_empty_is_zero(settings):
    from kovalo.cost.usage import price_usage

    got = price_usage([], settings.cost)
    assert got.ours_usd == 0 and got.reference_usd == 0 and got.unpriced_models == []


def test_same_function_as_reports(settings):
    """Licznik i raporty E3-E5 liczą tą samą funkcją `pricing.tokens_cost`."""
    import kovalo.cost.usage as usage_module
    from kovalo.cost.pricing import our_price, reference_price, tokens_cost
    from kovalo.cost.usage import ModelUsage, price_usage

    cost = settings.cost
    tokens_in, tokens_out = 6626, 1439  # E3 na żywo (reports/mp3: 0,5585 / 1,1445 centa)
    got = price_usage(
        [
            ModelUsage(
                model=cost.our_model, calls=14, input_tokens=tokens_in, output_tokens=tokens_out
            )
        ],
        cost,
    )
    assert got.ours_usd == tokens_cost(tokens_in, tokens_out, price=our_price(cost))
    assert got.reference_usd == tokens_cost(tokens_in, tokens_out, price=reference_price(cost))
    assert round(got.ours_usd * 100, 4) == 0.5585
    assert round(got.reference_usd * 100, 4) == 1.1445
    source = Path(usage_module.__file__).read_text(encoding="utf-8")
    assert "tokens_cost" in source, "nie liczy własnym wzorem"
