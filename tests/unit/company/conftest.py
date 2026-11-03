"""Fixture'y testów `kovalo.company` (kontrakt F4, 5). Moduły importowane leniwie."""

from __future__ import annotations

import pytest


@pytest.fixture
def lookup():  # type: ignore[no-untyped-def]
    import kovalo.company.lookup

    return kovalo.company.lookup


@pytest.fixture
def calc():  # type: ignore[no-untyped-def]
    import kovalo.company.penalty

    return kovalo.company.penalty
