"""Fixture'y testów workflow (kontrakt F3, 14.1). Atrapy: tests/workflow_fakes.py."""

from __future__ import annotations

import pytest

from tests.workflow_fakes import ReplayLLM, service_with, toy_answerer


@pytest.fixture
def replay():
    """Fabryka atrapy ReplayLLM."""
    return ReplayLLM


@pytest.fixture
def make_service(settings):
    """make_service(turns, **kw) -> CopilotService na grafie z atrapą modelu."""

    def make(turns=(), **kwargs):
        return service_with(settings, turns, **kwargs)

    return make


@pytest.fixture
def toy(settings):
    """toy(llm, **kw) -> DocumentAnswerer na toy retrieverze."""

    def make(llm, **kwargs):
        return toy_answerer(settings, llm, **kwargs)

    return make
