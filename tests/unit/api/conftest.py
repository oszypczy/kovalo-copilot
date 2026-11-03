"""Fixture'y API wątków (kontrakt F3, 14.1)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.workflow_fakes import service_with


@pytest.fixture
def make_client(settings):
    """make_client(turns, **kw) -> TestClient z create_app(settings, service=...) na atrapie."""
    from kovalo.api.main import create_app

    def make(turns=(), *, raise_server_exceptions=True, **kwargs):
        service = service_with(settings, turns, **kwargs)
        return TestClient(
            create_app(settings, service=service), raise_server_exceptions=raise_server_exceptions
        )

    return make


@pytest.fixture
def client(make_client):
    return make_client([])


@pytest.fixture
def d2_client(make_client, gold_dialogues):
    from tests.workflow_fakes import gold_turns

    return make_client(gold_turns(gold_dialogues, "D2"))
