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


@pytest.fixture
def sse_events():
    """sse_events(response) -> lista (id, event, data) z ciała text/event-stream; komentarze
    (`: ping`) są pomijane (zwracane osobno w `.pings`)."""
    import json

    class Events(list):
        pings = 0

    def parse(response) -> Events:
        events = Events()
        for block in response.text.split("\n\n"):
            if not block.strip():
                continue
            if block.startswith(":"):
                events.pings += 1
                continue
            fields = {}
            for line in block.splitlines():
                key, _, value = line.partition(": ")
                fields[key] = value
            events.append((int(fields["id"]), fields["event"], json.loads(fields["data"])))
        return events

    return parse


@pytest.fixture
def make_agent_client(settings):
    """make_agent_client(turns, trace) -> TestClient z serwisem z atrapą agenta."""
    from kovalo.api.main import create_app
    from tests.agent_fakes import agent_answerer

    def make(turns, trace, **kwargs):
        service = service_with(settings, turns, agent=agent_answerer(trace))
        return TestClient(create_app(settings, service=service), **kwargs)

    return make
