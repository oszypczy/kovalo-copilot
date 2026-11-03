"""Checkpointer: serializer z zarejestrowanymi typami (kontrakt F3, 6.6)."""

from __future__ import annotations

import importlib
import warnings

from tests.workflow_fakes import service_with, turn


def _sample_state():
    from kovalo.policy.models import Decision, DiscountRequest

    return {
        "turn": turn(kind="approval_request", request_action="new", discount_pct=22.0),
        "request": DiscountRequest(
            revision=2, discount_pct=22.0, margin_after_pct=7.0, clearance=True
        ),
        "decision": Decision(
            approvers=["CEO", "CFO"], reasons=["a", "b"], margin_checked=True, version="course-v1"
        ),
    }


def test_serializer_roundtrip_without_warnings():
    from kovalo.workflow.checkpoint import serializer

    serde = serializer()
    state = _sample_state()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = serde.loads_typed(serde.dumps_typed(state))
    assert restored == state
    assert (
        type(restored["turn"]).__name__ == "Turn"
        and type(restored["request"]).__name__ == "DiscountRequest"
    )


def test_memory_roundtrip_without_warnings(settings):
    """Stan z Turn, DiscountRequest i Decision zapisany na pauzie i odczytany po wznowieniu."""
    from kovalo.workflow.checkpoint import memory_checkpointer

    assert type(memory_checkpointer()).__name__ == "InMemorySaver"
    service = service_with(
        settings,
        [turn(kind="approval_request", request_action="new", discount_pct=22, margin_after_pct=10)],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        paused = service.send_message("t", "Prześlij rabat 22%.")
        state = service.get_thread("t")
        done = service.resume("t", value="approved")
        done = service.resume("t", value="approved")
    assert paused.pending.role == "CEO" and state.pending.role == "CEO"
    assert state.request["discount_pct"] == 22 and state.decision["approvers"] == ["CEO", "CFO"]
    assert done.status == "approved"


def test_checkpoint_types_importable():
    from kovalo.workflow.checkpoint import CHECKPOINT_TYPES

    assert {name for _, name in CHECKPOINT_TYPES} == {"Turn", "DiscountRequest", "Decision"}
    for module, name in CHECKPOINT_TYPES:
        assert isinstance(getattr(importlib.import_module(module), name), type), (module, name)


def test_open_checkpointer_memory(settings, monkeypatch):
    from kovalo.config import load_settings
    from kovalo.workflow.checkpoint import open_checkpointer

    monkeypatch.setenv("KOVALO_WORKFLOW__CHECKPOINTER", "memory")
    with open_checkpointer(load_settings(google_api_key="test-key")) as saver:
        assert type(saver).__name__ == "InMemorySaver"


def test_checkpointer_unavailable_error_type():
    from kovalo.workflow.checkpoint import CheckpointerUnavailableError

    assert issubclass(CheckpointerUnavailableError, RuntimeError)
