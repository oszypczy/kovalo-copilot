"""E5 z prawdziwym Gemini (marker live): rozmowa C5, najwyżej 6 wywołań (kontrakt F5, 5 i 13.2).

Ręcznie: make test-live T=tests/live/test_e5_live.py
C5 = 2 x `understand` + około 2 tury agenta (OTIF Mlecznej Doliny w październiku: brak danych).
Bez cache LLM-a; wspólny licznik `max_calls=6` pilnuje limitu. Serwer MCP w pamięci,
wyszukiwanie na lokalnych embeddingach fastembed (bez kosztu). Brak klucza -> skip.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


def test_c5_live(live_google_key: str | None):
    if not live_google_key:
        pytest.skip("brak GOOGLE_API_KEY w środowisku procesu")
    from kovalo.agents.trace import trace_from_record
    from kovalo.config import load_settings
    from kovalo.eval.assistant.e5 import run_e5

    settings = load_settings(google_api_key=live_google_key)
    report = run_e5(
        settings, live=True, ids=["C5"], transport="memory", max_calls=6, cache=None
    )  # bez cache: wywołania 1-6
    live = report.live
    assert report.stopped_by_max_calls is False
    first, second = live.turns
    assert (
        first["entry"]["turn"]["kind"] == "document_question"
        and first["entry"]["status"] == "answered"
    )
    trace = trace_from_record(first["entry"]["record"])
    otif = [
        e
        for e in trace.events
        if e.tool == "get_supplier_otif"
        and e.status == "ok"
        and "2026-10" in e.result["missing_months"]
    ]
    assert otif, "potrzeba get_supplier_otif z brakującym 2026-10"
    assert all(e.transport == "mcp" for e in trace.events if e.parent_call_id is None)
    assert second["entry"]["turn"]["kind"] == "other"
    calls = live.calls
    assert calls.understand_network + calls.agent_network <= 6 and calls.errors == 0
    assert sum(calls.mcp_failures.values()) == 0
