"""Fixture'y E5 (kontrakt F5, 13). Bez sieci i bez klucza: serwer MCP w pamięci, embeddingi
`DeterministicFakeEmbedding` (override), atrapy modeli z idealnymi skryptami."""

from __future__ import annotations

import pytest


def turn_ids(conversations, ids=None):
    return [
        f"{d.id}-t{n}"
        for d in conversations
        if ids is None or d.id in ids
        for n, _ in enumerate(d.turns, start=1)
    ]


@pytest.fixture(scope="session")
def e5_replay_report(session_settings):
    """Zgodność + odtworzenie (zero LLM-a), serwer w pamięci."""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from kovalo.eval.assistant.e5 import run_e5
    from kovalo.llm import override_embeddings

    with override_embeddings(DeterministicFakeEmbedding(size=16)):
        return run_e5(session_settings, transport="memory", backend="memory")


@pytest.fixture(scope="session")
def e5_live_report(session_settings, conversations, ctx_chunks):
    """Przebieg „na żywo” na idealnych atrapach: 25 x understand i 4 odpowiedzi agenta przez MCP."""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from kovalo.eval.assistant.e5 import run_e5
    from kovalo.eval.workflow.runner import ScriptedUnderstanding
    from kovalo.llm import override_embeddings
    from kovalo.workflow.schemas import Turn
    from tests.agent_fakes import ScriptedModel
    from tests.mcp_fakes import ideal_e5_agent_messages

    turns = [Turn.model_validate(t.understand) for d in conversations for t in d.turns]
    agent = ideal_e5_agent_messages(ctx_chunks)
    script = [m for key in ("C1-t1", "C1-t3", "C5-t1", "D7-t1") for m in agent[key]]
    with override_embeddings(DeterministicFakeEmbedding(size=16)):
        return run_e5(
            session_settings,
            live=True,
            transport="memory",
            backend="memory",
            max_calls=None,
            understand_model=ScriptedUnderstanding(turns).as_runnable(),
            agent_model=ScriptedModel(script=script),
        )
