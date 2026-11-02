"""Wspólne fixture'y testów ewaluacji E1."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

FIXED_NOW = datetime(2026, 11, 3, 10, 15, 0, tzinfo=UTC)


@pytest.fixture(scope="session")
def local_spec(session_settings):
    from kovalo.config import EmbeddingModelSpec

    local = session_settings.embeddings.local
    return EmbeddingModelSpec(
        model=local.model,
        dimensions=local.dimensions,
        query_prefix=local.query_prefix,
        passage_prefix=local.passage_prefix,
    )


@pytest.fixture(scope="session")
def e1_report(session_settings, local_spec):
    """Raport E1: referencja kursu (prawdziwy cache) + atrapa zamiast modelu z configu."""
    from kovalo.eval.retrieval.e1 import run_e1
    from langchain_core.embeddings import DeterministicFakeEmbedding

    fake = DeterministicFakeEmbedding(size=16)
    return run_e1(
        session_settings,
        specs=[local_spec],
        include_reference=True,
        embeddings_factory=lambda spec: fake,
        clock=lambda: FIXED_NOW,
    )
