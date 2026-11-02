"""Testy z prawdziwym modelem fastembed (marker `model`, 0,2-2,3 GB). Ręcznie: make test-model."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.model


def test_configured_model_loads_and_embeds():
    from kovalo.config import load_settings
    from kovalo.llm import get_embeddings

    settings = load_settings()
    emb = get_embeddings(settings=settings, cache=False)
    first = emb.embed_documents(["Rabaty powyżej 12% zatwierdza dyrektor handlowy."])
    second = emb.embed_documents(["Rabaty powyżej 12% zatwierdza dyrektor handlowy."])
    assert len(first[0]) == settings.embeddings.local.dimensions
    assert first[0] == pytest.approx(second[0], abs=1e-5)


def test_e5_prefix_changes_query_vector():
    from kovalo.config import load_settings
    from kovalo.llm import get_embeddings

    settings = load_settings()
    local = settings.embeddings.local
    if not (local.query_prefix or local.passage_prefix):
        pytest.skip("model z configu nie używa prefiksów")
    emb = get_embeddings(settings=settings, cache=False)
    text = "Kto zatwierdza rabat 15%?"
    assert emb.embed_query(text) != emb.embed_documents([text])[0]
