"""E2 z prawdziwym wywołaniem Gemini (marker live): dokładnie 2 wywołania (odpowiedź + sędzia).

Ręcznie: make test-live T=tests/live/test_e2_live.py
Bez cache LLM-a i bez pobierania modelu fastembed: wyszukiwanie z embeddingami z cache kursu.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


def test_answer_and_judge_live(live_google_key: str | None):
    if not live_google_key:
        pytest.skip("brak GOOGLE_API_KEY w środowisku procesu")
    from kovalo.config import load_settings
    from kovalo.eval.datasets import load_gold
    from kovalo.eval.rag.judge import judge_record
    from kovalo.knowledge.chunking import build_contextual_chunks
    from kovalo.knowledge.corpus import load_corpus
    from kovalo.knowledge.embeddings import build_course_cache_embeddings
    from kovalo.knowledge.retrieval import build_vector_store
    from kovalo.knowledge.retrievers import VectorRetriever
    from kovalo.llm import get_structured_model
    from kovalo.rag.answer import answer_question
    from kovalo.rag.records import Answer, Verdict

    settings = load_settings(google_api_key=live_google_key)
    chunks = build_contextual_chunks(load_corpus(settings), settings.chunking)
    embeddings = build_course_cache_embeddings(
        settings.paths.embedding_cache,
        model=settings.embeddings.model,
        underlying=None,  # tylko cache kursu: pytanie M05 jest w cache
    )
    retriever = VectorRetriever(build_vector_store(chunks, embeddings))
    gold = next(q for q in load_gold(settings.paths.mini_project_2_questions) if q.id == "M05")

    record = answer_question(
        gold.question,
        retriever,
        get_structured_model(Answer, "answer", settings=settings),  # bez cache
        system_prompt=settings.prompts.answer_system,
        k=settings.retrieval.top_k,
        chunks="ctx",
    )  # wywołanie 1
    assert record.error is None, record.error
    assert record.output is not None and record.output.answerable
    assert "PRO-202" in record.output.citations
    assert record.usage is not None and record.usage.input_tokens > 0

    verdict = judge_record(
        record,
        gold,
        get_structured_model(Verdict, "judge", settings=settings),
        system_prompt=settings.prompts.judge_system,
    )  # wywołanie 2
    assert verdict.error is None, verdict.error
    assert verdict.correct is not None and verdict.grounded is not None
