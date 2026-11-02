"""Program E1 (kontrakt F1, 7.3 i 8.1). Referencja kursu liczona z cache z data/embeddings."""

from __future__ import annotations

import pytest

REFERENCE_TOP3 = {
    "Q01": ["POL-001#002", "POL-001#001", "POL-001#000"],
    "Q02": ["POL-003#000", "POL-003#001", "PRO-201#002"],
    "Q03": ["CON-101#001", "POL-005#001", "POL-005#000"],
    "Q04": ["POL-002#000", "POL-002#001", "PRO-201#000"],
    "Q05": ["FAQ-501#000", "FAQ-501#001", "REP-301#001"],
    "Q06": ["PRO-203#001", "PRO-203#002", "PRO-203#000"],
    "Q07": ["PL-601#000", "PRO-203#001", "PRO-203#002"],
    "Q08": ["REP-302#000", "REP-302#002", "FAQ-501#002"],
    "Q09": ["MTG-402#000", "CON-102#000", "CON-101#000"],
    "Q10": ["PRO-203#002", "PRO-202#001", "CON-102#000"],
}


def test_reference_run_reproduces_course(settings):
    from kovalo.eval.retrieval.e1 import run_e1

    report = run_e1(settings, specs=[], include_reference=True)
    assert len(report.runs) == 1
    ref = report.runs[0]
    assert ref.label == "referencja kursu"
    assert ref.provider == "openai_cache"
    assert ref.model == settings.embeddings.model and ref.dimensions == 1536
    assert ref.embed_chunks_seconds is None and ref.query_ms_avg is None

    assert report.corpus.n_documents == 18 and report.corpus.total_tokens == 8762
    assert report.chunks.n_chunks == 47
    assert report.chunks.avg_chars == 603.7 and report.chunks.max_chars == 798
    assert report.chunks_not_in_course_cache == 0
    assert report.k == 3
    assert report.targets == {"hit_rate_at_1": 0.8, "hit_rate_at_3": 0.9}

    mp1 = ref.mp1
    assert mp1.set_name == "mp1" and len(mp1.results) == 10
    assert (mp1.hit_rate_at_1, mp1.hit_rate_at_3) == (0.9, 0.9)
    assert (mp1.hits_at_1, mp1.hits_at_3) == (9, 9)
    assert {r.query_id: r.top_ids for r in mp1.results} == REFERENCE_TOP3
    q01 = mp1.results[0]
    assert q01.top_scores == pytest.approx([0.5826, 0.5779, 0.5361], abs=1e-4)
    q10 = mp1.results[-1]
    assert q10.query_id == "Q10" and q10.hit_at_3 is False
    assert [r.query_id for r in mp1.results if not r.hit_at_3] == ["Q10"]

    mp2 = ref.mp2
    assert mp2.set_name == "mp2" and len(mp2.results) == 10
    assert (mp2.hit_rate_at_1, mp2.hit_rate_at_3) == (0.9, 0.9)
    m03 = next(r for r in mp2.results if r.query_id == "M03")
    assert m03.hit_at_3 is False
    assert m03.top_doc_ids == ["PRO-203", "PRO-202", "CON-102"]
    assert [r.query_id for r in mp2.results if not r.hit_at_3] == ["M03"]

    ref_cost = ref.cost_reference
    assert ref_cost.rag_avg_input_tokens == 722.7
    assert ref_cost.full_context_avg_input_tokens == 8988.5
    assert ref_cost.rag_monthly_usd == 17.78
    assert ref_cost.full_context_monthly_usd == 69.86
    assert ref_cost.full_to_rag_cost_ratio == 3.9
    ours = ref.cost_ours
    assert ours.llm_model == settings.cost.our_model == "gemini-3.6-flash"
    assert ours.rag_monthly_usd == 15.58
    assert ours.full_context_monthly_usd == 67.65
    assert ref.meets_target is True  # 0,9 / 0,9 >= 0,8 / 0,9
    assert ref.pg_consistent is None
    assert report.recommended_model is None  # tylko referencja
    assert report.llm_call is None
    assert report.configured_model == settings.embeddings.local.model


def test_run_with_fake_model(e1_report, local_spec):
    assert [r.provider for r in e1_report.runs] == [
        "openai_cache",
        "fastembed",
    ]  # referencja pierwsza
    assert e1_report.run_id == "20261103-101500"
    assert e1_report.created_at.year == 2026
    run = e1_report.runs[1]
    assert run.model == local_spec.model
    assert run.label == local_spec.model
    assert len(run.mp1.results) == 10 and len(run.mp2.results) == 10
    assert run.embed_chunks_seconds is not None and run.embed_chunks_seconds >= 0
    assert run.query_ms_avg is not None and run.query_ms_avg >= 0
    assert run.meets_target == (run.mp1.hit_rate_at_1 >= 0.8 and run.mp1.hit_rate_at_3 >= 0.9)
    assert e1_report.configured_model == local_spec.model
    assert e1_report.recommended_model == local_spec.model  # jedyny wiersz fastembed
    assert e1_report.chunks_not_in_course_cache == 0


def test_without_reference_has_only_given_models(session_settings, local_spec):
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from kovalo.eval.retrieval.e1 import run_e1

    fake = DeterministicFakeEmbedding(size=16)
    report = run_e1(
        session_settings,
        specs=[local_spec],
        include_reference=False,
        embeddings_factory=lambda spec: fake,
    )
    assert [r.provider for r in report.runs] == ["fastembed"]


def _run(label, provider, size_gb, *, h1, h3, h1b=None, h3b=None, mrr=0.5, meets=True):
    """Syntetyczny EmbeddingRun; MP1: h1/h3 trafień, MP2: h1b/h3b (domyślnie jak MP1)."""
    from kovalo.cost.pricing import CostEstimate
    from kovalo.eval.retrieval.e1 import EmbeddingRun
    from kovalo.eval.retrieval.metrics import RetrievalResult

    def result(name, a, b):
        return RetrievalResult(
            set_name=name, k=3, results=[], hits_at_1=a, hits_at_3=b,
            hit_rate_at_1=a / 10, hit_rate_at_3=b / 10, mrr=mrr,
        )  # fmt: skip

    cost = CostEstimate(
        llm_model="m", rag_avg_input_tokens=1.0, full_context_avg_input_tokens=1.0,
        rag_monthly_usd=1.0, full_context_monthly_usd=1.0, full_to_rag_cost_ratio=1.0,
    )  # fmt: skip
    return EmbeddingRun(
        label=label, provider=provider, model=label, dimensions=8, size_gb=size_gb,
        embed_chunks_seconds=None, query_ms_avg=None,
        mp1=result("mp1", h1, h3),
        mp2=result("mp2", h1 if h1b is None else h1b, h3 if h3b is None else h3b),
        cost_reference=cost, cost_ours=cost, meets_target=meets, pg_consistent=None,
    )  # fmt: skip


def test_recommend_model_rule():
    from kovalo.eval.retrieval.e1 import recommend_model

    ref = _run("ref", "openai_cache", None, h1=9, h3=9)

    # tylko referencja: brak rekomendacji
    assert recommend_model([ref]) is None

    # najlepszy po trafieniach (referencja nie jest brana pod uwagę)
    big = _run("big", "fastembed", 2.2, h1=9, h3=10)
    mid = _run("mid", "fastembed", 1.1, h1=7, h3=8)
    assert recommend_model([ref, big, mid]) == "big"

    # tolerancja 1 trafienia w hit@1 i hit@3 (suma MP1+MP2): wygrywa mniejszy model
    best = _run("best", "fastembed", 2.2, h1=9, h3=10)  # sumy: hit@1=18, hit@3=20
    close = _run("close", "fastembed", 1.1, h1=9, h3=10, h1b=8, h3b=9)  # 17 / 19: o 1 mniej
    assert recommend_model([ref, best, close]) == "close"

    # o 2 trafienia mniej w hit@1 (mimo równego hit@3): za duża strata
    far = _run("far", "fastembed", 0.2, h1=9, h3=10, h1b=7, h3b=10)  # 16 / 20
    assert recommend_model([ref, best, far]) == "best"

    # modele spełniające cel mają pierwszeństwo przed lepszym, ale niespełniającym
    good = _run("good", "fastembed", 2.2, h1=8, h3=9, meets=True)
    better_but_fails = _run("better", "fastembed", 0.2, h1=9, h3=10, meets=False)
    assert recommend_model([ref, good, better_but_fails]) == "good"

    # żaden nie spełnia celu: najlepszy ze wszystkich fastembed
    a = _run("a", "fastembed", 2.2, h1=6, h3=7, meets=False)
    b = _run("b", "fastembed", 1.1, h1=5, h3=5, meets=False)
    assert recommend_model([ref, a, b]) == "a"

    # remis w rozmiarze rozstrzyga kolejność z punktu 2 (więcej trafień)
    t1 = _run("t1", "fastembed", 1.1, h1=9, h3=10)
    t2 = _run("t2", "fastembed", 1.1, h1=9, h3=9, mrr=0.9)
    assert recommend_model([ref, t2, t1]) == "t1"


def test_recommend_model_prefers_hits_at_3_then_hits_at_1_then_mrr():
    from kovalo.eval.retrieval.e1 import recommend_model

    a = _run("a", "fastembed", 2.0, h1=9, h3=10, mrr=0.5)
    b = _run("b", "fastembed", 2.0, h1=9, h3=10, mrr=0.7)
    assert recommend_model([a, b]) == "b"  # remis sum trafień i rozmiaru: wyższy MRR


def test_with_llm_record(session_settings, local_spec, chunks):
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from kovalo.cost.tokens import count_tokens
    from kovalo.eval.datasets import load_eval_queries
    from kovalo.eval.retrieval.e1 import run_e1
    from kovalo.knowledge.prompts import build_prompt
    from kovalo.knowledge.retrieval import build_vector_store, search
    from tests.conftest import make_fake

    fake = DeterministicFakeEmbedding(size=16)
    report = run_e1(
        session_settings,
        specs=[local_spec],
        include_reference=False,
        with_llm=True,
        embeddings_factory=lambda spec: fake,
        chat_model=make_fake("Odpowiedź testowa [POL-001].", model_name="fake-model"),
    )
    q01 = load_eval_queries(session_settings.paths.eval_queries)[0]
    top = search(build_vector_store(chunks, fake), q01.question, k=3)
    prompt = build_prompt(session_settings.prompts.system, [c for c, _ in top], q01.question)
    rec = report.llm_call
    assert rec is not None
    assert rec.question == q01.question
    assert rec.estimated_input_tokens == count_tokens(prompt, "o200k_base")
    assert rec.estimate_tokenizer == "o200k_base"
    assert (rec.input_tokens, rec.output_tokens) == (9, 21)  # usage_metadata atrapy
    assert rec.answer == "Odpowiedź testowa [POL-001]."
    assert rec.answered_by == "fake-model"
    assert rec.seconds >= 0
    assert rec.context_chunks == [c.metadata["chunk_id"] for c, _ in top]


def _usage_model(reasoning):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    usage = {"input_tokens": 9, "output_tokens": 677, "total_tokens": 686}
    if reasoning is not None:
        usage["output_token_details"] = {"reasoning": reasoning}
    msg = AIMessage(
        content="Odpowiedź [POL-001].",
        usage_metadata=usage,
        response_metadata={"model_name": "fake-model"},
    )
    return GenericFakeChatModel(messages=iter([msg]))


@pytest.mark.parametrize(("details", "expected"), [(300, 300), (0, 0), (None, None)])
def test_llm_record_reasoning_tokens(session_settings, local_spec, details, expected):
    # P3: reasoning_tokens z usage_metadata["output_token_details"]["reasoning"]
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from kovalo.eval.retrieval.e1 import run_e1

    fake = DeterministicFakeEmbedding(size=16)
    report = run_e1(
        session_settings,
        specs=[local_spec],
        include_reference=False,
        with_llm=True,
        embeddings_factory=lambda spec: fake,
        chat_model=_usage_model(details),
    )
    rec = report.llm_call
    assert rec is not None and rec.output_tokens == 677
    assert not rec.reasoning_tokens if expected in (None, 0) else rec.reasoning_tokens == expected


def test_with_llm_context_comes_from_reference_in_openai_cache_mode(session_settings, local_spec):
    # P2: openai_cache + kandydaci: kontekst LLM-a = top-3 referencji kursu, nie modelu lokalnego
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from kovalo.eval.retrieval.e1 import run_e1
    from tests.conftest import make_fake

    emb = session_settings.embeddings.model_copy(update={"provider": "openai_cache"})
    settings = session_settings.model_copy(update={"embeddings": emb})
    fake = DeterministicFakeEmbedding(size=16)
    report = run_e1(
        settings,
        specs=[local_spec],
        include_reference=True,
        with_llm=True,
        embeddings_factory=lambda spec: fake,
        chat_model=make_fake("Odpowiedź [POL-001]."),
    )
    ref = report.runs[0]
    assert ref.label == "referencja kursu"
    assert report.llm_call is not None
    assert report.llm_call.context_chunks == ref.mp1.results[0].top_ids
    assert report.llm_call.context_chunks == ["POL-001#002", "POL-001#001", "POL-001#000"]


def test_pgvector_backend_requires_configured_model(session_settings):
    # P4: bez modelu z configu wśród ocenianych --backend pgvector to ConfigError (przed Postgresem)
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from kovalo.config import ConfigError, EmbeddingModelSpec
    from kovalo.eval.retrieval.e1 import run_e1

    other = EmbeddingModelSpec(model="inny/model", dimensions=16)
    fake = DeterministicFakeEmbedding(size=16)
    with pytest.raises(ConfigError, match="pgvector"):
        run_e1(
            session_settings,
            specs=[other],
            include_reference=True,
            backend="pgvector",
            embeddings_factory=lambda spec: fake,
        )
