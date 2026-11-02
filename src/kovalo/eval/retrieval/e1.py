"""Program E1: wyszukiwanie i koszty MP1 dla referencji kursu i modeli fastembed.

Kontrakt F1, sekcja 7.3. Koszt jak w run_pipeline.py kursu, wywołanie LLM-a jak
first_llm_call.py (krok 6), reguła wyboru modelu deterministyczna (recommend_model).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel

from kovalo.config import ConfigError, CostScenario, EmbeddingModelSpec, Settings
from kovalo.cost.pricing import CostEstimate, estimate_costs, our_price, reference_price
from kovalo.cost.tokens import count_tokens
from kovalo.eval.datasets import EvalQuery, load_eval_queries, load_mp2_retrieval_queries
from kovalo.eval.retrieval.metrics import RetrievalResult, evaluate_retrieval
from kovalo.knowledge.chunking import ChunkStats, chunk_documents, chunk_stats
from kovalo.knowledge.corpus import CorpusStats, corpus_stats, load_corpus
from kovalo.knowledge.embeddings import missing_from_course_cache, model_size_gb
from kovalo.knowledge.prompts import build_prompt
from kovalo.knowledge.retrieval import build_vector_store, search

__all__ = [
    "REFERENCE_LABEL",
    "TARGETS",
    "E1LlmCallError",
    "E1Report",
    "EmbeddingRun",
    "LlmCallRecord",
    "recommend_model",
    "run_e1",
]

REFERENCE_LABEL = "referencja kursu"
# Cele E1 (docs/06-evaluation.md): hit@1 ≥ 0,8 i hit@3 ≥ 0,9 na pytaniach MP1.
TARGETS: dict[str, float] = {"hit_rate_at_1": 0.8, "hit_rate_at_3": 0.9}

EmbeddingsFactory = Callable[[EmbeddingModelSpec | None], Embeddings]


class EmbeddingRun(BaseModel):
    label: str  # "referencja kursu" albo nazwa modelu
    provider: Literal["openai_cache", "fastembed"]
    model: str
    dimensions: int
    size_gb: float | None
    embed_chunks_seconds: float | None  # wszystkie chunki, model bez cache; None dla openai_cache
    query_ms_avg: float | None  # średnio na pytanie (MP1 + MP2), bez cache
    mp1: RetrievalResult
    mp2: RetrievalResult
    cost_reference: CostEstimate  # ceny modelu referencyjnego kursu (PINNED)
    cost_ours: CostEstimate  # ceny cost.our_model, te same tokeny
    meets_target: bool
    pg_consistent: bool | None  # tylko --backend pgvector


class LlmCallRecord(BaseModel):
    """Pola llm_call.json kursu (bez `credential`) + model, który odpowiedział, i czas."""

    date: date
    provider: str
    model: str
    answered_by: str
    question: str
    context_chunks: list[str]
    answer: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None = None  # output_token_details['reasoning'] (część output_tokens)
    estimated_input_tokens: int
    estimate_tokenizer: str
    assumed_output_tokens: int
    seconds: float


class E1Report(BaseModel):
    run_id: str  # YYYYMMDD-HHMMSS (UTC)
    created_at: datetime
    corpus: CorpusStats
    chunks: ChunkStats
    chunks_not_in_course_cache: int  # 0 = chunki identyczne z podziałem kursu
    k: int
    targets: dict[str, float]
    our_prices_checked_on: date
    cost_scenario: CostScenario  # rozszerzenie kontraktu: scenariusz kosztów do raportu MD
    runs: list[EmbeddingRun]  # referencja pierwsza
    configured_model: str  # embeddings.local.model
    recommended_model: str | None
    llm_call: LlmCallRecord | None


class E1LlmCallError(RuntimeError):
    """Wywołanie LLM-a (--with-llm) nie powiodło się; `report` to wynik bez sekcji LLM."""

    def __init__(self, report: E1Report, cause: BaseException) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.report = report
        self.cause = cause


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class _Context:
    """Dane wspólne dla wszystkich wierszy (liczone raz)."""

    settings: Settings
    docs: list[Document]
    chunks: list[Document]
    mp1: list[EvalQuery]
    mp2: list[EvalQuery]
    full_tokens: list[int]


def _costs(ctx: _Context, mp1: RetrievalResult) -> tuple[CostEstimate, CostEstimate]:
    settings = ctx.settings
    by_id = {c.metadata["chunk_id"]: c for c in ctx.chunks}
    encoding = settings.tokenizer.encoding
    rag_tokens = [
        count_tokens(
            build_prompt(settings.prompts.system, [by_id[cid] for cid in r.top_ids], r.question),
            encoding,
        )
        for r in mp1.results
    ]
    cost = settings.cost
    reference = estimate_costs(
        rag_tokens,
        ctx.full_tokens,
        model=cost.reference_model,
        price=reference_price(cost),
        scenario=cost.scenario,
    )
    ours = estimate_costs(
        rag_tokens,
        ctx.full_tokens,
        model=cost.our_model,
        price=our_price(cost),
        scenario=cost.scenario,
    )
    return reference, ours


def _meets_target(mp1: RetrievalResult) -> bool:
    return (
        mp1.hit_rate_at_1 >= TARGETS["hit_rate_at_1"]
        and mp1.hit_rate_at_3 >= TARGETS["hit_rate_at_3"]
    )


def _top_ids(store: VectorStore, queries: Sequence[EvalQuery], k: int) -> list[list[str]]:
    return [[c.metadata["chunk_id"] for c, _ in search(store, q.question, k)] for q in queries]


def _pg_store(
    settings: Settings, embeddings: Embeddings, *, provider: str, model: str, dimensions: int
) -> VectorStore:
    from kovalo.knowledge.pgvector import check_index, open_store, pg_engine

    table = settings.db.vector_collection_mp1
    check_index(settings, table, provider=provider, model=model, dimensions=dimensions)
    return open_store(pg_engine(settings), embeddings, table=table)


def _evaluate(
    ctx: _Context,
    store: VectorStore,
    *,
    label: str,
    provider: Literal["openai_cache", "fastembed"],
    model: str,
    dimensions: int,
    size_gb: float | None,
    embed_chunks_seconds: float | None,
    query_ms_avg: float | None,
    pg_store: VectorStore | None = None,
) -> EmbeddingRun:
    k = ctx.settings.retrieval.top_k
    pg_consistent: bool | None = None
    search_store = store
    if pg_store is not None:
        queries = [*ctx.mp1, *ctx.mp2]
        pg_consistent = _top_ids(pg_store, queries, k) == _top_ids(store, queries, k)
        search_store = pg_store  # model z configu wyszukuje w Postgresie
    mp1 = evaluate_retrieval(search_store, ctx.mp1, k=k, set_name="mp1")
    mp2 = evaluate_retrieval(search_store, ctx.mp2, k=k, set_name="mp2")
    cost_reference, cost_ours = _costs(ctx, mp1)
    return EmbeddingRun(
        label=label,
        provider=provider,
        model=model,
        dimensions=dimensions,
        size_gb=size_gb,
        embed_chunks_seconds=embed_chunks_seconds,
        query_ms_avg=query_ms_avg,
        mp1=mp1,
        mp2=mp2,
        cost_reference=cost_reference,
        cost_ours=cost_ours,
        meets_target=_meets_target(mp1),
        pg_consistent=pg_consistent,
    )


def _fastembed_settings(settings: Settings) -> Settings:
    """E1 porównuje modele fastembed także wtedy, gdy aplikacja działa w trybie openai_cache."""
    if settings.embeddings.provider == "fastembed":
        return settings
    embeddings = settings.embeddings.model_copy(update={"provider": "fastembed"})
    return settings.model_copy(update={"embeddings": embeddings})


def _llm_call(
    settings: Settings,
    store: VectorStore,
    question: EvalQuery,
    *,
    chat_model: Runnable | None,
    today: date,
) -> LlmCallRecord:
    """Port first_llm_call.py: prompt Q01 z top-k chunkami, jedno wywołanie, tokeny dostawcy."""
    hits = search(store, question.question, settings.retrieval.top_k)
    prompt = build_prompt(settings.prompts.system, [c for c, _ in hits], question.question)
    if chat_model is None:
        from kovalo.llm import get_resilient_chat_model

        chat_model = get_resilient_chat_model("default", settings=settings)
    start = time.perf_counter()
    message = chat_model.invoke(prompt)
    seconds = time.perf_counter() - start
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        # Kurs (krok 6): brak liczby tokenów od dostawcy to błąd; nigdy jej nie zgadujemy.
        raise ValueError("Odpowiedź modelu nie zawiera liczby tokenów (usage_metadata).")
    metadata = getattr(message, "response_metadata", None) or {}
    text = getattr(message, "text", None)
    answer = text if isinstance(text, str) else str(getattr(message, "content", message))
    return LlmCallRecord(
        date=today,
        provider=settings.llm.provider,
        model=settings.llm.model,
        answered_by=str(metadata.get("model_name") or "nieznany"),
        question=question.question,
        context_chunks=[c.metadata["chunk_id"] for c, _ in hits],
        answer=answer.strip(),
        input_tokens=int(usage["input_tokens"]),
        output_tokens=int(usage["output_tokens"]),
        reasoning_tokens=_reasoning_tokens(usage),
        estimated_input_tokens=count_tokens(prompt, settings.tokenizer.encoding),
        estimate_tokenizer=settings.tokenizer.encoding,
        assumed_output_tokens=settings.cost.scenario.output_tokens_per_answer,
        seconds=seconds,
    )


def _reasoning_tokens(usage: Mapping[str, Any]) -> int | None:
    reasoning = (usage.get("output_token_details") or {}).get("reasoning")
    return None if reasoning is None else int(reasoning)


def run_e1(
    settings: Settings,
    *,
    specs: Sequence[EmbeddingModelSpec],
    include_reference: bool = True,
    backend: Literal["memory", "pgvector"] = "memory",
    with_llm: bool = False,
    embeddings_factory: EmbeddingsFactory | None = None,
    chat_model: Runnable | None = None,
    clock: Callable[[], datetime] = _utcnow,
) -> E1Report:
    """Ewaluacja E1: referencja kursu (openai_cache) + podane modele fastembed.

    embeddings_factory(spec) zastępuje get_embeddings (testy): ten sam obiekt służy wtedy
    do pomiaru czasu i do wyszukiwania. Błąd --with-llm → E1LlmCallError z raportem bez LLM-a.
    """
    from kovalo.llm import get_embeddings, get_reference_embeddings

    provider = settings.embeddings.provider
    local = settings.embeddings.local
    use_pg = backend == "pgvector"
    if use_pg:
        # P4: bez modelu z configu wśród ocenianych Postgres byłby po cichu pominięty.
        configured = settings.embeddings.model if provider == "openai_cache" else local.model
        evaluated = (
            include_reference
            if provider == "openai_cache"
            else any(spec.model == local.model for spec in specs)
        )
        if not evaluated:
            raise ConfigError(
                f"--backend pgvector wymaga modelu z configu ({configured}) wśród ocenianych"
            )

    now = clock()
    if now.tzinfo is not None:
        now = now.astimezone(UTC)
    docs = load_corpus(settings)
    chunks = chunk_documents(docs, settings.chunking)
    texts = [c.page_content for c in chunks]
    system, encoding = settings.prompts.system, settings.tokenizer.encoding
    mp1_queries = load_eval_queries(settings.paths.eval_queries)
    ctx = _Context(
        settings=settings,
        docs=docs,
        chunks=chunks,
        mp1=mp1_queries,
        mp2=load_mp2_retrieval_queries(settings.paths.mini_project_2_questions),
        full_tokens=[
            count_tokens(build_prompt(system, docs, q.question), encoding) for q in mp1_queries
        ],
    )
    not_cached = missing_from_course_cache(
        settings.paths.embedding_cache, settings.embeddings.model, texts
    )
    runs: list[EmbeddingRun] = []
    stores: dict[str, VectorStore] = {}
    if include_reference:
        emb = get_reference_embeddings(settings=settings)
        store = build_vector_store(chunks, emb)
        pg = None
        if use_pg and provider == "openai_cache":
            pg = _pg_store(
                settings,
                emb,
                provider=provider,
                model=settings.embeddings.model,
                dimensions=settings.embeddings.dimensions,
            )
        runs.append(
            _evaluate(ctx, store, label=REFERENCE_LABEL, provider="openai_cache",
                      model=settings.embeddings.model, dimensions=settings.embeddings.dimensions,
                      size_gb=None, embed_chunks_seconds=None, query_ms_avg=None, pg_store=pg)
        )  # fmt: skip
        stores[REFERENCE_LABEL] = store

    fe_settings = _fastembed_settings(settings)
    for spec in specs:
        if embeddings_factory is not None:
            timing_emb = search_emb = embeddings_factory(spec)
        else:
            timing_emb = get_embeddings(settings=fe_settings, spec=spec, cache=False)
            search_emb = get_embeddings(settings=fe_settings, spec=spec)
        start = time.perf_counter()
        timing_emb.embed_documents(texts)
        embed_seconds = time.perf_counter() - start
        questions = [q.question for q in (*ctx.mp1, *ctx.mp2)]
        start = time.perf_counter()
        for question in questions:
            timing_emb.embed_query(question)
        query_ms = (time.perf_counter() - start) * 1000 / len(questions)

        store = build_vector_store(chunks, search_emb)
        pg = None
        if use_pg and provider == "fastembed" and spec.model == local.model:
            pg = _pg_store(
                settings,
                search_emb,
                provider=provider,
                model=spec.model,
                dimensions=spec.dimensions,
            )
        runs.append(
            _evaluate(ctx, store, label=spec.model, provider="fastembed", model=spec.model,
                      dimensions=spec.dimensions, size_gb=model_size_gb(spec.model),
                      embed_chunks_seconds=embed_seconds, query_ms_avg=query_ms, pg_store=pg)
        )  # fmt: skip
        stores[spec.model] = store

    report = E1Report(
        run_id=now.strftime("%Y%m%d-%H%M%S"),
        created_at=now,
        corpus=corpus_stats(docs, encoding=encoding),
        chunks=chunk_stats(chunks),
        chunks_not_in_course_cache=len(not_cached),
        k=settings.retrieval.top_k,
        targets=dict(TARGETS),
        our_prices_checked_on=settings.cost.our_prices_checked_on,
        cost_scenario=settings.cost.scenario,
        runs=runs,
        configured_model=local.model,
        recommended_model=recommend_model(runs),
        llm_call=None,
    )
    if not with_llm or not stores:
        return report

    # Chunki modelu z configu (w trybie openai_cache to referencja); bez niego - pierwszego wiersza.
    configured_label = REFERENCE_LABEL if provider == "openai_cache" else local.model
    llm_store = stores.get(configured_label) or next(iter(stores.values()))
    try:
        record = _llm_call(
            settings, llm_store, mp1_queries[0], chat_model=chat_model, today=now.date()
        )
    except Exception as exc:  # błąd dostawcy, brak klucza, brak usage_metadata
        raise E1LlmCallError(report, exc) from exc
    return report.model_copy(update={"llm_call": record})


def _totals(run: EmbeddingRun) -> tuple[int, int, float]:
    return (
        run.mp1.hits_at_3 + run.mp2.hits_at_3,
        run.mp1.hits_at_1 + run.mp2.hits_at_1,
        run.mp1.mrr + run.mp2.mrr,
    )


def recommend_model(runs: Sequence[EmbeddingRun]) -> str | None:
    """Reguła wyboru modelu z kontraktu F1, 7.3 (deterministyczna).

    1. Kandydaci: wiersze fastembed spełniające cel; gdy żaden, wszystkie fastembed.
    2. best = maksimum po (hit@3 MP1+MP2, hit@1 MP1+MP2, MRR MP1+MP2).
    3. Spośród kandydatów co najwyżej o 1 trafienie gorszych od best w hit@3 i w hit@1
       wygrywa najmniejszy size_gb (remis → kolejność z punktu 2).
    """
    fastembed = [r for r in runs if r.provider == "fastembed"]
    if not fastembed:
        return None
    candidates = [r for r in fastembed if r.meets_target] or fastembed
    best_h3, best_h1, _ = max(_totals(r) for r in candidates)
    close = [r for r in candidates if best_h3 - _totals(r)[0] <= 1 and best_h1 - _totals(r)[1] <= 1]

    def order(run: EmbeddingRun) -> tuple[float, tuple[int, int, float]]:
        size = run.size_gb if run.size_gb is not None else float("inf")
        h3, h1, mrr = _totals(run)
        return size, (-h3, -h1, -mrr)

    return min(close, key=order).model
