"""Program E2: odpowiedzi ze źródłami (MP2) w czterech poziomach (kontrakt F2, 8.5-8.6).

Poziomy 1-2 (wyszukiwanie i dowody) zawsze, bez LLM-a: trzy warianty (chunki MP1, chunki
z kontekstem, hybryda BM25 + wektory) x dwie strony embeddingów (cache kursu i model z configu).
Poziomy 3-4 (odpowiedź, sędzia, odwołania, odmowy) tylko z `with_llm`, dla wybranych wariantów
na jednej stronie. Port run_retrieval_eval.py, check.py i answer_questions.py kursu.
Reguła rekomendacji trybu wyszukiwania jest deterministyczna (`recommend_mode`).
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import mean
from typing import Any, Literal, cast

from langchain_core.caches import BaseCache
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from kovalo.config import ConfigError, EmbeddingModelSpec, RetrievalMode, Settings
from kovalo.cost.pricing import monthly_cost, monthly_cost_tokens, our_price, reference_price
from kovalo.cost.tokens import count_tokens
from kovalo.eval.datasets import GoldQuestion, load_gold
from kovalo.eval.rag.judge import JudgeResult, judge_record
from kovalo.eval.rag.reference import COURSE_LIVE_SUMMARY
from kovalo.eval.rag.scoring import all_required_found, score_record
from kovalo.eval.rag.summary import E2Summary, QuestionRow, summarise
from kovalo.knowledge.chunking import ChunkStats, ChunkVariant, build_chunks, chunk_stats
from kovalo.knowledge.corpus import load_corpus
from kovalo.knowledge.embeddings import missing_from_course_cache
from kovalo.knowledge.prompts import build_prompt
from kovalo.knowledge.retrieval import build_vector_store
from kovalo.knowledge.retrievers import Retriever, build_retriever
from kovalo.rag.answer import answer_question
from kovalo.rag.records import Answer, ContextItem, RunRecord, TokenUsage, Verdict

__all__ = [
    "LLM_LEVELS",
    "SIDES",
    "VARIANTS",
    "E2AllErrorsError",
    "E2Report",
    "EmbeddingsSide",
    "LlmCalls",
    "LlmVariantResult",
    "RetrievalRow",
    "RetrievalVariantResult",
    "VariantCost",
    "VariantName",
    "parse_variants",
    "plan_llm_calls",
    "recommend_mode",
    "run_e2",
    "select_questions",
]

VariantName = Literal["mp1", "ctx", "hybrid"]
EmbeddingsSide = Literal["ours", "reference"]
VARIANTS: dict[VariantName, tuple[ChunkVariant, RetrievalMode]] = {
    "mp1": ("mp1", "vector"),
    "ctx": ("ctx", "vector"),
    "hybrid": ("ctx", "hybrid"),
}
SIDES: tuple[EmbeddingsSide, ...] = ("reference", "ours")
DEFAULT_LLM_VARIANTS: tuple[VariantName, ...] = ("ctx", "hybrid")  # pytanie U1: odpowiedź A

# Poziomy porównywane z referencją kursu (cel: wynik ≥ referencja - 1; 06-evaluation).
LLM_LEVELS: tuple[str, ...] = (
    "retrieval_hit",
    "evidence_phrases_complete",
    "answered",
    "key_facts_complete",
    "judge_correct",
    "judge_grounded",
    "citations_valid",
    "cites_relevant",
    "refusals_correct",
)
_JUDGE_LEVELS = ("judge_correct", "judge_grounded")
TARGET_TOLERANCE = 1

EmbeddingsFactory = Callable[[EmbeddingModelSpec | None], Embeddings]


class RetrievalRow(BaseModel):
    """Wiersz retrieval_report.json kursu (+ nasze pola)."""

    id: str
    question: str
    relevant_docs: list[str]
    requires_all: bool
    top_chunk_ids: list[str]
    top_scores: list[float]  # round(…, 4): cosine (vector) albo RRF (hybrid)
    hit_at_1: bool
    hit_at_3: bool
    all_required_found: bool | None
    evidence_phrase_coverage: float | None  # krok 3 na rekordzie z samym kontekstem
    prompt_tokens: int  # o200k_base, build_prompt(prompts.answer_system, …)


class RetrievalVariantResult(BaseModel):
    side: EmbeddingsSide
    embedding_model: str
    variant: VariantName
    n_chunks: int
    hit_at_1: int
    hit_at_3: int
    evidence_phrases_complete: int
    all_required_found: tuple[int, int]  # (znalezione, pytań z requires_all)
    avg_prompt_tokens: float  # round(…, 1)
    rows: list[RetrievalRow]
    pg_consistent: bool | None = None  # tylko --backend pgvector: top-k z Postgresa = z pamięci


class LlmCalls(BaseModel):
    """Wywołania modelu bez trafień w cache (answer, judge), trafienia i nieudane wywołania."""

    answer: int = 0
    judge: int = 0
    from_cache: int = 0
    errors: int = 0


class LlmVariantResult(BaseModel):
    side: EmbeddingsSide
    variant: VariantName
    summary: E2Summary
    rows: list[QuestionRow]
    calls: LlmCalls
    answered_by: dict[str, int]  # model → liczba rekordów (fallback widoczny)
    judged_by: dict[str, int] = {}  # model sędziego → liczba ocen (z odpowiedzi dostawcy)
    avg_usage: TokenUsage | None  # średnie tokeny dostawcy (odpowiedzi, bez sędziego)
    meets_target: dict[str, bool]  # poziom → wynik ≥ referencja - 1
    from_cache: list[str] = []  # NASZE: id pytań, których odpowiedź przyszła z cache LLM-a


class VariantCost(BaseModel):
    variant: VariantName
    avg_prompt_tokens: float  # o200k_base (strona ours, bez niej reference)
    reference_monthly_usd: float  # ceny referencyjne (PINNED), 350 tokenów wyjścia
    ours_monthly_usd: float  # cost.our_model, 350 tokenów wyjścia
    ours_actual_monthly_usd: float | None  # faktyczne tokeny z avg_usage; None bez LLM-a


class E2Report(BaseModel):
    run_id: str  # YYYYMMDD-HHMMSS-<8 hex> (UTC)
    created_at: datetime
    provider: str
    model: str
    judge_model: str
    reference_model: str  # settings.reference_llm.model
    embedding_model: str  # model strony ours (embeddings.local.model)
    reference_embedding_model: str  # embeddings.model (cache kursu)
    our_cost_model: str  # cost.our_model
    our_prices_checked_on: str
    k: int
    judge: bool
    chunk_stats: dict[ChunkVariant, ChunkStats]
    chunks_not_in_course_cache: dict[ChunkVariant, int]
    retrieval: list[RetrievalVariantResult]
    llm: list[LlmVariantResult]
    llm_side: EmbeddingsSide | None  # strona poziomów 3-4 (None bez LLM-a)
    llm_calls_planned: int
    cache: dict[str, int] | None
    costs: list[VariantCost]
    recommended_mode: RetrievalMode | None
    configured_mode: RetrievalMode


class E2AllErrorsError(RuntimeError):
    """W którymś wariancie z LLM-em wszystkie odpowiedzi to błędy; `report` jest kompletny."""

    def __init__(self, report: E2Report, result: LlmVariantResult) -> None:
        first = next((r.error for r in result.rows if r.error), None)
        super().__init__(
            f"Żadna odpowiedź się nie udała ({result.side} {result.variant}, przebieg "
            f"{report.run_id}), np.: {first}"
        )
        self.report = report
        self.result = result
        self.first_error = first


def _utcnow() -> datetime:
    return datetime.now(UTC)


def plan_llm_calls(n_questions: int, n_variants: int, *, judge: bool) -> int:
    """Górna granica wywołań bez cache: pytania x warianty x (2 z sędzią, inaczej 1)."""
    return n_questions * n_variants * (2 if judge else 1)


def parse_variants(names: Sequence[str]) -> list[VariantName]:
    """Nazwy wariantów bez powtórzeń; nieznana → ConfigError z listą dozwolonych."""
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise ConfigError(
            f"Nieznany wariant: {', '.join(unknown)}. Dozwolone: {', '.join(VARIANTS)}"
        )
    if not names:
        raise ConfigError(f"Pusta lista wariantów. Dozwolone: {', '.join(VARIANTS)}")
    return [cast(VariantName, n) for n in dict.fromkeys(names)]


def select_questions(gold: Sequence[GoldQuestion], ids: Sequence[str] | None) -> list[GoldQuestion]:
    """Pytania o podanych id (kolejność zestawu); None → wszystkie. Nieznane id → ConfigError."""
    if ids is None:
        return list(gold)
    known = [q.id for q in gold]
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ConfigError(f"Nieznane id pytań: {', '.join(unknown)}. Znane: {', '.join(known)}")
    wanted = set(ids)
    return [q for q in gold if q.id in wanted]


def _judge_model_name(settings: Settings) -> str:
    override = settings.llm.purposes.get("judge")
    return override.model if override and override.model else settings.llm.model


@dataclass(frozen=True)
class _Side:
    name: EmbeddingsSide
    embedding_model: str
    retrievers: dict[VariantName, Retriever]
    pg_consistent: dict[VariantName, bool]


def _retrieval_row(
    settings: Settings, retriever: Retriever, gold: GoldQuestion, k: int
) -> RetrievalRow:
    hits = retriever.retrieve(gold.question, k)
    docs = [doc for doc, _ in hits]
    doc_ids = [str(d.metadata["doc_id"]) for d in docs]
    context = [
        ContextItem(
            doc_id=str(d.metadata["doc_id"]),
            chunk_id=str(d.metadata["chunk_id"]),
            text=d.page_content,
        )
        for d in docs
    ]
    # Poziom 2 jak w run_retrieval_eval.py kursu: krok 3 na rekordzie, który ma tylko kontekst.
    record = RunRecord(
        question=gold.question,
        context=context,
        output=Answer(answer="", citations=[], answerable=False),
    )
    prompt = build_prompt(settings.prompts.answer_system, docs, gold.question)
    return RetrievalRow(
        id=gold.id,
        question=gold.question,
        relevant_docs=list(gold.relevant_docs),
        requires_all=gold.requires_all,
        top_chunk_ids=[str(d.metadata["chunk_id"]) for d in docs],
        top_scores=[round(float(score), 4) for _, score in hits],
        hit_at_1=bool(doc_ids) and doc_ids[0] in gold.relevant_docs,
        hit_at_3=any(d in gold.relevant_docs for d in doc_ids),
        all_required_found=all_required_found(doc_ids, gold),
        evidence_phrase_coverage=score_record(record, gold).evidence_phrase_coverage,
        prompt_tokens=count_tokens(prompt, settings.tokenizer.encoding),
    )


def _evaluate_retrieval(
    settings: Settings,
    side: _Side,
    variant: VariantName,
    questions: Sequence[GoldQuestion],
    n_chunks: int,
) -> RetrievalVariantResult:
    k = settings.retrieval.top_k
    rows = [_retrieval_row(settings, side.retrievers[variant], q, k) for q in questions]
    required = [r.all_required_found for r in rows if r.all_required_found is not None]
    return RetrievalVariantResult(
        side=side.name,
        embedding_model=side.embedding_model,
        variant=variant,
        n_chunks=n_chunks,
        hit_at_1=sum(r.hit_at_1 for r in rows),
        hit_at_3=sum(r.hit_at_3 for r in rows),
        evidence_phrases_complete=sum(r.evidence_phrase_coverage == 1 for r in rows),
        all_required_found=(sum(required), len(required)),
        avg_prompt_tokens=round(mean(r.prompt_tokens for r in rows), 1),
        rows=rows,
        pg_consistent=side.pg_consistent.get(variant),
    )


def _memory_retrievers(
    settings: Settings, chunks: Mapping[ChunkVariant, Sequence[Document]], embeddings: Embeddings
) -> dict[VariantName, Retriever]:
    stores = {
        variant: build_vector_store(chunks[variant], embeddings) for variant in ("mp1", "ctx")
    }
    return {
        name: build_retriever(
            chunks[chunk_variant],
            stores[chunk_variant],
            mode=mode,
            hybrid=settings.retrieval.hybrid,
        )
        for name, (chunk_variant, mode) in VARIANTS.items()
    }


def _pg_retrievers(
    settings: Settings,
    embeddings: Embeddings,
    memory: Mapping[VariantName, Retriever],
    questions: Sequence[GoldQuestion],
) -> tuple[dict[VariantName, Retriever], dict[VariantName, bool]]:
    """Retrievery na Postgresie (kovalo_mp1/kovalo_ctx) i zgodność top-k z pamięcią."""
    from kovalo.knowledge.retrievers import open_retriever

    k = settings.retrieval.top_k
    retrievers: dict[VariantName, Retriever] = {}
    consistent: dict[VariantName, bool] = {}
    for name, (chunk_variant, mode) in VARIANTS.items():
        pg = open_retriever(settings, embeddings, variant=chunk_variant, mode=mode)

        def top(retriever: Retriever, question: str) -> list[str]:
            return [str(d.metadata["chunk_id"]) for d, _ in retriever.retrieve(question, k)]

        consistent[name] = all(
            top(pg, q.question) == top(memory[name], q.question) for q in questions
        )
        retrievers[name] = pg
    return retrievers, consistent


def _cache_counts(cache: BaseCache | None) -> tuple[int, int] | None:
    hits, misses = getattr(cache, "hits", None), getattr(cache, "misses", None)
    if isinstance(hits, int) and isinstance(misses, int):
        return hits, misses
    return None


class _CallCounter:
    """Klasyfikuje wywołania modelu: sieciowe (answer/judge), z cache, nieudane."""

    def __init__(self, cache: BaseCache | None) -> None:
        self.cache = cache
        self.calls = LlmCalls()
        self.last_from_cache = False

    def snapshot(self) -> tuple[int, int] | None:
        return _cache_counts(self.cache)

    def record(
        self, kind: Literal["answer", "judge"], before: tuple[int, int] | None, error: str | None
    ) -> None:
        after = _cache_counts(self.cache)
        from_cache = before is not None and after is not None and after[0] > before[0]
        cache_miss_only = error is not None and error.startswith("LlmCacheMiss")
        self.last_from_cache = from_cache
        if from_cache:
            self.calls.from_cache += 1
        elif not cache_miss_only:
            setattr(self.calls, kind, getattr(self.calls, kind) + 1)
        if error is not None:
            self.calls.errors += 1


def _avg_usage(records: Sequence[RunRecord]) -> TokenUsage | None:
    usages = [r.usage for r in records if r.usage is not None]
    if not usages:
        return None
    reasoning = [u.reasoning_tokens for u in usages if u.reasoning_tokens is not None]
    return TokenUsage(
        input_tokens=round(mean(u.input_tokens for u in usages)),
        output_tokens=round(mean(u.output_tokens for u in usages)),
        reasoning_tokens=round(mean(reasoning)) if reasoning else None,
    )


def _meets_target(summary: E2Summary, *, judge: bool) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for level in LLM_LEVELS:
        if not judge and level in _JUDGE_LEVELS:
            continue
        ours = cast(tuple[int, int], getattr(summary, level))
        reference = cast(tuple[int, int], COURSE_LIVE_SUMMARY[level])
        result[level] = ours[0] >= reference[0] - TARGET_TOLERANCE
    return result


def _run_llm_variant(
    settings: Settings,
    side: _Side,
    variant: VariantName,
    questions: Sequence[GoldQuestion],
    *,
    answer_model: Runnable[Any, Any],
    judge_model: Runnable[Any, Any] | None,
    cache: BaseCache | None,
) -> LlmVariantResult:
    chunk_variant, _ = VARIANTS[variant]
    retriever = side.retrievers[variant]
    counter = _CallCounter(cache)
    rows: list[QuestionRow] = []
    cached_ids: list[str] = []
    for gold in questions:  # sekwencyjnie: limiter i tak serializuje wywołania
        before = counter.snapshot()
        record = answer_question(
            gold.question,
            retriever,
            answer_model,
            system_prompt=settings.prompts.answer_system,
            k=settings.retrieval.top_k,
            chunks=chunk_variant,
        )
        counter.record("answer", before, record.error)
        if counter.last_from_cache:
            cached_ids.append(gold.id)
        score = score_record(record, gold)
        verdict = JudgeResult()
        output = record.output
        if judge_model is not None and record.error is None and output and output.answerable:
            before = counter.snapshot()
            verdict = judge_record(
                record, gold, judge_model, system_prompt=settings.prompts.judge_system
            )
            counter.record("judge", before, verdict.error)
        rows.append(
            QuestionRow(
                id=gold.id,
                question=gold.question,
                answerable=gold.answerable,
                retrieval_hit=any(c.doc_id in gold.relevant_docs for c in record.context),
                answer=output.answer if output else None,
                citations=output.citations if output else None,
                error=record.error,
                score=score,
                verdict=verdict,
                record=record,
            )
        )
    summary = summarise(rows)
    answered_by = Counter(r.record.answered_by for r in rows if r.record.answered_by)
    judged_by = Counter(r.verdict.judged_by for r in rows if r.verdict.judged_by)
    return LlmVariantResult(
        side=side.name,
        variant=variant,
        summary=summary,
        rows=rows,
        calls=counter.calls,
        answered_by=dict(answered_by),
        judged_by=dict(judged_by),
        avg_usage=_avg_usage([r.record for r in rows]),
        meets_target=_meets_target(summary, judge=judge_model is not None),
        from_cache=cached_ids,
    )


def _costs(
    settings: Settings,
    retrieval: Sequence[RetrievalVariantResult],
    llm: Sequence[LlmVariantResult],
) -> list[VariantCost]:
    cost = settings.cost
    ref_price, ours_price = reference_price(cost), our_price(cost)
    by_key = {(r.side, r.variant): r for r in retrieval}
    usage = {r.variant: r.avg_usage for r in llm}
    costs: list[VariantCost] = []
    for variant in VARIANTS:
        row = by_key.get(("ours", variant)) or by_key.get(("reference", variant))
        if row is None:
            continue
        tokens = row.avg_prompt_tokens
        actual = usage.get(variant)
        costs.append(
            VariantCost(
                variant=variant,
                avg_prompt_tokens=tokens,
                reference_monthly_usd=round(
                    monthly_cost(tokens, price=ref_price, scenario=cost.scenario), 2
                ),
                ours_monthly_usd=round(
                    monthly_cost(tokens, price=ours_price, scenario=cost.scenario), 2
                ),
                ours_actual_monthly_usd=(
                    round(
                        monthly_cost_tokens(
                            actual.input_tokens,
                            actual.output_tokens,
                            price=ours_price,
                            scenario=cost.scenario,
                        ),
                        2,
                    )
                    if actual is not None
                    else None
                ),
            )
        )
    return costs


def run_e2(
    settings: Settings,
    *,
    variants: Sequence[VariantName] = DEFAULT_LLM_VARIANTS,
    llm_side: EmbeddingsSide = "ours",
    with_llm: bool = False,
    judge: bool = True,
    ids: Sequence[str] | None = None,
    backend: Literal["memory", "pgvector"] = "memory",
    cache: BaseCache | None = None,
    answer_model: Runnable[Any, Any] | None = None,
    judge_model: Runnable[Any, Any] | None = None,
    embeddings_factory: EmbeddingsFactory | None = None,
    clock: Callable[[], datetime] = _utcnow,
) -> E2Report:
    """Ewaluacja E2 (kontrakt F2, 8.5). Bez `with_llm` zero wywołań LLM-a.

    `variants`, `llm_side`, `ids`, `judge` dotyczą tylko poziomów 3-4. Modele są budowane raz,
    przed pierwszym wywołaniem (brak klucza → ConfigError). Gdy w którymś wariancie wszystkie
    odpowiedzi to błędy → E2AllErrorsError z kompletnym raportem.
    """
    variant_names = parse_variants(list(variants))
    gold = load_gold(settings.paths.mini_project_2_questions)
    questions = select_questions(gold, ids)
    answerable = [q for q in gold if q.answerable]
    ours_enabled = settings.embeddings.provider == "fastembed"
    if with_llm and llm_side == "ours" and not ours_enabled:
        raise ConfigError(
            "Tryb embeddings.provider=openai_cache nie ma strony ours: użyj --llm-side reference."
        )

    if with_llm:
        from kovalo.llm import get_structured_model

        answer_model = answer_model or get_structured_model(
            Answer, "answer", settings=settings, cache=cache
        )
        if judge:
            judge_model = judge_model or get_structured_model(
                Verdict, "judge", settings=settings, cache=cache
            )

    now = clock()
    now = now.astimezone(UTC) if now.tzinfo is not None else now
    run_id = f"{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

    docs = load_corpus(settings)
    chunks: dict[ChunkVariant, list[Document]] = {
        variant: build_chunks(docs, settings.chunking, variant) for variant in ("mp1", "ctx")
    }
    not_cached = {
        variant: len(
            missing_from_course_cache(
                settings.paths.embedding_cache,
                settings.embeddings.model,
                [c.page_content for c in variant_chunks],
            )
        )
        for variant, variant_chunks in chunks.items()
    }

    from kovalo.llm import get_embeddings, get_reference_embeddings

    sides: dict[EmbeddingsSide, _Side] = {}
    reference_emb = get_reference_embeddings(settings=settings)
    sides["reference"] = _Side(
        name="reference",
        embedding_model=settings.embeddings.model,
        retrievers=_memory_retrievers(settings, chunks, reference_emb),
        pg_consistent={},
    )
    if ours_enabled:
        ours_emb = (
            embeddings_factory(None)
            if embeddings_factory is not None
            else get_embeddings(settings=settings)
        )
        retrievers = _memory_retrievers(settings, chunks, ours_emb)
        consistent: dict[VariantName, bool] = {}
        if backend == "pgvector":
            retrievers, consistent = _pg_retrievers(settings, ours_emb, retrievers, answerable)
        sides["ours"] = _Side(
            name="ours",
            embedding_model=settings.embeddings.local.model,
            retrievers=retrievers,
            pg_consistent=consistent,
        )
    elif backend == "pgvector":
        retrievers, consistent = _pg_retrievers(
            settings, reference_emb, sides["reference"].retrievers, answerable
        )
        sides["reference"] = _Side(
            name="reference",
            embedding_model=settings.embeddings.model,
            retrievers=retrievers,
            pg_consistent=consistent,
        )

    retrieval = [
        _evaluate_retrieval(settings, side, variant, answerable, len(chunks[VARIANTS[variant][0]]))
        for side in sides.values()
        for variant in VARIANTS
    ]

    llm: list[LlmVariantResult] = []
    if with_llm:
        assert answer_model is not None  # zbudowany wyżej
        for variant in variant_names:
            llm.append(
                _run_llm_variant(
                    settings,
                    sides[llm_side],
                    variant,
                    questions,
                    answer_model=answer_model,
                    judge_model=judge_model if judge else None,
                    cache=cache,
                )
            )

    counts = _cache_counts(cache) if with_llm else None
    report = E2Report(
        run_id=run_id,
        created_at=now,
        provider=settings.llm.provider,
        model=settings.llm.model,
        judge_model=_judge_model_name(settings),
        reference_model=settings.reference_llm.model,
        embedding_model=settings.embeddings.local.model,
        reference_embedding_model=settings.embeddings.model,
        our_cost_model=settings.cost.our_model,
        our_prices_checked_on=settings.cost.our_prices_checked_on.isoformat(),
        k=settings.retrieval.top_k,
        judge=judge,
        chunk_stats={variant: chunk_stats(c) for variant, c in chunks.items()},
        chunks_not_in_course_cache=not_cached,
        retrieval=retrieval,
        llm=llm,
        llm_side=llm_side if with_llm else None,
        llm_calls_planned=(
            plan_llm_calls(len(questions), len(variant_names), judge=judge) if with_llm else 0
        ),
        cache={"hits": counts[0], "misses": counts[1]} if counts is not None else None,
        costs=_costs(settings, retrieval, [r for r in llm if r.side == "ours"] or llm),
        recommended_mode=None,
        configured_mode=settings.retrieval.mode,
    )
    report = report.model_copy(update={"recommended_mode": recommend_mode(report)})
    for result in report.llm:
        if result.rows and all(r.score.outcome == "error" for r in result.rows):
            raise E2AllErrorsError(report, result)
    return report


def _retrieval_metrics(result: RetrievalVariantResult) -> tuple[int, ...]:
    return (result.hit_at_1, result.hit_at_3, result.evidence_phrases_complete)


def _llm_metrics(result: LlmVariantResult) -> tuple[int, ...]:
    s = result.summary
    return (s.answered[0], s.key_facts_complete[0], s.judge_correct[0])


def recommend_mode(report: E2Report) -> RetrievalMode | None:
    """Reguła z kontraktu F2, 8.5 (strona ours): 'hybrid', gdy hybryda nie jest gorsza od ctx
    w żadnej mierze (hit@1, hit@3, komplet dowodów; przy wynikach LLM-a obu wariantów także
    answered, key_facts_complete, judge_correct) i lepsza w co najmniej jednej; inaczej 'vector'.
    Brak strony ours → None."""
    retrieval = {r.variant: r for r in report.retrieval if r.side == "ours"}
    if "ctx" not in retrieval or "hybrid" not in retrieval:
        return None
    ctx = list(_retrieval_metrics(retrieval["ctx"]))
    hybrid = list(_retrieval_metrics(retrieval["hybrid"]))
    llm = {r.variant: r for r in report.llm if r.side == "ours"}
    if "ctx" in llm and "hybrid" in llm:
        ctx += _llm_metrics(llm["ctx"])
        hybrid += _llm_metrics(llm["hybrid"])
    not_worse = all(h >= c for h, c in zip(hybrid, ctx, strict=True))
    better = any(h > c for h, c in zip(hybrid, ctx, strict=True))
    return "hybrid" if not_worse and better else "vector"
