"""Program E4: agent z narzędziami (MP4; kontrakt F4, 11.3-11.5; ADR-0011 pkt 8).

Zawsze, zero wywołań LLM-a: (a) przeliczenie zapisanych przebiegów kursu (jeden agent i zespół)
naszym `score_trace` (musi dać wyniki kursu co do pytania), (b) odtworzenie wywołań przebiegu
kursu skryptowym modelem na NASZYCH narzędziach i danych. Na żywo (`live=True`): nasz agent
z modelem z configu (albo podanym, w testach), z cache LLM-a i limitem wywołań sieciowych
(`--max-calls`, CallMeter); po wyczerpaniu limitu reszta pytań jest „pominięta”.
Wyszukiwanie: indeks w pamięci z chunków `ctx` i embeddingów z configu (llm.get_embeddings),
budowany leniwie przy pierwszym wyszukiwaniu; `backend='pgvector'`: open_retriever.
F5 (kontrakt F5, 10.6; ADR-0014 pkt 4-5): `transport` inny niż `direct` → agent na żywo bierze
narzędzia przez MCP (odpowiednik `run_agent_mcp.py`; przeliczenie i odtworzenie kursu zostają
bezpośrednie); `count_errors=True` → nieudane próby zjadają `max_calls`.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any, Literal

from langchain_core.caches import BaseCache
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, field_validator

from kovalo.agents import Middleware
from kovalo.agents.agent import AgentRunner, build_agent
from kovalo.agents.middleware import CallMeter
from kovalo.agents.trace import Trace
from kovalo.company import load_company_data
from kovalo.config import ConfigError, Settings
from kovalo.cost.pricing import our_price, reference_price, tokens_cost
from kovalo.eval.datasets import AgentGold, load_agent_gold
from kovalo.eval.trace import reference
from kovalo.eval.trace.replay import ScriptedToolModel, script_from_trace
from kovalo.eval.trace.scoring import score_trace
from kovalo.knowledge.retrieval import ScoredChunk
from kovalo.knowledge.retrievers import Retriever
from kovalo.mcp_client import McpConnection
from kovalo.tools.langchain import langchain_tools
from kovalo.tools.specs import ToolContext

__all__ = [
    "SKIPPED_ERROR",
    "Backend",
    "E4AllErrorsError",
    "E4Calls",
    "E4Mode",
    "E4Report",
    "E4Run",
    "E4Transport",
    "LazySearch",
    "plan_llm_calls",
    "replay_problems",
    "run_e4",
    "select_questions",
    "summarize",
]

E4Mode = Literal["course", "course_supervisor", "replay", "live"]
Backend = Literal["memory", "pgvector"]
E4Transport = Literal["direct", "memory", "stdio", "http"]
TRANSPORTS: tuple[str, ...] = ("direct", "memory", "stdio", "http")
Variant = Literal["course", "ours"]

SKIPPED_ERROR = "pominięte: wyczerpany limit --max-calls"
EXPECTED_PENALTIES = {"P1": 729600, "P2": 362400, "P3": 0}  # grosze (kryterium 1 F4)
_TURNS_BY_KIND = {"penalty": 3, "termination": 3}  # pozostałe: 2 tury modelu


class E4Calls(BaseModel):
    network: int = 0  # wywołania sieciowe modelu (CallMeter)
    from_cache: int = 0
    errors: int = 0
    tool_calls: int = 0  # zdarzenia trace (ze składowymi)
    model_turns: int = 0  # sum(len(trace.model_calls))


class E4Run(BaseModel):
    mode: E4Mode
    run: dict[str, Any]  # format 'run' z save_traces kursu (+ NASZE tools_variant)
    summary: dict[str, Any]  # liczby z format_report kursu (P3)
    rows: list[dict[str, Any]]  # {'trace': Trace, 'score': dict}
    calls: E4Calls
    answered_by: dict[str, int] = {}  # model → liczba tur (tylko tury z model_name)
    seconds: float

    @field_validator("rows", mode="after")
    @classmethod
    def _traces(cls, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {**row, "trace": Trace.model_validate(row["trace"])}
            if not isinstance(row.get("trace"), Trace)
            else row
            for row in rows
        ]


class E4Report(BaseModel):
    created_at: datetime
    questions: list[str]
    backend: Backend
    transport: E4Transport = "direct"  # F5: narzędzia agenta na żywo (direct albo przez MCP)
    embedding_model: str | None
    reference: dict[str, Any]  # course_e4(...)
    course: E4Run
    course_supervisor: E4Run
    replay: E4Run
    live: E4Run | None = None
    llm_calls_planned: dict[str, int]
    max_calls: int | None = None
    stopped_by_max_calls: bool = False
    live_cost_cents: dict[str, float] | None = None  # {'ours', 'reference'}
    our_cost_model: str
    our_prices_checked_on: date
    reference_cost_model: str


class E4AllErrorsError(RuntimeError):
    """Każde (niepominięte) pytanie na żywo zakończyło się błędem; `report` jest kompletny."""

    def __init__(self, report: E4Report) -> None:
        live = report.live
        traces = [r["trace"] for r in live.rows] if live else []
        first = next((t.error for t in traces if t.error and t.error != SKIPPED_ERROR), None)
        run_id = live.run["run_id"] if live else "-"
        super().__init__(f"Żadne pytanie na żywo się nie udało (przebieg {run_id}), np.: {first}")
        self.report = report
        self.run_id = run_id
        self.first_error = first


# ---------------------------------------------------------------------------
# Plan, wybór pytań, podsumowanie
# ---------------------------------------------------------------------------


def plan_llm_calls(questions: Sequence[AgentGold], *, max_steps: int = 24) -> dict[str, int]:
    """{'estimate': tury modelu (kara i warunek rozwiązania 3, pozostałe 2),
    'upper_bound': len x ceil(agent.max_steps / 2)}."""
    estimate = sum(_TURNS_BY_KIND.get(q.kind, 2) for q in questions)
    return {"estimate": estimate, "upper_bound": len(questions) * math.ceil(max_steps / 2)}


def select_questions(gold: Sequence[AgentGold], ids: Sequence[str] | None) -> list[AgentGold]:
    """Pytania o podanych id (kolejność pliku); None → wszystkie; nieznane id → ConfigError."""
    if ids is None:
        return list(gold)
    known = [g.id for g in gold]
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ConfigError(f"Nieznane id pytań E4: {', '.join(unknown)}. Znane: {', '.join(known)}")
    wanted = set(ids)
    return [g for g in gold if g.id in wanted]


def _count(scores: Sequence[dict[str, Any]], key: str, ok: Any) -> list[int]:
    assessed = [s[key] for s in scores if s[key] is not None]
    return [sum(1 for v in assessed if ok(v)), len(assessed)]


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Liczby z format_report kursu: [trafione, ocenione], liczby bez źródła, wyniki, tokeny."""
    traces: list[Trace] = [r["trace"] for r in rows]
    scores: list[dict[str, Any]] = [r["score"] for r in rows]
    return {
        "tools": _count(scores, "tool_coverage", lambda v: v == 1),
        "args": _count(scores, "arguments_correct", bool),
        "facts": _count(scores, "answer_fact_coverage", lambda v: v == 1),
        "approval": [sum(1 for s in scores if s["approval_claim"]), len(scores)],
        "uncovered": sum(len(s["uncovered_numbers"]) for s in scores),
        "user_stated": sum(len(s["user_stated_numbers"]) for s in scores),
        "outcomes": {
            k: sum(1 for s in scores if s["outcome"] == k) for k in ("answered", "error", "limit")
        },
        "tool_calls": sum(s["tool_calls"] for s in scores),
        "failed": sum(s["failed_calls"] for s in scores),
        "tokens_in": sum(t.input_tokens for t in traces),
        "tokens_out": sum(t.output_tokens for t in traces),
    }


def _answered_by(traces: Sequence[Trace]) -> dict[str, int]:
    return dict(Counter(c.model for t in traces for c in t.model_calls if c.model))


def _make_run(
    mode: E4Mode,
    run: dict[str, Any],
    traces: Sequence[Trace],
    scores: Sequence[dict[str, Any]],
    calls: E4Calls,
    seconds: float,
) -> E4Run:
    rows = [{"trace": t, "score": s} for t, s in zip(traces, scores, strict=True)]
    calls = calls.model_copy(
        update={
            "tool_calls": sum(len(t.events) for t in traces),
            "model_turns": sum(len(t.model_calls) for t in traces),
        }
    )
    return E4Run(
        mode=mode,
        run=run,
        summary=summarize(rows),
        rows=rows,
        calls=calls,
        answered_by=_answered_by(traces),
        seconds=round(seconds, 2),
    )


def replay_problems(report: E4Report) -> list[str]:
    """Odtworzenie musi (kontrakt 11.3): każde pytanie answered, liczba wywołań jak w kursie,
    zero nieudanych, argumenty poprawne, kary P1-P3 = 729 600 / 362 400 / 0 groszy."""
    replay, course = report.replay, report.course
    problems = []
    outcomes = replay.summary["outcomes"]
    if outcomes["answered"] != len(replay.rows):
        problems.append(f"nie wszystkie pytania answered: {outcomes}")
    if replay.summary["tool_calls"] != course.summary["tool_calls"]:
        problems.append(
            f"wywołań {replay.summary['tool_calls']}, w kursie {course.summary['tool_calls']}"
        )
    if replay.summary["failed"]:
        problems.append(f"nieudane wywołania: {replay.summary['failed']}")
    args = replay.summary["args"]
    if args[0] != args[1]:
        problems.append(f"argumenty {args[0]}/{args[1]}")
    for row in replay.rows:
        trace: Trace = row["trace"]
        expected = EXPECTED_PENALTIES.get(trace.question_id or "")
        if expected is None:
            continue
        got = [
            e.result.get("penalty_grosze")
            for e in trace.events
            if e.tool == "calculate_otif_penalty" and isinstance(e.result, dict)
        ]
        if any(g != expected for g in got):
            problems.append(f"{trace.question_id}: penalty_grosze {got}, oczekiwano {expected}")
    return problems


# ---------------------------------------------------------------------------
# Wyszukiwanie i agent
# ---------------------------------------------------------------------------


class LazySearch:
    """Retriever budowany przy pierwszym wyszukiwaniu (wspólny dla odtworzenia i przebiegu na
    żywo): brak klucza albo błąd modelu wychodzi, zanim załadujemy model embeddingów."""

    def __init__(self, settings: Settings, backend: Backend) -> None:
        self.settings = settings
        self.backend = backend
        self._retriever: Retriever | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"e4-{self.backend}"

    def _open(self) -> Retriever:
        from kovalo.llm import get_embeddings

        embeddings = get_embeddings(settings=self.settings)
        if self.backend == "pgvector":
            from kovalo.knowledge.retrievers import open_retriever

            return open_retriever(self.settings, embeddings)
        from kovalo.knowledge.retrievers import build_memory_retriever

        return build_memory_retriever(self.settings, embeddings)

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        with self._lock:
            if self._retriever is None:
                self._retriever = self._open()
            retriever = self._retriever
        return retriever.retrieve(question, k)


def _context(settings: Settings, retriever: Retriever) -> ToolContext:
    return ToolContext(
        data=load_company_data(settings.paths.company_data_dir),
        policy=settings.policy,
        company=settings.company,
        retriever=retriever,
        top_k=settings.retrieval.top_k,
    )


def _live_runner(
    settings: Settings,
    *,
    retriever: Retriever,
    cache: BaseCache | None,
    offline: bool,
    variant: Variant,
    meter: CallMeter,
    model: BaseChatModel | None,
    tools: Sequence[BaseTool] | None = None,
) -> AgentRunner:
    """Agent na żywo; `tools` podane (narzędzia MCP) → zamiast bezpośrednich wariantu."""
    extra: list[Middleware] = [meter]
    if model is None:
        from kovalo.workflow.factory import build_agent_runner

        return build_agent_runner(
            settings,
            retriever=retriever,
            tools=tools,
            cache=cache,
            offline=offline,
            variant=variant,
            extra_middleware=extra,
        )
    agent = build_agent(
        model,
        tools if tools is not None else langchain_tools(_context(settings, retriever), variant),
        system_prompt=settings.prompts.agent_system,
        extra_middleware=extra,
    )
    return AgentRunner(agent, settings.agent)


# ---------------------------------------------------------------------------
# Przebieg
# ---------------------------------------------------------------------------


def _run_id(now: datetime) -> str:
    return f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _run_info(
    settings: Settings, now: datetime, *, provider: str | None, model: str | None, variant: str
) -> dict[str, Any]:
    return {
        "run_id": _run_id(now),
        "date": now.date().isoformat(),
        "provider": provider,
        "model": model,
        "architecture": "single agent",
        "policy_version": settings.policy.version,
        "terms_version": load_company_data(settings.paths.company_data_dir).terms_version,
        "budget": {"max_steps": settings.agent.max_steps, "timeout_s": settings.agent.timeout_s},
        "tools_variant": variant,
    }


def _course_run(
    settings: Settings, which: reference.CourseRun, mode: E4Mode, wanted: set[str]
) -> E4Run:
    start = time.perf_counter()
    run, traces, scores = reference.rescore_course(settings, which)
    picked = [(t, s) for t, s in zip(traces, scores, strict=True) if t.question_id in wanted]
    return _make_run(
        mode,
        run,
        [t for t, _ in picked],
        [s for _, s in picked],
        E4Calls(),
        time.perf_counter() - start,
    )


def _replay(
    settings: Settings,
    questions: Sequence[AgentGold],
    retriever: Retriever,
    now: datetime,
) -> E4Run:
    start = time.perf_counter()
    course_run, course_traces, _ = reference.rescore_course(settings, "single")
    by_id = {t.question_id: t for t in course_traces}
    tools = langchain_tools(_context(settings, retriever), "course")
    traces, scores = [], []
    for gold in questions:
        theirs = by_id[gold.id]
        agent = build_agent(
            ScriptedToolModel(script=script_from_trace(theirs)),
            tools,
            system_prompt=settings.prompts.agent_system,
        )
        trace = AgentRunner(agent, settings.agent)(gold.question, question_id=gold.id)
        traces.append(trace)
        scores.append(score_trace(trace, gold))
    info = _run_info(
        settings,
        now,
        provider="scripted",
        model=f"odtworzenie przebiegu kursu {course_run['run_id']}",
        variant="course",
    )
    return _make_run("replay", info, traces, scores, E4Calls(), time.perf_counter() - start)


def _skipped(gold: AgentGold) -> Trace:
    return Trace(question_id=gold.id, question=gold.question, status="error", error=SKIPPED_ERROR)


def _cost_cents(settings: Settings, tokens_in: int, tokens_out: int) -> dict[str, float]:
    cost = settings.cost
    return {
        side: round(100 * tokens_cost(tokens_in, tokens_out, price=price), 4)
        for side, price in (("ours", our_price(cost)), ("reference", reference_price(cost)))
    }


def _embedding_model(settings: Settings) -> str:
    emb = settings.embeddings
    return emb.local.model if emb.provider == "fastembed" else emb.model


def _live_traces(runner: AgentRunner, questions: Sequence[AgentGold]) -> tuple[list[Trace], bool]:
    """Pytania po kolei; CallBudgetExhausted → to pytanie i następne „pominięte”."""
    traces: list[Trace] = []
    stopped = False
    for gold in questions:
        if stopped:
            traces.append(_skipped(gold))
            continue
        trace = runner(gold.question, question_id=gold.id)
        if trace.status == "error" and (trace.error or "").startswith("CallBudgetExhausted"):
            stopped = True
            trace = _skipped(gold)
        traces.append(trace)
    return traces, stopped


def run_e4(
    settings: Settings,
    *,
    live: bool = False,
    ids: Sequence[str] | None = None,
    cache: BaseCache | None = None,
    offline: bool = False,
    max_calls: int | None = 20,
    backend: Backend = "memory",
    variant: Variant | None = None,
    model: BaseChatModel | None = None,
    transport: E4Transport = "direct",
    count_errors: bool = False,
) -> E4Report:
    """Ewaluacja E4 (kontrakt F4, 11.3-11.4; F5, 10.6). Bez `live` zero wywołań LLM-a.

    Na żywo agent jest budowany przed odtworzeniem (brak klucza → ConfigError, nic nie biegnie).
    transport != 'direct' (tylko z live; nieznany → ConfigError): narzędzia agenta przez jedno
    połączenie MCP (memory, stdio albo http), zamykane na końcu. count_errors: nieudane próby
    zjadają max_calls (R1-6; domyślnie nie, jak F4).
    Pytania po kolei; CallBudgetExhausted → to pytanie i następne „pominięte”,
    stopped_by_max_calls=True. ids: podzbiór (nieznane → ConfigError). Każde niepominięte
    pytanie na żywo z błędem → E4AllErrorsError(report). Przeliczenie ≠ kurs →
    reference.E4ReferenceMismatch.
    """
    questions = select_questions(load_agent_gold(settings.paths.mini_project_4_questions), ids)
    wanted = {q.id for q in questions}
    tools_variant: Variant = variant or settings.tools.variant
    search = LazySearch(settings, backend)

    if transport not in TRANSPORTS:
        raise ConfigError(f"Nieznany transport E4: {transport!r} ({', '.join(TRANSPORTS)})")
    if transport != "direct" and not live:
        raise ConfigError("--transport inny niż direct dotyczy tylko przebiegu na żywo (--live)")

    meter: CallMeter | None = None
    runner: AgentRunner | None = None
    connection: McpConnection | None = None
    if live:
        meter = CallMeter(cache=cache, max_calls=max_calls, count_errors=count_errors)
        tools: list[BaseTool] | None = None
        if transport != "direct":
            if model is None:  # brak klucza → MissingConfigError, zanim uruchomimy serwer
                from kovalo.llm import get_chat_model

                get_chat_model("agent", settings=settings, cache=cache, offline=offline)
            from kovalo.eval.assistant.e5 import open_connection
            from kovalo.tools.specs import tool_specs

            connection = open_connection(settings, transport, backend, _context(settings, search))
            tools = list(connection.tools([s.name for s in tool_specs(tools_variant)]))
        try:
            runner = _live_runner(
                settings,
                retriever=search,
                cache=cache,
                offline=offline,
                variant=tools_variant,
                meter=meter,
                model=model,
                tools=tools,
            )
        except BaseException:
            if connection is not None:
                connection.close()
            raise

    now = datetime.now(UTC)
    try:
        course = _course_run(settings, "single", "course", wanted)
        course_supervisor = _course_run(settings, "supervisor", "course_supervisor", wanted)
        replay = _replay(settings, questions, search, now)

        live_run: E4Run | None = None
        stopped = False
        if live:
            assert runner is not None and meter is not None
            start = time.perf_counter()
            traces, stopped = _live_traces(runner, questions)
            scores = [score_trace(t, g) for t, g in zip(traces, questions, strict=True)]
            info = _run_info(
                settings,
                datetime.now(UTC),
                provider=settings.llm.provider,
                model=settings.llm.model,
                variant=tools_variant,
            )
            calls = E4Calls(network=meter.network, from_cache=meter.from_cache, errors=meter.errors)
            live_run = _make_run("live", info, traces, scores, calls, time.perf_counter() - start)
    finally:
        if connection is not None:
            connection.close()

    report = E4Report(
        created_at=now,
        questions=[q.id for q in questions],
        backend=backend,
        transport=transport if live else "direct",
        embedding_model=_embedding_model(settings),
        reference=reference.course_e4(settings.reference_llm.model),
        course=course,
        course_supervisor=course_supervisor,
        replay=replay,
        live=live_run,
        llm_calls_planned=plan_llm_calls(questions, max_steps=settings.agent.max_steps),
        max_calls=max_calls if live else None,
        stopped_by_max_calls=stopped,
        live_cost_cents=(
            _cost_cents(settings, live_run.summary["tokens_in"], live_run.summary["tokens_out"])
            if live_run is not None
            else None
        ),
        our_cost_model=settings.cost.our_model,
        our_prices_checked_on=settings.cost.our_prices_checked_on,
        reference_cost_model=settings.cost.reference_model,
    )
    if live_run is not None:
        attempted = [r["trace"] for r in live_run.rows if r["trace"].error != SKIPPED_ERROR]
        if attempted and all(t.status == "error" for t in attempted):
            raise E4AllErrorsError(report)
    return report
