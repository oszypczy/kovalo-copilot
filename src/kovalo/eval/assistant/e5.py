"""Program E5: końcowy asystent przez MCP (MP5; kontrakt F5, 10.4; ADR-0014).

Zawsze, zero wywołań LLM-a: (a) zgodność MCP z wywołaniem bezpośrednim (20 wywołań), (b)
odtworzenie 13 rozmów (25 tur) z idealnym `understand` ze wzorca i agentem ze skryptu, przez
prawdziwy serwer MCP, przez CopilotService i run_turn. Na żywo (`live=True`): `understand'
z promptem `assistant_understand_system` i agent przez MCP; jeden licznik prób dla obu
(`--max-calls`, liczy też nieudane próby); po wyczerpaniu limitu reszta tur jest „pominięta”.
Jedno połączenie MCP na przebieg: w pamięci (testy), stdio (podproces `kovalo-mcp`, domyślnie)
albo streamable HTTP (`services.mcp_url`, compose).
"""

from __future__ import annotations

import math
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from typing import Any, Literal

from langchain_core.caches import BaseCache
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from kovalo.agents.agent import AgentRunner, build_agent
from kovalo.agents.middleware import CallBudgetExhausted, CallCounter, CallMeter
from kovalo.company import load_company_data
from kovalo.config import ConfigError, Settings
from kovalo.cost.pricing import our_price, reference_price, tokens_cost
from kovalo.eval.assistant.parity import ParityRow, run_parity
from kovalo.eval.assistant.reference import COURSE_E5
from kovalo.eval.assistant.scoring import score_assistant_turn, summarise
from kovalo.eval.datasets import Dialogue, DialogueTurn, load_agent_gold, load_conversations
from kovalo.eval.trace.replay import ScriptedToolModel
from kovalo.eval.trace.scoring import score_trace
from kovalo.eval.workflow.runner import ScriptedUnderstanding, run_turn
from kovalo.knowledge.retrievers import (
    LazyRetriever,
    Retriever,
    build_memory_retriever,
    open_retriever,
)
from kovalo.mcp_client import (
    McpConnection,
    http_connection,
    in_memory_connection,
    kovalo_stdio_args,
    kovalo_stdio_env,
    stdio_connection,
)
from kovalo.mcp_server.server import ServerDeps, build_server, read_catalog
from kovalo.tools.specs import ToolContext, tool_specs
from kovalo.workflow.checkpoint import memory_checkpointer
from kovalo.workflow.graph import CopilotDeps, build_copilot_graph
from kovalo.workflow.schemas import Turn
from kovalo.workflow.service import CopilotService

__all__ = [
    "MAX_CALLS_SKIPPED",
    "SCRIPTED_AGENT_ANSWER",
    "E5AllErrorsError",
    "E5Calls",
    "E5Mode",
    "E5Report",
    "E5Run",
    "E5Transport",
    "open_connection",
    "plan_llm_calls",
    "replay_problems",
    "run_e5",
    "select_conversations",
]

E5Mode = Literal["replay", "live"]
E5Transport = Literal["memory", "stdio", "http"]
Backend = Literal["memory", "pgvector"]

MAX_CALLS_SKIPPED = "pominięte: wyczerpany limit --max-calls"
SCRIPTED_AGENT_ANSWER = "(odpowiedź agenta)"
_BUDGET_EXHAUSTED = CallBudgetExhausted.__name__


class E5Calls(BaseModel):
    understand_network: int = 0
    understand_cache: int = 0
    agent_network: int = 0
    agent_cache: int = 0
    errors: int = 0
    mcp_calls: int = 0
    mcp_failures: dict[str, int] = {}  # warstwa → liczba (tool/protocol/transport/validation)


class E5Run(BaseModel):
    mode: E5Mode
    run: dict[str, Any]  # run_id, date, provider, model, transport, backend, policy_version,
    # terms_version, tools_variant, understand_prompt
    # {'dialogue', 'n', 'expected_kind', 'entry', 'score', 'mcp': [captures tury]}
    turns: list[dict[str, Any]]
    totals: dict[str, list[int]]  # summarise: wymiar → [zaliczone, ocenione]
    calls: E5Calls
    answered_by: dict[str, int] = {}
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float


class E5Report(BaseModel):
    created_at: datetime
    conversations: list[str]
    transport: E5Transport
    backend: Backend
    embedding_model: str | None
    reference: dict[str, Any]  # COURSE_E5
    parity: list[ParityRow]
    replay: E5Run
    live: E5Run | None = None
    llm_calls_planned: dict[str, int]
    max_calls: int | None = None
    stopped_by_max_calls: bool = False
    live_cost_cents: dict[str, float] | None = None
    our_cost_model: str
    our_prices_checked_on: date
    reference_cost_model: str


class E5AllErrorsError(RuntimeError):
    """Każda (niepominięta) tura na żywo zakończyła się błędem; `report` jest kompletny."""

    def __init__(self, report: E5Report) -> None:
        live = report.live
        entries = [t["entry"] for t in live.turns] if live else []
        first = next(
            (e.get("error") for e in entries if e.get("status") not in ("skipped", None)), None
        )
        run_id = live.run["run_id"] if live else "-"
        super().__init__(f"Żadna tura na żywo się nie udała (przebieg {run_id}), np.: {first}")
        self.report = report
        self.run_id = run_id
        self.first_error = first


# ---------------------------------------------------------------------------
# Plan i wybór rozmów
# ---------------------------------------------------------------------------


def _data_turns(conversations: Sequence[Dialogue]) -> list[DialogueTurn]:
    return [
        t for d in conversations for t in d.turns if t.expect.get("kind") == "document_question"
    ]


def plan_llm_calls(conversations: Sequence[Dialogue], *, max_steps: int = 24) -> dict[str, int]:
    """{'understand': liczba tur, 'agent_estimate': tury o dane (3, gdy wzorzec ma obliczenie,
    inaczej 2), 'estimate': suma, 'upper_bound': tury + tury_agenta x ceil(max_steps / 2)}."""
    understand = sum(len(d.turns) for d in conversations)
    data = _data_turns(conversations)
    agent = sum(3 if (t.expect.get("agent") or {}).get("calculations") else 2 for t in data)
    return {
        "understand": understand,
        "agent_estimate": agent,
        "estimate": understand + agent,
        "upper_bound": understand + len(data) * math.ceil(max_steps / 2),
    }


def select_conversations(
    conversations: Sequence[Dialogue], ids: Sequence[str] | None
) -> list[Dialogue]:
    """Rozmowy o podanych id (kolejność pliku); None → wszystkie; nieznane id → ConfigError."""
    if ids is None:
        return list(conversations)
    known = [d.id for d in conversations]
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ConfigError(f"Nieznane id rozmów E5: {', '.join(unknown)}. Znane: {', '.join(known)}")
    wanted = set(ids)
    return [d for d in conversations if d.id in wanted]


def replay_problems(report: E5Report) -> list[str]:
    """Wymagania bez LLM-a (kontrakt 10.4; kod 4): każdy wiersz zgodności `same`, odtworzenie
    100% w każdym wymiarze, `understand` raz na turę, agent raz na turę o dane, 0 nieudanych
    wywołań MCP."""
    problems: list[str] = []
    broken = [f"{r.tool}{r.arguments}" for r in report.parity if not r.same]
    if broken:
        problems.append(f"zgodność MCP {len(report.parity) - len(broken)}/{len(report.parity)}")
    replay = report.replay
    wrong = [f"{d} {p}/{n}" for d, (p, n) in replay.totals.items() if p != n]
    if wrong:
        problems.append("odtworzenie: " + ", ".join(wrong))
    expected = replay.run.get("expected_calls") or {}
    for key in ("understand", "agent"):
        got = replay.run.get(f"{key}_calls")
        if got != expected.get(key):
            problems.append(f"{key} wywołany {got} razy, oczekiwano {expected.get(key)}")
    failures = sum(replay.calls.mcp_failures.values())
    if failures:
        problems.append(f"nieudane wywołania MCP w odtworzeniu: {failures}")
    return problems


# ---------------------------------------------------------------------------
# Połączenie MCP i kontekst wywołań bezpośrednich
# ---------------------------------------------------------------------------


def _retriever(settings: Settings, backend: Backend) -> LazyRetriever:
    def opener() -> Retriever:
        from kovalo.llm import get_embeddings

        embeddings = get_embeddings(settings=settings)
        if backend == "pgvector":
            return open_retriever(settings, embeddings)
        return build_memory_retriever(settings, embeddings)

    return LazyRetriever(opener)


def _context(settings: Settings, retriever: Retriever) -> ToolContext:
    return ToolContext(
        data=load_company_data(settings.paths.company_data_dir),
        policy=settings.policy,
        company=settings.company,
        retriever=retriever,
        top_k=settings.retrieval.top_k,
    )


def _policy_yaml(settings: Settings) -> str:
    import yaml

    raw = yaml.safe_load((settings.root / "config.yaml").read_text(encoding="utf-8"))
    return yaml.safe_dump(raw["policy"], allow_unicode=True, sort_keys=False)


def open_connection(
    settings: Settings, transport: E5Transport, backend: Backend, ctx: ToolContext
) -> McpConnection:
    """Jedno połączenie na przebieg (bez limitu zapisów; E5 i E4 przez MCP). memory: serwer
    z TYM SAMYM kontekstem (ten sam retriever), więc zgodność search_documents jest dokładna;
    stdio: podproces `kovalo-mcp --search <backend>` bez kluczy; http: services.mcp_url."""
    mcp = settings.mcp
    startup, call, cleanup = mcp.startup_timeout_s, mcp.call_timeout_s, mcp.cleanup_timeout_s
    if transport == "memory":
        deps = ServerDeps(
            ctx=ctx,
            corpus_dir=settings.paths.corpus_dir,
            catalog=read_catalog(settings.paths.corpus_catalog),
            policy_yaml=_policy_yaml(settings),
            variant=settings.tools.variant,
            search=backend,
        )
        return in_memory_connection(
            build_server(deps, settings=settings),
            startup_timeout_s=startup,
            call_timeout_s=call,
            cleanup_timeout_s=cleanup,
        )
    if transport == "stdio":
        return stdio_connection(
            kovalo_stdio_args(settings, search=backend),
            settings.root,
            startup_timeout_s=startup,
            call_timeout_s=call,
            cleanup_timeout_s=cleanup,
            env=kovalo_stdio_env(),
        )
    if transport == "http":
        return http_connection(
            settings.services.mcp_url,
            startup_timeout_s=startup,
            call_timeout_s=call,
            cleanup_timeout_s=cleanup,
        )
    raise ConfigError(f"Nieznany transport E5: {transport!r} (memory, stdio, http)")


@contextmanager
def _turn_captures(connection: McpConnection) -> Iterator[list[dict[str, Any]]]:
    """Zapisy wywołań MCP z czasu tury (connection.captures bez limitu: lista)."""
    out: list[dict[str, Any]] = []
    before = len(connection.captures)
    try:
        yield out
    finally:
        out.extend(list(connection.captures)[before:])


# ---------------------------------------------------------------------------
# Liczenie wywołań understand (wspólny licznik z agentem)
# ---------------------------------------------------------------------------


def _cache_hits(cache: BaseCache | None) -> int | None:
    hits = getattr(cache, "hits", None)
    return hits if isinstance(hits, int) else None


def _parse_failed(result: Any) -> bool:
    if isinstance(result, Mapping) and "parsed" in result:
        return result.get("parsing_error") is not None or result.get("parsed") is None
    return False


class _UnderstandTracker:
    """Owija model `understand` (jak _CallTracker E3): admit() wspólnego licznika przed próbą,
    potem klasyfikacja (sieć, cache, błąd). Nieudana próba zjada limit (count_errors)."""

    def __init__(self, cache: BaseCache | None, counter: CallCounter) -> None:
        self.cache = cache
        self.counter = counter
        self.network = 0
        self.from_cache = 0
        self.errors = 0

    def wrap(self, model: Runnable[LanguageModelInput, Any]) -> Runnable[LanguageModelInput, Any]:
        def invoke(messages: Any) -> Any:
            self.counter.admit()  # CallBudgetExhausted → tura 'error' → „pominięta”
            before = _cache_hits(self.cache)
            try:
                result = model.invoke(messages)
            except Exception:
                self.errors += 1
                self.counter.record("error")
                raise
            after = _cache_hits(self.cache)
            if before is not None and after is not None and after > before:
                self.from_cache += 1
                self.counter.record("cache")
            else:
                self.network += 1
                self.counter.record("network")
            if _parse_failed(result):
                self.errors += 1
            return result

        return RunnableLambda(invoke, name="e5_understand")


# ---------------------------------------------------------------------------
# Tury
# ---------------------------------------------------------------------------


def _scripted_answers(turn: DialogueTurn) -> Callable[[dict[str, Any]], Any]:
    scripted = iter(turn.resume)

    def answer(payload: dict[str, Any]) -> Any:
        try:
            return next(scripted)
        except StopIteration:
            raise RuntimeError(
                "skrypt nie ma odpowiedzi na tę pauzę (więcej pauz, niż przewidziano)"
            ) from None

    return answer


def _skipped(user: str, error: str) -> dict[str, Any]:
    """Wpis tury pominiętej (P8): kształt run_turn ze statusem 'skipped'."""
    return {
        "user": user,
        "turn": None,
        "request": None,
        "approvers": None,
        "reasons": None,
        "pauses": [],
        "approvals": {},
        "status": "skipped",
        "record": None,
        "error": error,
        "reply": None,
        "calls": [],
        "seconds": 0,
    }


class _Conversations:
    """Prowadzi rozmowy tura po turze przez run_turn z połączeniem MCP; zbiera wpisy."""

    def __init__(self, connection: McpConnection, run_id: str) -> None:
        self.connection = connection
        self.run_id = run_id
        self.turns: list[dict[str, Any]] = []
        self.stopped = False

    def run(self, service: CopilotService, dialogue: Dialogue) -> None:
        thread_id = f"{self.run_id}-{dialogue.id}"
        failed: str | None = None
        for n, t in enumerate(dialogue.turns, start=1):
            captured: list[dict[str, Any]] = []
            if self.stopped:
                entry = _skipped(t.user, MAX_CALLS_SKIPPED)
            elif failed is not None:
                entry = _skipped(t.user, f"pominięte: {failed}")
            else:
                with _turn_captures(self.connection) as captured:
                    try:
                        entry = run_turn(
                            service,
                            thread_id,
                            t.user,
                            _scripted_answers(t),
                            connection=self.connection,
                            turn=n,
                        )
                    except RuntimeError as exc:  # wątek w pauzie (nie powinno się zdarzyć)
                        entry = _skipped(t.user, f"pominięte: {exc}")
                        failed = str(exc)
                if _BUDGET_EXHAUSTED in (entry.get("error") or ""):
                    self.stopped = True
                    entry = _skipped(t.user, MAX_CALLS_SKIPPED)
                elif entry["status"] == "error" and "przy pauzie" in (entry.get("error") or ""):
                    failed = entry["error"]
            self.turns.append(
                {
                    "dialogue": dialogue.id,
                    "n": n,
                    "entry": entry,
                    "expect": t.expect,
                    "mcp": captured,
                }
            )


def _score_turns(
    turns: list[dict[str, Any]], claim_phrases: Sequence[str], *, with_agent: bool
) -> list[dict[str, Any]]:
    scored = []
    for t in turns:
        expect = dict(t.pop("expect"))
        if not with_agent:
            expect.pop("agent", None)  # agent ze skryptu: sprawdzenia agenta tylko na żywo
        t["score"] = score_assistant_turn(t["entry"], expect, claim_phrases, score_trace)
        t["expected_kind"] = expect.get("kind")
        scored.append(
            {k: t[k] for k in ("dialogue", "n", "expected_kind", "entry", "score", "mcp")}
        )
    return scored


def _usage(turns: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    usages = [c["usage"] for t in turns for c in t["entry"].get("calls") or [] if c.get("usage")]
    return (
        sum(int(u.get("input_tokens") or 0) for u in usages),
        sum(int(u.get("output_tokens") or 0) for u in usages),
    )


def _answered_by(turns: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(
        Counter(
            str(c["model"]) for t in turns for c in t["entry"].get("calls") or [] if c.get("model")
        )
    )


def _mcp_counts(turns: Sequence[Mapping[str, Any]]) -> tuple[int, dict[str, int]]:
    captures = [c for t in turns for c in t["mcp"]]
    failures = Counter(str(c["outcome"]) for c in captures if c.get("outcome") != "ok")
    return len(captures), dict(failures)


def _make_run(
    mode: E5Mode,
    run: dict[str, Any],
    turns: list[dict[str, Any]],
    calls: E5Calls,
    seconds: float,
) -> E5Run:
    tokens_in, tokens_out = _usage(turns)
    mcp_calls, failures = _mcp_counts(turns)
    return E5Run(
        mode=mode,
        run=run,
        turns=turns,
        totals=summarise([t["score"] for t in turns]),
        calls=calls.model_copy(update={"mcp_calls": mcp_calls, "mcp_failures": failures}),
        answered_by=_answered_by(turns) if mode == "live" else {},
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        seconds=round(seconds, 2),
    )


# ---------------------------------------------------------------------------
# Przebieg
# ---------------------------------------------------------------------------


def _run_id(now: datetime) -> str:
    return f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _run_info(
    settings: Settings,
    now: datetime,
    *,
    provider: str | None,
    model: str | None,
    transport: E5Transport,
    backend: Backend,
) -> dict[str, Any]:
    return {
        "run_id": _run_id(now),
        "date": now.date().isoformat(),
        "provider": provider,
        "model": model,
        "transport": transport,
        "backend": backend,
        "policy_version": settings.policy.version,
        "terms_version": load_company_data(settings.paths.company_data_dir).terms_version,
        "tools_variant": settings.tools.variant,
        "understand_prompt": "assistant_understand_system",
        "agent_prompt": "agent_system",
        "budget": {"max_steps": settings.agent.max_steps, "timeout_s": settings.agent.timeout_s},
    }


def _replay(
    settings: Settings,
    conversations: Sequence[Dialogue],
    connection: McpConnection,
    names: Sequence[str],
    claim_phrases: Sequence[str],
    *,
    transport: E5Transport,
    backend: Backend,
) -> E5Run:
    """P2: każda rozmowa na świeżym grafie (jak _replay E3) z idealnym understand i agentem
    ze skryptu na narzędziach MCP; ocena bez wzorca agenta."""
    start = time.perf_counter()
    info = _run_info(
        settings,
        datetime.now(UTC),
        provider="scripted",
        model="idealne rozumienie ze wzorca, agent ze skryptu",
        transport=transport,
        backend=backend,
    )
    runner = _Conversations(connection, info["run_id"])
    tools = connection.tools(names)
    understand_calls = agent_calls = 0
    for dialogue in conversations:
        understand = ScriptedUnderstanding(
            [Turn.model_validate(t.understand) for t in dialogue.turns]
        )
        n_data = sum(1 for t in dialogue.turns if t.expect.get("kind") == "document_question")
        model = ScriptedToolModel(script=[AIMessage(SCRIPTED_AGENT_ANSWER)] * n_data)
        agent = AgentRunner(
            build_agent(model, tools, system_prompt=settings.prompts.agent_system), settings.agent
        )
        deps = CopilotDeps(
            understand_model=understand.as_runnable(),
            agent=agent,
            policy=settings.policy,
            understand_prompt=settings.prompts.assistant_understand_system,
        )
        graph = build_copilot_graph(deps, checkpointer=memory_checkpointer())
        service = CopilotService(graph, max_message_chars=settings.workflow.max_message_chars)
        try:
            runner.run(service, dialogue)
        finally:
            service.close()
        understand_calls += understand.calls
        agent_calls += model.calls
    info["understand_calls"], info["agent_calls"] = understand_calls, agent_calls
    info["expected_calls"] = {
        "understand": sum(len(d.turns) for d in conversations),
        "agent": len(_data_turns(conversations)),
    }
    turns = _score_turns(runner.turns, claim_phrases, with_agent=False)
    return _make_run("replay", info, turns, E5Calls(), time.perf_counter() - start)


def _cost_cents(settings: Settings, tokens_in: int, tokens_out: int) -> dict[str, float]:
    cost = settings.cost
    return {
        side: round(100 * tokens_cost(tokens_in, tokens_out, price=price), 4)
        for side, price in (("ours", our_price(cost)), ("reference", reference_price(cost)))
    }


def _embedding_model(settings: Settings) -> str:
    emb = settings.embeddings
    return emb.local.model if emb.provider == "fastembed" else emb.model


def _claim_phrases(settings: Settings) -> list[str]:
    gold = load_agent_gold(settings.paths.mini_project_4_questions)
    return list(gold[0].approval_claim_phrases) if gold else []


def run_e5(
    settings: Settings,
    *,
    live: bool = False,
    ids: Sequence[str] | None = None,
    cache: BaseCache | None = None,
    offline: bool = False,
    max_calls: int | None = 40,
    transport: E5Transport = "stdio",
    backend: Backend = "memory",
    understand_model: Runnable[LanguageModelInput, Any] | None = None,
    agent_model: BaseChatModel | None = None,
) -> E5Report:
    """Ewaluacja E5 (kontrakt F5, 10.4). Bez `live` zero wywołań LLM-a.

    Na żywo modele są budowane przed połączeniem (brak klucza → ConfigError, nic nie biegnie).
    Rozmowy po kolei, tura po turze, pauzy z `resume`; wyczerpany wspólny licznik → ta tura
    i wszystkie następne 'skipped' (w mianowniku), stopped_by_max_calls=True. Tura z błędem
    przy pauzie → reszta rozmowy 'skipped'. ids: nieznane → ConfigError. McpStartupError
    (serwer nie wystartował) leci wyżej. Każda niepominięta tura na żywo z błędem →
    E5AllErrorsError(report).
    """
    conversations = select_conversations(
        load_conversations(
            settings.paths.mini_project_5_conversations, settings.paths.mini_project_3_dialogues
        ),
        ids,
    )
    names = [spec.name for spec in tool_specs(settings.tools.variant)]
    claim_phrases = _claim_phrases(settings)

    counter: CallCounter | None = None
    tracker: _UnderstandTracker | None = None
    meter: CallMeter | None = None
    live_understand: Runnable[LanguageModelInput, Any] | None = None
    if live:
        from kovalo.workflow.factory import build_understand_model

        counter = CallCounter(max_calls=max_calls)
        tracker = _UnderstandTracker(cache, counter)
        meter = CallMeter(cache=cache, count_errors=True, counter=counter)
        base = understand_model or build_understand_model(settings, cache=cache, offline=offline)
        live_understand = tracker.wrap(base)
        if agent_model is None:  # brak klucza → MissingConfigError, zanim cokolwiek pobiegnie
            from kovalo.llm import get_chat_model

            get_chat_model("agent", settings=settings, cache=cache, offline=offline)

    now = datetime.now(UTC)
    ctx = _context(settings, _retriever(settings, backend))
    connection = open_connection(settings, transport, backend, ctx)
    try:
        parity = run_parity(ctx, connection)
        replay = _replay(
            settings,
            conversations,
            connection,
            names,
            claim_phrases,
            transport=transport,
            backend=backend,
        )
        live_run: E5Run | None = None
        stopped = False
        if live:
            assert tracker is not None and meter is not None and live_understand is not None
            live_run, stopped = _live(
                settings,
                conversations,
                connection,
                names,
                claim_phrases,
                understand=live_understand,
                tracker=tracker,
                meter=meter,
                cache=cache,
                offline=offline,
                agent_model=agent_model,
                transport=transport,
                backend=backend,
            )
    finally:
        connection.close()

    report = E5Report(
        created_at=now,
        conversations=[d.id for d in conversations],
        transport=transport,
        backend=backend,
        embedding_model=_embedding_model(settings),
        reference=dict(COURSE_E5),
        parity=parity,
        replay=replay,
        live=live_run,
        llm_calls_planned=plan_llm_calls(conversations, max_steps=settings.agent.max_steps),
        max_calls=max_calls if live else None,
        stopped_by_max_calls=stopped,
        live_cost_cents=(
            _cost_cents(settings, live_run.tokens_in, live_run.tokens_out)
            if live_run is not None
            else None
        ),
        our_cost_model=settings.cost.our_model,
        our_prices_checked_on=settings.cost.our_prices_checked_on,
        reference_cost_model=settings.cost.reference_model,
    )
    if live_run is not None:
        attempted = [t["entry"] for t in live_run.turns if t["entry"]["status"] != "skipped"]
        if attempted and all(e["status"] == "error" for e in attempted):
            raise E5AllErrorsError(report)
    return report


def _live(
    settings: Settings,
    conversations: Sequence[Dialogue],
    connection: McpConnection,
    names: Sequence[str],
    claim_phrases: Sequence[str],
    *,
    understand: Runnable[LanguageModelInput, Any],
    tracker: _UnderstandTracker,
    meter: CallMeter,
    cache: BaseCache | None,
    offline: bool,
    agent_model: BaseChatModel | None,
    transport: E5Transport,
    backend: Backend,
) -> tuple[E5Run, bool]:
    start = time.perf_counter()
    tools = connection.tools(names)
    if agent_model is None:
        from kovalo.workflow.factory import build_agent_runner

        agent = build_agent_runner(
            settings, tools=tools, cache=cache, offline=offline, extra_middleware=[meter]
        )
    else:
        agent = AgentRunner(
            build_agent(
                agent_model,
                tools,
                system_prompt=settings.prompts.agent_system,
                extra_middleware=[meter],
            ),
            settings.agent,
        )
    deps = CopilotDeps(
        understand_model=understand,
        agent=agent,
        policy=settings.policy,
        understand_prompt=settings.prompts.assistant_understand_system,
    )
    graph = build_copilot_graph(deps, checkpointer=memory_checkpointer())
    service = CopilotService(graph, max_message_chars=settings.workflow.max_message_chars)
    info = _run_info(
        settings,
        datetime.now(UTC),
        provider=settings.llm.provider,
        model=settings.llm.model,
        transport=transport,
        backend=backend,
    )
    runner = _Conversations(connection, info["run_id"])
    try:
        for dialogue in conversations:
            runner.run(service, dialogue)
    finally:
        service.close()
    calls = E5Calls(
        understand_network=tracker.network,
        understand_cache=tracker.from_cache,
        agent_network=meter.network,
        agent_cache=meter.from_cache,
        errors=tracker.errors + meter.errors,
    )
    turns = _score_turns(runner.turns, claim_phrases, with_agent=True)
    return _make_run("live", info, turns, calls, time.perf_counter() - start), runner.stopped
