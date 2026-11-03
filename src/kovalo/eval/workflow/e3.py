"""Program E3: workflow decyzyjny MP3 (kontrakt F3, 10.3; port check.py i run_dialogues.py kursu).

Odtworzenie (zawsze, zero wywołań LLM-a): każdy dialog na nowym grafie z idealnym
`understand` (ScriptedUnderstanding); pytanie o dokumenty przez rag.answer_question na chunkach
z kontekstem w pamięci z embeddingami referencyjnymi (cache kursu, bez sieci) i stałą odpowiedzią
(ScriptedAnswerModel). Na żywo (`live=True`): jeden graf z modelem `understand` z configu i RAG
z modelem odpowiedzi z configu. Na pauzy odpowiada skrypt (`resume` każdej tury), jak w kursie.
Ocena: score_transcript wobec wzorca sprawdzonego ręcznie (nie wobec naszego silnika polityki).
"""

from __future__ import annotations

import functools
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any, Literal

from langchain_core.caches import BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from kovalo.config import ConfigError, Settings
from kovalo.cost.pricing import our_price, reference_price, tokens_cost
from kovalo.eval.datasets import Dialogue, DialogueTurn, load_dialogues
from kovalo.eval.workflow.reference import course_e3
from kovalo.eval.workflow.runner import ScriptedAnswerModel, ScriptedUnderstanding, run_turn
from kovalo.eval.workflow.scoring import STAGES, score_transcript
from kovalo.knowledge.chunking import ChunkVariant
from kovalo.knowledge.retrievers import Retriever, build_memory_retriever
from kovalo.rag.answer import DocumentAnswerer, answer_question
from kovalo.rag.records import Answer, TokenUsage
from kovalo.workflow import (
    CopilotDeps,
    CopilotService,
    Turn,
    build_copilot_graph,
    memory_checkpointer,
)

__all__ = [
    "Backend",
    "E3AllErrorsError",
    "E3Calls",
    "E3Mode",
    "E3Report",
    "E3Run",
    "plan_llm_calls",
    "replay_perfect",
    "run_e3",
    "select_dialogues",
]

E3Mode = Literal["replay", "live"]
Backend = Literal["memory", "pgvector"]
CHUNKS: ChunkVariant = "ctx"


class E3Calls(BaseModel):
    """Wywołania modelu bez trafień w cache (understand, answer), trafienia i nieudane."""

    understand: int = 0
    answer: int = 0
    from_cache: int = 0
    errors: int = 0


class E3Run(BaseModel):
    mode: E3Mode
    run_id: str  # YYYYMMDDTHHMMSS-<8 hex>, jak w kursie
    provider: str | None
    model: str | None
    policy: str
    score: dict[str, Any]  # score_transcript bez 'details'
    details: list[dict[str, Any]]
    transcript: list[dict[str, Any]]  # wpisy run_turn + 'dialogue'
    calls: E3Calls
    answered_by: dict[str, int] = {}  # model → liczba wywołań (fallback widoczny)
    usage: TokenUsage | None = None  # suma tokenów dostawcy
    seconds: float


class E3Report(BaseModel):
    created_at: datetime
    dialogues: list[str]
    backend: Backend
    embedding_model: str | None  # live: model wyszukiwania dla pytań o dokumenty
    reference: dict[str, Any]  # COURSE_E3
    llm_calls_planned: dict[str, int]  # NASZE: plan_llm_calls (raport: plan / wykonane)
    replay: E3Run
    live: E3Run | None
    our_cost_model: str
    our_prices_checked_on: date
    reference_cost_model: str
    live_cost_cents: dict[str, float] | None = (
        None  # NASZE: tokeny na żywo po cenach ours/reference
    )


class E3AllErrorsError(RuntimeError):
    """Każda tura na żywo zakończyła się błędem; `report` jest kompletny (odtworzenie też)."""

    def __init__(self, report: E3Report) -> None:
        live = report.live
        first = (
            next((e.get("error") for e in live.transcript if e.get("error")), None)
            if live
            else None
        )
        run_id = live.run_id if live else "-"
        super().__init__(f"Żadna tura na żywo się nie udała (przebieg {run_id}), np.: {first}")
        self.report = report
        self.run_id = run_id
        self.first_error = first


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _run_id(now: datetime) -> str:
    return f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


def plan_llm_calls(dialogues: Sequence[Dialogue]) -> dict[str, int]:
    """{'understand': tury, 'answer': tury z expect.kind == 'document_question', 'total'}."""
    turns = [t for d in dialogues for t in d.turns]
    understand = len(turns)
    answer = sum(1 for t in turns if t.expect.get("kind") == "document_question")
    return {"understand": understand, "answer": answer, "total": understand + answer}


def select_dialogues(dialogues: Sequence[Dialogue], ids: Sequence[str] | None) -> list[Dialogue]:
    """Dialogi o podanych id (kolejność pliku); None → wszystkie. Nieznane id → ConfigError."""
    if ids is None:
        return list(dialogues)
    known = [d.id for d in dialogues]
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ConfigError(f"Nieznane id dialogów: {', '.join(unknown)}. Znane: {', '.join(known)}")
    wanted = set(ids)
    return [d for d in dialogues if d.id in wanted]


def replay_perfect(run: E3Run) -> bool:
    """Każdy etap trafny w każdej ocenianej turze i zero błędów."""
    return run.score["errors"] == 0 and all(run.score[s][0] == run.score[s][1] for s in STAGES)


# ---------------------------------------------------------------------------
# Liczenie wywołań na żywo (cache: trafienia z liczników JsonFileLlmCache)
# ---------------------------------------------------------------------------


def _cache_hits(cache: BaseCache | None) -> int | None:
    hits = getattr(cache, "hits", None)
    return hits if isinstance(hits, int) else None


def _result_error(result: Any) -> bool:
    """Słownik include_raw z błędem parsowania albo bez obiektu → nieudane wywołanie."""
    if isinstance(result, Mapping) and "parsed" in result:
        return result.get("parsing_error") is not None or result.get("parsed") is None
    return False


class _CallTracker:
    """Opakowuje modele przebiegu na żywo: klasyfikuje każde wywołanie (sieć, cache, błąd)."""

    def __init__(self, cache: BaseCache | None) -> None:
        self.cache = cache
        self.calls = E3Calls()

    def _record(
        self, purpose: Literal["understand", "answer"], before: int | None, error: str | None
    ) -> None:
        after = _cache_hits(self.cache)
        from_cache = before is not None and after is not None and after > before
        if from_cache:
            self.calls.from_cache += 1
        elif not (error or "").startswith("LlmCacheMiss"):
            setattr(self.calls, purpose, getattr(self.calls, purpose) + 1)
        if error is not None:
            self.calls.errors += 1

    def wrap(
        self, purpose: Literal["understand", "answer"], model: Runnable[LanguageModelInput, Any]
    ) -> Runnable[LanguageModelInput, Any]:
        def invoke(messages: Any) -> Any:
            before = _cache_hits(self.cache)
            try:
                result = model.invoke(messages)
            except Exception as exc:
                self._record(purpose, before, f"{type(exc).__name__}: {exc}")
                raise
            self._record(purpose, before, "parsing" if _result_error(result) else None)
            return result

        return RunnableLambda(invoke, name=f"e3_{purpose}")


# ---------------------------------------------------------------------------
# Przebieg
# ---------------------------------------------------------------------------


def _scripted_answers(turn: DialogueTurn) -> Callable[[dict[str, Any]], Any]:
    scripted: Iterator[Any] = iter(turn.resume)

    def answer(payload: dict[str, Any]) -> Any:
        try:
            return next(scripted)
        except StopIteration:
            raise RuntimeError(
                "skrypt nie ma odpowiedzi na tę pauzę (więcej pauz, niż przewidziano)"
            ) from None

    return answer


def _skipped(user: str, failed: str) -> dict[str, Any]:
    """Wpis tury pominiętej po nieudanej pauzie (zasada run_dialogues.py kursu)."""
    return {
        "user": user,
        "turn": None,
        "request": None,
        "approvers": None,
        "reasons": None,
        "pauses": [],
        "approvals": {},
        "status": "error",
        "record": None,
        "error": f"pominięte: wcześniejsza tura się nie udała ({failed})",
        "reply": None,
        "calls": [],
        "seconds": 0.0,
    }


def _run_dialogue(
    service: CopilotService, thread_id: str, dialogue: Dialogue
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    failed: str | None = None
    for t in dialogue.turns:
        if failed:
            entry = _skipped(t.user, failed)
        else:
            entry = run_turn(service, thread_id, t.user, _scripted_answers(t))
            error = entry.get("error") or ""
            if entry["status"] == "error" and "przy pauzie" in error:
                failed = error
        entries.append({"dialogue": dialogue.id, **entry})
    return entries


def _usage(transcript: Sequence[Mapping[str, Any]]) -> TokenUsage | None:
    usages = [c["usage"] for e in transcript for c in e.get("calls") or [] if c.get("usage")]
    if not usages:
        return None
    reasoning = [u["reasoning_tokens"] for u in usages if u.get("reasoning_tokens") is not None]
    return TokenUsage(
        input_tokens=sum(int(u["input_tokens"]) for u in usages),
        output_tokens=sum(int(u["output_tokens"]) for u in usages),
        reasoning_tokens=sum(reasoning) if reasoning else None,
    )


def _answered_by(transcript: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(
        Counter(str(c["model"]) for e in transcript for c in e.get("calls") or [] if c.get("model"))
    )


def _make_run(
    settings: Settings,
    mode: E3Mode,
    run_id: str,
    transcript: list[dict[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    calls: E3Calls,
    seconds: float,
) -> E3Run:
    score = score_transcript(transcript, gold)
    live = mode == "live"
    return E3Run(
        mode=mode,
        run_id=run_id,
        provider=settings.llm.provider if live else None,
        model=settings.llm.model if live else None,
        policy=settings.policy.version,
        score={k: v for k, v in score.items() if k != "details"},
        details=score["details"],
        transcript=transcript,
        calls=calls,
        answered_by=_answered_by(transcript) if live else {},
        usage=_usage(transcript) if live else None,
        seconds=seconds,
    )


def _rag_answerer(
    settings: Settings, retriever: Retriever, model: Runnable[LanguageModelInput, Any]
) -> DocumentAnswerer:
    return functools.partial(
        answer_question,
        retriever=retriever,
        model=model,
        system_prompt=settings.prompts.answer_system,
        k=settings.retrieval.top_k,
        chunks=CHUNKS,
    )


def _replay(settings: Settings, dialogues: Sequence[Dialogue], run_id: str) -> list[dict[str, Any]]:
    """Idealne understand, RAG offline na embeddingach referencyjnych, zero wywołań LLM-a."""
    from kovalo.llm import get_reference_embeddings

    needs_documents = any(
        t.expect.get("kind") == "document_question" for d in dialogues for t in d.turns
    )
    answerer: DocumentAnswerer
    if needs_documents:
        retriever = build_memory_retriever(
            settings, get_reference_embeddings(settings=settings), variant=CHUNKS, mode="vector"
        )
        answerer = _rag_answerer(settings, retriever, ScriptedAnswerModel())
    else:
        answerer = _no_documents
    transcript: list[dict[str, Any]] = []
    for dialogue in dialogues:
        scripted = ScriptedUnderstanding(
            [Turn.model_validate(t.understand) for t in dialogue.turns]
        )
        deps = CopilotDeps(
            understand_model=scripted.as_runnable(),
            document_answerer=answerer,
            policy=settings.policy,
            understand_prompt=settings.prompts.understand_system,
        )
        graph = build_copilot_graph(deps, checkpointer=memory_checkpointer())
        service = CopilotService(graph, max_message_chars=settings.workflow.max_message_chars)
        transcript += _run_dialogue(service, f"{run_id}-{dialogue.id}", dialogue)
    return transcript


def _no_documents(question: str) -> Any:
    raise AssertionError("odtworzenie tych dialogów nie przewiduje pytań o dokumenty")


def _live_retriever(settings: Settings, embeddings: Embeddings, backend: Backend) -> Retriever:
    if backend == "pgvector":
        from kovalo.knowledge.retrievers import open_retriever

        return open_retriever(settings, embeddings, variant=CHUNKS)
    return build_memory_retriever(settings, embeddings, variant=CHUNKS)


def _cost_cents(settings: Settings, usage: TokenUsage | None) -> dict[str, float] | None:
    """Koszt tokenów przebiegu na żywo w centach: ceny naszego modelu i referencyjne kursu."""
    if usage is None:
        return None
    cost = settings.cost
    return {
        side: round(100 * tokens_cost(usage.input_tokens, usage.output_tokens, price=price), 4)
        for side, price in (("ours", our_price(cost)), ("reference", reference_price(cost)))
    }


def _embedding_model(settings: Settings) -> str:
    emb = settings.embeddings
    return emb.local.model if emb.provider == "fastembed" else emb.model


def run_e3(
    settings: Settings,
    *,
    live: bool = False,
    ids: Sequence[str] | None = None,
    cache: BaseCache | None = None,
    offline: bool = False,
    backend: Backend = "memory",
    understand_model: Runnable[LanguageModelInput, Any] | None = None,
    document_answerer: DocumentAnswerer | None = None,
    clock: Callable[[], datetime] = _utcnow,
) -> E3Report:
    """Ewaluacja E3 (kontrakt F3, 10.3). Bez `live` zero wywołań LLM-a.

    Na żywo modele są budowane przed odtworzeniem (brak klucza → ConfigError, nic nie
    biegnie). Zasada kursu: po nieudanej pauzie reszta dialogu jest pominięta (wpis 'error',
    liczony w mianowniku). ids: podzbiór dialogów (nieznane → ConfigError). Każda tura na
    żywo z błędem → E3AllErrorsError(report).
    """
    dialogues = select_dialogues(load_dialogues(settings.paths.mini_project_3_dialogues), ids)
    gold = [t.expect for d in dialogues for t in d.turns]

    tracker: _CallTracker | None = None
    live_understand: Runnable[LanguageModelInput, Any] | None = None
    live_answerer: DocumentAnswerer | None = None
    if live:
        from kovalo.llm import get_embeddings, get_structured_model
        from kovalo.workflow.factory import build_understand_model

        tracker = _CallTracker(cache)
        base_understand = understand_model or build_understand_model(
            settings, cache=cache, offline=offline
        )
        live_understand = tracker.wrap("understand", base_understand)
        if document_answerer is not None:
            live_answerer = document_answerer
        else:
            answer_model = get_structured_model(
                Answer, "answer", settings=settings, cache=cache, offline=offline
            )
            retriever = _live_retriever(settings, get_embeddings(settings=settings), backend)
            live_answerer = _rag_answerer(settings, retriever, tracker.wrap("answer", answer_model))

    now = clock()
    now = now.astimezone(UTC) if now.tzinfo is not None else now

    start = time.perf_counter()
    replay_id = _run_id(now)
    replay_transcript = _replay(settings, dialogues, replay_id)
    replay = _make_run(
        settings, "replay", replay_id, replay_transcript, gold, E3Calls(),
        time.perf_counter() - start,
    )  # fmt: skip

    live_run: E3Run | None = None
    if live:
        assert live_understand is not None and live_answerer is not None and tracker is not None
        start = time.perf_counter()
        live_id = _run_id(clock())
        deps = CopilotDeps(
            understand_model=live_understand,
            document_answerer=live_answerer,
            policy=settings.policy,
            understand_prompt=settings.prompts.understand_system,
        )
        graph = build_copilot_graph(deps, checkpointer=memory_checkpointer())
        service = CopilotService(graph, max_message_chars=settings.workflow.max_message_chars)
        transcript: list[dict[str, Any]] = []
        for dialogue in dialogues:
            transcript += _run_dialogue(service, f"{live_id}-{dialogue.id}", dialogue)
        live_run = _make_run(
            settings, "live", live_id, transcript, gold, tracker.calls,
            time.perf_counter() - start,
        )  # fmt: skip

    report = E3Report(
        created_at=now,
        dialogues=[d.id for d in dialogues],
        backend=backend,
        embedding_model=_embedding_model(settings) if live else None,
        reference=course_e3(settings.reference_llm.model),
        llm_calls_planned=plan_llm_calls(dialogues),
        replay=replay,
        live=live_run,
        our_cost_model=settings.cost.our_model,
        our_prices_checked_on=settings.cost.our_prices_checked_on,
        reference_cost_model=settings.cost.reference_model,
        live_cost_cents=_cost_cents(settings, live_run.usage) if live_run else None,
    )
    if (
        live_run is not None
        and live_run.transcript
        and all(e["status"] == "error" for e in live_run.transcript)
    ):
        raise E3AllErrorsError(report)
    return report
