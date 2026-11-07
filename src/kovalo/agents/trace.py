"""Uruchomienie agenta na jednym pytaniu i zapis tego, co zrobił (FR-4.4, FR-4.5).

Port helpers/agent_trace.py kursu 1:1 (kontrakt: docs/contracts/f4.md, sekcja 7.1). Zapis
przebiegu (trace): ostateczna odpowiedź (albo błąd) i każde wywołanie narzędzia po kolei,
z argumentami, wynikiem i POCHODZENIEM (origin):
  data            rekord z narzędzia danych albo fragmenty z wyszukiwania: dowód ze źródła
  calculation     deterministyczne obliczenie (kara, polityka, warunek umowy): dowód ze źródła
  worker_summary  wygenerowana odpowiedź agenta-specjalisty: NIE jest dowodem ze źródła
  unknown         wywołanie narzędzia, które nie istnieje
Jeden budżet (kroki i czas, PINNED) obejmuje całe pytanie, razem ze specjalistami.

NASZE (z wartościami domyślnymi, więc zapisy kursu wczytują się bez zmian): `model_calls`
(tura modelu i jej tokeny, fallback widoczny), wywołanie zwrotne `on_event` (strumień kroków
do UI), zdarzenia składowe (narzędzie, które samo czyta dane, oddaje rekord odczytu:
`check_termination_condition` → `get_supplier_otif`), koperta `agent_trace` z MP5.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.callbacks import Callbacks
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from kovalo.config import AgentConfig
from kovalo.quantities import Quantity, source_values

if TYPE_CHECKING:
    from langchain.agents.middleware import ToolErrorMiddleware

__all__ = [
    "COURSE_ORIGINS",
    "DERIVED_EVENTS",
    "ENVELOPE_KIND",
    "ENVELOPE_VERSION",
    "EVIDENCE_FIELDS",
    "ORIGINS",
    "SOURCE_ORIGINS",
    "Budget",
    "BudgetExceeded",
    "EventCallback",
    "ModelCallInfo",
    "Trace",
    "TraceEvent",
    "evidence_values",
    "load_traces",
    "run_question",
    "save_traces",
    "tool_errors",
    "trace_envelope",
    "trace_from_record",
    "worker_result",
]

logger = logging.getLogger(__name__)

# Narzędzia kursu i rodzaj dowodu, jaki każde z nich daje (agents/ nie importuje tools/: test
# pilnuje, że ORIGINS == tools.specs.ORIGINS). Żadne nie zatwierdza, nie zapisuje, nie obiecuje.
COURSE_ORIGINS: dict[str, str] = {
    "search_documents": "data",
    "get_supplier_otif": "data",
    "get_invoice_value": "data",
    "get_contract_terms": "data",
    "calculate_otif_penalty": "calculation",
    "check_discount_policy": "calculation",
}
ORIGINS: dict[str, str] = {**COURSE_ORIGINS, "check_termination_condition": "calculation"}
SOURCE_ORIGINS = frozenset({"data", "calculation"})

# Część wyniku każdego narzędzia, która jest dowodem: tylko wartości znalezione lub obliczone.
EVIDENCE_FIELDS: dict[str, tuple[str, ...]] = {
    "search_documents": ("passages",),  # znaleziony tekst, a nie zapytanie
    "get_supplier_otif": ("observations",),  # a nie okres z zapytania ani komunikat
    "get_invoice_value": ("invoiced_grosze",),
    "get_contract_terms": ("otif_target_pct", "rate_per_full_point_pct", "cap_pct"),
    "calculate_otif_penalty": (
        "shortfall_points",
        "full_points",
        "penalty_pct_uncapped",
        "penalty_pct",
        "penalty_grosze",
    ),  # wyniki, a nie powtórzone dane wejściowe
    "check_discount_policy": ("reasons",),  # znane ograniczenie: uzasadnienie powtarza rabat
    # NASZE: rekord `otif` jest dowodem zdarzenia składowego get_supplier_otif, nie tego narzędzia
    "check_termination_condition": (
        "otif_below_pct",
        "consecutive_months",
        "notice_days",
        "months",
        "months_below",
    ),
}

# NASZE: narzędzie, które samo czyta dane, oddaje rekord odczytu w polu wyniku:
# {narzędzie: (pole wyniku, narzędzie pochodne)}.
DERIVED_EVENTS: dict[str, tuple[str, str]] = {
    "check_termination_condition": ("otif", "get_supplier_otif")
}

ENVELOPE_KIND, ENVELOPE_VERSION = "agent_trace", 1


def evidence_values(tool: str, result: Any) -> list[Quantity]:
    """Wielkości, które wynik narzędzia podaje jako dowód (EVIDENCE_FIELDS); [] dla reszty."""
    if not isinstance(result, dict) or tool not in EVIDENCE_FIELDS:
        return []
    return source_values({k: result[k] for k in EVIDENCE_FIELDS[tool] if k in result})


class TraceEvent(BaseModel):
    call_id: str
    parent_call_id: str | None = None  # wywołanie, wewnątrz którego zaszło to zdarzenie
    agent: str = "agent"  # który agent (albo narzędzie, dla zdarzeń składowych) je wykonał
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    origin: Literal["data", "calculation", "worker_summary", "unknown"]
    status: Literal["ok", "error", "no_result"]  # error: złe argumenty, nieznane narzędzie…
    result: Any = None  # rekord narzędzia (sparsowany JSON), gdy status to "ok"
    error: str | None = None
    transport: str | None = None  # "mcp", gdy wywołanie przeszło przez serwer MCP (MP5)
    invocation_id: str | None = None  # identyfikator wywołania MCP


class ModelCallInfo(BaseModel):
    """NASZE: jedna tura modelu (FR-6.5; fallback widoczny w `model`)."""

    model: str | None = None  # response_metadata['model_name']
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int | None = None


class Trace(BaseModel):
    question_id: str | None = None
    question: str
    answer: str | None = None
    status: Literal["answered", "error", "limit"]
    error: str | None = None
    events: list[TraceEvent] = Field(default_factory=list)
    steps: int = 0  # tury modelu + wyniki narzędzi, razem ze specjalistami
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    model_calls: list[ModelCallInfo] = Field(default_factory=list)  # NASZE


class BudgetExceeded(RuntimeError):
    """Pytanie wyczerpało limit kroków lub czasu."""


class Budget:
    """Kroki i sekundy dla JEDNEGO pytania, wspólne dla wszystkich agentów, którzy nad nim
    pracują."""

    def __init__(self, max_steps: int, timeout_s: float) -> None:
        self.max_steps, self.timeout_s = max_steps, timeout_s
        self.steps, self.deadline = 0, time.monotonic() + timeout_s

    @classmethod
    def from_settings(cls, agent: AgentConfig) -> Budget:
        """agent.max_steps i agent.timeout_s (PINNED)."""
        return cls(agent.max_steps, agent.timeout_s)

    def count(self) -> None:
        """Liczy jeden krok: turę modelu (także ostateczną odpowiedź) albo wynik narzędzia."""
        self.steps += 1

    def admit(self) -> None:
        """Przed nową pracą: BudgetExceeded, gdy liczba kroków OSIĄGNĘŁA maksimum albo minął
        czas."""
        if self.steps >= self.max_steps:
            raise BudgetExceeded(f"osiągnięto limit kroków ({self.max_steps} kroków na to pytanie)")
        if time.monotonic() > self.deadline:
            raise BudgetExceeded(f"osiągnięto limit czasu ({self.timeout_s:g} s na to pytanie)")

    def check(self) -> None:
        """Po zakończonej pracy: BudgetExceeded, gdy PRZEKROCZONO limit kroków lub czasu."""
        if self.steps > self.max_steps:
            raise BudgetExceeded(f"osiągnięto limit kroków ({self.max_steps} kroków na to pytanie)")
        if time.monotonic() > self.deadline:
            raise BudgetExceeded(f"osiągnięto limit czasu ({self.timeout_s:g} s na to pytanie)")

    def tick(self) -> None:
        """Liczy jeden krok, potem sprawdza budżet."""
        self.count()
        self.check()


def tool_errors() -> ToolErrorMiddleware[Any, Any]:
    """Middleware każdego agenta Kovalo: wyjątek WEWNĄTRZ wywołania narzędzia staje się
    ToolMessage z błędem („Błąd: <Typ>: <treść>”), który model może przeczytać, a trace notuje
    jako zdarzenie z błędem. Narzędzia Kovalo nie ujawniają w wyjątkach sekretów."""
    from langchain.agents.middleware import ToolErrorMiddleware

    return ToolErrorMiddleware(on_error=lambda exc, request: f"Błąd: {type(exc).__name__}: {exc}")


def _origin(tool: str, result: Any) -> str:
    if tool in ORIGINS:
        return ORIGINS[tool]
    if isinstance(result, dict) and "summary" in result and "events" in result:
        return "worker_summary"
    return "unknown"


def _parse(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return content
    return content


def _record_result(trace: Trace, events: dict[str, TraceEvent], message: ToolMessage) -> None:
    event = events.get(message.tool_call_id)
    if event is None:
        return
    if message.status == "error":
        event.status, event.error = "error", str(message.content)
        return
    event.status, event.result = "ok", _parse(message.content)
    if isinstance(event.result, dict) and "__mcp__" in event.result:  # MCP: rekord + pochodzenie
        event.transport = "mcp"
        event.invocation_id = event.result["__mcp__"].get("invocation_id")
        event.result = event.result["record"]
    artifact = message.artifact  # NASZE (ADR-0013 pkt 4): pochodzenie MCP poza treścią
    if isinstance(artifact, dict) and "transport" in artifact:
        event.transport = artifact["transport"]
        event.invocation_id = artifact.get("invocation_id")
    event.origin = _origin(event.tool, event.result)  # type: ignore[assignment]
    if event.origin == "worker_summary":  # specjalista: jego wywołania jako zagnieżdżone
        worker = event.result
        for child in worker["events"]:
            child = TraceEvent.model_validate(child)
            child.parent_call_id = child.parent_call_id or event.call_id
            trace.events.append(child)
        trace.input_tokens += worker.get("input_tokens", 0)
        trace.output_tokens += worker.get("output_tokens", 0)
        event.result = {"summary": worker["summary"]}


def _derived_event(event: TraceEvent) -> TraceEvent | None:
    """NASZE: zdarzenie składowe z rekordu odczytu w wyniku narzędzia (DERIVED_EVENTS)."""
    spec = DERIVED_EVENTS.get(event.tool)
    if spec is None or event.status != "ok" or not isinstance(event.result, dict):
        return None
    field, tool = spec
    record = event.result.get(field)
    if not isinstance(record, dict):
        return None
    requested = record.get("requested") or {}
    return TraceEvent(
        call_id=f"{event.call_id}#{tool}",
        parent_call_id=event.call_id,
        agent=event.tool,
        tool=tool,
        arguments={
            "supplier": record.get("supplier_id"),
            "from_month": requested.get("from"),
            "to_month": requested.get("to"),
        },
        origin=ORIGINS[tool],  # type: ignore[arg-type]
        status="ok",
        result=record,
        transport=event.transport,  # ten sam odczyt na serwerze (P8)
        invocation_id=event.invocation_id,
    )


EventCallback = Callable[[Literal["start", "end"], TraceEvent], None]


def _notify(on_event: EventCallback | None, phase: Literal["start", "end"], ev: TraceEvent) -> None:
    if on_event is None:
        return
    try:
        on_event(phase, ev)
    except Exception:  # wywołanie zwrotne (UI) nie może przerwać przebiegu
        logger.exception("on_event(%s, %s) zgłosiło wyjątek", phase, ev.tool)


def _model_call(message: AIMessage) -> Any:
    usage: dict[str, Any] = dict(message.usage_metadata or {})
    details = usage.get("output_token_details") or {}
    reasoning = details.get("reasoning") if isinstance(details, dict) else None
    model = (message.response_metadata or {}).get("model_name")
    return ModelCallInfo(
        model=str(model) if model else None,
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        reasoning_tokens=int(reasoning) if reasoning is not None else None,
    )


def run_question(
    agent: Any,
    question: str,
    budget: Budget,
    question_id: str | None = None,
    agent_name: str = "agent",
    *,
    on_event: EventCallback | None = None,
    callbacks: Callbacks = None,
) -> Trace:
    """Uruchamia skompilowanego agenta na jednym pytaniu, od świeżego stanu, i zwraca Trace.

    Nigdy nie zgłasza wyjątku z powodu błędów samego agenta: błąd modelu, błąd narzędzia, limit
    kroków lub czasu kończą się Trace ze statusem "error" albo "limit", który zachowuje
    wszystkie dotychczasowe zdarzenia. NASZE: `model_calls`, `on_event('start'|'end', event)`
    (wyjątek w nim jest logowany i połykany) i zdarzenia składowe (start rodzica, end rodzica,
    start składowego, end składowego; nie liczą się do budżetu). `callbacks` (F5, R1-2) trafiają
    do konfiguracji przebiegu agenta (zagnieżdżenie pod runem grafu); None → jak F4.
    """
    trace = Trace(question_id=question_id, question=question, status="error")
    events: dict[str, TraceEvent] = {}
    started = time.monotonic()
    try:
        budget.admit()  # wyczerpany budżet nie zaczyna nowej pracy
        config: RunnableConfig = {"recursion_limit": 2 * budget.max_steps + 1}
        if callbacks is not None:
            config["callbacks"] = callbacks
        stream = agent.stream(
            {"messages": [HumanMessage(question)]},
            config,
            stream_mode=["updates", "tasks"],
        )
        for mode, chunk in stream:
            if mode == "tasks":
                if "result" not in chunk:  # zadanie zaraz wystartuje
                    budget.admit()
                continue
            for value in chunk.values():
                for message in (value or {}).get("messages", []):
                    if isinstance(message, AIMessage):
                        usage: dict[str, Any] = dict(message.usage_metadata or {})
                        trace.input_tokens += usage.get("input_tokens", 0)
                        trace.output_tokens += usage.get("output_tokens", 0)
                        trace.model_calls.append(_model_call(message))
                        for tool_call in message.tool_calls:
                            event = TraceEvent(
                                call_id=str(tool_call["id"]),
                                agent=agent_name,
                                tool=tool_call["name"],
                                arguments=tool_call["args"],
                                status="no_result",
                                origin=ORIGINS.get(tool_call["name"], "unknown"),  # type: ignore[arg-type]
                            )
                            events[event.call_id] = event
                            trace.events.append(event)
                            _notify(on_event, "start", event)
                        if not message.tool_calls:
                            # Kurs: str(content). Gemini 3 zwraca bloki z podpisem myśli,
                            # więc dla nie-str bierzemy sam tekst bloków (E4 na żywo, P1).
                            content = message.content
                            trace.answer = content if isinstance(content, str) else message.text
                        budget.count()
                    elif isinstance(message, ToolMessage):
                        _record_result(trace, events, message)
                        budget.count()
                        recorded = events.get(message.tool_call_id)
                        if recorded is not None:
                            _notify(on_event, "end", recorded)
                            derived = _derived_event(recorded)
                            if derived is not None:
                                trace.events.append(derived)
                                _notify(on_event, "start", derived)
                                _notify(on_event, "end", derived)
        budget.check()  # ostateczna odpowiedź ponad budżet: zachowana, ale "limit"
        if trace.answer is None:
            trace.error = "agent zakończył pracę bez ostatecznej odpowiedzi"
        else:
            trace.status = "answered"
    except BudgetExceeded as exc:
        trace.status, trace.error = "limit", str(exc)
    except Exception as exc:  # zawiódł model albo graf: zachowaj to, co zaobserwowano
        name = type(exc).__name__
        trace.status = "limit" if name == "GraphRecursionError" else "error"
        trace.error = f"{name}: {exc}"
    trace.steps = budget.steps
    trace.seconds = round(time.monotonic() - started, 2)
    return trace


def worker_result(trace: Trace) -> dict[str, Any]:
    """Co narzędzie-specjalista zwraca supervisorowi: `summary` i własne zdarzenia."""
    summary = (
        trace.answer
        if trace.status == "answered"
        else f"(specjalista zakończył pracę bez odpowiedzi: {trace.status}: {trace.error})"
    )
    return {
        "summary": summary,
        "events": [e.model_dump() for e in trace.events],
        "input_tokens": trace.input_tokens,
        "output_tokens": trace.output_tokens,
    }


def save_traces(
    path: Path, run: dict[str, Any], traces: list[Trace], scores: list[dict[str, Any]] | None = None
) -> None:
    """Zapisuje jeden przebieg jako JSON (format kursu): pochodzenie i każdy trace (z oceną)."""
    optional: list[dict[str, Any] | None] = list(scores) if scores else [None] * len(traces)
    paired = zip(traces, optional, strict=True)
    rows = [{"trace": t.model_dump(), **({"score": s} if scores else {})} for t, s in paired]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps({"run": run, "traces": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_traces(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Odwrotność save_traces: (run, rows), każdy wiersz z Trace pod kluczem "trace"."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["run"], [
        {**row, "trace": Trace.model_validate(row["trace"])} for row in data["traces"]
    ]


def trace_envelope(trace: Trace) -> dict[str, Any]:
    """Koperta MP5 w kanale `record`: {'kind', 'version', 'trace'} + NASZE 'trace_id'
    (uuid4 hex) i 'created_at' (ISO, UTC)."""
    return {
        "kind": ENVELOPE_KIND,
        "version": ENVELOPE_VERSION,
        "trace": trace.model_dump(mode="json"),
        "trace_id": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
    }


def trace_from_record(record: dict[str, Any] | None) -> Trace | None:
    """Trace z koperty w `record` albo None (tura bez agenta). Zła wersja → ValueError."""
    if isinstance(record, dict) and record.get("kind") == ENVELOPE_KIND:
        if record.get("version") != ENVELOPE_VERSION:
            raise ValueError(f"nieznana wersja koperty agent_trace: {record.get('version')!r}")
        return Trace.model_validate(record["trace"])
    return None
