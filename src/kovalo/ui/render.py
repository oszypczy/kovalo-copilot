"""Renderowanie tury dla Chainlita: czyste funkcje, bez chainlit (kontrakt F3, 9.2; F4, 10.2).

UI nie formatuje logiki biznesowej: opis wniosku (`summary`), pytania i odpowiedzi przychodzą
z api. Tu tylko układ wiadomości, przyciski pauzy i etykiety statusów po polsku.
F6: koszt tury i suma rozmowy w panelu bocznym „Koszt tury” (kontrakt F6, 6.4): kwoty liczy
api, UI je sumuje i formatuje (bez cen i bez kovalo.cost).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from decimal import ROUND_HALF_UP, Decimal

from kovalo.text_pl import number
from kovalo.ui.client import (
    ApiCallError,
    PendingView,
    ToolEventView,
    TurnView,
    UsageView,
    pause_ref,
)

__all__ = [
    "ACTION_LABELS",
    "CLARIFY_HINT",
    "COST_STEP",
    "EVIDENCE_LINK",
    "EVIDENCE_STEP",
    "INTERRUPTED",
    "MCP_SUFFIX",
    "NOT_PAUSED",
    "NO_TEXT",
    "RESUME_REJECTED",
    "STATUS_LABELS",
    "TRACE_STATUS_LABELS",
    "UiAction",
    "UiMessage",
    "UiSource",
    "UiStep",
    "UsageTotal",
    "add_usage",
    "message_text",
    "render_cost",
    "render_error",
    "render_evidence",
    "render_pending",
    "render_tool_end",
    "render_tool_start",
    "render_turn",
    "render_usage",
]

ACTION_LABELS = {"approved": "Zatwierdź", "rejected": "Odrzuć", "cancel": "Anuluj"}
STATUS_LABELS = {
    "answered": "odpowiedź",
    "pending_clarification": "czeka na doprecyzowanie",
    "pending_approval": "czeka na akceptację",
    "approved": "zatwierdzony (symulacja)",
    "rejected": "odrzucony",
    "cancelled": "anulowany",
    "error": "błąd",
}
TRACE_STATUS_LABELS = {
    "answered": "odpowiedź",
    "error": "błąd",
    "limit": "limit kroków lub czasu",
}
EVIDENCE_STEP = "Ścieżka dowodowa"
COST_STEP = "Koszt tury"  # F6: panel boczny z kosztem tury i sumą rozmowy
# linia w treści odpowiedzi: Chainlit linkuje nazwę elementu bocznego (cl.Text display="side")
EVIDENCE_LINK = f"Szczegóły: {EVIDENCE_STEP}"
INTERRUPTED = "(przerwane)"
# wiadomość bez tekstu (np. sam załącznik): copilot odpowiada tylko na pytania tekstowe
NO_TEXT = "Wpisz pytanie tekstem: copilot nie czyta załączników ani obrazów."
MCP_SUFFIX = " (przez MCP)"  # F5: krok narzędzia wywołanego przez serwer MCP
CLARIFY_HINT = "Wpisz marżę w polu czatu (np. 7,5)."
NOT_PAUSED = "Ta decyzja jest nieaktualna: rozmowa nie czeka już na odpowiedź."
RESUME_REJECTED = (
    "Nie rozpoznałem odpowiedzi. Kliknij przycisk albo napisz „zatwierdź”, „odrzuć” lub "
    "„anuluj” (przy pytaniu o marżę: liczbę, np. 7,5)."
)


@dataclass(frozen=True)
class UiAction:
    value: str
    label: str
    pause_ref: dict[str, object]


@dataclass(frozen=True)
class UiSource:
    name: str  # chunk_id
    content: str


@dataclass(frozen=True)
class UiMessage:
    content: str
    actions: list[UiAction] = field(default_factory=list)
    sources: list[UiSource] = field(default_factory=list)


@dataclass(frozen=True)
class UiStep:
    """Krok agenta w UI: wywołanie narzędzia (cl.Step) albo ścieżka dowodowa (element boczny)."""

    key: str  # call_id (rodzic: parent_call_id)
    parent_key: str | None
    name: str  # nazwa narzędzia; składowe: '↳ get_supplier_otif (odczyt narzędzia)'
    input: str  # argumenty jako JSON (ensure_ascii=False, wcięcie 2)
    output: str  # tool_end: brief, przy błędzie 'Błąd: …'; tool_start: ''
    is_error: bool = False
    icon: str | None = None  # ikona lucide zamiast pustego awatara kroku (Chainlit: metadata.icon)


def _step_name(event: ToolEventView) -> str:
    if event.parent_call_id is not None:
        return f"↳ {event.tool} (odczyt narzędzia)"
    return event.tool


# ikony lucide przy krokach narzędzi (inaczej Chainlit rysuje szare kółko pustego awatara)
TOOL_ICONS = {"search_documents": "file-search"}
ORIGIN_ICONS = {"data": "database", "calculation": "calculator"}
DEFAULT_TOOL_ICON = "wrench"


def step_icon(event: ToolEventView) -> str:
    """Ikona kroku: wyszukiwanie w dokumentach, odczyt danych, obliczenie, inne narzędzie."""
    return TOOL_ICONS.get(event.tool) or ORIGIN_ICONS.get(event.origin or "", DEFAULT_TOOL_ICON)


def _arguments(event: ToolEventView) -> str:
    return json.dumps(event.arguments, ensure_ascii=False, indent=2)


def render_tool_start(event: ToolEventView) -> UiStep:
    """Początek wywołania narzędzia: nazwa i argumenty, wynik jeszcze pusty."""
    return UiStep(
        key=event.call_id,
        parent_key=event.parent_call_id,
        name=_step_name(event),
        input=_arguments(event),
        output="",
        icon=step_icon(event),
    )


def render_tool_end(event: ToolEventView) -> UiStep:
    """Koniec wywołania: brief z api albo komunikat błędu („Błąd: …”). F5: przy
    transport == "mcp" nazwa z dopiskiem „ (przez MCP)” i w wyniku linia
    „Wywołanie MCP: <invocation_id>” (zdarzenie składowe dziedziczy transport rodzica)."""
    is_error = event.status not in (None, "ok")
    if is_error:
        error = event.error or event.status or "nieznany błąd"
        output = error if error.startswith("Błąd: ") else f"Błąd: {error}"
    else:
        output = event.brief or ""
    name = _step_name(event)
    if event.transport == "mcp":
        name = f"{name}{MCP_SUFFIX}"
        if event.invocation_id:
            line = f"Wywołanie MCP: {event.invocation_id}"
            output = f"{output}\n{line}" if output else line
    return UiStep(
        key=event.call_id,
        parent_key=event.parent_call_id,
        name=name,
        input=_arguments(event),
        output=output,
        is_error=is_error,
    )


def render_evidence(view: TurnView, number: int = 1) -> UiStep | None:
    """„Ścieżka dowodowa” z evidence_path (w UI: panel boczny podlinkowany w odpowiedzi);
    None, gdy w turze nie było agenta. `number` > 1 dopisuje numer: Chainlit łączy link
    z elementem po nazwie, więc kolejne ścieżki w rozmowie muszą mieć różne nazwy."""
    if not view.evidence_path:
        return None
    return UiStep(
        key="evidence",
        parent_key=None,
        name=EVIDENCE_STEP if number == 1 else f"{EVIDENCE_STEP} {number}",
        input="",
        output=view.evidence_path,
    )


def message_text(
    message: UiMessage, evidence: UiStep | None = None, cost: UiStep | None = None
) -> str:
    """Treść wiadomości w czacie: odpowiedź, potem nazwy elementów bocznych (źródła, ścieżka
    dowodowa, F6: koszt tury), żeby Chainlit podlinkował ich panele."""
    parts = [message.content] if message.content else []
    if message.sources:
        parts.append("Fragmenty: " + ", ".join(s.name for s in message.sources))
    details = [step.name for step in (evidence, cost) if step is not None]
    if details:
        parts.append("Szczegóły: " + ", ".join(details))
    return "\n\n".join(parts)


def render_cost(details: str | None, number: int = 1) -> UiStep | None:
    """Element boczny „Koszt tury” z listą z render_usage; numerowany w rozmowie jak ścieżka
    dowodowa (Chainlit łączy link z panelem po nazwie). None bez usage."""
    if details is None:
        return None
    return UiStep(
        key="cost",
        parent_key=None,
        name=COST_STEP if number == 1 else f"{COST_STEP} {number}",
        input="",
        output=details,
    )


def render_pending(pending: PendingView) -> UiMessage:
    """Wiadomość pauzy z przyciskami z `options` (każdy z pause_ref bieżącej pauzy)."""
    ref = pause_ref(pending)
    actions = [
        UiAction(value=option, label=ACTION_LABELS.get(option, option), pause_ref=ref)
        for option in pending.options
    ]
    if pending.type == "approval":
        required = ", ".join(pending.approvers) or pending.role or "-"
        lines = [f"**Akceptacja (symulacja): {pending.role}** (wymagani: {required})"]
        if pending.summary:
            lines.append(pending.summary)
    else:
        lines = [pending.question or "Podaj brakującą informację.", CLARIFY_HINT]
    return UiMessage(content="\n".join(lines), actions=actions)


def render_turn(view: TurnView) -> list[UiMessage]:
    """Odpowiedzi tury (źródła przy ostatniej), potem ewentualne „nie rozpoznałem”, pauza
    albo linia statusu końcowego („Status: …”) w ostatniej wiadomości."""
    messages = [UiMessage(content=reply) for reply in view.replies]
    sources = [UiSource(name=s.chunk_id, content=s.text) for s in view.sources]
    if sources:
        if messages:
            last = messages[-1]
            messages[-1] = UiMessage(content=last.content, actions=last.actions, sources=sources)
        else:
            messages.append(UiMessage(content="Źródła:", sources=sources))
    if view.resume_rejected:
        messages.append(UiMessage(content=RESUME_REJECTED))
    if view.pending is not None:
        messages.append(render_pending(view.pending))
        return messages
    if view.status is not None:
        label = STATUS_LABELS.get(view.status, view.status)
        if view.trace is not None and view.trace.get("status") == "limit":
            label = TRACE_STATUS_LABELS["limit"]
        status = f"Status: {label}"
        if messages:
            last = messages[-1]
            messages[-1] = UiMessage(
                content=f"{last.content}\n\n{status}", actions=last.actions, sources=last.sources
            )
        else:
            messages.append(UiMessage(content=status))
    return messages


def render_error(error: ApiCallError) -> UiMessage:
    """Błąd api po polsku; sesja i wątek zostają bez zmian."""
    if error.error == "api_unreachable" or error.status == 0:
        text = (
            "Api nie odpowiada (brak połączenia albo przekroczony limit czasu). "
            "Sprawdź, czy usługa api działa (make up), i spróbuj ponownie."
        )
    elif error.status == 503:
        text = (
            "Copilot jest chwilowo niedostępny: sprawdź klucz GOOGLE_API_KEY w .env i czy "
            f"działa Postgres. Szczegóły: {error.detail}"
        )
    elif error.error == "stale_pause":
        text = "Ta decyzja jest nieaktualna: wątek czeka już na inną odpowiedź."
    elif error.error == "not_paused":
        text = NOT_PAUSED
    elif error.error == "thread_busy":
        text = "Poprzednia wiadomość jest jeszcze przetwarzana. Poczekaj chwilę."
    elif error.error == "thread_stuck":
        text = f"Ta rozmowa utknęła po błędzie. Odśwież stronę, aby zacząć nową. ({error.detail})"
    elif error.status == 422:
        text = f"Nie mogę wysłać tej wiadomości: {error.detail}"
    else:
        text = f"Wystąpił błąd ({error.status} {error.error}): {error.detail}"
    return UiMessage(content=text)


# --- F6: licznik tokenów i kosztu (kontrakt F6, 6.4) ---


@dataclass(frozen=True)
class UsageTotal:
    """Suma rozmowy (sesja Chainlita). Kwoty w USD, jak w `usage.cost`."""

    turns: int = 0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    ours_usd: float = 0.0
    reference_usd: float = 0.0
    partial: bool = False  # któraś tura bez ceny (ours_usd None albo cost None)


def add_usage(total: UsageTotal, usage: UsageView | None) -> UsageTotal:
    """Dodaje turę do sumy; None (wznowienie pauzy, tura bez LLM-a) → suma bez zmian."""
    if usage is None:
        return total
    cost = usage.cost
    ours = cost.ours_usd if cost is not None else None
    return replace(
        total,
        turns=total.turns + 1,
        calls=total.calls + usage.calls,
        input_tokens=total.input_tokens + usage.input_tokens,
        output_tokens=total.output_tokens + usage.output_tokens,
        ours_usd=total.ours_usd + (ours or 0.0),
        reference_usd=total.reference_usd + (cost.reference_usd if cost is not None else 0.0),
        partial=total.partial or ours is None,
    )


def _plural(count: int, one: str, few: str, many: str) -> str:
    """Odmiana po liczebniku: 1; 2-4 (poza 12-14); reszta."""
    if count == 1:
        return one
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return few
    return many


_USD_STEP = Decimal("0.000001")


def _usd(usd: float) -> str:
    """USD jako ułamek dolara: do 6 miejsc po przecinku, bez zer na końcu („0,003412 $”);
    dodatnia kwota poniżej 0,0000005 $ → „< 0,000001 $”."""
    value = Decimal(str(usd))
    rounded = value.quantize(_USD_STEP, rounding=ROUND_HALF_UP)
    if value > 0 and rounded == 0:
        return "< 0,000001 $"
    text = f"{rounded:f}".rstrip("0").rstrip(".")
    return f"{text.replace('.', ',')} $"


def _models(usage: UsageView) -> str:
    names = [m.model for m in usage.by_model if m.model]
    return ", ".join(dict.fromkeys(names))


def render_usage(usage: UsageView | None, total: UsageTotal) -> str | None:
    """Koszt tury jako lista Markdown do panelu bocznego „Koszt tury” (z sumą rozmowy `total`,
    już z tą turą).

    None, gdy tura nie ma usage. Bez cennika w api (cost None): bez części o koszcie.
    """
    if usage is None:
        return None
    calls = f"{number(usage.calls)} {_plural(usage.calls, 'wywołanie', 'wywołania', 'wywołań')}"
    models = _models(usage)
    if models:
        calls = f"{calls} ({models})"
    lines = [
        f"- **Tura:** {calls}",
        f"- **Tokeny:** {number(usage.input_tokens)} wejścia, "
        f"{number(usage.output_tokens)} wyjścia",
    ]
    cost = usage.cost
    if cost is not None:  # tylko cena faktycznie użytego modelu, bez cennika kursu
        if cost.ours_usd is None:
            missing = ", ".join(cost.unpriced_models) or "nieznany model"
            lines.append(f"- **Koszt tury:** brak ceny dla {missing}")
        else:
            lines.append(f"- **Koszt tury:** {_usd(cost.ours_usd)}")
        turns = f"{number(total.turns)} {_plural(total.turns, 'tura', 'tury', 'tur')}"
        conversation = f"- **Rozmowa:** {turns}, {_usd(total.ours_usd)}"
        if total.partial:
            conversation += " (część tur bez ceny)"
        lines.append(conversation)
    return "\n".join(lines)
