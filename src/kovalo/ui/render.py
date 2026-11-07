"""Renderowanie tury dla Chainlita: czyste funkcje, bez chainlit (kontrakt F3, 9.2; F4, 10.2).

UI nie formatuje logiki biznesowej: opis wniosku (`summary`), pytania i odpowiedzi przychodzą
z api. Tu tylko układ wiadomości, przyciski pauzy i etykiety statusów po polsku.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from kovalo.ui.client import ApiCallError, PendingView, ToolEventView, TurnView, pause_ref

__all__ = [
    "ACTION_LABELS",
    "CLARIFY_HINT",
    "EVIDENCE_LINK",
    "EVIDENCE_STEP",
    "INTERRUPTED",
    "MCP_SUFFIX",
    "RESUME_REJECTED",
    "STATUS_LABELS",
    "TRACE_STATUS_LABELS",
    "UiAction",
    "UiMessage",
    "UiSource",
    "UiStep",
    "message_text",
    "render_error",
    "render_evidence",
    "render_pending",
    "render_tool_end",
    "render_tool_start",
    "render_turn",
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
# linia w treści odpowiedzi: Chainlit linkuje nazwę elementu bocznego (cl.Text display="side")
EVIDENCE_LINK = f"Szczegóły: {EVIDENCE_STEP}"
INTERRUPTED = "(przerwane)"
MCP_SUFFIX = " (przez MCP)"  # F5: krok narzędzia wywołanego przez serwer MCP
CLARIFY_HINT = "Wpisz marżę w polu czatu (np. 7,5)."
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


def _step_name(event: ToolEventView) -> str:
    if event.parent_call_id is not None:
        return f"↳ {event.tool} (odczyt narzędzia)"
    return event.tool


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


def message_text(message: UiMessage, evidence: UiStep | None = None) -> str:
    """Treść wiadomości w czacie: odpowiedź, potem nazwy elementów bocznych (źródła i ścieżka
    dowodowa), żeby Chainlit podlinkował ich panele."""
    parts = [message.content] if message.content else []
    if message.sources:
        parts.append("Fragmenty: " + ", ".join(s.name for s in message.sources))
    if evidence is not None:
        parts.append(f"Szczegóły: {evidence.name}")
    return "\n\n".join(parts)


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
    elif error.error == "thread_busy":
        text = "Poprzednia wiadomość jest jeszcze przetwarzana. Poczekaj chwilę."
    elif error.error == "thread_stuck":
        text = f"Ta rozmowa utknęła po błędzie. Odśwież stronę, aby zacząć nową. ({error.detail})"
    elif error.status == 422:
        text = f"Nie mogę wysłać tej wiadomości: {error.detail}"
    else:
        text = f"Wystąpił błąd ({error.status} {error.error}): {error.detail}"
    return UiMessage(content=text)
