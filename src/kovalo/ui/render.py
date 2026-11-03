"""Renderowanie tury dla Chainlita: czyste funkcje, bez chainlit (kontrakt F3, 9.2).

UI nie formatuje logiki biznesowej: opis wniosku (`summary`), pytania i odpowiedzi przychodzą
z api. Tu tylko układ wiadomości, przyciski pauzy i etykiety statusów po polsku.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kovalo.ui.client import ApiCallError, PendingView, TurnView, pause_ref

__all__ = [
    "ACTION_LABELS",
    "CLARIFY_HINT",
    "RESUME_REJECTED",
    "STATUS_LABELS",
    "UiAction",
    "UiMessage",
    "UiSource",
    "render_error",
    "render_pending",
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
        status = f"Status: {STATUS_LABELS.get(view.status, view.status)}"
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
