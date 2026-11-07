"""Teksty odpowiedzi grafu, bez LLM-a (FR-3.3).

Kontrakt: docs/contracts/f3.md, sekcja 6.4. Deterministyczne szablony po polsku; liczby przez
text_pl.number. `describe_request` to port chat.describe_request kursu.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kovalo.policy.models import EXCEPTION_NAMES, Decision, DiscountRequest
from kovalo.rag.records import RunRecord
from kovalo.text_pl import number
from kovalo.workflow.schemas import Status

__all__ = [
    "AGENT_ERROR_REPLY",
    "AGENT_LIMIT_REPLY",
    "CANCELLED_REPLY",
    "CLARIFY_QUESTION",
    "DOCUMENT_ERROR_REPLY",
    "ERROR_REPLY",
    "MISSING_DISCOUNT",
    "OTHER_REPLY",
    "already_approved_reply",
    "approved_reply",
    "decision_reply",
    "describe_request",
    "document_reply",
    "join_roles",
    "margin_message",
    "policy_answer",
    "rejected_reply",
]

# F4 (kontrakt F4, 8.1): agent bez odpowiedzi
AGENT_ERROR_REPLY = (
    "Nie udało się odpowiedzieć na to pytanie. Spróbuj ponownie za chwilę albo zadaj je inaczej."
)
AGENT_LIMIT_REPLY = (
    "Przerwałem pracę: pytanie wymagało więcej kroków albo czasu, niż pozwala limit. "
    "Zadaj węższe pytanie."
)
MISSING_DISCOUNT = "Podaj wysokość rabatu w procentach, a sprawdzę, kto go zatwierdza."
CANCELLED_REPLY = "Wniosek anulowany. Nic nie zostało złożone."
CLARIFY_QUESTION = (
    "Jaka będzie marża brutto po rabacie (w procentach)? Wpisz liczbę, np. 7,5, albo „anuluj”."
)
OTHER_REPLY = (
    "Jestem Kovalo Decision Copilot. Mogę:\n"
    "- odpowiedzieć na pytania o dokumenty Kovalo (ze źródłami),\n"
    "- sprawdzić, kto zatwierdza rabat według polityki course-v1,\n"
    "- przesłać wniosek rabatowy do akceptacji."
)
ERROR_REPLY = (
    "Nie zrozumiałem wiadomości. Napisz ją inaczej, np. "
    "„Kto zatwierdza rabat 15% dla zwykłego klienta?”."
)
DOCUMENT_ERROR_REPLY = (
    "Nie udało się odpowiedzieć na pytanie o dokumenty. Spróbuj ponownie za chwilę."
)
_SIMULATION = "To symulacja: nic nie trafia do CRM-u."


def join_roles(roles: Sequence[str]) -> str:
    """['CEO'] → 'CEO'; ['CEO', 'CFO'] → 'CEO i CFO'; ['A', 'B', 'C'] → 'A, B i C'."""
    items = list(roles)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} i {items[-1]}"


def _as_dict(request: DiscountRequest | Mapping[str, Any]) -> Mapping[str, Any]:
    return request.model_dump() if isinstance(request, DiscountRequest) else request


def _pct(value: Any) -> str:
    return "-" if value is None else f"{number(value)}%"


def describe_request(request: DiscountRequest | Mapping[str, Any]) -> str:
    """Wniosek tak, jak zrozumiał go copilot (port chat.describe_request kursu)."""
    data = _as_dict(request)
    flags = [label for name, label in EXCEPTION_NAMES.items() if data.get(name)]
    margin = data.get("margin_after_pct")
    return (
        f"klient: {data.get('customer') or '-'}, produkt: {data.get('product') or '-'}, "
        f"rabat: {_pct(data.get('discount_pct'))}, "
        f"marża po rabacie: {'nie podano' if margin is None else _pct(margin)}, "
        f"wyjątki: {', '.join(flags) or 'brak'}, wersja wniosku: {data.get('revision')}"
    )


def _reasons(decision: Decision) -> str:
    return "\n".join(f"- {reason}" for reason in decision.reasons)


def policy_answer(request: DiscountRequest, decision: Decision) -> str:
    """Odpowiedź na pytanie o politykę: kto zatwierdza, uzasadnienia, zastrzeżenie."""
    conditions = [label for name, label in EXCEPTION_NAMES.items() if getattr(request, name)]
    if request.margin_after_pct is not None:
        conditions.append(f"marża po rabacie {_pct(request.margin_after_pct)}")
    scope = f" ({', '.join(conditions)})" if conditions else ""
    return (
        f"Rabat {_pct(request.discount_pct)}{scope} zatwierdza: "
        f"{join_roles(decision.approvers)}.\n"
        f"Uzasadnienie ({decision.version}):\n{_reasons(decision)}\n\n"
        "To tylko informacja: żaden wniosek nie został złożony."
    )


def decision_reply(request: DiscountRequest, decision: Decision) -> str:
    """Wniosek przed akceptacją: opis, wymagane role, uzasadnienia."""
    return (
        f"Wniosek: {describe_request(request)}.\n"
        f"Wymagana akceptacja: {join_roles(decision.approvers)} (symulacja).\n"
        f"Uzasadnienie ({decision.version}):\n{_reasons(decision)}"
    )


def already_approved_reply(request: DiscountRequest, decision: Decision) -> str:
    return (
        f"Już zatwierdzone (wersja {request.revision}), akceptacja: "
        f"{join_roles(decision.approvers)}. Wniosek się nie zmienił, więc nie trzeba go "
        f"akceptować ponownie. {_SIMULATION}"
    )


def approved_reply(request: DiscountRequest, decision: Decision) -> str:
    return (
        f"Wniosek zatwierdzony (wersja {request.revision}): "
        f"{join_roles(decision.approvers)}. {_SIMULATION}"
    )


def rejected_reply(request: DiscountRequest, role: str) -> str:
    return f"Wniosek odrzucony (wersja {request.revision}) przez: {role}. {_SIMULATION}"


def margin_message(value: float) -> str:
    """Odpowiedź z doprecyzowania jako wiadomość menedżera: 'Marża po rabacie: 7,37%'."""
    return f"Marża po rabacie: {number(value)}%"


def document_reply(record: RunRecord) -> tuple[str, Status]:
    """(tekst, status) dla odpowiedzi o dokumentach: błąd → 'error'; odmowa to odpowiedź."""
    output = record.output
    if record.error is not None or output is None:
        return DOCUMENT_ERROR_REPLY, "error"
    if not output.answerable or not output.citations:
        return output.answer, "answered"
    return f"{output.answer}\n\nŹródła: {', '.join(output.citations)}", "answered"
