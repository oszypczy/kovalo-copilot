"""Teksty odpowiedzi grafu bez LLM-a (kontrakt F3, 6.4). Sprawdzamy fragmenty, nie całe zdania."""

from __future__ import annotations

from tests.fakes import toy_record


def _request(**fields):
    from kovalo.policy.models import DiscountRequest

    return DiscountRequest(**fields)


def _decision(settings, request):
    from kovalo.policy import required_approvers

    return required_approvers(request, settings.policy)


def test_describe_request_shows_fields():
    """Asercje `test_chat_shows_the_interpreted_request_before_asking` kursu."""
    from kovalo.workflow.replies import describe_request

    payload = {
        "revision": 2,
        "discount_pct": 22.0,
        "margin_after_pct": 7.0,
        "clearance": False,
        "strategic_account": True,
        "competitor_offer": False,
        "customer": "Carrefour Express",
        "product": None,
    }
    for source in (payload, _request(**payload)):  # Mapping i DiscountRequest
        shown = describe_request(source)
        for value in ("Carrefour Express", "22%", "7%", "klient strategiczny", "wersja wniosku: 2"):
            assert value in shown, f"opis wniosku musi pokazać {value!r}"
        assert "produkt: -" in shown


def test_describe_request_missing_values():
    from kovalo.workflow.replies import describe_request

    shown = describe_request(_request(discount_pct=8))
    assert "nie podano" in shown, "marża None"
    assert "wyjątki: brak" in shown and "rabat: 8%" in shown


def test_join_roles_and_numbers():
    from kovalo.workflow.replies import join_roles, margin_message

    assert join_roles(["CEO"]) == "CEO"
    assert join_roles(["CEO", "CFO"]) == "CEO i CFO"
    assert join_roles(["A", "B", "C"]) == "A, B i C"
    assert margin_message(7.37) == "Marża po rabacie: 7,37%"


def test_policy_answer_is_information_only(settings):
    from kovalo.workflow.replies import policy_answer

    request = _request(discount_pct=22)
    text = policy_answer(request, _decision(settings, request))
    assert "22%" in text and "CEO i CFO" in text
    assert "To tylko informacja: żaden wniosek nie został złożony." in text
    # osobny akapit: inaczej Markdown dokleja notkę do ostatniego punktu uzasadnienia
    assert "\n\nTo tylko informacja:" in text


def test_decision_and_approval_replies(settings):
    from kovalo.workflow.replies import (
        already_approved_reply,
        approved_reply,
        decision_reply,
        rejected_reply,
    )

    request = _request(revision=2, discount_pct=22, margin_after_pct=7, strategic_account=True)
    decision = _decision(settings, request)
    assert "Wymagana akceptacja" in decision_reply(request, decision)
    assert "(symulacja)" in decision_reply(request, decision)
    assert "Już zatwierdzone" in already_approved_reply(request, decision)
    approved = approved_reply(request, decision)
    assert "Wniosek zatwierdzony (wersja 2)" in approved and "nic nie trafia do CRM-u" in approved
    assert "CFO" in rejected_reply(request, "CFO")


def test_fixed_texts():
    from kovalo.workflow import replies

    assert replies.CANCELLED_REPLY == "Wniosek anulowany. Nic nie zostało złożone."
    assert replies.CLARIFY_QUESTION.startswith("Jaka będzie marża brutto po rabacie")
    assert "anuluj" in replies.CLARIFY_QUESTION
    assert replies.MISSING_DISCOUNT.startswith("Podaj wysokość rabatu")
    assert "dokumenty Kovalo" in replies.OTHER_REPLY
    assert "Nie zrozumiałem wiadomości" in replies.ERROR_REPLY


def test_document_reply():
    from kovalo.workflow.replies import document_reply

    text, status = document_reply(
        toy_record(answer="CFO zatwierdza.", citations=["POL-001", "POL-004"])
    )
    assert status == "answered"
    assert text == "CFO zatwierdza.\n\nŹródła: POL-001, POL-004"

    text, status = document_reply(toy_record(answer="Brak informacji.", answerable=False))
    assert (text, status) == ("Brak informacji.", "answered"), "odmowa to odpowiedź, nie błąd"

    text, status = document_reply(toy_record(error="ValueError: x"))
    assert status == "error" and "Nie udało się odpowiedzieć na pytanie o dokumenty" in text
