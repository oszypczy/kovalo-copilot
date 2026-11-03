"""Rozumienie wiadomości z prawdziwym Gemini (marker live): 1 wywołanie (kontrakt F3, 11, 14.2).

Ręcznie: make test-live T=tests/live/test_e3_live.py
Bez cache LLM-a. Wywołanie 1: D1-t2 („A 22%?”) z historią D1-t1 (pytanie i odpowiedź policy_answer).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


def test_understand_follow_up_live(live_google_key: str | None):
    if not live_google_key:
        pytest.skip("brak GOOGLE_API_KEY w środowisku procesu")
    from langchain_core.messages import AIMessage, HumanMessage

    from kovalo.config import load_settings
    from kovalo.llm import get_structured_model
    from kovalo.policy import DiscountRequest, required_approvers
    from kovalo.workflow.replies import policy_answer
    from kovalo.workflow.schemas import Turn
    from kovalo.workflow.understand import understand_turn

    settings = load_settings(google_api_key=live_google_key)
    first_question = "Kto zatwierdza rabat 15% dla zwykłego klienta?"
    request = DiscountRequest(discount_pct=15)
    history = [
        HumanMessage(first_question),
        AIMessage(policy_answer(request, required_approvers(request, settings.policy))),
    ]
    understanding = understand_turn(
        history,
        "A 22%?",
        get_structured_model(Turn, "understand", settings=settings),  # bez cache
        system_prompt=settings.prompts.understand_system,
    )  # wywołanie 1
    assert understanding.call.error is None, understanding.call.error
    turn = understanding.turn
    assert turn.kind == "policy_question"
    assert turn.discount_pct == 22
    assert turn.request_action == "none"
    assert understanding.call.usage is not None and understanding.call.usage.input_tokens > 0
