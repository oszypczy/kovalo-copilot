"""Pełny graf copilota z pauzami (kontrakt F3, 6.5; port kroku 4 MP3, 13 testów, przez run_turn)
oraz ścieżka czatu (`test_provided_polish_input.py`, 2 testy) przez CopilotService."""

from __future__ import annotations

from tests.workflow_fakes import (
    NoAnswerer,
    ReplayLLM,
    answers,
    service_and_llm,
    service_with,
    toy_answerer,
    turn,
)


def run_turn(*args, **kwargs):
    from kovalo.eval.workflow.runner import run_turn as _run_turn

    return _run_turn(*args, **kwargs)


def new_request(**fields):
    return turn(kind="approval_request", request_action="new", **fields)


def update_request(**fields):
    return turn(kind="approval_request", request_action="update", **fields)


def test_clarification_then_role_by_role_approval(settings):
    service, llm = service_and_llm(settings, [new_request(discount_pct=8)])
    t = run_turn(
        service, "t", "Prześlij rabat 8% na chipsy.", answers("nie wiem", 7, "approved", "approved")
    )
    assert t["pauses"] == [
        ["clarification", "margin_after_pct"],
        ["clarification", "margin_after_pct"],
        ["approval", "Regionalny kierownik sprzedaży"],
        ["approval", "CFO"],
    ], "zapytaj o marżę (ponownie, gdy to nie liczba), potem jedna pauza na każdą rolę"
    assert t["status"] == "approved" and t["request"]["margin_after_pct"] == 7
    assert t["request"]["revision"] == 1
    assert llm.calls["Turn"] == 1, "wznowienie nie może ponownie uruchomić zrozumienia wiadomości"


def test_rejection_ends_and_cancel_cancels(settings):
    service = service_with(
        settings,
        [
            new_request(discount_pct=22, margin_after_pct=10),
            new_request(discount_pct=22, margin_after_pct=10),
        ],
    )
    t = run_turn(service, "t", "Prześlij rabat 22%.", answers("approved", "rejected"))
    assert t["pauses"] == [["approval", "CEO"], ["approval", "CFO"]] and t["status"] == "rejected"
    t = run_turn(service, "t", "Prześlij ponownie rabat 22%.", answers("cancel"))
    assert t["pauses"] == [["approval", "CEO"]] and t["status"] == "cancelled"


def test_an_update_needs_new_approval(settings):
    service = service_with(
        settings,
        [new_request(discount_pct=14, margin_after_pct=10), update_request(discount_pct=18)],
    )
    first = run_turn(service, "t", "Prześlij rabat 14%.", answers("approved"))
    assert first["status"] == "approved"
    second = run_turn(service, "t", "Zmień na 18%.", answers("approved"))
    assert second["request"]["revision"] == 2
    assert second["pauses"] == [["approval", "Dyrektor handlowy"]], (
        "nowa wersja wniosku wymaga nowej zgody"
    )


def test_clearance_without_margin_needs_no_clarification(settings):
    service = service_with(settings, [new_request(discount_pct=30, clearance=True)])
    t = run_turn(service, "t", "Prześlij rabat 30% na wyprzedaż końcówek.", answers("approved"))
    assert t["pauses"] == [["approval", "Regionalny kierownik sprzedaży"]], (
        "minimalna marża nie dotyczy wyprzedaży końcówek"
    )


def test_policy_questions_never_pause(settings):
    service = service_with(
        settings, [turn(kind="policy_question", discount_pct=22, margin_after_pct=7)]
    )
    t = run_turn(service, "t", "Kto zatwierdza rabat 22% przy marży 7%?", answers())
    assert t["pauses"] == [] and t["status"] == "answered" and t["approvers"] == ["CEO", "CFO"]


# --- Przejścia cyklu życia: approved znaczy, że każda wymagana rola zatwierdziła BIEŻĄCY wniosek.


def assert_fully_approved(t):
    assert t["status"] == "approved"
    assert t["approvers"] and all(
        t["approvals"].get(role) == "approved" for role in t["approvers"]
    ), f"'approved' wymaga 'approved' u każdej wymaganej roli; jest {t['approvals']}"


def test_rejection_then_unchanged_resubmission_starts_a_fresh_cycle(settings):
    service = service_with(
        settings,
        [new_request(discount_pct=22, margin_after_pct=10), update_request(discount_pct=22)],
    )
    first = run_turn(
        service, "t", "Prześlij rabat 22%, marża 10%.", answers("approved", "rejected")
    )
    assert first["status"] == "rejected"
    again = run_turn(service, "t", "Prześlij go ponownie, 22%.", answers("approved", "approved"))
    assert again["pauses"] == [["approval", "CEO"], ["approval", "CFO"]], (
        "odrzucenie nie zamienia się w zgodę"
    )
    assert again["request"]["revision"] == 1
    assert_fully_approved(again)


def test_rejection_by_the_first_role_ends_the_attempt(settings):
    service = service_with(settings, [new_request(discount_pct=22, margin_after_pct=10)])
    t = run_turn(service, "t", "Prześlij rabat 22%.", answers("rejected"))
    assert t["pauses"] == [["approval", "CEO"]] and t["status"] == "rejected"


def test_cancel_after_one_approval_then_retry_starts_fresh(settings):
    service = service_with(
        settings,
        [new_request(discount_pct=22, margin_after_pct=10), update_request(discount_pct=22)],
    )
    assert (
        run_turn(service, "t", "Prześlij rabat 22%.", answers("approved", "cancel"))["status"]
        == "cancelled"
    )
    retry = run_turn(service, "t", "Prześlij go ponownie.", answers("approved", "approved"))
    assert retry["pauses"] == [["approval", "CEO"], ["approval", "CFO"]], (
        "zgody z anulowanej próby nie przechodzą"
    )
    assert_fully_approved(retry)


def test_approved_unchanged_resubmission_needs_no_new_pause(settings):
    service = service_with(
        settings,
        [new_request(discount_pct=14, margin_after_pct=10), update_request(discount_pct=14)],
    )
    run_turn(service, "t", "Prześlij rabat 14%.", answers("approved"))
    again = run_turn(service, "t", "Prześlij ponownie rabat 14%.", answers())
    assert again["pauses"] == [] and "Już zatwierdzone" in again["reply"]
    assert_fully_approved(again)


def test_cancellation_does_not_leak_into_the_next_request(settings):
    service = service_with(
        settings, [new_request(discount_pct=8, margin_after_pct=10), new_request(discount_pct=8)]
    )
    assert (
        run_turn(service, "t", "Prześlij rabat 8%, marża 10%.", answers("cancel"))["status"]
        == "cancelled"
    )
    t = run_turn(service, "t", "Nowe zamówienie: rabat 8% na chipsy.", answers(10, "approved"))
    assert t["pauses"] == [
        ["clarification", "margin_after_pct"],
        ["approval", "Regionalny kierownik sprzedaży"],
    ]
    assert_fully_approved(t)


def test_cancel_at_the_clarification_itself(settings):
    service = service_with(settings, [new_request(discount_pct=8)])
    t = run_turn(service, "t", "Prześlij rabat 8%.", answers("cancel"))
    assert t["pauses"] == [["clarification", "margin_after_pct"]] and t["status"] == "cancelled"


def test_the_clarification_answer_is_in_the_next_prompt(settings):
    service, llm = service_and_llm(
        settings,
        [
            new_request(discount_pct=22),
            turn(kind="policy_question", discount_pct=18, margin_after_pct=7.37),
        ],
    )
    run_turn(service, "t", "Prześlij rabat 22%.", answers(7.37, "approved", "approved"))
    run_turn(service, "t", "Kto zatwierdziłby rabat 18% przy tej samej marży?", answers())
    human_lines = [m.content for m in llm.prompts[-1] if m.type == "human"]
    assert any("7,37" in line for line in human_lines[:-1]), (
        "odpowiedź z doprecyzowania trafia do rozmowy jako wiadomość menedżera"
    )


# --- Ścieżka czatu (port 2 testów test_provided_polish_input.py): service.resume(text=...)


def test_the_chat_path_accepts_a_polish_margin_and_polish_commands(settings):
    service = service_with(settings, [new_request(discount_pct=8)])
    first = service.send_message("t", "Prześlij rabat 8% na chipsy.")
    assert first.status == "pending_clarification"
    assert (first.pending.type, first.pending.field) == ("clarification", "margin_after_pct")
    assert first.pending.options == ["cancel"] and first.pending.question

    result = service.resume("t", text="7,5")
    assert result.request["margin_after_pct"] == 7.5 and result.resume_rejected is False
    assert (
        result.pending.type == "approval"
        and result.pending.role == "Regionalny kierownik sprzedaży"
    )

    result = service.resume("t", text="może")
    assert result.resume_rejected is True, "„może” to nie decyzja"
    assert result.pending.role == "Regionalny kierownik sprzedaży", (
        "graf ponownie pyta tę samą rolę"
    )

    result = service.resume("t", text="zatwierdź")
    assert result.resume_rejected is False and result.pending.role == "CFO"
    result = service.resume("t", text="Zatwierdź")
    assert result.status == "approved" and result.pending is None
    assert result.approvals == {"Regionalny kierownik sprzedaży": "approved", "CFO": "approved"}


def test_the_chat_path_cancels_in_polish(settings):
    service = service_with(settings, [new_request(discount_pct=8)])
    service.send_message("t", "Prześlij rabat 8% na chipsy.")
    result = service.resume("t", text="anuluj")
    assert result.status == "cancelled" and result.pending is None


def test_clarify_out_of_range_reasks(settings):
    service = service_with(settings, [new_request(discount_pct=8)])
    service.send_message("t", "Prześlij rabat 8%.")
    result = service.resume("t", value=150)
    assert result.resume_rejected is True, "150 nie jest marżą"
    assert result.pending.type == "clarification" and result.pending.field == "margin_after_pct"
    assert result.request["margin_after_pct"] is None, "wniosek bez marży"
    assert service.resume("t", value=-100.5).pending.type == "clarification"
    ok = service.resume("t", value=100)
    assert ok.pending.type == "approval" and ok.request["margin_after_pct"] == 100


def test_boolean_is_not_a_margin(settings):
    service = service_with(settings, [new_request(discount_pct=8)])
    service.send_message("t", "Prześlij rabat 8%.")
    result = service.resume("t", value=True)  # type: ignore[arg-type]
    assert result.pending.type == "clarification"


def test_approval_request_without_discount_is_error(settings):
    service = service_with(
        settings,
        [new_request(discount_pct=8, margin_after_pct=10), new_request(margin_after_pct=10)],
    )
    service.send_message("t", "Prześlij rabat 8%, marża 10%.")
    service.resume("t", value="cancel")
    first = service.send_message("t", "Prześlij rabat.")
    assert first.status == "error" and first.pending is None and first.error
    state = service.get_thread("t")
    assert state.request["discount_pct"] == 8 and state.request["revision"] == 1


def test_document_errors_become_status_error(settings):
    from kovalo.knowledge.pgvector import IndexUnavailableError
    from kovalo.rag.records import RunRecord

    def unavailable(question):
        raise IndexUnavailableError("brak Postgresa")

    service = service_with(
        settings,
        [turn(kind="document_question", standalone="Pytanie?")],
        answerer=unavailable,
    )
    result = service.send_message("t", "Pytanie?")
    assert result.status == "error" and result.error and "brak Postgresa" in result.error
    assert result.replies and "Nie udało się odpowiedzieć" in result.replies[0]

    def failed_record(question):
        return RunRecord(question=question, context=[], error="ValueError: model")

    service = service_with(
        settings, [turn(kind="document_question", standalone="Pytanie?")], answerer=failed_record
    )
    entry = run_turn(service, "t", "Pytanie?", answers())
    assert entry["status"] == "error"
    assert entry["record"]["error"] == "ValueError: model", "record zapisany także przy błędzie"


def test_calls_recorded_per_turn(settings):
    llm = ReplayLLM([turn(kind="document_question", standalone="Co z wodą?"), turn(kind="other")])
    service = service_with(settings, llm=llm, answerer=toy_answerer(settings, llm))
    first = run_turn(service, "t", "Co z wodą?", answers())
    assert [c["purpose"] for c in first["calls"]] == ["understand", "answer"]
    second = run_turn(service, "t", "Dzięki!", answers())
    assert [c["purpose"] for c in second["calls"]] == ["understand"], "kolejna tura zaczyna od nowa"


def test_policy_question_does_not_touch_existing_request_or_approvals(settings):
    service = service_with(
        settings,
        [
            new_request(discount_pct=22, margin_after_pct=10),
            turn(kind="policy_question", discount_pct=3),
        ],
    )
    run_turn(service, "t", "Prześlij rabat 22%.", answers("approved", "rejected"))
    run_turn(service, "t", "Kto zatwierdza 3%?", answers())
    state = service.get_thread("t")
    assert state.request["discount_pct"] == 22
    assert state.approvals == {"CEO": "approved", "CFO": "rejected"}


def test_error_turn_replies_with_hint(settings):
    service = service_with(settings, [turn(kind="error")])
    result = service.send_message("t", "???")
    assert result.status == "error" and "Nie zrozumiałem wiadomości" in result.replies[0]


def test_resume_does_not_call_the_model_again(settings):
    """ScriptedUnderstanding z jednym Turn: ponowne wywołanie understand = AssertionError."""
    from kovalo.eval.workflow.runner import ScriptedUnderstanding
    from kovalo.workflow.checkpoint import memory_checkpointer
    from kovalo.workflow.graph import build_copilot_graph
    from kovalo.workflow.service import CopilotService
    from tests.workflow_fakes import make_deps

    scripted = ScriptedUnderstanding([new_request(discount_pct=22, margin_after_pct=10)])
    deps, _ = make_deps(settings, understand_model=scripted.as_runnable(), answerer=NoAnswerer())
    service = CopilotService(build_copilot_graph(deps, checkpointer=memory_checkpointer()))
    service.send_message("t", "Prześlij rabat 22%.")
    service.resume("t", value="approved")
    result = service.resume("t", value="approved")
    assert result.status == "approved" and scripted.calls == 1


def test_run_turn_entry_keys(settings):
    service = service_with(settings, [new_request(discount_pct=14, margin_after_pct=10)])
    t = run_turn(service, "t", "Prześlij rabat 14%.", answers("approved"))
    assert {
        "user", "turn", "request", "approvers", "reasons", "pauses", "approvals",
        "status", "record", "error", "reply", "calls", "seconds",
    } <= set(t)  # fmt: skip
    assert t["user"] == "Prześlij rabat 14%." and t["turn"]["kind"] == "approval_request"
    assert t["approvals"] == {"Dyrektor handlowy": "approved"} and t["error"] is None
    assert isinstance(t["reasons"], list) and t["reasons"]
    assert t["calls"][0]["purpose"] == "understand" and t["seconds"] >= 0
