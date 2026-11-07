"""Agent przez narzędzia MCP w grafie asystenta (kontrakt F5, 9.1 i 13.2): port kroków 3 i 4
kursu (`test_step3_answer_with_tools.py`, `test_step4_build_assistant.py`). Serwer MCP w pamięci,
model ze skryptem (`ScriptedChat`), graf `build_copilot` z gotowym `AgentRunner`; JEDEN punkt
wejścia: `run_turn(..., connection=..., turn=...)` (odpowiednik `assistant_turn` kursu)."""

from __future__ import annotations

import threading

import pytest

from tests.mcp_fakes import ScriptedChat, answer, answers, call, turn

DATA_Q = turn(
    kind="document_question",
    standalone="Na jaką kwotę Mleczna Dolina wystawiła fakturę za lipiec 2026 r.?",
)
SUBMIT_14 = turn(
    kind="approval_request",
    request_action="new",
    standalone="Prześlij do akceptacji rabat 14%, marża po rabacie 10%.",
    discount_pct=14,
    margin_after_pct=10,
    customer="Biedronka",
)
INVOICE_CALL = call("get_invoice_value", supplier="SUP-114", month="2026-07")


def build(settings, connection, turns, script=()):
    """(model, CopilotService, graf) = asystent z agentem na narzędziach MCP (`build_assistant`
    kursu na naszym API: build_copilot z AgentRunner i promptem assistant_understand_system)."""
    from kovalo.agents.agent import AgentRunner, build_agent
    from kovalo.llm import override_chat_model
    from kovalo.workflow.checkpoint import memory_checkpointer
    from kovalo.workflow.factory import build_copilot
    from kovalo.workflow.service import CopilotService

    llm = ScriptedChat(turns=list(turns), script=list(script))
    runner = AgentRunner(
        build_agent(llm, connection.tools(), system_prompt=settings.prompts.agent_system),
        settings.agent,
    )
    with override_chat_model(llm):
        graph = build_copilot(
            settings,
            checkpointer=memory_checkpointer(),
            agent=runner,
            understand_prompt=settings.prompts.assistant_understand_system,
        )
    service = CopilotService(graph, max_message_chars=settings.workflow.max_message_chars)
    return llm, service, graph


def assistant_with(settings, connection, turns, script=()):
    llm, service, _ = build(settings, connection, turns, script)
    return llm, service


def turn_on(service, thread, text, pause, connection=None, n=None):
    from kovalo.eval.workflow.runner import run_turn

    return run_turn(service, thread, text, pause, connection=connection, turn=n)


def trace_of(record):
    from kovalo.agents.trace import trace_from_record

    return trace_from_record(record)


# --- krok 3: węzeł odpowiada agentem przez narzędzia MCP --------------------------------------


def test_the_agent_uses_the_mcp_tools_and_the_trace_is_kept(settings, connection):
    from kovalo.tools.specs import COURSE_TOOLS

    llm, service = assistant_with(
        settings,
        connection,
        [DATA_Q],
        [
            call("get_contract_terms", supplier="SUP-114"),
            INVOICE_CALL,
            answer("Faktura za lipiec: 182 400 PLN (warunki CON-101 sprawdzone)."),
        ],
    )
    result = service.send_message("t", "Ile zafakturowała Mleczna Dolina w lipcu?")
    assert result.status == "answered"
    assert result.replies == ["Faktura za lipiec: 182 400 PLN (warunki CON-101 sprawdzone)."]
    values = service.get_values("t")
    assert values["status"] == "answered" and values["error"] is None
    trace = trace_of(values["record"])
    assert trace.question == DATA_Q.standalone, (
        "pytanie przeredagowane (standalone), nie surowa wiadomość"
    )
    assert [(e.tool, e.status, e.transport) for e in trace.events] == [
        ("get_contract_terms", "ok", "mcp"),
        ("get_invoice_value", "ok", "mcp"),
    ]
    captured = [c["invocation_id"] for c in connection.captures]
    assert [e.invocation_id for e in trace.events] == captured, (
        "każde zdarzenie powiązane ze swoim wywołaniem MCP przez identyfikator"
    )
    assert trace.events[1].result["invoiced_grosze"] == 18240000
    assert sorted(llm.bound) == sorted(COURSE_TOOLS), "agent dostaje dokładnie narzędzia MCP"


def test_the_system_prompt_is_the_agents(settings, connection):
    seen: list[str] = []

    def reply(messages):
        seen.append(messages[0].content)
        return answer("ok")

    _, service = assistant_with(settings, connection, [DATA_Q], [reply])
    service.send_message("t", "q")
    assert seen == [settings.prompts.agent_system]


def test_a_limit_is_an_error_turn_that_keeps_the_trace(settings, connection):
    from kovalo.workflow.replies import AGENT_LIMIT_REPLY

    looping = [lambda m: call("get_contract_terms", supplier="SUP-114")] * 40
    _, service = assistant_with(settings, connection, [DATA_Q], looping)
    result = service.send_message("t", "warunki?")
    values = service.get_values("t")
    trace = trace_of(values["record"])
    assert values["status"] == "error" and trace.status == "limit", (
        "koperta zachowuje własny wynik zapisu przebiegu"
    )
    assert trace.events and "limit kroków" in values["error"]
    assert result.replies == [AGENT_LIMIT_REPLY]


def test_a_model_failure_after_a_call_keeps_the_call(settings, connection):
    _, service = assistant_with(
        settings,
        connection,
        [DATA_Q],
        [call("get_contract_terms", supplier="SUP-207"), RuntimeError("provider down")],
    )
    service.send_message("t", "Warunki Aqua?")
    values = service.get_values("t")
    trace = trace_of(values["record"])
    assert values["status"] == "error" and trace.status == "error"
    assert [(e.tool, e.status) for e in trace.events] == [("get_contract_terms", "ok")]


def test_every_turn_starts_fresh(settings, connection):
    seen: list[list[str]] = []

    def reply(messages):
        seen.append([str(m.content) for m in messages])
        return answer("ok")

    first = turn(kind="document_question", standalone="pierwsze")
    second = turn(kind="document_question", standalone="drugie")
    _, service = assistant_with(settings, connection, [first, second], [reply, reply])
    service.send_message("t", "pierwsze")
    service.send_message("t", "drugie")
    assert "pierwsze" not in seen[1]


def test_an_exception_inside_a_tool_call_is_an_error_event(settings):
    """Narzędzie, którego proces zawodzi (nie MCP): zdarzenie z błędem, a agent odpowiada dalej."""
    from langchain_core.tools import tool

    from kovalo.agents.agent import AgentRunner, build_agent

    @tool
    def get_contract_terms(supplier: str) -> dict:
        """Warunki umowy (dubler testowy, którego proces zawodzi)."""
        raise RuntimeError("connection reset")

    llm = ScriptedChat(
        script=[
            call("get_contract_terms", supplier="SUP-114"),
            answer("Nie udało mi się pobrać warunków."),
        ]
    )
    runner = AgentRunner(
        build_agent(llm, [get_contract_terms], system_prompt=settings.prompts.agent_system),
        settings.agent,
    )
    trace = runner("warunki?")
    assert trace.status == "answered" and llm.agent_calls == 2
    assert [(e.tool, e.status) for e in trace.events] == [("get_contract_terms", "error")]


def test_a_dead_mcp_connection_is_an_error_event_and_the_turn_continues(settings, server):
    """Połączenie padło w trakcie rozmowy: `[błąd transportu]` w zdarzeniu, odpowiedź przeprosin."""
    from tests.mcp_fakes import connect

    conn = connect(server)
    llm, service = assistant_with(
        settings, conn, [DATA_Q], [INVOICE_CALL, answer("Nie mogę teraz sprawdzić faktury.")]
    )
    conn.close()
    result = service.send_message("t", "Ile zafakturowała Mleczna Dolina?")
    trace = trace_of(service.get_values("t")["record"])
    assert result.status == "answered"
    assert trace.events[0].status == "error" and "[błąd transportu]" in trace.events[0].error
    assert llm.agent_calls == 2


# --- krok 4: cały asystent przez jeden punkt wejścia ------------------------------------------


def test_a_data_question_goes_to_the_agent_over_mcp(settings, connection):
    llm, service = assistant_with(
        settings, connection, [DATA_Q], [INVOICE_CALL, answer("182 400 PLN.")]
    )
    entry = turn_on(
        service, "t", "Ile zafakturowała Mleczna Dolina w lipcu?", answers(), connection, 1
    )
    assert entry["turn"]["kind"] == "document_question" and entry["status"] == "answered"
    assert entry["reply"] == "182 400 PLN."
    trace = trace_of(entry["record"])
    assert trace.events[0].transport == "mcp" and llm.understand_calls == 1
    assert connection.captures[0]["thread_id"] == "t" and connection.captures[0]["turn"] == 1


def test_the_mp5_prompt_is_used_and_the_config_is_untouched(settings, connection):
    before = (settings.prompts.understand_system, settings.workflow.understand_prompt)
    llm, service = assistant_with(settings, connection, [turn(kind="other")])
    turn_on(service, "t", "Cześć", answers())
    assert llm.seen_systems == [settings.prompts.assistant_understand_system]
    assert settings.prompts.assistant_understand_system != settings.prompts.understand_system
    assert (settings.prompts.understand_system, settings.workflow.understand_prompt) == before


def test_a_policy_question_uses_mp3s_rules_with_no_tool_call(settings, connection):
    policy = turn(
        kind="policy_question", discount_pct=22, margin_after_pct=7, strategic_account=True
    )
    llm, service = assistant_with(settings, connection, [policy])
    entry = turn_on(
        service,
        "t",
        "Kto zatwierdza rabat 22% dla klienta strategicznego przy marży po rabacie 7%?",
        answers(),
    )
    assert entry["approvers"] == ["Dyrektor handlowy", "CFO"] and entry["pauses"] == []
    assert llm.agent_calls == 0 and not connection.captures and trace_of(entry["record"]) is None


def test_policy_then_submit_that(settings, connection):
    policy = turn(
        kind="policy_question", discount_pct=22, margin_after_pct=7, strategic_account=True
    )
    submit = turn(
        kind="approval_request",
        request_action="new",
        discount_pct=22,
        margin_after_pct=7,
        strategic_account=True,
    )
    llm, service = assistant_with(settings, connection, [policy, submit])
    turn_on(service, "t", "Kto zatwierdza 22%...?", answers())
    entry = turn_on(service, "t", "Prześlij to.", answers("approved", "approved"))
    assert entry["pauses"] == [["approval", "Dyrektor handlowy"], ["approval", "CFO"]]
    assert entry["status"] == "approved"
    assert llm.understand_calls == 2, (
        "wznowienie nigdy nie uruchamia ponownie rozumienia wiadomości"
    )
    assert llm.agent_calls == 0


def test_approval_then_a_data_question_then_an_unchanged_resubmission(settings, connection):
    again = turn(
        kind="approval_request", request_action="update", discount_pct=14, margin_after_pct=10
    )
    llm, service = assistant_with(
        settings, connection, [SUBMIT_14, DATA_Q, again], [INVOICE_CALL, answer("182 400 PLN.")]
    )
    first = turn_on(service, "t", "Prześlij do akceptacji rabat 14%.", answers("approved"))
    assert first["status"] == "approved"
    assert first["pauses"] == [["approval", "Dyrektor handlowy"]]
    question = turn_on(service, "t", "Ile zafakturowała Mleczna Dolina w lipcu?", answers())
    assert question["request"]["revision"] == 1
    assert question["approvals"] == {"Dyrektor handlowy": "approved"}, (
        "pytanie o dane nie zmienia zatwierdzonego wniosku"
    )
    resubmitted = turn_on(service, "t", "Prześlij jeszcze raz ten sam rabat.", answers())
    assert resubmitted["pauses"] == [] and resubmitted["status"] == "approved"
    assert resubmitted["request"]["revision"] == 1
    assert trace_of(resubmitted["record"]) is None, "na innej ścieżce nie zostaje nieaktualny trace"
    assert llm.understand_calls == 3


def test_a_revised_resubmission_needs_new_approval(settings, connection):
    revised = turn(
        kind="approval_request", request_action="update", discount_pct=22, margin_after_pct=10
    )
    _, service = assistant_with(settings, connection, [SUBMIT_14, revised])
    turn_on(service, "t", "Prześlij rabat 14%.", answers("approved"))
    entry = turn_on(service, "t", "Jednak zróbmy 22%.", answers("approved", "approved"))
    assert entry["request"]["revision"] == 2
    assert entry["pauses"] == [["approval", "CEO"], ["approval", "CFO"]]


@pytest.mark.parametrize("outcome", ["rejected", "cancel"])
def test_a_rejected_or_cancelled_request_retries_from_the_first_role(settings, connection, outcome):
    again = turn(
        kind="approval_request", request_action="update", discount_pct=14, margin_after_pct=10
    )
    _, service = assistant_with(settings, connection, [SUBMIT_14, again])
    turn_on(service, "t", "Prześlij rabat 14%.", answers(outcome))
    retry = turn_on(
        service, "t", "Spróbuj jeszcze raz na tych samych warunkach.", answers("approved")
    )
    assert retry["pauses"] == [["approval", "Dyrektor handlowy"]] and retry["status"] == "approved"


def test_a_clarified_margin_is_used_by_a_later_question(settings, connection):
    missing = turn(kind="approval_request", request_action="new", discount_pct=8)
    later = turn(kind="policy_question", discount_pct=12, margin_after_pct=7)
    llm, service = assistant_with(settings, connection, [missing, later])
    first = turn_on(
        service,
        "t",
        "Prześlij do akceptacji rabat 8% na chipsy.",
        answers(7, "approved", "approved"),
    )
    assert first["pauses"][0] == ["clarification", "margin_after_pct"]
    assert first["status"] == "approved"
    entry = turn_on(service, "t", "Przy tej samej marży kto zatwierdziłby 12%?", answers())
    assert entry["approvers"] == ["Regionalny kierownik sprzedaży", "CFO"]
    assert llm.understand_calls == 2
    assert any("Marża po rabacie: 7" in text for text in llm.seen_understand[1]), (
        "doprecyzowana marża jest w historii, którą widzi następne rozumienie"
    )


def test_a_paused_thread_refuses_a_new_message(settings, connection):
    _, service = assistant_with(settings, connection, [SUBMIT_14])
    first = service.send_message("t", "Prześlij rabat 14%.")  # wątek zostaje zatrzymany
    assert first.status == "pending_approval"
    with pytest.raises(RuntimeError, match="pauza"):
        turn_on(service, "t", "Ile zafakturowała Mleczna Dolina?", answers())
    with pytest.raises(RuntimeError, match="pauza"):
        turn_on(service, "t", "Ile zafakturowała Mleczna Dolina?", answers(), connection, 2)
    assert not connection.captures


def test_two_interleaved_threads_keep_their_own_state(settings, connection):
    _, service = assistant_with(
        settings,
        connection,
        [DATA_Q, SUBMIT_14, DATA_Q],
        [
            INVOICE_CALL,
            answer("A"),
            call("get_invoice_value", supplier="SUP-207", month="2026-07"),
            answer("B"),
        ],
    )
    one = turn_on(service, "one", "q1", answers(), connection, 1)
    two = turn_on(service, "two", "Prześlij rabat 14%.", answers("approved"), connection, 1)
    one_again = turn_on(service, "one", "q2", answers(), connection, 2)
    assert two["request"]["discount_pct"] == 14 and one_again["request"] is None, (
        "wnioski nie przeciekają między wątkami"
    )
    assert trace_of(one["record"]).answer == "A" and trace_of(one_again["record"]).answer == "B"
    assert [(c["thread_id"], c["turn"]) for c in connection.captures] == [("one", 1), ("one", 2)]


def test_the_trace_survives_a_checkpoint(settings, connection):
    _, service = assistant_with(
        settings, connection, [DATA_Q], [INVOICE_CALL, answer("182 400 PLN.")]
    )
    turn_on(service, "t", "q", answers())
    stored = service.get_values("t")["record"]
    assert trace_of(stored).events[0].result["invoiced_grosze"] == 18240000
    assert trace_of(stored).events[0].transport == "mcp"


def test_overlapping_turns_on_one_connection_are_refused_and_captures_keep_their_owner(
    settings, connection
):
    """Dwie rozmowy, jedno połączenie. Pierwsza tura jest wstrzymana wewnątrz wywołania agenta;
    druga na tym samym połączeniu musi zostać odrzucona (`turn_lock`) i nie może błędnie oznaczyć
    dowodów pierwszej."""
    entered, release = threading.Event(), threading.Event()

    def held(messages):
        entered.set()
        release.wait(10)
        return answer("A")

    _, first = assistant_with(settings, connection, [DATA_Q], [INVOICE_CALL, held])
    _, second = assistant_with(settings, connection, [DATA_Q], [answer("B")])
    results: dict = {}
    worker = threading.Thread(
        target=lambda: results.setdefault(
            "one", turn_on(first, "one", "q", answers(), connection, 1)
        )
    )
    worker.start()
    assert entered.wait(10)
    with pytest.raises(RuntimeError, match="jedna tura naraz"):
        turn_on(second, "two", "q", answers(), connection, 1)
    release.set()
    worker.join(10)
    assert results["one"]["status"] == "answered"
    assert [(c["thread_id"], c["turn"]) for c in connection.captures] == [("one", 1)]
    event = trace_of(results["one"]["record"]).events[0]
    assert event.invocation_id == connection.captures[0]["invocation_id"]


def test_the_chat_path_accepts_polish_answers_at_the_pauses(settings, connection):
    """Odpowiedzi w pauzach po polsku przez `text_pl.human_answer` (jak `chat.ask_human` kursu)."""
    from kovalo.text_pl import human_answer

    submit = turn(
        kind="approval_request",
        request_action="new",
        standalone="Prześlij rabat 8%.",
        discount_pct=8,
    )
    _, service = assistant_with(settings, connection, [submit])
    typed = iter(["7,5", "zatwierdź", "zatwierdź"])

    def ask_human(payload):
        return human_answer(payload["type"], next(typed))

    entry = turn_on(service, "t", "Prześlij rabat 8% na chipsy.", ask_human)
    assert entry["request"]["margin_after_pct"] == 7.5 and entry["status"] == "approved"


def test_run_turn_without_a_connection_is_unchanged(settings, connection):
    """`connection` opcjonalne: bez niego run_turn działa jak w F3 (E3 nie zmienia się)."""
    _, service = assistant_with(settings, connection, [turn(kind="other")])
    entry = turn_on(service, "t", "Dzięki!", answers())
    assert entry["status"] == "answered" and not connection.captures
