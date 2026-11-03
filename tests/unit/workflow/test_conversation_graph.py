"""Punkt kontrolny 3a: graf rozmowy bez pauz (kontrakt F3, 6.5; port kroku 3 MP3, 6 testów)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from tests.workflow_fakes import NoAnswerer, ReplayLLM, make_deps, toy_answerer, turn


def conversation_graph(settings, turns, *, answerer=None, llm=None):
    from kovalo.workflow.checkpoint import memory_checkpointer
    from kovalo.workflow.graph import build_conversation_graph

    deps, llm = make_deps(settings, turns, llm=llm, answerer=answerer)
    return build_conversation_graph(deps, checkpointer=memory_checkpointer()), llm


def ask(graph, thread, text):
    cfg = {"configurable": {"thread_id": thread}}
    graph.invoke({"messages": [HumanMessage(text)]}, cfg)
    return graph.get_state(cfg).values


def test_policy_question_answers_without_touching_the_request(settings):
    graph, _ = conversation_graph(settings, [turn(kind="policy_question", discount_pct=22)])
    state = ask(graph, "t", "Kto zatwierdza rabat 22%?")
    assert state["decision"].approvers == ["CEO", "CFO"] and state["status"] == "answered"
    assert state.get("request") is None, "pytanie o politykę nie może utworzyć ani zmienić wniosku"
    assert state["messages"][-1].type == "ai", "każda tura kończy się jedną odpowiedzią"


def test_document_question_keeps_the_run_record(settings):
    llm = ReplayLLM(
        [turn(kind="document_question", standalone="Co komitet postanowił w sprawie wody?")]
    )
    graph, _ = conversation_graph(settings, [], llm=llm, answerer=toy_answerer(settings, llm))
    state = ask(graph, "t", "A ceny wody?")
    assert state["status"] == "answered"
    assert state["record"]["question"] == "Co komitet postanowił w sprawie wody?", (
        "odpowiedz na pytanie STANDALONE i zachowaj RunRecord w 'record'"
    )
    assert state["record"]["context"][0]["chunk_id"] == "TST-001#ctx000"


def test_other_and_error_never_search(settings):
    graph, _ = conversation_graph(
        settings, [turn(kind="other"), turn(kind="error")], answerer=NoAnswerer()
    )
    assert ask(graph, "t", "Dzięki!")["status"] == "answered"
    assert ask(graph, "t", "???")["status"] == "error"


def test_approval_request_updates_state_and_decides(settings):
    graph, _ = conversation_graph(
        settings,
        [
            turn(
                kind="approval_request", request_action="new", discount_pct=14, margin_after_pct=10
            ),
            turn(kind="approval_request", request_action="update", discount_pct=22),
        ],
    )
    state = ask(graph, "t", "Prześlij rabat 14%, marża 10%.")
    assert state["status"] == "pending_approval" and state["decision"].approvers == [
        "Dyrektor handlowy"
    ]
    assert state["request"].revision == 1
    state = ask(graph, "t", "Zmień na 22%.")
    assert state["request"].revision == 2 and state["request"].margin_after_pct == 10, (
        "aktualizacja zachowuje pozostałe pola"
    )
    assert state["decision"].approvers == ["CEO", "CFO"]


def test_threads_do_not_share_state(settings):
    graph, _ = conversation_graph(
        settings,
        [
            turn(
                kind="approval_request", request_action="new", discount_pct=14, margin_after_pct=10
            ),
            turn(kind="policy_question", discount_pct=3),
        ],
    )
    ask(graph, "manager-a", "Prześlij rabat 14%.")
    other = ask(graph, "manager-b", "Kto zatwierdza rabat 3%?")
    assert other.get("request") is None and len(other["messages"]) == 2


def test_an_ambiguous_approval_request_changes_nothing(settings):
    graph, _ = conversation_graph(
        settings,
        [
            turn(
                kind="approval_request", request_action="new", discount_pct=8, margin_after_pct=10
            ),
            turn(kind="approval_request", request_action="none", discount_pct=22),
        ],
    )
    ask(graph, "t", "Prześlij rabat 8%, marża 10%.")
    state = ask(graph, "t", "22%?")
    assert state["status"] == "error" and state["request"].discount_pct == 8
    assert state["request"].revision == 1
    assert "nowy wniosek" in state["messages"][-1].content, "graf zamienia pytanie w odpowiedź"


def test_policy_question_without_discount_asks_for_it(settings):
    graph, _ = conversation_graph(settings, [turn(kind="policy_question")])
    state = ask(graph, "t", "Kto zatwierdza rabat?")
    assert state["status"] == "answered" and state.get("decision") is None
    assert "Podaj wysokość rabatu" in state["messages"][-1].content


def test_policy_question_with_invalid_discount_is_error(settings):
    graph, _ = conversation_graph(settings, [turn(kind="policy_question", discount_pct=150)])
    state = ask(graph, "t", "Kto zatwierdza rabat 150%?")
    assert state["status"] == "error" and state.get("error")
    assert state.get("request") is None


def test_node_names_and_routing_functions():
    from kovalo.workflow import graph as g

    assert g.NODES == (
        "understand",
        "answer_documents",
        "answer_policy",
        "reply_other",
        "update_request",
        "clarify",
        "decide",
        "approve",
    )
    by_kind = {
        "document_question": "answer_documents",
        "policy_question": "answer_policy",
        "other": "reply_other",
        "error": "reply_other",
        "approval_request": "update_request",
    }
    for kind, node in by_kind.items():
        assert g.route_by_kind({"turn": turn(kind=kind)}) == node, kind


def test_only_conversation_graph_stops_without_pauses(settings):
    """Graf 3a nie ma węzłów clarify i approve: wniosek bez marży idzie wprost do decyzji."""
    graph, _ = conversation_graph(
        settings, [turn(kind="approval_request", request_action="new", discount_pct=8)]
    )
    cfg = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [HumanMessage("Prześlij rabat 8%.")]}, cfg)
    assert not graph.get_state(cfg).next, "graf 3a nie wstrzymuje się"
