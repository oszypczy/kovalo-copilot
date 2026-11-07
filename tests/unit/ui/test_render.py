"""render_turn i render_error: czyste funkcje widoku (kontrakt F3, 9.2)."""

from __future__ import annotations

import httpx
import pytest


def view(**fields):
    from kovalo.ui.client import PendingView, SourceView, TurnView

    base = {"thread_id": "T1", "status": "answered", "replies": []}
    base.update(fields)
    if isinstance(base.get("pending"), dict):
        base["pending"] = PendingView(**base["pending"])
    base["sources"] = [
        SourceView(**s) if isinstance(s, dict) else s for s in base.get("sources", [])
    ]
    return TurnView(**base)


APPROVAL = {
    "type": "approval",
    "role": "CFO",
    "revision": 2,
    "approvers": ["CEO", "CFO"],
    "summary": "klient: Carrefour Express, rabat: 22%, wersja wniosku: 2",
    "options": ["approved", "rejected", "cancel"],
}


def test_render_approval_pause():
    from kovalo.ui.render import render_turn

    messages = render_turn(
        view(status="pending_approval", replies=["Opis wniosku."], pending=APPROVAL)
    )
    assert messages[0].content == "Opis wniosku." and not messages[0].actions
    pause = messages[-1]
    assert "**Akceptacja (symulacja): CFO**" in pause.content
    assert "CEO" in pause.content, "wymagani zatwierdzający"
    assert "klient: Carrefour Express, rabat: 22%, wersja wniosku: 2" in pause.content
    assert [(a.value, a.label) for a in pause.actions] == [
        ("approved", "Zatwierdź"),
        ("rejected", "Odrzuć"),
        ("cancel", "Anuluj"),
    ]
    for action in pause.actions:
        assert action.pause_ref == {"type": "approval", "key": "CFO", "revision": 2}
    assert "Status:" not in pause.content, "status końcowy tylko bez pauzy"


def test_render_resume_rejected_comes_before_the_pause():
    from kovalo.ui.render import render_turn

    messages = render_turn(view(status="pending_approval", resume_rejected=True, pending=APPROVAL))
    texts = [m.content for m in messages]
    rejected = next(i for i, t in enumerate(texts) if t.startswith("Nie rozpoznałem odpowiedzi."))
    pause = next(i for i, t in enumerate(texts) if "Akceptacja (symulacja)" in t)
    assert rejected < pause


def test_render_clarification_and_status():
    from kovalo.ui.render import render_turn

    pending = {
        "type": "clarification",
        "field": "margin_after_pct",
        "question": "Jaka będzie marża brutto po rabacie (w procentach)?",
        "options": ["cancel"],
    }
    pause = render_turn(view(status="pending_clarification", replies=[], pending=pending))[-1]
    assert "Jaka będzie marża brutto po rabacie" in pause.content
    assert "Wpisz marżę w polu czatu (np. 7,5)." in pause.content
    assert [(a.value, a.label) for a in pause.actions] == [("cancel", "Anuluj")]
    assert pause.actions[0].pause_ref["type"] == "clarification"
    assert pause.actions[0].pause_ref["key"] == "margin_after_pct"


@pytest.mark.parametrize(
    ("status", "label"),
    [
        ("answered", "odpowiedź"),
        ("approved", "zatwierdzony (symulacja)"),
        ("rejected", "odrzucony"),
        ("cancelled", "anulowany"),
        ("error", "błąd"),
    ],
)
def test_final_status_is_last_line_in_polish(status, label):
    from kovalo.ui.render import render_turn

    messages = render_turn(view(status=status, replies=["Pierwsza.", "Druga."]))
    assert len(messages) == 2
    assert messages[-1].content.splitlines()[-1] == f"Status: {label}"
    assert "Status:" not in messages[0].content, "status tylko w ostatniej wiadomości"
    assert not messages[-1].actions


def test_sources_attached_to_last_reply():
    from kovalo.ui.render import UiSource, render_turn

    sources = [
        {"doc_id": "MTG-402", "chunk_id": "MTG-402#ctx001", "text": "Komitet utrzymał ceny."}
    ]
    messages = render_turn(view(replies=["Ceny bez zmian."], sources=sources))
    assert messages[-1].sources == [
        UiSource(name="MTG-402#ctx001", content="Komitet utrzymał ceny.")
    ]
    two = render_turn(view(replies=["A", "B"], sources=sources))
    assert not two[0].sources and len(two[-1].sources) == 1


def test_label_tables():
    from kovalo.ui.render import ACTION_LABELS, STATUS_LABELS

    assert ACTION_LABELS == {"approved": "Zatwierdź", "rejected": "Odrzuć", "cancel": "Anuluj"}
    assert STATUS_LABELS["pending_approval"] == "czeka na akceptację"
    assert STATUS_LABELS["pending_clarification"] == "czeka na doprecyzowanie"


async def _api_error(status: int, error: str, detail: str):
    from kovalo.ui.client import ApiCallError, CopilotApiClient

    def handler(request: httpx.Request) -> httpx.Response:
        if status == 0:
            raise httpx.ConnectError("odmowa")
        return httpx.Response(status, json={"error": error, "detail": detail, "pending": None})

    client = CopilotApiClient(
        "http://api.test", timeout_s=1, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ApiCallError) as info:
        await client.send_message("T1", "x")
    return info.value


@pytest.mark.anyio
async def test_render_error_messages():
    from kovalo.ui.render import render_error

    unavailable = render_error(await _api_error(503, "copilot_unavailable", "Brak klucza LLM-a."))
    text = unavailable.content
    assert any(word in text for word in ("GOOGLE_API_KEY", "klucz", "Postgres")), text
    unreachable = render_error(await _api_error(0, "api_unreachable", "timeout"))
    assert "api nie odpowiada" in unreachable.content.lower()
    assert not unavailable.actions and not unreachable.actions


# ---------------------------------------------------------------------------
# F4: kroki agenta (kontrakt F4, 10.2)
# ---------------------------------------------------------------------------


def tool_view(**fields):
    from kovalo.ui.client import ToolEventView

    base = {
        "call_id": "c1",
        "tool": "get_contract_terms",
        "arguments": {"supplier": "Mleczna Dolina"},
        "origin": "data",
    }
    base.update(fields)
    return ToolEventView(**base)


def test_render_tool_steps():
    import json

    from kovalo.ui.render import UiStep, render_tool_end, render_tool_start

    start = render_tool_start(tool_view())
    assert isinstance(start, UiStep)
    assert (start.key, start.parent_key, start.name) == ("c1", None, "get_contract_terms")
    assert json.loads(start.input) == {"supplier": "Mleczna Dolina"}
    assert start.output == "" and start.is_error is False
    assert start.input == json.dumps({"supplier": "Mleczna Dolina"}, ensure_ascii=False, indent=2)

    unicode_args = render_tool_start(tool_view(arguments={"question": "wypowiedzenie umowy żółć"}))
    assert "żółć" in unicode_args.input, "ensure_ascii=False"

    end = render_tool_end(tool_view(status="ok", brief='ok: {"supplier_id": "SUP-114"}'))
    assert (end.key, end.output, end.is_error) == ("c1", 'ok: {"supplier_id": "SUP-114"}', False)

    derived = render_tool_start(
        tool_view(
            call_id="c2#get_supplier_otif",
            parent_call_id="c2",
            tool="get_supplier_otif",
            agent="check_termination_condition",
        )
    )
    assert derived.parent_key == "c2" and derived.key == "c2#get_supplier_otif"
    assert derived.name == "↳ get_supplier_otif (odczyt narzędzia)"
    derived_end = render_tool_end(
        tool_view(
            call_id="c2#get_supplier_otif",
            parent_call_id="c2",
            tool="get_supplier_otif",
            brief="ok: x",
        )
    )
    assert derived_end.parent_key == "c2" and derived_end.name == derived.name

    failed = render_tool_end(
        tool_view(status="error", error="Błąd: RuntimeError: embedding service unavailable")
    )
    assert failed.is_error is True
    assert "RuntimeError: embedding service unavailable" in failed.output
    assert failed.output.startswith("Błąd: ") and not failed.output.startswith("Błąd: Błąd:")


def test_render_evidence():
    from kovalo.ui.render import render_evidence

    agent_turn = view(
        replies=["Kara wynosi 7296,00 PLN."],
        evidence_path="P1  Ile wynosi kara?\nstatus: answered",
        trace={"status": "answered"},
    )
    step = render_evidence(agent_turn)
    assert step is not None and step.name == "Ścieżka dowodowa"
    assert step.output == "P1  Ile wynosi kara?\nstatus: answered"
    assert render_evidence(view(replies=["Odpowiedź RAG."])) is None, "tylko tura z agentem"
    assert render_evidence(view(status="pending_approval", pending=APPROVAL)) is None


def test_trace_status_labels_and_limit_line():
    from kovalo.ui.render import TRACE_STATUS_LABELS, render_turn

    assert TRACE_STATUS_LABELS == {
        "answered": "odpowiedź",
        "error": "błąd",
        "limit": "limit kroków lub czasu",
    }
    limited = render_turn(
        view(status="error", replies=["Przerwałem pracę."], trace={"status": "limit"})
    )
    assert limited[-1].content.splitlines()[-1] == "Status: limit kroków lub czasu"
    normal = render_turn(
        view(status="answered", replies=["Odpowiedź."], trace={"status": "answered"})
    )
    assert normal[-1].content.splitlines()[-1] == "Status: odpowiedź"


def test_message_text_links_side_elements():
    from kovalo.ui.render import EVIDENCE_LINK, UiMessage, UiSource, UiStep, message_text

    evidence = UiStep(
        key="evidence", parent_key=None, name="Ścieżka dowodowa", input="", output="x"
    )
    plain = UiMessage(content="Odpowiedź.\n\nStatus: odpowiedź")
    assert message_text(plain) == plain.content
    source = UiSource(name="CON-114#ctx002", content="t")
    sourced = UiMessage(content="Odpowiedź.", sources=[source])
    assert message_text(sourced) == "Odpowiedź.\n\nFragmenty: CON-114#ctx002"
    assert message_text(sourced, evidence).endswith("\n\nSzczegóły: Ścieżka dowodowa")
    assert EVIDENCE_LINK == "Szczegóły: Ścieżka dowodowa"
    assert message_text(UiMessage(content=""), evidence) == EVIDENCE_LINK


# ---------------------------------------------------------------------------
# F5: krok „przez MCP” (kontrakt F5, 9.2 i 13.2)
# ---------------------------------------------------------------------------


def test_mcp_step_label():
    """`transport == "mcp"` → nazwa kroku z dopiskiem „(przez MCP)”, a w wyniku
    „Wywołanie MCP: <invocation_id>”; bez transportu jak F4; zdarzenie składowe dziedziczy
    dopisek rodzica; `tool_start` bez zmian."""
    from kovalo.ui.render import render_tool_end, render_tool_start

    plain = render_tool_end(tool_view(status="ok", brief="ok: x"))
    assert plain.name == "get_contract_terms" and "MCP" not in plain.output

    via = render_tool_end(
        tool_view(status="ok", brief="ok: x", transport="mcp", invocation_id="abc123def456")
    )
    assert via.name == "get_contract_terms (przez MCP)"
    assert "ok: x" in via.output and "Wywołanie MCP: abc123def456" in via.output
    assert via.key == "c1" and via.is_error is False

    start = render_tool_start(tool_view(transport="mcp", invocation_id="abc123def456"))
    assert start.name == "get_contract_terms", "tool_start: transport jeszcze nieznany"

    derived = render_tool_end(
        tool_view(
            call_id="c2#get_supplier_otif",
            parent_call_id="c2",
            tool="get_supplier_otif",
            brief="ok: x",
            transport="mcp",
            invocation_id="abc123def456",
        )
    )
    assert derived.name == "↳ get_supplier_otif (odczyt narzędzia) (przez MCP)"
    assert derived.parent_key == "c2"


def test_mcp_step_without_invocation_id_has_no_empty_line():
    from kovalo.ui.render import render_tool_end

    step = render_tool_end(tool_view(status="ok", brief="ok: x", transport="mcp"))
    assert step.name.endswith("(przez MCP)") and "Wywołanie MCP: None" not in step.output


def test_mcp_error_step_keeps_the_label_and_the_error_text():
    """Nieudane wywołanie przez MCP (np. `[błąd transportu]`): krok oznaczony błędem, z dopiskiem
    „(przez MCP)”, komunikatem i identyfikatorem wywołania, bez podwójnego „Błąd: ”."""
    from kovalo.ui.render import render_tool_end

    step = render_tool_end(
        tool_view(
            status="error",
            error="[błąd transportu] serwer MCP niedostępny: x (wywołanie f00d)",
            transport="mcp",
            invocation_id="f00d",
        )
    )
    assert step.is_error is True and step.name == "get_contract_terms (przez MCP)"
    assert step.output.startswith("Błąd: [błąd transportu]")
    assert step.output.count("Błąd: ") == 1 and step.output.endswith("Wywołanie MCP: f00d")
