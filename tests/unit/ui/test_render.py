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


# ---------------------------------------------------------------------------
# F6: linia kosztu tury i suma rozmowy (kontrakt F6, 6.4), tekst not_paused (7)
# ---------------------------------------------------------------------------


def usage_view(
    *,
    calls=5,
    input_tokens=9120,
    output_tokens=612,
    models=("gemini-3.5-flash-lite",),
    ours=0.004266,
    reference=0.009594,
    unpriced=(),
    cost=True,
):
    from kovalo.ui.client import ModelUsageView, TurnCostView, UsageView

    return UsageView(
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=210,
        models=list(models),
        by_model=[
            ModelUsageView(
                model=m,
                calls=max(1, calls // len(models)),
                input_tokens=input_tokens // len(models),
                output_tokens=output_tokens // len(models),
            )
            for m in models
        ],
        cost=(
            TurnCostView(ours_usd=ours, reference_usd=reference, unpriced_models=list(unpriced))
            if cost
            else None
        ),
    )


def squash(text: str) -> str:
    """Zawinięcia linii i twarde spacje nie są częścią testowanego kontraktu."""
    return " ".join(text.split())  # split() traktuje też twarde spacje jak białe znaki


def total_after(*usages):
    from kovalo.ui.render import UsageTotal, add_usage

    total = UsageTotal()
    for u in usages:
        total = add_usage(total, u)
    return total


def test_render_usage_details():
    """Lista do panelu „Koszt tury”: polskie liczby, odmiana, koszt faktycznie użytego modelu
    jako ułamek dolara (bez darmowego tieru i cennika kursu), suma rozmowy."""
    from kovalo.ui.render import render_usage

    first = usage_view(
        ours=0.002834, reference=0.005606, calls=2, input_tokens=4000, output_tokens=300
    )
    this = usage_view()
    total = total_after(first, this)
    details = render_usage(this, total)
    assert details == (
        "- **Tura:** 5 wywołań (gemini-3.5-flash-lite)\n"
        "- **Tokeny:** 9120 wejścia, 612 wyjścia\n"
        "- **Koszt tury:** 0,004266 $\n"
        "- **Rozmowa:** 2 tury, 0,0071 $"
    )


def test_render_cost_element_is_numbered_in_conversation():
    from kovalo.ui.render import COST_STEP, UiMessage, message_text, render_cost

    assert render_cost(None) is None, "tura bez LLM-a nie ma panelu kosztu"
    first, second = render_cost("x"), render_cost("y", number=2)
    assert first is not None and second is not None
    assert (first.name, second.name) == (COST_STEP, f"{COST_STEP} 2")
    assert message_text(UiMessage(content="Odp."), cost=first) == "Odp.\n\nSzczegóły: Koszt tury"


@pytest.mark.parametrize(
    ("calls", "word"),
    [
        (1, "1 wywołanie"),
        (2, "2 wywołania"),
        (4, "4 wywołania"),
        (5, "5 wywołań"),
        (12, "12 wywołań"),
        (14, "14 wywołań"),
        (22, "22 wywołania"),
        (25, "25 wywołań"),
    ],
)
def test_render_usage_call_declension(calls, word):
    from kovalo.ui.render import render_usage

    usage = usage_view(calls=calls, input_tokens=100, output_tokens=10)
    assert word + " " in squash(render_usage(usage, total_after(usage)))


def test_render_usage_models_listed_without_duplicates():
    from kovalo.ui.client import ModelUsageView
    from kovalo.ui.render import render_usage

    usage = usage_view(models=("gemini-3.5-flash-lite", "gemini-3.6-flash"))
    usage = usage.model_copy(
        update={
            "by_model": [
                *usage.by_model,
                ModelUsageView(
                    model="gemini-3.5-flash-lite", calls=1, input_tokens=1, output_tokens=1
                ),
            ],
            "models": ["bez-znaczenia"],
        }
    )
    text = squash(render_usage(usage, total_after(usage)))
    assert "(gemini-3.5-flash-lite, gemini-3.6-flash)" in text, (
        "modele z by_model (nie z models), bez powtórzeń, przecinkami (fallback widoczny)"
    )
    assert "bez-znaczenia" not in text


def test_render_usage_thousands_and_tiny_cost():
    from kovalo.ui.render import render_usage

    usage = usage_view(input_tokens=1234567, output_tokens=89, ours=1e-8, reference=2e-8)
    text = squash(render_usage(usage, total_after(usage)))
    assert "1 234 567 wejścia" in text and "89 wyjścia" in text
    assert "**Koszt tury:** < 0,000001 $" in text and "**Rozmowa:** 1 tura, < 0,000001 $" in text


def test_render_usage_dollar_fraction_format():
    from kovalo.ui.render import render_usage

    for ours, shown in ((0.0034, "0,0034 $"), (0.12, "0,12 $"), (1.5, "1,5 $"), (0.0, "0 $")):
        usage = usage_view(ours=ours)
        assert f"**Koszt tury:** {shown}" in squash(render_usage(usage, total_after(usage)))


def test_render_usage_unpriced_and_missing_cost():
    from kovalo.ui.render import add_usage, render_usage

    assert render_usage(None, total_after()) is None, "tura bez LLM-a nie ma kosztu"

    unpriced = usage_view(ours=None, unpriced=("model-x",), models=("model-x",))
    total = add_usage(total_after(usage_view()), unpriced)
    assert total.partial is True
    text = squash(render_usage(unpriced, total))
    assert "brak ceny dla model-x" in text
    assert "cennik kursu" not in text and "darmow" not in text
    assert "(część tur bez ceny)" in text

    no_cost = usage_view(cost=False)
    total = add_usage(total_after(), no_cost)
    text = squash(render_usage(no_cost, total))
    assert text.startswith("- **Tura:** 5 wywołań")
    assert "Koszt" not in text and "$" not in text


def test_add_usage_accumulates():
    from kovalo.ui.render import UsageTotal, add_usage

    first = usage_view(calls=5, input_tokens=9120, output_tokens=612, ours=0.004, reference=0.01)
    second = usage_view(calls=1, input_tokens=100, output_tokens=10, ours=0.001, reference=0.002)
    total = add_usage(UsageTotal(), first)
    assert (total.turns, total.calls, total.input_tokens, total.output_tokens) == (1, 5, 9120, 612)
    assert total.ours_usd == pytest.approx(0.004) and total.reference_usd == pytest.approx(0.01)
    assert total.partial is False

    unchanged = add_usage(total, None)
    assert unchanged == total, "tura bez usage (wznowienie pauzy) nie zmienia sumy"

    total = add_usage(unchanged, second)
    assert (total.turns, total.calls, total.input_tokens, total.output_tokens) == (2, 6, 9220, 622)
    assert total.ours_usd == pytest.approx(0.005) and total.reference_usd == pytest.approx(0.012)
    assert UsageTotal().turns == 0, "UsageTotal jest niezmienny (kopie)"


def test_add_usage_marks_partial_when_a_turn_has_no_price():
    from kovalo.ui.render import UsageTotal, add_usage

    priced = usage_view(ours=0.004, reference=0.01)
    unpriced = usage_view(ours=None, unpriced=("x",))
    no_cost = usage_view(cost=False)
    total = add_usage(add_usage(UsageTotal(), priced), unpriced)
    assert total.partial and total.turns == 2
    assert total.ours_usd == pytest.approx(0.004), "tura bez ceny nie dodaje do kwoty"
    assert total.reference_usd == pytest.approx(0.01 + 0.009594)
    assert add_usage(UsageTotal(), no_cost).partial is True, "cost None też oznacza brak ceny"


def test_not_paused_text():
    from kovalo.ui.client import ApiCallError
    from kovalo.ui.render import render_error

    message = render_error(ApiCallError(409, "not_paused", "Wątek T1 nie czeka na odpowiedź."))
    assert message.content == "Ta decyzja jest nieaktualna: rozmowa nie czeka już na odpowiedź."
    assert not message.actions
    stale = render_error(ApiCallError(409, "stale_pause", "inna pauza"))
    assert stale.content != message.content, "stale_pause ma własny tekst"


def test_turn_view_carries_usage():
    from kovalo.ui.client import TurnView

    view = TurnView.model_validate(
        {
            "thread_id": "T1",
            "status": "answered",
            "replies": ["x"],
            "usage": {
                "calls": 2,
                "input_tokens": 10,
                "output_tokens": 3,
                "reasoning_tokens": None,
                "models": ["m"],
                "by_model": [
                    {"model": "m", "calls": 2, "input_tokens": 10, "output_tokens": 3},
                ],
                "cost": {
                    "ours_usd": 0.1,
                    "reference_usd": 0.2,
                    "unpriced_models": [],
                    "our_prices_checked_on": "2026-11-02",
                },
            },
        }
    )
    assert view.usage.calls == 2 and view.usage.by_model[0].model == "m"
    assert view.usage.cost.ours_usd == 0.1 and view.usage.cost.reference_usd == 0.2
    assert TurnView(thread_id="T1", status=None).usage is None


def test_tool_steps_get_icons_instead_of_empty_avatars():
    """Chainlit rysuje szare kółko, gdy krok nie ma awatara; ikona lucide w metadata.icon."""
    from kovalo.ui.client import ToolEventView
    from kovalo.ui.render import DEFAULT_TOOL_ICON, render_tool_start

    def icon(tool: str, origin: str | None) -> str | None:
        return render_tool_start(ToolEventView(call_id="c", tool=tool, origin=origin)).icon

    assert icon("search_documents", "data") == "file-search"
    assert icon("get_supplier_otif", "data") == "database"
    assert icon("calculate_otif_penalty", "calculation") == "calculator"
    assert icon("cos_nowego", None) == DEFAULT_TOOL_ICON
