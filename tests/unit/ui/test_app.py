"""Tura w Chainlicie (ui/app.py) z podmienionym modułem `cl` (kontrakt F4, 10.3; A8).

Chainlit 2.12 rysuje kroki-dzieci wiadomości pod jej treścią, a kroki bez rodzica jako osobne
wpisy z kropką ładowania nad nimi. Dlatego kroki narzędzi są dziećmi kroku `on_message` tury,
a odpowiedź (jedna wiadomość) idzie po nich. „Ścieżka dowodowa” to element boczny, nie krok.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from tests.unit.ui.conftest import sse_body, tool_end, tool_start, turn_result

pytestmark = pytest.mark.anyio

_ids = itertools.count(1)


class FakeChainlit:
    """Minimalny `chainlit` dla app.py: zapisuje send/update w kolejności wywołań."""

    def __init__(self) -> None:
        self.log: list[tuple[str, Any]] = []
        self.session: dict[str, Any] = {}
        self.run = self.Step(name="on_message", type="run")
        fake = self

        class UserSession:
            def get(self, key: str, default: Any = None) -> Any:
                return fake.session.get(key, default)

            def set(self, key: str, value: Any) -> None:
                fake.session[key] = value

        class Context:
            current_run = fake.run

        self.user_session = UserSession()
        self.context = Context()

    # klasy jako atrybuty, jak w module chainlit
    class Step:
        log: list[tuple[str, Any]]

        def __init__(self, name: str = "", type: str = "undefined", **kwargs: Any) -> None:
            self.id = f"step-{next(_ids)}"
            self.name = name
            self.type = type
            self.parent_id = kwargs.pop("parent_id", None)
            self.kwargs = kwargs
            self.input = ""
            self.output = ""
            self.is_error = False

        async def send(self) -> FakeChainlit.Step:
            self.log.append(("step.send", self))
            return self

        async def update(self) -> None:
            self.log.append(("step.update", self))

    class Message:
        log: list[tuple[str, Any]]

        def __init__(
            self,
            content: str = "",
            actions: list[Any] | None = None,
            elements: list[Any] | None = None,
        ) -> None:
            self.id = f"msg-{next(_ids)}"
            self.content = content
            self.actions = actions or []
            self.elements = elements or []

        async def send(self) -> FakeChainlit.Message:
            self.log.append(("message.send", self))
            return self

        async def update(self) -> None:
            self.log.append(("message.update", self))

        async def remove_actions(self) -> None:
            self.log.append(("message.remove_actions", self))

    class Text:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class Action:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    def bind(self) -> FakeChainlit:
        self.Step.log = self.log
        self.Message.log = self.log
        return self

    def sent(self, kind: str) -> list[Any]:
        return [obj for op, obj in self.log if op == kind]


@pytest.fixture
def fake_cl(monkeypatch):
    from kovalo.ui import app

    fake = FakeChainlit().bind()
    monkeypatch.setattr(app, "cl", fake)
    return fake


def _start(fake: FakeChainlit, recorder) -> None:
    from kovalo.ui.client import CopilotApiClient

    fake.session.update(
        client=CopilotApiClient("http://api.test", timeout_s=1, transport=recorder.transport),
        thread_id="T1",
        pending=None,
        pause_message=None,
    )


class _UserMessage:
    def __init__(self, content: str) -> None:
        self.content = content


EVIDENCE = "P1  Czy można rozwiązać umowę?\nstatus: answered; kroki: 3, 4 s\n    1. x"


def _agent_turn() -> bytes:
    final = turn_result(
        replies=["Nie: OTIF Mlecznej Doliny wynosi 94,6%."],
        sources=[{"doc_id": "CON-114", "chunk_id": "CON-114#ctx002", "text": "Próg OTIF 92%."}],
        trace={"status": "answered", "events": []},
        evidence_path=EVIDENCE,
    )
    return sse_body(
        ("tool_start", tool_start("c1", "search_documents")),
        ("tool_end", tool_end("c1", "search_documents")),
        ("tool_start", tool_start("c2", "check_termination_condition")),
        (
            "tool_start",
            tool_start("c2#get_supplier_otif", "get_supplier_otif", parent_call_id="c2"),
        ),
        (
            "tool_end",
            tool_end("c2#get_supplier_otif", "get_supplier_otif", parent_call_id="c2"),
        ),
        ("tool_end", tool_end("c2", "check_termination_condition")),
        ("final", final),
    )


async def test_agent_steps_nested_in_turn_and_answer_below(fake_cl, mock_stream):
    from kovalo.ui import app

    _start(fake_cl, mock_stream(("sse", _agent_turn())))
    await app.on_message(_UserMessage("Czy Kovalo może rozwiązać umowę z Mleczną Doliną?"))

    steps = {s.name: s for s in fake_cl.sent("step.send")}
    assert set(steps) == {
        "search_documents",
        "check_termination_condition",
        "↳ get_supplier_otif (odczyt narzędzia)",
    }
    assert all(s.type == "tool" for s in steps.values())
    run = fake_cl.run.id
    assert steps["search_documents"].parent_id == run, "krok w turze, nie osierocony"
    assert steps["check_termination_condition"].parent_id == run
    derived = steps["↳ get_supplier_otif (odczyt narzędzia)"]
    assert derived.parent_id == steps["check_termination_condition"].id

    messages = fake_cl.sent("message.send")
    assert len(messages) == 1, "jedna wiadomość z odpowiedzią tury"
    answer = messages[0]
    ops = [op for op, _ in fake_cl.log]
    assert ops.index("message.send") > max(
        i for i, op in enumerate(ops) if op.startswith("step.")
    ), "odpowiedź pod krokami narzędzi"
    assert all(s.parent_id != answer.id for s in steps.values()), (
        "Chainlit rysuje dzieci wiadomości pod jej treścią"
    )
    assert answer.content.startswith("Nie: OTIF Mlecznej Doliny wynosi 94,6%.")
    assert "Fragmenty: CON-114#ctx002" in answer.content
    assert answer.content.endswith("Szczegóły: Ścieżka dowodowa")
    assert fake_cl.session["pending"] is None


async def test_evidence_path_is_side_element_not_tool_step(fake_cl, mock_stream):
    from kovalo.ui import app

    _start(fake_cl, mock_stream(("sse", _agent_turn())))
    await app.on_message(_UserMessage("Czy Kovalo może rozwiązać umowę?"))

    assert "Ścieżka dowodowa" not in {s.name for s in fake_cl.sent("step.send")}
    (answer,) = fake_cl.sent("message.send")
    evidence = [e for e in answer.elements if e.name == "Ścieżka dowodowa"]
    assert len(evidence) == 1
    assert evidence[0].display == "side" and evidence[0].content == EVIDENCE
    assert evidence[0].language == "text", "wcięcia ścieżki bez formatowania markdown"
    sources = [e for e in answer.elements if e.name == "CON-114#ctx002"]
    assert sources and sources[0].display == "side"


async def test_turn_without_tools_looks_like_f3(fake_cl, mock_stream):
    from kovalo.ui import app

    final = turn_result(
        replies=["Ceny bez zmian."],
        sources=[{"doc_id": "MTG-402", "chunk_id": "MTG-402#ctx001", "text": "Komitet."}],
    )
    _start(fake_cl, mock_stream(("sse", sse_body(("final", final)))))
    await app.on_message(_UserMessage("Co postanowił komitet?"))

    assert fake_cl.sent("step.send") == []
    (answer,) = fake_cl.sent("message.send")
    assert answer.content == "Ceny bez zmian.\n\nStatus: odpowiedź\n\nFragmenty: MTG-402#ctx001"
    assert [e.name for e in answer.elements] == ["MTG-402#ctx001"]


async def test_interrupted_stream_marks_open_steps_and_reports_error(fake_cl, mock_stream):
    from kovalo.ui import app

    body = sse_body(
        ("tool_start", tool_start("c1", "search_documents")),
        ("tool_start", tool_start("c2", "get_contract_terms")),
        ("tool_end", tool_end("c2", "get_contract_terms")),
    )
    _start(fake_cl, mock_stream(("sse", body)))
    await app.on_message(_UserMessage("Pytanie"))

    (search,) = [s for s in fake_cl.sent("step.send") if s.name == "search_documents"]
    assert search.output == "(przerwane)"
    (terms,) = [s for s in fake_cl.sent("step.send") if s.name == "get_contract_terms"]
    assert terms.output.startswith("ok:")
    (error,) = fake_cl.sent("message.send")
    assert "Api nie odpowiada" in error.content, "stream_interrupted ma status 0"
    ops = [op for op, _ in fake_cl.log]
    assert ops[-1] == "message.send", "komunikat błędu pod krokami"


async def test_pause_turn_keeps_buttons_on_last_message(fake_cl, mock_stream):
    from kovalo.ui import app
    from tests.unit.ui.conftest import approval_pending

    final = turn_result(
        status="pending_approval", replies=["Opis wniosku."], pending=approval_pending()
    )
    _start(fake_cl, mock_stream(("sse", sse_body(("final", final)))))
    await app.on_message(_UserMessage("Rabat 22% dla Carrefour Express"))

    first, pause = fake_cl.sent("message.send")
    assert first.content == "Opis wniosku." and not first.actions
    assert [a.payload["value"] for a in pause.actions] == ["approved", "rejected", "cancel"]
    assert "Ścieżka dowodowa" not in pause.content
    assert fake_cl.session["pause_message"] is pause
    assert fake_cl.session["pending"] is not None


async def test_chat_start_sends_no_welcome(fake_cl, monkeypatch):
    """Nowa rozmowa zostawia domyślny pusty ekran Chainlita; listę możliwości copilot
    podaje dopiero w odpowiedzi na pytanie spoza zakresu (OTHER_REPLY)."""
    from kovalo.ui import app

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def create_thread(self):
            return "T-new"

    monkeypatch.setattr(app, "CopilotApiClient", _Client)
    await app.on_chat_start()
    assert fake_cl.sent("message.send") == []
    assert fake_cl.session["thread_id"] == "T-new"


async def test_mcp_step_is_labelled_after_the_call(fake_cl, mock_stream):
    """F5: krok dostaje „(przez MCP)” po tool_end (transport znany dopiero po wywołaniu), a w wyniku
    jest identyfikator wywołania; krok bez transportu zostaje bez zmian."""
    from kovalo.ui import app

    final = turn_result(
        replies=["Kara wynosi 7296,00 PLN."], trace={"status": "answered", "events": []}
    )
    body = sse_body(
        ("tool_start", tool_start("c1", "get_contract_terms")),
        ("tool_end", tool_end("c1", "get_contract_terms", transport="mcp", invocation_id="inv-77")),
        ("tool_start", tool_start("c2", "get_invoice_value")),
        ("tool_end", tool_end("c2", "get_invoice_value")),
        ("final", final),
    )
    _start(fake_cl, mock_stream(("sse", body)))
    await app.on_message(_UserMessage("Ile wynosi kara?"))

    sent = fake_cl.sent("step.send")
    by_key = {s.id: s for s in sent}
    names = sorted(s.name for s in by_key.values())
    assert names == ["get_contract_terms (przez MCP)", "get_invoice_value"]
    mcp_step = next(s for s in sent if s.name.startswith("get_contract_terms"))
    assert "Wywołanie MCP: inv-77" in mcp_step.output
    plain = next(s for s in sent if s.name == "get_invoice_value")
    assert "MCP" not in plain.output


async def test_each_evidence_path_in_chat_has_its_own_name(fake_cl, mock_stream):
    """Chainlit łączy link „Szczegóły: …” z elementem po nazwie: druga ścieżka dowodowa
    w tej samej rozmowie dostaje numer, żeby link nie otwierał ścieżki z wcześniejszej tury."""
    from kovalo.ui import app

    _start(fake_cl, mock_stream(("sse", _agent_turn()), ("sse", _agent_turn())))
    fake_cl.session["evidence_count"] = 0
    await app.on_message(_UserMessage("Pierwsze pytanie"))
    await app.on_message(_UserMessage("Drugie pytanie"))

    first, second = fake_cl.sent("message.send")
    names = [[e.name for e in m.elements if e.name.startswith("Ścieżka")] for m in (first, second)]
    assert names == [["Ścieżka dowodowa"], ["Ścieżka dowodowa 2"]]
    assert first.content.endswith("Szczegóły: Ścieżka dowodowa")
    assert second.content.endswith("Szczegóły: Ścieżka dowodowa 2")


# ---------------------------------------------------------------------------
# F6: linia kosztu na końcu ostatniej wiadomości tury, suma rozmowy, not_paused
# ---------------------------------------------------------------------------


def _usage_payload(*, ours=0.004266, reference=0.009594, calls=5, tokens_in=9120, tokens_out=612):
    return {
        "calls": calls,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "reasoning_tokens": 210,
        "models": ["gemini-3.5-flash-lite"],
        "by_model": [
            {
                "model": "gemini-3.5-flash-lite",
                "calls": calls,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "reasoning_tokens": 210,
            }
        ],
        "cost": {
            "ours_usd": ours,
            "reference_usd": reference,
            "unpriced_models": [],
            "our_prices_checked_on": "2026-11-02",
        },
    }


def _agent_turn_with_usage(**usage_kwargs) -> bytes:
    final = turn_result(
        replies=["Nie: OTIF Mlecznej Doliny wynosi 94,6%."],
        sources=[{"doc_id": "CON-114", "chunk_id": "CON-114#ctx002", "text": "Próg OTIF 92%."}],
        trace={"status": "answered", "events": []},
        evidence_path=EVIDENCE,
        usage=_usage_payload(**usage_kwargs),
    )
    return sse_body(
        ("tool_start", tool_start("c1", "search_documents")),
        ("tool_end", tool_end("c1", "search_documents")),
        ("final", final),
    )


def _fresh_conversation(fake) -> None:
    """Jak on_chat_start: pusta suma rozmowy w sesji."""
    from kovalo.ui.render import UsageTotal

    fake.session["usage_total"] = UsageTotal()


async def test_chat_start_resets_usage_total(fake_cl, monkeypatch):
    from kovalo.ui import app
    from kovalo.ui.render import UsageTotal

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def create_thread(self):
            return "T-new"

    fake_cl.session["usage_total"] = UsageTotal(turns=3, calls=9)
    monkeypatch.setattr(app, "CopilotApiClient", _Client)
    await app.on_chat_start()
    assert fake_cl.session["usage_total"] == UsageTotal(), "nowa rozmowa = nowa suma"


def _cost(message) -> Any:
    """Element boczny „Koszt tury” wiadomości albo None."""
    found = [e for e in message.elements if e.name.startswith("Koszt tury")]
    assert len(found) <= 1
    return found[0] if found else None


async def test_cost_panel_on_last_message(fake_cl, mock_stream):
    """F6 (po przeklikaniu): koszt tury to element boczny obok ścieżki dowodowej, podlinkowany
    w „Szczegóły: …”, numerowany w rozmowie; suma rozmowy w sesji."""
    from kovalo.ui import app

    _start(
        fake_cl, mock_stream(("sse", _agent_turn_with_usage()), ("sse", _agent_turn_with_usage()))
    )
    _fresh_conversation(fake_cl)
    await app.on_message(_UserMessage("Pierwsze pytanie"))
    await app.on_message(_UserMessage("Drugie pytanie"))

    first, second = fake_cl.sent("message.send")
    assert first.content.endswith("Szczegóły: Ścieżka dowodowa, Koszt tury")
    assert second.content.endswith("Szczegóły: Ścieżka dowodowa 2, Koszt tury 2")
    assert "**Tura:**" not in first.content, "koszt nie w treści wiadomości"
    cost1, cost2 = _cost(first), _cost(second)
    assert cost1.display == "side" and cost2.name == "Koszt tury 2"
    text1 = " ".join(cost1.content.split())
    assert text1.startswith("- **Tura:** 5 wywołań (gemini-3.5-flash-lite)")
    assert "9120 wejścia" in text1
    assert "**Koszt tury:** 0,004266 $" in text1 and "cennik kursu" not in text1
    assert "**Rozmowa:** 1 tura" in text1
    assert "**Rozmowa:** 2 tury, 0,008532 $" in " ".join(cost2.content.split())

    total = fake_cl.session["usage_total"]
    assert (total.turns, total.calls, total.input_tokens, total.output_tokens) == (
        2,
        10,
        18240,
        1224,
    )
    assert total.ours_usd == pytest.approx(0.008532) and total.reference_usd == pytest.approx(
        0.019188
    )
    assert total.partial is False


async def test_cost_panel_without_session_total_still_renders(fake_cl, mock_stream):
    """Sesja bez `usage_total` (np. wznowiona po restarcie UI) nie wywala tury."""
    from kovalo.ui import app

    _start(fake_cl, mock_stream(("sse", _agent_turn_with_usage())))
    fake_cl.session.pop("usage_total", None)
    await app.on_message(_UserMessage("Pytanie"))
    (answer,) = fake_cl.sent("message.send")
    assert _cost(answer) is not None
    assert fake_cl.session["usage_total"].turns == 1


async def test_turn_without_usage_has_no_cost_panel(fake_cl, mock_stream):
    from kovalo.ui import app

    final = turn_result(replies=["Ceny bez zmian."], usage=None)
    _start(fake_cl, mock_stream(("sse", sse_body(("final", final)))))
    _fresh_conversation(fake_cl)
    await app.on_message(_UserMessage("Co postanowił komitet?"))
    (answer,) = fake_cl.sent("message.send")
    assert _cost(answer) is None and "Koszt tury" not in answer.content
    assert fake_cl.session["usage_total"].turns == 0


async def test_cost_panel_on_approval_question_and_none_on_resume(fake_cl, mock_stream):
    """Przy pauzie koszt jest przy wiadomości z pytaniem o akceptację; wznowienie nie woła
    LLM-a (usage None), więc nie ma panelu, a suma rozmowy się nie zmienia."""
    from kovalo.ui import app
    from tests.unit.ui.conftest import approval_pending

    paused = turn_result(
        status="pending_approval",
        replies=["Opis wniosku."],
        pending=approval_pending(),
        usage=_usage_payload(),
    )
    resumed = turn_result(status="approved", replies=["Zatwierdzone."], usage=None)
    recorder = mock_stream(("sse", sse_body(("final", paused))), (200, resumed))
    _start(fake_cl, recorder)
    _fresh_conversation(fake_cl)
    await app.on_message(_UserMessage("Rabat 22% dla Carrefour Express"))

    first, pause = fake_cl.sent("message.send")
    assert _cost(first) is None, "koszt tylko przy ostatniej wiadomości tury"
    assert _cost(pause) is not None and pause.content.endswith("Szczegóły: Koszt tury")
    assert pause.actions, "przyciski zostają przy wiadomości z pauzą"
    before = fake_cl.session["usage_total"]
    assert before.turns == 1

    class _Action:
        def __init__(self, payload):
            self.payload = payload

        async def remove(self):
            pass

    action = pause.actions[0]
    await app.on_resume(_Action(action.payload))
    done = fake_cl.sent("message.send")[-1]
    assert "Zatwierdzone." in done.content
    assert _cost(done) is None and "Koszt tury" not in done.content
    assert fake_cl.session["usage_total"] == before


async def test_not_paused_removes_pause_buttons_and_explains(fake_cl, mock_stream):
    from kovalo.ui import app
    from tests.unit.ui.conftest import approval_pending

    paused = turn_result(
        status="pending_approval", replies=["Opis wniosku."], pending=approval_pending()
    )
    not_paused = {
        "error": "not_paused",
        "detail": "Wątek T1 nie czeka na odpowiedź.",
        "pending": None,
    }
    _start(fake_cl, mock_stream(("sse", sse_body(("final", paused))), (409, not_paused)))
    _fresh_conversation(fake_cl)
    await app.on_message(_UserMessage("Rabat 22% dla Carrefour Express"))
    _, pause = fake_cl.sent("message.send")

    class _Action:
        def __init__(self, payload):
            self.payload = payload

        async def remove(self):
            pass

    await app.on_resume(_Action(pause.actions[0].payload))
    last = fake_cl.sent("message.send")[-1]
    assert last.content == "Ta decyzja jest nieaktualna: rozmowa nie czeka już na odpowiedź."
    assert ("message.remove_actions", pause) in fake_cl.log, "przyciski pauzy usunięte"
    assert fake_cl.session["pause_message"] is None


async def test_tool_step_sends_icon_metadata(fake_cl, mock_stream):
    from kovalo.ui import app

    _start(fake_cl, mock_stream(("sse", _agent_turn())))
    await app.on_message(_UserMessage("Czy Kovalo może rozwiązać umowę?"))
    steps = {s.name: s for s in fake_cl.sent("step.send")}
    assert steps["search_documents"].kwargs["metadata"] == {"icon": "file-search"}


@pytest.mark.parametrize("content", ["", "   \n"])
async def test_message_without_text_gets_hint_and_no_api_call(fake_cl, mock_stream, content):
    """Sam obraz/załącznik (pusty tekst): podpowiedź po polsku zamiast 422 z api."""
    from kovalo.ui import app
    from kovalo.ui.render import NO_TEXT

    recorder = mock_stream()
    _start(fake_cl, recorder)
    await app.on_message(_UserMessage(content))
    (reply,) = fake_cl.sent("message.send")
    assert reply.content == NO_TEXT
    assert fake_cl.sent("step.send") == []
