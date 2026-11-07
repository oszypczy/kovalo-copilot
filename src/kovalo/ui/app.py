"""Chainlit: czat copilota przez HTTP do api (FR-6.2; kontrakt F3, 9.3; F4, 10.3; ADR-0010,
ADR-0012).

UI nie importuje backendu (workflow, api): tylko kovalo.ui.client i kovalo.ui.render.
- Odświeżenie strony = nowa sesja = nowy wątek (decyzja U9: stary wątek zostaje w Postgresie).
- Nowa wiadomość idzie strumieniem (SSE): każde wywołanie narzędzia agenta to cl.Step
  zagnieżdżony w kroku `on_message` tury (zdarzenie składowe pod krokiem rodzica). Odpowiedź
  idzie po `final`, więc stoi pod krokami. Chainlit 2.12 rysuje kroki-dzieci wiadomości POD jej
  treścią, dlatego kroki nie są dziećmi wiadomości odpowiedzi.
- „Ścieżka dowodowa” to element boczny (cl.Text display="side") podlinkowany w odpowiedzi,
  nie cl.Step: nagłówek kroku zawsze ma prefiks „Używam/Użyto”, a ścieżka nie jest narzędziem.
- Gdy wątek czeka na pauzę, tekst wpisany w czacie idzie do /resume jako `text` (marża „7,5”,
  „zatwierdź”, „odrzuć”, „anuluj”), jak chat.py kursu; przyciski to cl.Action z callbackiem.
- F6: koszt tury jako element boczny „Koszt tury” (obok ścieżki dowodowej, link w linii
  „Szczegóły: …” ostatniej wiadomości tury); suma rozmowy w cl.user_session["usage_total"].
  Panel boczny otwiera się tylko po kliknięciu (public/kovalo.js).
"""

from __future__ import annotations

from typing import Any

import chainlit as cl

from kovalo.config import get_settings
from kovalo.ui.client import ApiCallError, CopilotApiClient, PendingView, TurnView, pause_ref
from kovalo.ui.render import (
    INTERRUPTED,
    NO_TEXT,
    UiMessage,
    UiStep,
    UsageTotal,
    add_usage,
    message_text,
    render_cost,
    render_error,
    render_evidence,
    render_pending,
    render_tool_end,
    render_tool_start,
    render_turn,
    render_usage,
)

STALE = "Ta decyzja jest nieaktualna: wątek czeka już na inną odpowiedź."
RESUME_ACTION = "resume"


def _client() -> CopilotApiClient:
    client = cl.user_session.get("client")
    assert isinstance(client, CopilotApiClient)
    return client


async def _send(
    message: UiMessage, evidence: UiStep | None = None, cost: UiStep | None = None
) -> cl.Message:
    actions = [
        cl.Action(
            name=RESUME_ACTION,
            payload={"value": a.value, "pause_ref": a.pause_ref},
            label=a.label,
        )
        for a in message.actions
    ]
    elements: list[Any] = [
        cl.Text(name=s.name, content=s.content, display="side") for s in message.sources
    ]
    if evidence is not None:  # tekst z wcięciami: blok bez formatowania markdown
        elements.append(
            cl.Text(name=evidence.name, content=evidence.output, display="side", language="text")
        )
    if cost is not None:  # lista Markdown: tura, tokeny, koszt tury, suma rozmowy
        elements.append(cl.Text(name=cost.name, content=cost.output, display="side"))
    # nazwy elementów w treści (message_text): Chainlit podlinkuje ich panele boczne
    msg = cl.Message(
        content=message_text(message, evidence, cost), actions=actions, elements=elements
    )
    await msg.send()
    return msg


async def _clear_pause_actions() -> None:
    """Usuwa przyciski z poprzedniej wiadomości z pauzą (żeby nie kliknąć drugi raz)."""
    msg = cl.user_session.get("pause_message")
    if isinstance(msg, cl.Message):
        await msg.remove_actions()
    cl.user_session.set("pause_message", None)


def _usage_details(view: TurnView) -> UiStep | None:
    """Dodaje turę do sumy rozmowy w sesji i zwraca element „Koszt tury” (None bez usage)."""
    if view.usage is None:
        return None
    total = cl.user_session.get("usage_total")
    if not isinstance(total, UsageTotal):  # sesja bez sumy: liczymy od zera
        total = UsageTotal()
    total = add_usage(total, view.usage)
    cl.user_session.set("usage_total", total)
    shown = cl.user_session.get("cost_count", 0)
    cl.user_session.set("cost_count", shown + 1)
    return render_cost(render_usage(view.usage, total), number=shown + 1)


async def _show(view: TurnView) -> None:
    """Renderuje turę i zapamiętuje bieżącą pauzę (albo jej brak) w sesji."""
    await _clear_pause_actions()
    messages = render_turn(view)
    shown = cl.user_session.get("evidence_count", 0)
    evidence = render_evidence(view, number=shown + 1)
    if evidence is not None:
        cl.user_session.set("evidence_count", shown + 1)
    cost = _usage_details(view)
    if (evidence is not None or cost is not None) and not messages:
        messages = [UiMessage(content="")]
    last: cl.Message | None = None
    for n, message in enumerate(messages, start=1):  # ścieżka dowodowa i koszt przy ostatniej
        is_last = n == len(messages)
        last = await _send(message, evidence if is_last else None, cost if is_last else None)
    cl.user_session.set("pending", view.pending)
    if view.pending is not None:
        cl.user_session.set("pause_message", last)


class _ToolSteps:
    """cl.Step na każde wywołanie narzędzia: tool_start tworzy krok, tool_end go uzupełnia.

    Krok wywołania agenta ma rodzica `root_id` (krok `on_message` tury): Chainlit rysuje go
    w kolejności przyjścia, z awatarem asystenta, a odpowiedź wysłana po `final` stoi pod nim.
    Zdarzenie składowe (np. get_supplier_otif) idzie pod krok rodzica.
    """

    def __init__(self, root_id: str | None) -> None:
        self.root_id = root_id
        self.steps: dict[str, cl.Step] = {}
        self.open: set[str] = set()

    async def start(self, ui: UiStep) -> None:
        parent = self.steps.get(ui.parent_key) if ui.parent_key else None
        step = cl.Step(
            name=ui.name,
            type="tool",
            parent_id=parent.id if parent is not None else self.root_id,
            show_input="json",
            language="json",
            metadata={"icon": ui.icon} if ui.icon else None,
        )
        step.input = ui.input
        await step.send()
        self.steps[ui.key] = step
        self.open.add(ui.key)

    async def end(self, ui: UiStep) -> None:
        step = self.steps.get(ui.key)
        if step is None:  # tool_end bez tool_start (nie powinno się zdarzyć)
            await self.start(ui)
            step = self.steps[ui.key]
        step.name = ui.name  # F5: „(przez MCP)” znane dopiero w tool_end (P8)
        step.output = ui.output
        step.is_error = ui.is_error
        await step.update()
        self.open.discard(ui.key)

    async def interrupt(self) -> None:
        """Kroki z tool_start bez tool_end dostają „(przerwane)”."""
        for key in sorted(self.open):
            step = self.steps[key]
            step.output = INTERRUPTED
            await step.update()
        self.open.clear()


async def _stream_turn(thread_id: str, text: str) -> TurnView:
    run = cl.context.current_run  # krok `on_message`: bez rodzica kroki byłyby osierocone
    steps = _ToolSteps(run.id if run is not None else None)
    try:
        async for kind, item in _client().stream_message(thread_id, text):
            if kind == "tool_start":
                await steps.start(render_tool_start(item))  # type: ignore[arg-type]
            elif kind == "tool_end":
                await steps.end(render_tool_end(item))  # type: ignore[arg-type]
            elif kind == "final":
                assert isinstance(item, TurnView)
                return item
    except ApiCallError:
        await steps.interrupt()
        raise
    await steps.interrupt()
    raise ApiCallError(0, "stream_interrupted", "Strumień api skończył się bez wyniku tury.")


async def _refresh_pending(thread_id: str) -> None:
    """Pauza z api (np. sesja i wątek się rozjechały): pokaż ją ponownie."""
    state = await _client().get_thread(thread_id)
    cl.user_session.set("pending", state.pending)
    await _clear_pause_actions()
    if state.pending is not None:
        msg = await _send(render_pending(state.pending))
        cl.user_session.set("pause_message", msg)


async def _handle_error(error: ApiCallError, thread_id: str) -> None:
    await _send(render_error(error))
    if error.error == "not_paused":  # F6: np. po zamknięciu przerwanej tury
        cl.user_session.set("pending", None)
        await _clear_pause_actions()
    if error.error in ("thread_paused", "stale_pause"):
        try:
            await _refresh_pending(thread_id)
        except ApiCallError as again:
            await _send(render_error(again))


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = get_settings()
    client = CopilotApiClient(
        settings.services.api_url, timeout_s=settings.services.ui_api_timeout_s
    )
    cl.user_session.set("client", client)
    cl.user_session.set("pending", None)
    cl.user_session.set("pause_message", None)
    cl.user_session.set("evidence_count", 0)
    cl.user_session.set("usage_total", UsageTotal())
    cl.user_session.set("cost_count", 0)
    cl.user_session.set("thread_id", None)
    # bez powitania: domyślny pusty ekran Chainlita; możliwości opisuje OTHER_REPLY
    try:
        cl.user_session.set("thread_id", await client.create_thread())
    except ApiCallError as error:
        await _send(render_error(error))


async def _thread_id() -> str | None:
    """Bieżący wątek; gdy api nie działało przy starcie czatu, próbuje założyć go teraz."""
    thread_id = cl.user_session.get("thread_id")
    if isinstance(thread_id, str):
        return thread_id
    try:
        thread_id = await _client().create_thread()
    except ApiCallError as error:
        await _send(render_error(error))
        return None
    cl.user_session.set("thread_id", thread_id)
    return thread_id


@cl.on_message
async def on_message(message: cl.Message) -> None:
    if not (message.content or "").strip():  # np. sam obraz: api wymaga tekstu (422)
        await _send(UiMessage(content=NO_TEXT))
        return
    thread_id = await _thread_id()
    if thread_id is None:
        return
    pending = cl.user_session.get("pending")
    try:
        if isinstance(pending, PendingView):
            view = await _client().resume(
                thread_id, text=message.content, pause_ref=pause_ref(pending)
            )
        else:
            view = await _stream_turn(thread_id, message.content)
    except ApiCallError as error:
        await _handle_error(error, thread_id)
        return
    await _show(view)


@cl.action_callback(RESUME_ACTION)
async def on_resume(action: cl.Action) -> None:
    thread_id = cl.user_session.get("thread_id")
    if not isinstance(thread_id, str):
        return
    payload = action.payload or {}
    try:
        view = await _client().resume(
            thread_id, value=payload.get("value"), pause_ref=payload.get("pause_ref")
        )
    except ApiCallError as error:
        if error.error == "stale_pause":
            await action.remove()
            await cl.Message(content=STALE).send()
            return
        await _handle_error(error, thread_id)
        return
    await _show(view)
