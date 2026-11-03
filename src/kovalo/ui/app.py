"""Chainlit: czat copilota przez HTTP do api (FR-6.2, część; kontrakt F3, 9.3; ADR-0010).

UI nie importuje backendu (workflow, api): tylko kovalo.ui.client i kovalo.ui.render.
- Odświeżenie strony = nowa sesja = nowy wątek (decyzja U9: stary wątek zostaje w Postgresie).
- Gdy wątek czeka na pauzę, tekst wpisany w czacie idzie do /resume jako `text` (marża „7,5”,
  „zatwierdź”, „odrzuć”, „anuluj”), jak chat.py kursu; przyciski to cl.Action z callbackiem.
"""

from __future__ import annotations

from typing import Any

import chainlit as cl

from kovalo.config import get_settings
from kovalo.ui.client import ApiCallError, CopilotApiClient, PendingView, TurnView, pause_ref
from kovalo.ui.render import UiMessage, render_error, render_pending, render_turn

WELCOME = (
    "Dzień dobry, tu Kovalo Decision Copilot. Mogę:\n"
    "- odpowiedzieć na pytania o dokumenty Kovalo (ze źródłami),\n"
    "- sprawdzić, kto zatwierdza rabat według polityki course-v1,\n"
    "- przesłać wniosek rabatowy do akceptacji."
)
STALE = "Ta decyzja jest nieaktualna: wątek czeka już na inną odpowiedź."
RESUME_ACTION = "resume"


def _client() -> CopilotApiClient:
    client = cl.user_session.get("client")
    assert isinstance(client, CopilotApiClient)
    return client


async def _send(message: UiMessage) -> cl.Message:
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
    if elements:  # nazwy źródeł w treści, żeby Chainlit podlinkował panele boczne
        names = ", ".join(s.name for s in message.sources)
        content = f"{message.content}\n\nFragmenty: {names}"
    else:
        content = message.content
    msg = cl.Message(content=content, actions=actions, elements=elements)
    await msg.send()
    return msg


async def _clear_pause_actions() -> None:
    """Usuwa przyciski z poprzedniej wiadomości z pauzą (żeby nie kliknąć drugi raz)."""
    msg = cl.user_session.get("pause_message")
    if isinstance(msg, cl.Message):
        await msg.remove_actions()
    cl.user_session.set("pause_message", None)


async def _show(view: TurnView) -> None:
    """Renderuje turę i zapamiętuje bieżącą pauzę (albo jej brak) w sesji."""
    await _clear_pause_actions()
    last: cl.Message | None = None
    for message in render_turn(view):
        last = await _send(message)
    cl.user_session.set("pending", view.pending)
    if view.pending is not None:
        cl.user_session.set("pause_message", last)


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
    cl.user_session.set("thread_id", None)
    await cl.Message(content=WELCOME).send()
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
            view = await _client().send_message(thread_id, message.content)
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
