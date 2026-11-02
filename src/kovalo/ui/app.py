"""Chainlit: szkielet czatu F0 (powitanie i stała odpowiedź, bez LLM-a).

Od F3 UI rozmawia z api przez HTTP; nie importuje logiki backendu.
"""

from __future__ import annotations

import chainlit as cl

WELCOME = (
    "Dzień dobry, tu Kovalo Decision Copilot. Aplikacja jest w budowie: "
    "odpowiedzi na pytania pojawią się w kolejnych etapach."
)
UNDER_CONSTRUCTION = "Ta funkcja jest w budowie. Wkrótce odpowiem na pytania o dokumenty i rabaty."


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    await cl.Message(content=UNDER_CONSTRUCTION).send()
