"""Odtworzenie wywołań przebiegu kursu na naszych narzędziach (kontrakt F4, 11.3; zero LLM-a).

Skrypt z trace kursu: po jednym AIMessage z jednym wywołaniem narzędzia na każde zdarzenie
najwyższego poziomu (id, nazwa, argumenty jak w kursie), na końcu odpowiedź kursu. Wyniki
narzędzi pochodzą z NASZEGO kodu i danych, więc odtworzenie sprawdza port, nie model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from kovalo.agents.trace import Trace

__all__ = ["ScriptedToolModel", "script_from_trace"]


class ScriptedToolModel(BaseChatModel):
    """Atrapa modelu z wywołaniami narzędzi: bind_tools zwraca siebie, kolejne _generate
    zwracają AIMessage ze skryptu; wyczerpany skrypt → AssertionError; licznik `calls`."""

    script: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedToolModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.calls >= len(self.script):
            raise AssertionError("skryptowi odtworzenia skończyły się odpowiedzi modelu")
        message = self.script[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def script_from_trace(trace: Trace) -> list[AIMessage]:
    """Zdarzenia bez parent_call_id po kolei → AIMessage z jednym tool_call (id = call_id,
    name = tool, args = arguments), na końcu AIMessage(trace.answer). Zdarzenia worker_summary
    albo unknown → ValueError (odtwarzamy tylko przebieg jednego agenta)."""
    top = [e for e in trace.events if e.parent_call_id is None]
    if any(e.origin in ("worker_summary", "unknown") for e in top):
        raise ValueError(
            f"{trace.question_id}: odtwarzamy tylko przebieg jednego agenta "
            "(zdarzenia worker_summary albo unknown)"
        )
    script = [
        AIMessage(
            content="",
            tool_calls=[{"id": e.call_id, "name": e.tool, "args": dict(e.arguments)}],
        )
        for e in top
    ]
    script.append(AIMessage(content=trace.answer or ""))
    return script
