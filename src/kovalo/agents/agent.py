"""Agent z narzędziami: create_agent z obsługą błędów narzędzi i fallbackiem (FR-4.4; krok 3 MP4).

Kontrakt: docs/contracts/f4.md, sekcja 7.3. Model, narzędzia i wyjątki fallbacku przychodzą
przez parametry (agents/ nie importuje kovalo.llm ani kovalo.tools). Agent bez pamięci między
pytaniami: każde pytanie ma nowy stan i nowy Budget.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from kovalo.agents.middleware import (
    AvailabilityFallbackMiddleware,
    Middleware,
    StableUsageMiddleware,
)
from kovalo.agents.trace import Budget, EventCallback, Trace, run_question, tool_errors
from kovalo.config import AgentConfig

__all__ = ["AgentRunner", "build_agent"]


def build_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    system_prompt: str,
    fallbacks: Sequence[BaseChatModel] = (),
    fallback_exceptions: tuple[type[BaseException], ...] = (),
    extra_middleware: Sequence[Middleware] = (),
) -> Any:
    """create_agent(model, tools, system_prompt, middleware, checkpointer=False).

    Kolejność middleware (od zewnątrz): tool_errors, fallback (gdy są modele zapasowe), extra
    (np. CallMeter: liczy każdą próbę), na końcu zawsze StableUsageMiddleware (P1).
    """
    middleware: list[Middleware] = [tool_errors()]
    if fallbacks:
        middleware.append(AvailabilityFallbackMiddleware(fallbacks, exceptions=fallback_exceptions))
    middleware.extend(extra_middleware)
    middleware.append(StableUsageMiddleware())
    return create_agent(
        model,
        list(tools),
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=False,
    )


class AgentRunner:
    """Jedno pytanie = nowy stan i nowy Budget.from_settings(agent_config).

    `callbacks` (F5, R1-2): callbacki konfiguracji węzła grafu, żeby run agenta zagnieżdżał się
    pod turą grafu (LangSmith); None → jak F4."""

    def __init__(self, agent: Any, agent_config: AgentConfig) -> None:
        self.agent = agent
        self.agent_config = agent_config

    def __call__(
        self,
        question: str,
        *,
        question_id: str | None = None,
        on_event: EventCallback | None = None,
        callbacks: Callbacks = None,
    ) -> Trace:
        return run_question(
            self.agent,
            question,
            Budget.from_settings(self.agent_config),
            question_id=question_id,
            on_event=on_event,
            callbacks=callbacks,
        )
