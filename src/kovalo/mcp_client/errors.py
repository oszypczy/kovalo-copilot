"""Błędy klienta MCP (port `helpers/mcp_runtime.py` 1:1; publicznie przez `connection`).

Osobny moduł, żeby `contract.check_record` i `connection.McpConnection` używały tych samych
klas bez cyklu importów.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import ToolException

from kovalo.config import ConfigError

__all__ = ["LAYER_NAMES", "Layer", "McpCallError", "McpStartupError"]

Layer = Literal["tool", "protocol", "transport", "validation"]
LAYER_NAMES: dict[str, str] = {
    "tool": "błąd narzędzia",
    "protocol": "błąd protokołu",
    "transport": "błąd transportu",
    "validation": "błąd walidacji lokalnej",
}


class McpCallError(ToolException):
    """Nieudane wywołanie MCP; `layer` to "tool", "protocol", "transport" albo "validation".

    Podklasa ToolException: narzędzie LangChain z `handle_tool_error=True` zamienia go na
    ToolMessage z błędem, który model może przeczytać.
    """

    def __init__(self, layer: Layer | str, message: str, invocation_id: str | None = None):
        self.layer, self.invocation_id = layer, invocation_id
        suffix = f" (wywołanie {invocation_id})" if invocation_id else ""
        super().__init__(f"[{LAYER_NAMES[layer]}] {message}{suffix}")


class McpStartupError(ConfigError):
    """Nie udało się uruchomić serwera (albo połączyć się z nim), nie dokończył inicjalizacji na
    czas albo nie ma wymaganych narzędzi."""
