"""Źródło narzędzi agenta przez MCP: stabilne narzędzia nad wymienialnym połączeniem (FR-5.3).

Kontrakt: docs/contracts/f5.md, sekcja 8.3 (P5); ADR-0013 pkt 5 i 6 (U3: jedno połączenie na
proces `api`, wywołania z wielu tur równolegle, nowe połączenie po błędzie transportu).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Collection
from typing import Any, Literal

from langchain_core.tools import BaseTool

from kovalo.config import Settings
from kovalo.mcp_client.connection import (
    McpCallError,
    McpConnection,
    http_connection,
    kovalo_stdio_args,
    kovalo_stdio_env,
    stdio_connection,
    tool_from_info,
)

__all__ = ["McpToolSource", "open_tool_source"]


class McpToolSource:
    """Stabilne narzędzia LangChain nad wymienialnym połączeniem (U3).

    Pierwsze tools() łączy się (connect()) i buduje narzędzia z list_tools() (nazwa, opis,
    inputSchema); każde narzędzie woła self.call(name, args). call(): jeśli bieżące połączenie
    jest None, zamknięte albo `unusable`, pod blokadą zamyka je i otwiera NOWE (connect()); błąd
    otwarcia → McpCallError('transport', 'serwer MCP niedostępny: ...'). Połączenie po błędzie
    transportu nigdy nie jest używane ponownie (FR-5.3). Wywołania równoległe z wielu tur idą
    jednym połączeniem.
    """

    def __init__(self, connect: Callable[[], McpConnection], *, names: Collection[str]) -> None:
        self._connect = connect
        self.names = frozenset(names)
        self._connection: McpConnection | None = None
        self._tools: list[BaseTool] | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _usable(conn: McpConnection | None) -> bool:
        return conn is not None and conn.alive and not conn.closed and not conn.unusable

    def tools(self) -> list[BaseTool]:
        """Narzędzia (budowane raz). McpStartupError przy pierwszym połączeniu leci wyżej."""
        with self._lock:
            if self._tools is None:
                if not self._usable(self._connection):
                    self._replace()
                assert self._connection is not None
                infos = self._connection.tool_infos(self.names)  # sprawdza brakujące nazwy
                self._tools = [tool_from_info(info, self.call) for info in infos]
            return list(self._tools)

    def _replace(self) -> McpConnection:
        """Pod blokadą: zamyka bieżące połączenie i otwiera nowe (wyjątek connect() leci wyżej)."""
        old, self._connection = self._connection, None
        if old is not None:
            old.close()
        self._connection = self._connect()
        return self._connection

    def _live(self) -> McpConnection:
        with self._lock:
            conn = self._connection
            if conn is not None and self._usable(conn):
                return conn
            try:
                return self._replace()
            except Exception as exc:
                raise McpCallError(
                    "transport", f"serwer MCP niedostępny: {type(exc).__name__}: {exc}"
                ) from None

    def call(self, tool: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """(rekord, pochodzenie) przez bieżące albo nowe połączenie."""
        return self._live().call(tool, arguments)

    @property
    def connection(self) -> McpConnection | None:
        return self._connection

    def close(self) -> None:
        """Zamyka bieżące połączenie; idempotentne."""
        with self._lock:
            conn, self._connection = self._connection, None
        if conn is not None:
            conn.close()


def open_tool_source(
    settings: Settings,
    *,
    names: Collection[str],
    transport: Literal["http", "stdio"] | None = None,
    search: Literal["pgvector", "memory"] | None = None,
    capture_limit: int | None = 100,
) -> McpToolSource:
    """Niczego nie otwiera (P5): zwraca McpToolSource z funkcją connect; połączenie i błąd startu
    (McpStartupError) pojawiają się dopiero w source.tools(). transport None →
    tools.mcp_transport: http → http_connection(services.mcp_url) z limitami z sekcji `mcp`;
    stdio → podproces `python -m kovalo.mcp_server --search <search>` w katalogu repozytorium,
    ze środowiskiem bez kluczy (kovalo_stdio_env)."""
    chosen = transport or settings.tools.mcp_transport
    limits = settings.mcp

    def connect_http() -> McpConnection:
        # nazwa modułu czytana w chwili połączenia (testy podmieniają http_connection)
        return http_connection(
            settings.services.mcp_url,
            startup_timeout_s=limits.startup_timeout_s,
            call_timeout_s=limits.call_timeout_s,
            cleanup_timeout_s=limits.cleanup_timeout_s,
            capture_limit=capture_limit,
        )

    def connect_stdio() -> McpConnection:
        args = kovalo_stdio_args(settings, search=search or settings.mcp_server.search)
        return stdio_connection(
            args,
            settings.root,
            startup_timeout_s=limits.startup_timeout_s,
            call_timeout_s=limits.call_timeout_s,
            cleanup_timeout_s=limits.cleanup_timeout_s,
            env=kovalo_stdio_env(),
            capture_limit=capture_limit,
        )

    return McpToolSource(connect_http if chosen == "http" else connect_stdio, names=names)
