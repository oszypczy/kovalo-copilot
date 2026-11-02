"""Serwer MCP Kovalo (FastMCP). W F0 bez narzędzi, tylko /health (kontrakt F0, sekcja 6).

Transporty: stdio (domyślny, np. MCP Inspector) i streamable HTTP (`/mcp`, w compose).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Literal

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from kovalo.config import ConfigError, Settings, get_settings
from kovalo.health import build_report, http_status

SERVER_NAME = "kovalo"
Transport = Literal["stdio", "streamable-http"]


def build_server(
    settings: Settings, *, host: str = "127.0.0.1", port: int | None = None
) -> FastMCP:
    """FastMCP w trybie stateless HTTP. Przy host 0.0.0.0 SDK nie włącza ochrony DNS rebinding."""
    server = FastMCP(
        SERVER_NAME,
        host=host,
        port=port if port is not None else settings.services.mcp_port,
        stateless_http=True,
    )

    @server.custom_route("/health", methods=["GET"], include_in_schema=False)
    async def health(request: Request) -> Response:
        # F0: brak sprawdzeń (od F5: indeks w pgvector).
        report = await build_report("mcp-server", {}, timeout_s=settings.services.health_timeout_s)
        return JSONResponse(report.model_dump(mode="json"), status_code=http_status(report))

    return server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kovalo-mcp", description="Serwer MCP Kovalo.")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="adres nasłuchu (streamable-http)")
    parser.add_argument("--port", type=int, default=None, help="port (domyślnie services.mcp_port)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    transport: Transport = args.transport
    build_server(settings, host=args.host, port=args.port).run(transport=transport)
    return 0


def entrypoint() -> None:
    """Punkt wejścia `kovalo-mcp`."""
    sys.exit(main())
