"""Zapisuje nazwy zmiennych środowiska procesu (jedna na wiersz) do pliku z argv[1], potem działa
jak mały prawdziwy serwer MCP z jednym narzędziem. Do testu: serwer nie dostaje kluczy (NFR-7)."""

# ruff: noqa  (skrypt-zabawka: zwięzły styl jak w serwerach kursu)
import os
import sys

open(sys.argv[1], "w").write("\n".join(sorted(os.environ)))
from mcp.server.fastmcp import FastMCP

server = FastMCP("env-report", log_level="WARNING")


@server.tool()
def ping() -> str:
    """Odpowiada."""
    return "pong"


server.run("stdio")
