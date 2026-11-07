"""`python -m kovalo.mcp_server` = `kovalo-mcp` (stdio dla klientów, którzy podają interpreter)."""

from kovalo.mcp_server.server import entrypoint

if __name__ == "__main__":
    entrypoint()
