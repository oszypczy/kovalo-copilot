"""Filtr środowiska podprocesu serwera Kovalo (kontrakt F5, 8.2, P8, NFR-7).

`test_stdio_env_has_no_keys` (test_stdio.py) sprawdza ogólny `stdio_connection` z `env=None`, czyli
domyślne środowisko SDK; ono samo nie przenosi kluczy, więc ten test przechodziłby także bez
`kovalo_stdio_env`. Tu sprawdzamy sam filtr i to, że oba miejsca otwierające stdio (źródło narzędzi
agenta, E5) naprawdę go używają. Bez sieci i bez podprocesów."""

from __future__ import annotations

import pytest

SECRETS = {
    "GOOGLE_API_KEY": "g",
    "GEMINI_API_KEY": "g2",
    "OPENAI_API_KEY": "o",
    "ANTHROPIC_API_KEY": "a",
    "LANGSMITH_API_KEY": "l",
    "LANGSMITH_TRACING": "true",
    "LANGSMITH_PROJECT": "p",
    "LANGCHAIN_API_KEY": "lc",
    "LANGCHAIN_TRACING_V2": "true",
    # także klucz przekazany jako KOVALO_*: przedrostek nie jest przepustką
    "KOVALO_LLM__GOOGLE_API_KEY": "k",
}


def test_kovalo_stdio_env_drops_keys_and_keeps_the_rest(monkeypatch):
    from kovalo.mcp_client.connection import kovalo_stdio_env

    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("KOVALO_MCP_SERVER__SEARCH", "memory")
    monkeypatch.setenv("KOVALO_DB__HOST", "db")
    monkeypatch.setenv("NIEZWIAZANA_ZMIENNA", "x")
    env = kovalo_stdio_env()
    leaked = sorted(set(env) & set(SECRETS))
    assert leaked == [], f"klucze w środowisku serwera: {leaked}"
    assert env["KOVALO_MCP_SERVER__SEARCH"] == "memory" and env["KOVALO_DB__HOST"] == "db"
    assert "PATH" in env, "bezpieczne zmienne domyślne SDK zostają"
    assert "NIEZWIAZANA_ZMIENNA" not in env, "tylko domyślne zmienne SDK i KOVALO_*"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())


def test_kovalo_stdio_env_does_not_modify_os_environ(monkeypatch):
    import os

    from kovalo.mcp_client.connection import kovalo_stdio_env

    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    kovalo_stdio_env()
    assert os.environ["GOOGLE_API_KEY"] == "g", "filtr działa na kopii"


def _capture_stdio(monkeypatch, module):
    from kovalo.mcp_client.connection import McpStartupError

    seen: dict = {}

    def fake(args, cwd, **kwargs):
        seen.update(args=args, cwd=cwd, **kwargs)
        raise McpStartupError("stop")

    monkeypatch.setattr(module, "stdio_connection", fake)
    return seen


def test_open_tool_source_stdio_passes_the_filtered_env(settings, monkeypatch):
    from kovalo.mcp_client import source as source_module
    from kovalo.mcp_client.connection import McpStartupError

    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    monkeypatch.setenv("LANGSMITH_API_KEY", "l")
    monkeypatch.setenv("KOVALO_MCP_SERVER__SEARCH", "memory")
    seen = _capture_stdio(monkeypatch, source_module)
    src = source_module.open_tool_source(settings, names=["get_invoice_value"], transport="stdio")
    with pytest.raises(McpStartupError, match="stop"):
        src.tools()
    env = seen["env"]
    assert env is not None, "env=None oznaczałoby środowisko domyślne SDK bez KOVALO_*"
    assert "GOOGLE_API_KEY" not in env and "LANGSMITH_API_KEY" not in env
    assert env.get("KOVALO_MCP_SERVER__SEARCH") == "memory"
    assert seen["args"][:2] == ["-m", "kovalo.mcp_server"]
    assert (seen["startup_timeout_s"], seen["call_timeout_s"], seen["cleanup_timeout_s"]) == (
        60,
        120,
        10,
    )


def test_e5_stdio_passes_the_filtered_env(settings, monkeypatch, conversations, fake_embeddings):
    """E5 przez stdio (domyślny transport lokalny) też bez kluczy; nie startuje serwera: atrapa
    `stdio_connection` zgłasza błąd startu, więc `run_e5` kończy się `McpStartupError`."""
    from kovalo.eval.assistant import e5 as e5_module
    from kovalo.llm import override_embeddings
    from kovalo.mcp_client.connection import McpStartupError

    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    seen = _capture_stdio(monkeypatch, e5_module)
    with override_embeddings(fake_embeddings), pytest.raises(McpStartupError):
        e5_module.run_e5(settings, transport="stdio", ids=["C5"])
    env = seen["env"]
    assert env is not None and "GOOGLE_API_KEY" not in env and "OPENAI_API_KEY" not in env
