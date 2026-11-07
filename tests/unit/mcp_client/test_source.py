"""McpToolSource i open_tool_source (kontrakt F5, 8.3 i 13.2): ponowne łączenie po błędzie
transportu (FR-5.3), błąd startu, zamykanie. Serwer w pamięci; `http_connection` na prawdziwym
uvicornie w wątku; jedno połączenie stdio w podprocesie z zewnętrznym limitem czasu."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kovalo.company import InvoiceResult
from tests.mcp_fakes import connect, mcp_server

REPO_ROOT = Path(__file__).resolve().parents[3]
ARGS = {"supplier": "SUP-114", "month": "2026-07"}


def make_source(settings, connects, names=None):
    from kovalo.mcp_client.source import McpToolSource
    from kovalo.tools.specs import COURSE_TOOLS

    server = mcp_server(settings, "course")
    made: list = []

    def factory():
        made.append(connect(server))
        connects.append(made[-1])
        return made[-1]

    return McpToolSource(factory, names=names or COURSE_TOOLS), made


def test_tools_are_built_once_from_the_first_connection(settings):
    connects: list = []
    source, made = make_source(settings, connects)
    try:
        tools = source.tools()
        assert sorted(t.name for t in tools) == sorted(
            [
                "search_documents",
                "get_supplier_otif",
                "get_invoice_value",
                "get_contract_terms",
                "calculate_otif_penalty",
                "check_discount_policy",
            ]
        )
        assert len(made) == 1 and source.connection is made[0]
        assert [t.name for t in source.tools()] == [t.name for t in tools]
    finally:
        source.close()


def test_reconnects_after_transport_error(settings):
    """Połączenie `unusable` nigdy nie jest używane ponownie; następne wywołanie dostaje nowe,
    a stare zostaje zamknięte. Narzędzia z `tools()` są stabilne i idą przez nowe połączenie."""
    connects: list = []
    source, made = make_source(settings, connects)
    try:
        tools = {t.name: t for t in source.tools()}
        record, provenance = source.call("get_invoice_value", ARGS)
        assert record["invoiced_grosze"] == 18240000 and provenance["transport"] == "mcp"
        first = made[0]
        first.unusable = "symulowany błąd transportu"
        record, _ = source.call("get_invoice_value", ARGS)
        assert record["invoiced_grosze"] == 18240000
        assert len(made) == 2 and source.connection is made[1]
        assert not first.alive, "stare połączenie zamknięte"
        assert len(first.captures) == 1 and len(made[1].captures) == 1
        message = tools["get_invoice_value"].invoke(
            {"type": "tool_call", "id": "t1", "name": "get_invoice_value", "args": ARGS}
        )
        assert json.loads(message.content)["invoiced_grosze"] == 18240000
        assert len(made[1].captures) == 2 and len(made) == 2
    finally:
        source.close()


def test_a_closed_connection_is_replaced(settings):
    connects: list = []
    source, made = make_source(settings, connects)
    try:
        source.tools()
        made[0].close()
        source.call("get_invoice_value", ARGS)
        assert len(made) == 2
    finally:
        source.close()


def test_connect_failure_is_transport_error(settings):
    from kovalo.mcp_client.connection import McpCallError, McpStartupError
    from kovalo.mcp_client.source import McpToolSource
    from kovalo.tools.specs import COURSE_TOOLS

    server = mcp_server(settings, "course")
    first = connect(server)
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] == 1:
            return first
        raise McpStartupError("serwer się nie uruchomił")

    source = McpToolSource(factory, names=COURSE_TOOLS)
    try:
        source.tools()
        first.unusable = "zerwane połączenie"
        with pytest.raises(McpCallError, match="serwer MCP niedostępny") as info:
            source.call("get_invoice_value", ARGS)
        assert info.value.layer == "transport" and str(info.value).startswith("[błąd transportu]")
        assert not first.alive
    finally:
        source.close()


def test_first_tools_failure_propagates():
    from kovalo.mcp_client.connection import McpStartupError
    from kovalo.mcp_client.source import McpToolSource
    from kovalo.tools.specs import COURSE_TOOLS

    def factory():
        raise McpStartupError("serwer MCP się nie uruchomił")

    source = McpToolSource(factory, names=COURSE_TOOLS)
    with pytest.raises(McpStartupError, match="nie uruchomił"):
        source.tools()
    source.close()  # idempotentne także bez połączenia


def test_missing_names_are_a_startup_error(settings):
    from kovalo.mcp_client.connection import McpStartupError
    from kovalo.mcp_client.source import McpToolSource

    source = McpToolSource(
        lambda: connect(mcp_server(settings, "course")),
        names={"get_invoice_value", "check_termination_condition"},
    )
    try:
        with pytest.raises(McpStartupError, match="check_termination_condition"):
            source.tools()
    finally:
        source.close()


def test_concurrent_calls_after_a_transport_error_reconnect_once(settings):
    """Kilka tur naraz trafia na `unusable` połączenie: pod blokadą powstaje JEDNO nowe, a nie po
    jednym na wywołanie (FR-5.3, U3); wszystkie wywołania kończą się sukcesem."""
    connects: list = []
    source, made = make_source(settings, connects)
    try:
        source.tools()
        made[0].unusable = "zerwane połączenie"
        with ThreadPoolExecutor(6) as pool:
            results = list(pool.map(lambda _: source.call("get_invoice_value", ARGS), range(6)))
        assert all(rec["invoiced_grosze"] == 18240000 for rec, _ in results)
        assert len(made) == 2, "jedno nowe połączenie, nie sześć"
        assert not made[0].alive and len(made[1].captures) == 6
    finally:
        source.close()


def test_reconnect_is_retried_after_a_failed_reconnect(settings):
    """Nieudane ponowne połączenie nie zostawia źródła w stanie trwale zepsutym: następne wywołanie
    próbuje znowu (serwer wstał po restarcie), a stare, zepsute połączenie nie wraca do użycia."""
    from kovalo.mcp_client.connection import McpCallError, McpStartupError
    from kovalo.mcp_client.source import McpToolSource
    from kovalo.tools.specs import COURSE_TOOLS

    server = mcp_server(settings, "course")
    made: list = []
    outage = {"on": False}

    def factory():
        if outage["on"]:
            raise McpStartupError("serwer jeszcze nie wstał")
        made.append(connect(server))
        return made[-1]

    source = McpToolSource(factory, names=COURSE_TOOLS)
    try:
        source.tools()
        made[0].unusable = "zerwane połączenie"
        outage["on"] = True
        for _ in range(2):  # dwie nieudane próby z rzędu, bez zawieszenia
            with pytest.raises(McpCallError, match="serwer MCP niedostępny") as info:
                source.call("get_invoice_value", ARGS)
            assert info.value.layer == "transport"
        assert len(made) == 1 and len(made[0].captures) == 0, (
            "stare połączenie nie dostało wywołania"
        )
        outage["on"] = False
        record, _ = source.call("get_invoice_value", ARGS)
        assert record["invoiced_grosze"] == 18240000 and len(made) == 2
    finally:
        source.close()


def test_a_call_timeout_makes_the_next_call_use_a_new_connection(settings):
    """Nie flaga ustawiona ręcznie, tylko prawdziwy limit czasu: połączenie, na którym wywołanie
    przekroczyło `call_timeout_s`, oznacza się `unusable` i następne wywołanie dostaje nowe."""
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from kovalo.mcp_client.connection import McpCallError
    from kovalo.mcp_client.source import McpToolSource

    delays = iter([1.0, 0.0])  # pierwsze wywołanie za wolne, kolejne natychmiast

    def make_server():
        server = FastMCP("slow", log_level="WARNING")
        delay = next(delays)

        @server.tool()
        async def get_invoice_value(supplier: str, month: str) -> InvoiceResult:
            """Faktura (dubler testowy)."""
            await asyncio.sleep(delay)
            return InvoiceResult(
                status="ok", supplier_id="SUP-114", supplier="M", month=month, invoiced_grosze=1
            )

        return server

    made: list = []

    def factory():
        made.append(connect(make_server(), call_timeout_s=0.2, cleanup_timeout_s=3))
        return made[-1]

    source = McpToolSource(factory, names={"get_invoice_value"})
    try:
        source.tools()
        with pytest.raises(McpCallError, match="limit czasu"):
            source.call("get_invoice_value", ARGS)
        assert made[0].unusable
        record, _ = source.call("get_invoice_value", ARGS)
        assert record["invoiced_grosze"] == 1 and len(made) == 2 and not made[0].alive
    finally:
        source.close()


def test_close_is_idempotent(settings):
    connects: list = []
    source, made = make_source(settings, connects)
    source.tools()
    source.close()
    source.close()
    assert not made[0].alive


def test_parallel_calls_share_one_connection(settings):
    connects: list = []
    source, made = make_source(settings, connects)
    try:
        source.tools()
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(lambda _: source.call("get_invoice_value", ARGS), range(4)))
        assert len(made) == 1
        assert len({prov["invocation_id"] for _, prov in results}) == 4
    finally:
        source.close()


# --- http_connection ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class UvicornThread:
    def __init__(self, app, port):
        import uvicorn

        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert self.server.started, "uvicorn nie wystartował"

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=15)
        assert not self.thread.is_alive()


def test_http_connection_in_process(settings):
    """`http_connection` do `streamable_http_app()` w wątku: wywołanie działa; po zatrzymaniu
    serwera `[błąd transportu]`, a połączenie jest `unusable`."""
    from kovalo.mcp_client.connection import McpCallError, http_connection

    port = free_port()
    server = mcp_server(settings, "course", host="127.0.0.1", port=port)
    host = UvicornThread(server.streamable_http_app(), port)
    host.start()
    conn = http_connection(
        f"http://127.0.0.1:{port}/mcp",
        startup_timeout_s=10,
        call_timeout_s=10,
        cleanup_timeout_s=5,
    )
    try:
        record, provenance = conn.call("get_invoice_value", ARGS)
        assert record["invoiced_grosze"] == 18240000 and provenance["transport"] == "mcp"
        host.stop()
        with pytest.raises(McpCallError, match=r"\[błąd transportu\]"):
            conn.call("get_invoice_value", ARGS)
        assert conn.unusable
        with pytest.raises(McpCallError, match="nie nadaje się do użycia"):
            conn.call("get_invoice_value", ARGS)
    finally:
        conn.close()
        host.server.should_exit = True


def test_http_connection_refused_is_a_startup_error_with_the_address():
    from kovalo.mcp_client.connection import McpStartupError, http_connection

    with pytest.raises(McpStartupError, match=r"127\.0\.0\.1:1"):
        http_connection(
            "http://127.0.0.1:1/mcp", startup_timeout_s=5, call_timeout_s=5, cleanup_timeout_s=5
        )


# --- open_tool_source: transport z configu -----------------------------------------------------

SCENARIO = r"""
import json, sys
from kovalo.config import load_settings
from kovalo.mcp_client.source import open_tool_source
from kovalo.tools.specs import tool_specs

settings = load_settings()
names = [s.name for s in tool_specs("ours")]
source = open_tool_source(settings, names=names, transport="stdio", search="memory")
out = {}
try:
    out["tools"] = sorted(t.name for t in source.tools())
    args = {"supplier": "SUP-114", "month": "2026-07"}
    record, provenance = source.call("get_invoice_value", args)
    out["grosze"] = record["invoiced_grosze"]
    out["transport"] = provenance["transport"]
    out["captures"] = len(source.connection.captures)
finally:
    source.close()
print(json.dumps(out))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="podprocesy stdio: tylko POSIX")
def test_open_tool_source_over_stdio(monkeypatch):
    """Prawdziwy podproces `kovalo-mcp --search memory` z zewnętrznym limitem czasu."""
    monkeypatch.setenv("GOOGLE_API_KEY", "klucz-testowy-nieprawdziwy")
    proc = subprocess.run(
        [sys.executable, "-c", SCENARIO],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert len(out["tools"]) == 7 and out["grosze"] == 18240000
    assert out["transport"] == "mcp" and out["captures"] == 1


def test_open_tool_source_http_uses_the_configured_url(settings, monkeypatch):
    """transport None → tools.mcp_transport; http → services.mcp_url i limity z sekcji mcp (PINNED
    60 / 120 / 10 s) oraz capture_limit 100 (api)."""
    from kovalo.mcp_client import connection as connection_module
    from kovalo.mcp_client import source as source_module
    from kovalo.mcp_client.connection import McpStartupError

    seen: dict = {}

    def fake_http(url, **kwargs):
        seen.update(url=url, **kwargs)
        raise McpStartupError("stop")

    monkeypatch.setattr(connection_module, "http_connection", fake_http)
    monkeypatch.setattr(source_module, "http_connection", fake_http, raising=False)
    src = source_module.open_tool_source(settings, names=["get_invoice_value"], transport="http")
    with pytest.raises(McpStartupError, match="stop"):
        src.tools()
    assert seen["url"] == settings.services.mcp_url
    assert (seen["startup_timeout_s"], seen["call_timeout_s"], seen["cleanup_timeout_s"]) == (
        60,
        120,
        10,
    )
    assert seen["capture_limit"] == 100


def test_a_dead_http_server_is_detected_fast_not_after_the_call_timeout(settings):
    """D2 (F5): po zatrzymaniu serwera sesja SDK kończy się od razu, a oczekujące wywołanie nie
    dostaje odpowiedzi. Koniec sesji ma oznaczyć połączenie jako `unusable` i zwolnić wywołania:
    `[błąd transportu]` w kilka sekund, nie po `call_timeout_s` (tu 60 s, jak dla wolnych
    narzędzi); następne wywołanie przez McpToolSource łączy się ponownie."""
    from kovalo.mcp_client.connection import McpCallError, http_connection
    from kovalo.mcp_client.source import McpToolSource

    port = free_port()
    host = UvicornThread(
        mcp_server(settings, "course", host="127.0.0.1", port=port).streamable_http_app(), port
    )
    host.start()
    url = f"http://127.0.0.1:{port}/mcp"
    conn = http_connection(url, startup_timeout_s=10, call_timeout_s=60, cleanup_timeout_s=5)
    source = McpToolSource(lambda: conn, names=["get_invoice_value"])
    try:
        source.tools()
        assert source.call("get_invoice_value", ARGS)[0]["invoiced_grosze"] == 18240000
        host.stop()
        started = time.monotonic()
        with pytest.raises(McpCallError, match=r"\[błąd transportu\]") as info:
            conn.call("get_invoice_value", ARGS)
        assert time.monotonic() - started < 10, "martwy serwer wykryty szybko"
        assert info.value.layer == "transport" and conn.unusable
        assert conn.captures[-1]["outcome"] == "transport"
    finally:
        source.close()
        host.server.should_exit = True
