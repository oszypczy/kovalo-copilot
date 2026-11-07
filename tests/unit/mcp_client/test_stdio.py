"""Prawdziwa ścieżka stdio (kontrakt F5, 8.2 i 13.2): port `test_provided_stdio.py` kursu.

stdout serwera Kovalo przenosi wyłącznie komunikaty protokołu, a czas życia połączenia jest
ograniczony bez względu na to, co robi serwer: kończy działanie, nigdy nie odpowiada, ulega
awarii w trakcie wywołania, zawiesza się albo ignoruje łagodne zamknięcie. Każdy scenariusz
działa we własnym procesie Pythona z ZEWNĘTRZNYM limitem czasu, więc błąd zwalniania zasobów
kończy test wynikiem negatywnym, zamiast zawiesić cały zestaw. Serwery-zabawki zapisują PID
(tests/fixtures/mcp_toy_servers/). Bez klucza API i bez sieci."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOYS = ROOT / "tests" / "fixtures" / "mcp_toy_servers"
POSIX = pytest.mark.skipif(sys.platform == "win32", reason="sprawdzanie procesów: tylko POSIX")

SCENARIO = r"""
import json, os, sys, time
from kovalo.mcp_client.connection import McpCallError, McpStartupError, stdio_connection
toy, pid_file, action, cleanup = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
out = {}
started = time.monotonic()
try:
    conn = stdio_connection([toy, pid_file], os.path.dirname(toy), startup_timeout_s=2,
                            call_timeout_s=1, cleanup_timeout_s=cleanup)
    out["started"] = True
    if action == "crash":
        try: conn.call("crash", {})
        except McpCallError as e: out["error"] = str(e)
    if action == "hang":
        try: conn.call("hang", {})
        except McpCallError as e: out["error"] = str(e)
        try: conn.call("ping", {})
        except McpCallError as e: out["after"] = str(e)
    t0 = time.monotonic()
    conn.close()
    conn.close()
    out["close_s"] = round(time.monotonic() - t0, 2)
    out["thread_alive"] = conn.alive
except McpStartupError as e:
    out["startup_error"] = str(e)
out["seconds"] = round(time.monotonic() - started, 2)
print(json.dumps(out))
"""


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_scenario(
    tmp_path: Path, toy: str, action: str, deadline: float = 40, cleanup: float = 6
) -> tuple[dict, Path]:
    folder = tmp_path / "toy servers"  # ścieżka ze spacją
    folder.mkdir(exist_ok=True)
    shutil.copy(TOYS / toy, folder / toy)
    pid_file = folder / f"{toy}.pid"
    proc = subprocess.run(
        [sys.executable, "-c", SCENARIO, str(folder / toy), str(pid_file), action, str(cleanup)],
        capture_output=True,
        text=True,
        timeout=deadline,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    return json.loads(proc.stdout.strip().splitlines()[-1]), pid_file


@POSIX
def test_a_server_that_exits_at_start_fails_promptly(tmp_path):
    out, pid_file = run_scenario(tmp_path, "exits_at_start.py", "none")
    assert "startup_error" in out and out["seconds"] < 10, out
    assert "celowo nie startuje" in out["startup_error"], "końcówka stderr serwera w komunikacie"
    assert not pid_alive(int(pid_file.read_text()))


@POSIX
def test_a_silent_server_hits_the_startup_deadline_and_is_ended(tmp_path):
    out, pid_file = run_scenario(tmp_path, "silent.py", "none")
    assert "nie dokończył inicjalizacji w ciągu 2 s" in out.get("startup_error", ""), out
    assert out["seconds"] < 2 + 6 + 8, "ograniczone: termin startu plus sprzątanie"
    assert not pid_alive(int(pid_file.read_text())), "proces milczącego serwera zakończony"


@POSIX
def test_a_server_that_ignores_a_graceful_close_is_killed(tmp_path):
    out, pid_file = run_scenario(tmp_path, "stubborn.py", "none")
    assert out.get("started") and out["close_s"] < 6 + 8, out
    assert out["thread_alive"] is False and not pid_alive(int(pid_file.read_text())), (
        "SIGTERM zignorowany: SDK przechodzi do kill"
    )


@POSIX
def test_a_server_that_dies_during_a_call(tmp_path):
    out, pid_file = run_scenario(tmp_path, "fragile.py", "crash")
    assert "[błąd transportu]" in out.get("error", ""), out
    assert out["thread_alive"] is False and not pid_alive(int(pid_file.read_text()))


@POSIX
def test_a_hanging_call_times_out_and_the_connection_is_not_reused(tmp_path):
    out, pid_file = run_scenario(tmp_path, "fragile.py", "hang")
    assert "przekroczyło limit czasu 1 s" in out.get("error", ""), out
    assert "nie nadaje się do użycia" in out.get("after", "")
    assert out["thread_alive"] is False and not pid_alive(int(pid_file.read_text()))


@POSIX
def test_the_smallest_allowed_cleanup_deadline_still_ends_a_stubborn_server(tmp_path):
    from kovalo.mcp_client.connection import MIN_STDIO_CLEANUP_S

    assert MIN_STDIO_CLEANUP_S == 5.0
    out, pid_file = run_scenario(tmp_path, "stubborn.py", "none", cleanup=MIN_STDIO_CLEANUP_S)
    assert out.get("started") and out["thread_alive"] is False, out
    assert not pid_alive(int(pid_file.read_text())), "zakończył się proces, nie tylko wątek"


def test_a_cleanup_deadline_too_short_for_the_sdks_teardown_is_refused(tmp_path):
    from kovalo.config import ConfigError
    from kovalo.mcp_client.connection import stdio_connection

    with pytest.raises(ConfigError, match="co najmniej 5 s"):
        stdio_connection(
            [TOYS / "stubborn.py", tmp_path / "pid"],
            tmp_path,
            startup_timeout_s=2,
            call_timeout_s=1,
            cleanup_timeout_s=0.05,
        )
    assert not (tmp_path / "pid").exists(), "odrzucone, zanim uruchomiono jakikolwiek proces"


# --- NASZE ----------------------------------------------------------------------------------------


def test_kovalo_stdio_args(settings):
    from kovalo.mcp_client.connection import kovalo_stdio_args

    assert kovalo_stdio_args(settings, search="memory") == [
        "-m",
        "kovalo.mcp_server",
        "--transport",
        "stdio",
        "--search",
        "memory",
    ]
    assert kovalo_stdio_args(settings, search="pgvector")[-2:] == ["--search", "pgvector"]


@POSIX
def test_stdio_env_has_no_keys(tmp_path, monkeypatch):
    """NFR-7: podproces serwera nie dostaje kluczy LLM-a ani LangSmitha mimo ustawienia ich
    w środowisku testu; zachowuje bezpieczne zmienne domyślne SDK (np. PATH)."""
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.setenv(name, "sekret-testowy")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    folder = tmp_path / "toy"
    folder.mkdir()
    shutil.copy(TOYS / "env_report.py", folder / "env_report.py")
    names_file = folder / "env.txt"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            SCENARIO,
            str(folder / "env_report.py"),
            str(names_file),
            "none",
            "6",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    names = set(names_file.read_text().split())
    assert "PATH" in names
    leaked = [n for n in names if n.startswith("LANGSMITH") or n.endswith("_API_KEY")]
    assert leaked == [], f"klucze w środowisku serwera: {leaked}"


@POSIX
def test_the_kovalo_servers_stdout_is_protocol_only(monkeypatch):
    """`python -m kovalo.mcp_server --transport stdio --search memory`: każda linia stdout to
    JSON-RPC (żadnych logów ani printów); `get_invoice_value` nie liczy embeddingów."""
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "kovalo.mcp_server", "--transport", "stdio", "--search", "memory"],
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    watchdog = threading.Timer(90, proc.kill)  # zewnętrzny limit czasu
    watchdog.start()
    try:

        def send(msg):
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()

        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "t", "version": "0"},
                },
            }
        )
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        send(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "get_invoice_value",
                    "arguments": {"supplier": "SUP-114", "month": "2026-07"},
                },
            }
        )
        frames = []
        while len([f for f in frames if "id" in f]) < 3:
            line = proc.stdout.readline()
            assert line, f"serwer zamknął stdout; stderr: {proc.stderr.read()[-800:]}"
            frame = json.loads(line)  # każdy wiersz stdout musi być komunikatem JSON-RPC
            assert frame.get("jsonrpc") == "2.0", line
            frames.append(frame)
        tools = next(f for f in frames if f.get("id") == 2)["result"]["tools"]
        assert len(tools) == 7, "wariant z configu: ours"
        answer = next(f for f in frames if f.get("id") == 3)
        assert answer["result"]["structuredContent"]["invoiced_grosze"] == 18240000
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)
        watchdog.cancel()
