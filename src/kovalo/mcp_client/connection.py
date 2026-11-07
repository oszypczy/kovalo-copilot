"""Połączenie z serwerem MCP w zwykłym (synchronicznym) kodzie (FR-5.2, FR-5.3).

Port `helpers/mcp_runtime.py` kursu 1:1 (kontrakt: docs/contracts/f5.md, sekcja 8.2; ADR-0013
pkt 2 i 4). McpConnection prowadzi jedną sesję klienta MCP w pętli zdarzeń działającej w tle,
więc reszta projektu (LangGraph, run_question) pozostaje synchroniczna. Udostępnia:

  call()     wiążący rekord (surowy structuredContent sprawdzony z kanonicznym rekordem, zanim
             cokolwiek uzupełni wartości domyślne) i pochodzenie wywołania;
  tools()    narzędzia LangChain (content_and_artifact): model widzi sam rekord, a pochodzenie
             idzie do artefaktu ToolMessage (NASZE, ADR-0013 pkt 4);
  captures   każde wywołanie tak, jak przebiegło (identyfikator, wątek i tura, narzędzie,
             argumenty, surowy structuredContent, tekst, isError, wynik);
  close()    zamknięcie z limitem czasu, idempotentne.

Cztery warstwy błędów są rozdzielone i nazwane w każdym komunikacie:
  [błąd narzędzia]          serwer odpowiedział z isError=true (złe argumenty, wyjątek w narzędziu)
  [błąd protokołu]          serwer odpowiedział obiektem błędu JSON-RPC
  [błąd transportu]         brak odpowiedzi: serwer się zakończył, potok albo połączenie HTTP się
                            zerwało, minął limit czasu wywołania
  [błąd walidacji lokalnej] odpowiedź jest niezgodna z kanonicznym kontraktem
Zwykły wynik dziedzinowy ("no_data", "not_applicable", ...) nie jest żadnym z nich: to wynik.

NASZE (poza kursem): `call()` zwraca (rekord, pochodzenie) zamiast koperty `__mcp__`,
`tools(names)`, `capture_limit`, `http_connection` (streamable HTTP), `kovalo_stdio_args`
i `kovalo_stdio_env` (środowisko podprocesu bez kluczy, NFR-7).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import os
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable, Collection, Coroutine
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from mcp import ClientSession

from kovalo.config import ConfigError, Settings
from kovalo.mcp_client.contract import check_record
from kovalo.mcp_client.errors import McpCallError, McpStartupError

__all__ = [
    "MIN_STDIO_CLEANUP_S",
    "McpCallError",
    "McpConnection",
    "McpStartupError",
    "http_connection",
    "in_memory_connection",
    "kovalo_stdio_args",
    "kovalo_stdio_env",
    "stdio_connection",
]

# Zamykanie stdio w przypiętej wersji SDK: zamknij stdin, czekaj do 2 s, zakończ drzewo procesów,
# po kolejnych 2 s je zabij. Krótszy limit sprzątania zatrzymałby pętlę, zanim proces potomny
# zniknie.
MIN_STDIO_CLEANUP_S = 5.0

# Zmienne, których podproces serwera Kovalo nigdy nie dostaje (NFR-7): klucze LLM-a i LangSmitha.
_SECRET_PREFIXES = ("LANGSMITH_", "LANGCHAIN_")
_SECRET_SUFFIXES = ("_API_KEY",)

OpenSession = Callable[[], AbstractAsyncContextManager[ClientSession]]


def _leaf(exc: BaseException) -> BaseException:
    """Pierwszy prawdziwy wyjątek wewnątrz (zagnieżdżonych) grup wyjątków anyio."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


class McpConnection:
    """Jedna sesja klienta MCP prowadzona przez pętlę zdarzeń w tle (menedżer kontekstu).

    open_session: fabryka asynchronicznych menedżerów kontekstu, która daje ClientSession
    (stdio, streamable HTTP albo sesja w pamięci). Limity czasu z sekcji `mcp` configu (PINNED).
    capture_limit: None = wszystkie zapisy (E5, testy); N = ostatnie N (api: 100).
    """

    def __init__(
        self,
        open_session: OpenSession,
        *,
        startup_timeout_s: float,
        call_timeout_s: float,
        cleanup_timeout_s: float,
        server_name: str = "kovalo",
        stderr_tail: Callable[[], str] | None = None,
        capture_limit: int | None = None,
    ) -> None:
        self.call_timeout_s, self.cleanup_timeout_s = call_timeout_s, cleanup_timeout_s
        self.server_name = server_name
        self.captures: list[dict[str, Any]] | deque[dict[str, Any]] = (
            [] if capture_limit is None else deque(maxlen=capture_limit)
        )
        self.context: dict[str, Any] = {"thread_id": None, "turn": None}
        self.unusable: str | None = None
        self._stderr_tail = stderr_tail or (lambda: "")
        self._closed = False
        self.turn_lock = threading.Lock()  # jedna tura asystenta naraz (E5, run_turn)
        self._captures_lock = threading.Lock()
        self._pending: set[concurrent.futures.Future[Any]] = set()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="mcp-connection", daemon=True
        )
        self._thread.start()
        self._stop: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self.session: ClientSession | None = None
        try:
            ready = asyncio.run_coroutine_threadsafe(
                self._start(open_session, startup_timeout_s), self._loop
            )
            self._tools_info: list[Any] = ready.result(
                timeout=startup_timeout_s + cleanup_timeout_s + 10
            )
        except BaseException as exc:
            self.close()
            if isinstance(exc, McpStartupError):
                raise
            tail = self._stderr_tail()
            raise McpStartupError(
                f"serwer MCP się nie uruchomił: {type(exc).__name__}: {exc}"
                + (f"\nstderr serwera (ostatnie wiersze):\n{tail}" if tail else "")
            ) from None

    async def _start(self, open_session: OpenSession, startup_timeout_s: float) -> list[Any]:
        self._stop = asyncio.Event()
        stop = self._stop
        ready: asyncio.Future[list[Any]] = asyncio.get_running_loop().create_future()

        async def hold() -> None:
            lost: BaseException | None = None
            try:
                async with open_session() as session:
                    try:
                        await asyncio.wait_for(session.initialize(), startup_timeout_s)
                    except TimeoutError:
                        raise McpStartupError(
                            f"serwer MCP nie dokończył inicjalizacji w ciągu "
                            f"{startup_timeout_s:g} s (config mcp.startup_timeout_s)"
                        ) from None
                    self.session = session
                    listed = await asyncio.wait_for(session.list_tools(), startup_timeout_s)
                    ready.set_result(list(listed.tools))
                    await stop.wait()
            except BaseException as exc:  # błąd startu: zgłoś od razu, nigdy nie czekaj na limit
                lost = _leaf(exc)
                if not ready.done():
                    ready.set_exception(
                        lost if isinstance(lost, Exception) else RuntimeError(str(lost))
                    )
                if not isinstance(exc, Exception | asyncio.CancelledError):
                    raise
            finally:
                if ready.done() and not stop.is_set():  # sesja padła sama, nie przez close()
                    self._session_lost(lost)

        self._task = asyncio.create_task(hold())
        return await ready

    def _session_lost(self, cause: BaseException | None) -> None:
        """Sesja skończyła się bez close() (serwer zniknął, transport zerwany; D2 F5): połączenie
        jest bezużyteczne, a oczekujące wywołania dostają od razu błąd transportu zamiast czekać
        do mcp.call_timeout_s (SDK nie odpowiada na nie po śmierci transportu)."""
        reason = f"{type(cause).__name__}: {cause}" if cause is not None else "koniec strumienia"
        with self._captures_lock:
            self.unusable = (
                self.unusable or f"sesja MCP zakończona przez serwer albo transport ({reason})"
            )
            pending = list(self._pending)
        for future in pending:
            future.cancel()

    # --- wywołania -----------------------------------------------------------------------------

    def _capture(self, capture: dict[str, Any]) -> None:
        with self._captures_lock:
            self.captures.append(capture)

    def call(self, tool: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """(rekord, pochodzenie): rekord = check_record(...) (dokładnie to, co zwraca narzędzie
        bezpośrednie), pochodzenie = {"transport": "mcp", "invocation_id", "server",
        "thread_id", "turn", "seconds"}. Błąd → McpCallError z warstwą."""
        invocation_id = uuid.uuid4().hex[:12]
        identity = dict(self.context)  # ustalone teraz, nigdy nie czytane ponownie
        capture: dict[str, Any] = {
            "invocation_id": invocation_id,
            **identity,
            "tool": tool,
            "arguments": arguments,
            "server": self.server_name,
            "structured": None,
            "text": None,
            "is_error": None,
            "outcome": None,
        }
        self._capture(capture)
        if self._closed or self.unusable:
            capture["outcome"] = "transport"
            raise McpCallError(
                "transport",
                f"połączenie nie nadaje się do użycia: {self.unusable or 'zamknięte'}",
                invocation_id,
            )
        started = time.monotonic()
        future = asyncio.run_coroutine_threadsafe(self._call(tool, arguments), self._loop)
        with self._captures_lock:
            self._pending.add(future)
            if self.unusable:  # sesja padła między sprawdzeniem a wysłaniem
                future.cancel()
        try:
            result = future.result(timeout=self.call_timeout_s + self.cleanup_timeout_s + 5)
        except (concurrent.futures.CancelledError, asyncio.CancelledError):  # close() albo utrata
            capture["outcome"] = "transport"
            reason = self.unusable or "połączenie zamknięto, gdy wywołanie czekało na odpowiedź"
            raise McpCallError("transport", f"{tool}: {reason}", invocation_id) from None
        except TimeoutError:  # wywołanie anulowane i dokończone w pętli (wait_for)
            self.unusable = f"wywołanie {tool} przekroczyło limit czasu {self.call_timeout_s:g} s"
            capture["outcome"] = "transport"
            raise McpCallError(
                "transport",
                self.unusable + " (config mcp.call_timeout_s); połącz się ponownie",
                invocation_id,
            ) from None
        except Exception as exc:
            leaf = _leaf(exc)
            from mcp.shared.exceptions import McpError
            from mcp.types import CONNECTION_CLOSED

            # SDK zgłasza zamknięty potok jako McpError(CONNECTION_CLOSED): nic nie odebrano
            if isinstance(leaf, McpError) and leaf.error.code != CONNECTION_CLOSED:
                capture["outcome"] = "protocol"
                raise McpCallError(
                    "protocol",
                    f"{tool}: {leaf.error.message} (kod {leaf.error.code})",
                    invocation_id,
                ) from None
            self.unusable = f"{type(leaf).__name__}: {leaf}"
            capture["outcome"] = "transport"
            raise McpCallError(
                "transport", f"{tool}: serwer nie odpowiedział ({self.unusable})", invocation_id
            ) from None
        finally:
            self._pending.discard(future)
        text = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        capture.update(
            structured=result.structuredContent, text=text, is_error=bool(result.isError)
        )
        if result.isError:
            capture["outcome"] = "tool"
            raise McpCallError("tool", f"{tool}: {' '.join(text)[:300]}", invocation_id)
        try:
            record = check_record(tool, result.structuredContent, text)
        except McpCallError as exc:
            capture["outcome"] = "validation"
            raise McpCallError(
                "validation", str(exc).removeprefix("[błąd walidacji lokalnej] "), invocation_id
            ) from None
        capture["outcome"] = "ok"
        provenance = {
            "transport": "mcp",
            "invocation_id": invocation_id,
            "server": self.server_name,
            **identity,
            "seconds": round(time.monotonic() - started, 3),
        }
        return record, provenance

    async def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        assert self.session is not None
        return await asyncio.wait_for(self.session.call_tool(tool, arguments), self.call_timeout_s)

    def run(
        self,
        coro_factory: Callable[[ClientSession], Coroutine[Any, Any, Any]],
        timeout: float | None = None,
    ) -> Any:
        """Uruchamia session.<coś> w pętli, np. conn.run(lambda s: s.list_resources())."""
        if self._closed or self.unusable or self.session is None:
            raise McpCallError(
                "transport", f"połączenie nie nadaje się do użycia: {self.unusable or 'zamknięte'}"
            )
        session = self.session
        limit = timeout or self.call_timeout_s

        async def bounded() -> Any:
            return await asyncio.wait_for(coro_factory(session), limit)

        future = asyncio.run_coroutine_threadsafe(bounded(), self._loop)
        with self._captures_lock:
            self._pending.add(future)
        try:
            return future.result(timeout=limit + self.cleanup_timeout_s + 5)
        except concurrent.futures.CancelledError:
            raise McpCallError(
                "transport", f"połączenie nie nadaje się do użycia: {self.unusable or 'zamknięte'}"
            ) from None
        finally:
            self._pending.discard(future)

    def tool_infos(self, names: Collection[str] | None = None) -> list[Any]:
        """Definicje narzędzi z list_tools() (opcjonalnie tylko `names`; brak któregoś na
        serwerze → McpStartupError z listą brakujących)."""
        infos = list(self._tools_info)
        if names is None:
            return infos
        wanted = set(names)
        missing = sorted(wanted - {info.name for info in infos})
        if missing:
            raise McpStartupError(
                f"serwer MCP {self.server_name} nie ma wymaganych narzędzi: " + ", ".join(missing)
            )
        return [info for info in infos if info.name in wanted]

    def tools(self, names: Collection[str] | None = None) -> list[StructuredTool]:
        """Narzędzia z tool_infos(names) jako StructuredTool z
        response_format='content_and_artifact' i handle_tool_error=True: model widzi rekord
        (tę samą treść co narzędzie bezpośrednie), trace czyta pochodzenie z artefaktu."""
        return [tool_from_info(info, self.call) for info in self.tool_infos(names)]

    # --- cykl życia ----------------------------------------------------------------------------

    def close(self) -> None:
        """Z limitem czasu i idempotentne: prosi sesję o zakończenie (SDK zamyka stdin, czeka,
        potem kończy i zabija drzewo procesów), anuluje ją, jeśli się nie skończy, zatrzymuje
        pętlę."""
        if self._closed:
            return
        self._closed = True
        for future in list(self._pending):  # zwolnij każdego, kto wciąż czeka na odpowiedź
            future.cancel()
        task, stop = self._task, self._stop
        if task is not None and stop is not None:
            self._loop.call_soon_threadsafe(stop.set)
            self._wait_for(task)
            if not task.done():  # łagodne zamknięcie się nie skończyło: anuluj (SDK sprząta)
                self._loop.call_soon_threadsafe(task.cancel)
                self._wait_for(task)
            if not task.done():
                print(
                    "ostrzeżenie: sesja MCP nie zakończyła zamykania w ciągu "
                    f"{2 * self.cleanup_timeout_s:g} s",
                    file=sys.stderr,
                )
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=self.cleanup_timeout_s + 5)

    def _wait_for(self, task: asyncio.Task[None]) -> None:
        async def wait() -> None:
            await asyncio.wait({task}, timeout=self.cleanup_timeout_s)

        done = asyncio.run_coroutine_threadsafe(wait(), self._loop)
        with contextlib.suppress(Exception):  # limit zamykania: wywołujący sprawdza task.done()
            done.result(timeout=self.cleanup_timeout_s + 5)

    def __enter__(self) -> McpConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def alive(self) -> bool:
        """True, dopóki działa wątek pętli w tle."""
        return self._thread.is_alive()

    @property
    def closed(self) -> bool:
        """True po close() (także nieudanym starcie)."""
        return self._closed


def tool_from_info(
    info: Any, call: Callable[[str, dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]
) -> StructuredTool:
    """StructuredTool z definicji narzędzia MCP (nazwa, opis, inputSchema), który woła `call`.

    handle_tool_error=True: McpCallError (ToolException) staje się ToolMessage z błędem, który
    model może przeczytać (węzeł narzędzi LangGraph 1 nie łapie wyjątków za nas)."""
    name = info.name

    def run(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return call(name, kwargs)

    return StructuredTool.from_function(
        func=run,
        name=name,
        description=info.description or name,
        args_schema=info.inputSchema,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )


def stdio_connection(
    args: list[Any],
    cwd: Path,
    *,
    startup_timeout_s: float,
    call_timeout_s: float,
    cleanup_timeout_s: float,
    server_name: str = "kovalo",
    env: dict[str, str] | None = None,
    capture_limit: int | None = None,
) -> McpConnection:
    """Uruchamia `python <args...>` (tym interpreterem) jako serwer MCP przez stdio i łączy się
    z nim. env None → get_default_environment() SDK (jak kurs)."""
    from mcp import StdioServerParameters
    from mcp.client.stdio import get_default_environment, stdio_client

    if cleanup_timeout_s < MIN_STDIO_CLEANUP_S:
        raise ConfigError(
            f"mcp.cleanup_timeout_s musi wynosić co najmniej {MIN_STDIO_CLEANUP_S:g} s dla "
            f"serwera stdio (jest {cleanup_timeout_s:g}): zamykanie w samym SDK może trwać do 4 s "
            "(2 s na łagodne zakończenie, potem terminate, a po kolejnych 2 s kill)"
        )
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(a) for a in args],
        cwd=str(cwd),
        env=env if env is not None else get_default_environment(),
    )
    errlog = _StderrTail()

    @asynccontextmanager
    async def open_session() -> AsyncIterator[ClientSession]:
        async with (
            stdio_client(params, errlog=errlog) as (read, write),  # type: ignore[arg-type]
            ClientSession(read, write) as session,
        ):
            yield session

    return McpConnection(
        open_session,
        startup_timeout_s=startup_timeout_s,
        call_timeout_s=call_timeout_s,
        cleanup_timeout_s=cleanup_timeout_s,
        server_name=server_name,
        stderr_tail=errlog.tail,
        capture_limit=capture_limit,
    )


def http_connection(
    url: str,
    *,
    startup_timeout_s: float,
    call_timeout_s: float,
    cleanup_timeout_s: float,
    server_name: str = "kovalo",
    capture_limit: int | None = None,
) -> McpConnection:
    """Streamable HTTP (`streamable_http_client(url)` + ClientSession). Odmowa połączenia, HTTP
    5xx, zerwane połączenie i brak odpowiedzi w call_timeout_s → McpCallError('transport')
    i `unusable`; błąd przy starcie → McpStartupError z adresem (bez nagłówków)."""
    from mcp.client.streamable_http import streamable_http_client

    @asynccontextmanager
    async def open_session() -> AsyncIterator[ClientSession]:
        async with (
            streamable_http_client(url) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            yield session

    try:
        return McpConnection(
            open_session,
            startup_timeout_s=startup_timeout_s,
            call_timeout_s=call_timeout_s,
            cleanup_timeout_s=cleanup_timeout_s,
            server_name=server_name,
            capture_limit=capture_limit,
        )
    except McpStartupError as exc:
        raise McpStartupError(f"{exc} (adres: {url})") from None


def in_memory_connection(
    server: Any,
    *,
    startup_timeout_s: float = 10,
    call_timeout_s: float = 10,
    cleanup_timeout_s: float = 5,
    capture_limit: int | None = None,
) -> McpConnection:
    """Łączy się z obiektem serwera FastMCP w tym procesie (bez podprocesu): testy i E5
    w pamięci. Serwer przychodzi przez parametr (mcp_client/ nie importuje mcp_server/)."""
    import anyio
    from mcp.shared.memory import create_client_server_memory_streams

    low = server._mcp_server

    @asynccontextmanager
    async def open_session() -> AsyncIterator[ClientSession]:
        async with (
            create_client_server_memory_streams() as (client_streams, server_streams),
            anyio.create_task_group() as group,
        ):

            async def serve() -> None:
                await low.run(
                    server_streams[0],
                    server_streams[1],
                    low.create_initialization_options(),
                    raise_exceptions=False,
                )

            group.start_soon(serve)
            try:
                async with ClientSession(*client_streams) as session:
                    yield session
            finally:
                group.cancel_scope.cancel()

    return McpConnection(
        open_session,
        startup_timeout_s=startup_timeout_s,
        call_timeout_s=call_timeout_s,
        cleanup_timeout_s=cleanup_timeout_s,
        server_name=getattr(server, "name", "kovalo"),
        capture_limit=capture_limit,
    )


def kovalo_stdio_args(settings: Settings, *, search: Literal["pgvector", "memory"]) -> list[str]:
    """Argumenty interpretera dla serwera Kovalo przez stdio (ten sam interpreter)."""
    return ["-m", "kovalo.mcp_server", "--transport", "stdio", "--search", search]


def kovalo_stdio_env() -> dict[str, str]:
    """Środowisko podprocesu serwera Kovalo (P8, NFR-7): domyślne zmienne SDK + `KOVALO_*`,
    bez kluczy LLM-a (`*_API_KEY`) i LangSmitha (`LANGSMITH_*`, `LANGCHAIN_*`)."""
    from mcp.client.stdio import get_default_environment

    env = dict(get_default_environment())
    env.update({k: v for k, v in os.environ.items() if k.startswith("KOVALO_")})
    return {
        k: v
        for k, v in env.items()
        if not k.startswith(_SECRET_PREFIXES) and not k.endswith(_SECRET_SUFFIXES)
    }


class _StderrTail:
    """Plikopodobne ujście stderr procesu serwera, które zachowuje ostatnie wiersze."""

    def __init__(self, lines: int = 15) -> None:
        self._file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")  # noqa: SIM115
        self._lines = lines

    def fileno(self) -> int:
        return self._file.fileno()

    def write(self, text: str) -> int:
        return self._file.write(text)

    def flush(self) -> None:
        self._file.flush()

    def tail(self) -> str:
        try:
            self._file.flush()
            self._file.seek(0)
            return "\n".join(self._file.read().splitlines()[-self._lines :])
        except Exception:
            return ""
