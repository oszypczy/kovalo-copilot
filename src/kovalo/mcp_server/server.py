"""Serwer MCP Kovalo (FastMCP): narzędzia, zasoby, prompt i /health (FR-5.1, FR-5.5, NFR-7).

Kontrakt: docs/contracts/f5.md, sekcja 7; ADR-0005, ADR-0013 pkt 1. Transporty: stdio (domyślny:
Claude Code, MCP Inspector, E5) i streamable HTTP (`/mcp`, usługa `mcp-server` w compose).
Serwer nie ma modelu czatu ani klucza LLM-a: embeddingi zapytań są lokalne (`get_embeddings`),
a indeks otwiera się leniwie przy pierwszym wyszukiwaniu. Przy stdio stdout przenosi wyłącznie
protokół: logi idą na stderr.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import anyio.to_thread
import yaml
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from kovalo.company import load_company_data
from kovalo.config import ConfigError, SearchBackend, Settings, ToolsVariant, get_settings
from kovalo.health import CheckResult, build_report, http_status
from kovalo.knowledge.corpus import CorpusError
from kovalo.knowledge.retrievers import (
    LazyRetriever,
    Retriever,
    build_memory_retriever,
    open_retriever,
)
from kovalo.llm import get_embeddings
from kovalo.mcp_server.registry import register_prompts, register_resources, register_tools
from kovalo.tools.specs import ToolContext

__all__ = [
    "SERVER_NAME",
    "ServerDeps",
    "build_server",
    "entrypoint",
    "main",
    "open_server_deps",
    "read_catalog",
]

logger = logging.getLogger(__name__)

SERVER_NAME = "kovalo"
Transport = Literal["stdio", "streamable-http"]
MEMORY_INDEX_DETAIL = "indeks w pamięci przy pierwszym wyszukiwaniu"


@dataclass(frozen=True)
class ServerDeps:
    """Wszystko, czego serwer potrzebuje (wczytane przy starcie, poza leniwym indeksem)."""

    ctx: ToolContext  # dane firmy, polityka, warunki, retriever (leniwy), top_k
    corpus_dir: Path  # settings.paths.corpus_dir
    catalog: dict[str, str]  # doc_id → nazwa pliku (paths.corpus_catalog)
    policy_yaml: str  # yaml.safe_dump sekcji `policy` (równość z config.yaml)
    variant: ToolsVariant  # settings.tools.variant (aplikacja: ours, 7 narzędzi)
    search: SearchBackend | None = None  # backend wyszukiwania (/health); None → mcp_server.search


def read_catalog(path: Path) -> dict[str, str]:
    """Katalog korpusu jako {doc_id: plik}; brak pliku albo kolumn → CorpusError."""
    if not path.is_file():
        raise CorpusError(f"Nie znaleziono katalogu korpusu: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    catalog: dict[str, str] = {}
    for row in rows:
        doc_id, file = (row.get("doc_id") or "").strip(), (row.get("file") or "").strip()
        if not doc_id or not file:
            raise CorpusError(f"{path.name}: wiersz bez doc_id albo nazwy pliku.")
        catalog[doc_id] = file
    return catalog


def _policy_yaml(settings: Settings) -> str:
    """Sekcja `policy` z config.yaml tak, jak jest w pliku (yaml.safe_load(...) == sekcja)."""
    raw = yaml.safe_load((settings.root / "config.yaml").read_text(encoding="utf-8"))
    return yaml.safe_dump(raw["policy"], allow_unicode=True, sort_keys=False)


def open_server_deps(settings: Settings, *, search: SearchBackend | None = None) -> ServerDeps:
    """load_company_data (CompanyDataError leci wyżej), katalog korpusu i retriever leniwy:
    pgvector → open_retriever(settings, get_embeddings(...)); memory → build_memory_retriever.
    search None → mcp_server.search. Żadnego modelu czatu i klucza LLM-a (NFR-7)."""
    backend: SearchBackend = search or settings.mcp_server.search

    def opener() -> Retriever:
        embeddings = get_embeddings(settings=settings)
        if backend == "memory":
            return build_memory_retriever(settings, embeddings)
        return open_retriever(settings, embeddings)

    ctx = ToolContext(
        data=load_company_data(settings.paths.company_data_dir),
        policy=settings.policy,
        company=settings.company,
        retriever=LazyRetriever(opener),
        top_k=settings.retrieval.top_k,
    )
    return ServerDeps(
        ctx=ctx,
        corpus_dir=settings.paths.corpus_dir,
        catalog=read_catalog(settings.paths.corpus_catalog),
        policy_yaml=_policy_yaml(settings),
        variant=settings.tools.variant,
        search=backend,
    )


def _check_pgvector_index(settings: Settings) -> CheckResult:
    """Wiersz metadanych indeksu z modelem z configu (bez wyszukiwania)."""
    from kovalo.knowledge.pgvector import check_index, configured_embedding_model, index_table

    table = index_table(settings, "ctx")
    model, dimensions = configured_embedding_model(settings)
    try:
        meta = check_index(
            settings,
            table,
            provider=settings.embeddings.provider,
            model=model,
            dimensions=dimensions,
        )
    except Exception as exc:  # brak indeksu, niezgodny model albo Postgres niedostępny
        return CheckResult(status="error", detail=f"{type(exc).__name__}: {exc}"[:200])
    return CheckResult(status="ok", detail=f"{table}: {meta.n_chunks} chunków")


def transport_security(settings: Settings) -> TransportSecuritySettings:
    """Ochrona przed DNS rebinding zawsze włączona, także przy --host 0.0.0.0 w kontenerze
    (SDK włącza ją sam tylko dla hosta loopback). Dozwolone: loopback i host z
    `services.mcp_url` (w compose `mcp-server`), dowolny port."""
    hosts = ["127.0.0.1", "localhost", "[::1]"]
    own = urlsplit(settings.services.mcp_url).hostname
    if own and own not in hosts and f"[{own}]" not in hosts:
        hosts.append(own)
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[f"{h}:*" for h in hosts],
        allowed_origins=[f"http://{h}:*" for h in hosts],
    )


def build_server(
    deps: ServerDeps | Settings,
    *,
    settings: Settings | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
) -> FastMCP:
    """FastMCP(SERVER_NAME, stateless HTTP) z narzędziami wariantu, zasobami, promptem i /health.
    Ochrona przed DNS rebinding zawsze włączona (`transport_security`), także przy 0.0.0.0.

    Zgodność z F0: `build_server(settings)` (bez `deps`) = `build_server(open_server_deps(
    settings), settings=settings)`."""
    if isinstance(deps, Settings):
        settings = settings or deps
        deps = open_server_deps(settings)
    if settings is None:
        raise TypeError("build_server(deps, *, settings=...): brak settings")
    server = FastMCP(
        SERVER_NAME,
        host=host,
        port=port if port is not None else settings.services.mcp_port,
        stateless_http=True,
        log_level="WARNING",
        transport_security=transport_security(settings),
    )
    register_tools(server, deps.ctx, deps.variant)
    register_resources(
        server,
        corpus_dir=deps.corpus_dir,
        catalog=deps.catalog,
        policy_version=settings.policy.version,
        policy_yaml=deps.policy_yaml,
    )
    register_prompts(server)
    backend = deps.search or settings.mcp_server.search

    async def company_data() -> CheckResult:
        # dane wczytane przy starcie (inaczej serwer by nie wystartował)
        return CheckResult(status="ok", critical=True)

    async def index() -> CheckResult:
        if backend == "memory":
            return CheckResult(status="ok", detail=MEMORY_INDEX_DETAIL)
        return await anyio.to_thread.run_sync(_check_pgvector_index, settings)

    @server.custom_route("/health", methods=["GET"], include_in_schema=False)
    async def health(request: Request) -> Response:
        report = await build_report(
            "mcp-server",
            {"company_data": company_data, "index": index},
            timeout_s=settings.services.health_timeout_s,
            critical=("company_data",),
        )
        return JSONResponse(report.model_dump(mode="json"), status_code=http_status(report))

    return server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kovalo-mcp", description="Serwer MCP Kovalo.")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument(
        "--search",
        choices=["pgvector", "memory"],
        default=None,
        help="wyszukiwanie w search_documents (domyślnie mcp_server.search z configu)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="adres nasłuchu (streamable-http)")
    parser.add_argument("--port", type=int, default=None, help="port (domyślnie services.mcp_port)")
    return parser


def _configure_logging() -> None:
    """Logi tylko na stderr; nasze wywołania narzędzi na poziomie INFO, biblioteki od WARNING."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("kovalo").setLevel(logging.INFO)


def main(argv: Sequence[str] | None = None) -> int:
    """`kovalo-mcp` / `python -m kovalo.mcp_server`. ConfigError (w tym CompanyDataError)
    i CorpusError → jedna linia na stderr, kod 2."""
    args = _parser().parse_args(argv)
    _configure_logging()
    try:
        settings = get_settings()
        deps = open_server_deps(settings, search=args.search)
    except (ConfigError, CorpusError) as exc:
        first = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        print(f"kovalo-mcp: {first}", file=sys.stderr)
        return 2
    transport: Transport = args.transport
    logger.info(
        "serwer MCP %s: transport %s, wyszukiwanie %s, narzędzia %s",
        SERVER_NAME,
        transport,
        deps.search,
        deps.variant,
    )
    build_server(deps, settings=settings, host=args.host, port=args.port).run(transport=transport)
    return 0


def entrypoint() -> None:
    """Punkt wejścia `kovalo-mcp`."""
    sys.exit(main())
