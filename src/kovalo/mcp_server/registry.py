"""Rejestracja narzędzi, zasobów i promptu serwera MCP Kovalo (FR-5.1, FR-5.5; krok 1 MP5).

Kontrakt: docs/contracts/f5.md, sekcje 7.2-7.3 (P3, P4); ADR-0013 pkt 1. Narzędzia to te same
funkcje co u agenta (`tools.specs.TOOL_SPECS`, `spec.make(ctx)`), opakowane w asynchroniczną
nakładkę, która woła je w wątku: FastMCP 1.30 wywołuje funkcję synchroniczną wprost w pętli
zdarzeń, więc wyszukiwanie zablokowałoby serwer HTTP. Nic tu nie zatwierdza ani nie zapisuje.
"""

from __future__ import annotations

import inspect
import logging
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import anyio.to_thread
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from kovalo.tools.specs import ToolContext, ToolSpec, ToolsVariant, tool_specs

__all__ = [
    "DOCUMENT_ID",
    "DOCUMENT_URI",
    "POLICY_URI",
    "SUPPLIER_REVIEW",
    "register_prompts",
    "register_resources",
    "register_tools",
]

logger = logging.getLogger(__name__)

# Identyfikator dokumentu z katalogu korpusu (np. CON-103); dopasowanie zawsze `fullmatch`.
DOCUMENT_ID = re.compile(r"[A-Z]{2,4}-\d{3}")
DOCUMENT_URI = "kovalo://documents/{doc_id}"
POLICY_URI = "kovalo://policy/{version}"
SUPPLIER_REVIEW = (
    "Podaj warunki umowy z dostawcą {supplier}, jego ostatni miesięczny OTIF i ewentualną należną "
    "karę umowną. Przy każdej liczbie podaj źródło (narzędzie albo dokument)."
)
_MAX_LOGGED_CHARS = 200
_READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)


def _short(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_LOGGED_CHARS:
        return value[:_MAX_LOGGED_CHARS] + "…"
    return value


def _request_id(server: FastMCP) -> Any:
    try:
        return server.get_context().request_id
    except (LookupError, ValueError):  # poza żądaniem (nie powinno się zdarzyć)
        return None


def _wrapper(server: FastMCP, spec: ToolSpec, function: Callable[..., BaseModel]) -> Any:
    """Asynchroniczna nakładka z nazwą, docstringiem i sygnaturą `function` (wynik: spec.record)."""

    async def tool(**kwargs: Any) -> BaseModel:
        request_id = _request_id(server)
        started = time.perf_counter()
        try:
            record = await anyio.to_thread.run_sync(lambda: function(**kwargs))
        except Exception as exc:
            logger.warning(
                "narzędzie %s: błąd %s (request_id=%s, %.3f s, argumenty %s)",
                spec.name,
                type(exc).__name__,
                request_id,
                time.perf_counter() - started,
                {k: _short(v) for k, v in kwargs.items()},
            )
            raise
        logger.info(
            "narzędzie %s: status %s (request_id=%s, %.3f s, argumenty %s)",
            spec.name,
            getattr(record, "status", None),
            request_id,
            time.perf_counter() - started,
            {k: _short(v) for k, v in kwargs.items()},
        )
        return record

    signature = inspect.signature(function, eval_str=True)
    tool.__name__ = tool.__qualname__ = spec.name
    tool.__doc__ = function.__doc__
    tool.__signature__ = signature.replace(return_annotation=spec.record)  # type: ignore[attr-defined]
    tool.__annotations__ = {
        **inspect.get_annotations(function, eval_str=True),
        "return": spec.record,
    }
    return tool


def register_tools(server: FastMCP, ctx: ToolContext, variant: ToolsVariant) -> list[str]:
    """Każda specyfikacja wariantu (course: 6, ours: 7) jako narzędzie tylko do odczytu
    z `outputSchema` rekordu. Opis = `(fn.__doc__ or "").strip()`, jak w tools/langchain.py
    (P3: równe opisy to warunek trafień w cache F4). Zwraca zarejestrowane nazwy."""
    names: list[str] = []
    for spec in tool_specs(variant):
        function = spec.make(ctx)
        server.tool(
            name=spec.name,
            description=(function.__doc__ or "").strip(),
            annotations=_READ_ONLY,
            structured_output=True,
        )(_wrapper(server, spec, function))
        names.append(spec.name)
    return names


def register_resources(
    server: FastMCP,
    *,
    corpus_dir: Path,
    catalog: Mapping[str, str],
    policy_version: str,
    policy_yaml: str,
) -> None:
    """Szablon `kovalo://documents/{doc_id}` (dokładny identyfikator z katalogu, nic innego)
    i zasób statyczny `kovalo://policy/<wersja>` (YAML sekcji `policy` configu)."""

    @server.resource(
        DOCUMENT_URI,
        name="document",
        description="Pełny tekst dokumentu z korpusu Kovalo po identyfikatorze z katalogu "
        "(np. CON-103).",
        mime_type="text/markdown",
    )
    def document(doc_id: str) -> str:
        if not DOCUMENT_ID.fullmatch(doc_id) or doc_id not in catalog:
            raise ValueError(f"nieznany dokument: {doc_id!r} (podaj identyfikator z katalogu)")
        return (corpus_dir / catalog[doc_id]).read_text(encoding="utf-8")

    @server.resource(
        POLICY_URI.format(version=policy_version),
        name="policy",
        description=f"Polityka rabatowa Kovalo w wersji {policy_version} (YAML, tylko do odczytu).",
        mime_type="application/yaml",
    )
    def policy() -> str:
        return policy_yaml


def register_prompts(server: FastMCP) -> None:
    """Prompt `supplier_review(supplier)`: jedna wiadomość użytkownika."""

    @server.prompt(
        name="supplier_review",
        description="Przegląd dostawcy: warunki umowy, ostatni miesięczny OTIF i ewentualna "
        "kara umowna, ze źródłem każdej liczby.",
    )
    def supplier_review(supplier: str) -> str:
        return SUPPLIER_REVIEW.format(supplier=supplier)
