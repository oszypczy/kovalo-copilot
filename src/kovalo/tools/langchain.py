"""Narzędzia agenta jako StructuredTool LangChaina (FR-4.3; krok 3 MP4).

Kontrakt: docs/contracts/f4.md, sekcja 6.4. Nazwa, opis (docstring) i schemat argumentów
pochodzą z funkcji `spec.make(ctx)`; narzędzie zwraca rekord jako słownik JSON.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model

from kovalo.tools.specs import ToolContext, ToolSpec, ToolsVariant, tool_specs

__all__ = ["langchain_tools"]


def _args_schema(name: str, function: Callable[..., Any]) -> type[BaseModel]:
    """Model argumentów z sygnatury funkcji (te same typy, opisy i wartości domyślne)."""
    fields: dict[str, Any] = {}
    hints = inspect.get_annotations(function, eval_str=True)
    for param in inspect.signature(function).parameters.values():
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (hints[param.name], default)
    return create_model(name, **fields)


def _tool(spec: ToolSpec, ctx: ToolContext) -> BaseTool:
    function = spec.make(ctx)

    def run(**kwargs: Any) -> dict[str, Any]:
        return function(**kwargs).model_dump(mode="json")

    return StructuredTool.from_function(
        func=run,
        name=spec.name,
        description=(function.__doc__ or "").strip(),
        args_schema=_args_schema(spec.name, function),
    )


def langchain_tools(ctx: ToolContext, variant: ToolsVariant = "ours") -> list[BaseTool]:
    """StructuredTool dla każdej specyfikacji wariantu: 6 (course) albo 7 (ours) narzędzi."""
    return [_tool(spec, ctx) for spec in tool_specs(variant)]
