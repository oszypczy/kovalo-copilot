"""Zgodność MCP z wywołaniem bezpośrednim (E5; krok 2 MP5).

Kontrakt: docs/contracts/f5.md, sekcja 10.2; ADR-0014 pkt 1a. To samo wywołanie wykonane
bezpośrednio (`spec.make(ctx)`, te same funkcje co serwer) i przez serwer MCP musi dać ten sam
rekord: te same pola, wartości i typy JSON. Zgodność sprawdza nakładkę i transport, nie
odpowiedź biznesową (tę ocenia wzorzec E3/E4).
"""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel

from kovalo.mcp_client import McpCallError, McpConnection
from kovalo.tools.specs import TOOL_SPECS, ToolContext

__all__ = [
    "PARITY_MATRIX",
    "SKIPPED_SEARCH",
    "DiffKind",
    "ParityRow",
    "compare_call",
    "direct_record",
    "run_parity",
]

DiffKind = Literal["missing", "extra", "unit", "type", "changed"]

# 17 wywołań kursu (test_step2_compare_call.py) 1:1 + 3 NASZE (check_termination_condition).
PARITY_MATRIX: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "get_supplier_otif",
        {"supplier": "Mleczna Dolina", "from_month": "2026-08", "to_month": "2026-10"},
    ),
    ("get_supplier_otif", {"supplier": "SUP-114", "from_month": "2026-05", "to_month": "2026-07"}),
    ("get_supplier_otif", {"supplier": "SUP-114", "from_month": "2026-10", "to_month": "2026-10"}),
    (
        "get_supplier_otif",
        {"supplier": "Nordic Fish", "from_month": "2026-07", "to_month": "2026-07"},
    ),
    ("get_supplier_otif", {"supplier": "SUP-114", "from_month": "2026-7", "to_month": "2026-07"}),
    ("get_invoice_value", {"supplier": "SUP-114", "month": "2026-07"}),
    ("get_invoice_value", {"supplier": "SUP-207", "month": "2026-08"}),
    ("get_contract_terms", {"supplier": "Mleczna Dolina"}),
    ("get_contract_terms", {"supplier": "SUP-207"}),
    ("get_contract_terms", {"supplier": "TransBałt"}),
    (
        "calculate_otif_penalty",
        {
            "otif_pct": 94.6,
            "target_pct": 97,
            "rate_per_full_point_pct": 2,
            "cap_pct": 10,
            "invoice_grosze": 18240000,
        },
    ),
    (
        "calculate_otif_penalty",
        {
            "otif_pct": 101,
            "target_pct": 97,
            "rate_per_full_point_pct": 2,
            "cap_pct": 10,
            "invoice_grosze": 1,
        },
    ),
    (
        "check_discount_policy",
        {"discount_pct": 22, "margin_after_pct": 7, "strategic_account": True},
    ),
    ("check_discount_policy", {"discount_pct": 30, "margin_after_pct": 4, "clearance": True}),
    ("check_discount_policy", {"discount_pct": 14, "competitor_offer": True}),
    ("check_discount_policy", {"discount_pct": 4, "contractual_volume_rebate": True}),
    ("search_documents", {"question": "TransBałt terminowe dostawy palet"}),
    # NASZE: warunek rozwiązania umowy (condition_met False, brak października, not_applicable)
    ("check_termination_condition", {"supplier": "Mleczna Dolina", "as_of_month": "2026-09"}),
    ("check_termination_condition", {"supplier": "SUP-114", "as_of_month": "2026-10"}),
    ("check_termination_condition", {"supplier": "SUP-207", "as_of_month": "2026-07"}),
)


def _is_int(value: Any) -> bool:
    """Liczba całkowita JSON (bool nie jest tu int, choć w Pythonie jest jego podklasą)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _walk(direct: Any, mcp: Any, path: str, out: list[dict[str, Any]]) -> None:
    def add(kind: DiffKind, d: Any, m: Any) -> None:
        out.append({"path": path, "kind": kind, "direct": d, "mcp": m})

    if isinstance(direct, dict) and isinstance(mcp, dict):
        for key, value in direct.items():
            if key in mcp:
                _walk(value, mcp[key], _join(path, key), out)
            else:
                out.append(
                    {"path": _join(path, key), "kind": "missing", "direct": value, "mcp": None}
                )
        for key, value in mcp.items():
            if key not in direct:
                out.append(
                    {"path": _join(path, key), "kind": "extra", "direct": None, "mcp": value}
                )
        return
    if isinstance(direct, list) and isinstance(mcp, list):
        for i, value in enumerate(direct):
            item = f"{path}[{i}]"
            if i < len(mcp):
                _walk(value, mcp[i], item, out)
            else:
                out.append({"path": item, "kind": "missing", "direct": value, "mcp": None})
        for i in range(len(direct), len(mcp)):
            out.append({"path": f"{path}[{i}]", "kind": "extra", "direct": None, "mcp": mcp[i]})
        return
    if type(direct) is type(mcp) and direct == mcp:
        return
    field = path.rsplit(".", 1)[-1].split("[", 1)[0]
    if field.endswith("_grosze") and (not _is_int(mcp) or mcp * 100 == direct):
        add("unit", direct, mcp)  # nakładka zamieniła grosze na złotówki (albo na float)
    elif type(direct) is not type(mcp):
        add("type", direct, mcp)  # 97 vs 97.0, "5" vs 5, true vs 1, null vs "None"
    else:
        add("changed", direct, mcp)


def compare_call(direct: Any, mcp: Any) -> dict[str, Any]:
    """{'same': bool, 'differences': [{'path', 'kind', 'direct', 'mcp'}]} (krok 2 kursu).

    Pierwszeństwo: missing/extra > unit (pole *_grosze: mcp nie-int albo mcp*100 == direct) >
    type > changed; kolejność: pola direct, potem nadmiarowe z mcp; wejść nie modyfikuje."""
    differences: list[dict[str, Any]] = []
    _walk(copy.deepcopy(direct), copy.deepcopy(mcp), "", differences)
    return {"same": not differences, "differences": differences}


class ParityRow(BaseModel):
    """Jeden wiersz zgodności: wywołanie, wynik porównania i ewentualny błąd wywołania MCP."""

    tool: str
    arguments: dict[str, Any]
    same: bool
    differences: list[dict[str, Any]]
    error: str | None = None
    skipped: bool = False  # nieporównane (brak lokalnego retrievera dla search_documents)


SKIPPED_SEARCH = "pominięte: brak lokalnego retrievera (wywołanie bezpośrednie niemożliwe)"


def direct_record(ctx: ToolContext, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """spec.make(ctx)(**args).model_dump(mode='json') (te same funkcje co serwer)."""
    spec = next((s for s in TOOL_SPECS if s.name == tool), None)
    if spec is None:
        raise KeyError(f"nieznane narzędzie: {tool}")
    return spec.make(ctx)(**args).model_dump(mode="json")


def run_parity(
    ctx: ToolContext,
    connection: McpConnection,
    matrix: tuple[tuple[str, dict[str, Any]], ...] = PARITY_MATRIX,
) -> list[ParityRow]:
    """Każdy wiersz: direct_record vs surowy structuredContent z `captures` (także gdy
    check_record odrzucił wynik: wtedy `error` = komunikat walidacji). ctx.retriever None (np.
    zgodność przez HTTP z kontenerem na pgvector) → wiersze search_documents jawnie pominięte
    (`skipped`, same=False), bez wywołania bezpośredniego i MCP."""
    rows: list[ParityRow] = []
    for tool, args in matrix:
        if tool == "search_documents" and ctx.retriever is None:
            rows.append(
                ParityRow(
                    tool=tool,
                    arguments=dict(args),
                    same=False,
                    differences=[],
                    error=SKIPPED_SEARCH,
                    skipped=True,
                )
            )
            continue
        error: str | None = None
        try:
            connection.call(tool, dict(args))
        except McpCallError as exc:
            error = str(exc)
        structured = connection.captures[-1]["structured"]
        result = compare_call(direct_record(ctx, tool, args), structured)
        rows.append(
            ParityRow(
                tool=tool,
                arguments=dict(args),
                same=result["same"] and error is None,
                differences=result["differences"],
                error=error,
            )
        )
    return rows
