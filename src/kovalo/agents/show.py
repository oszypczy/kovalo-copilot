"""Ścieżka dowodowa jednego przebiegu agenta (FR-4.5; port show_trace.py MP4).

Kontrakt: docs/contracts/f4.md, sekcja 7.4. Każde wywołanie narzędzia po kolei (zagnieżdżone
z wcięciem pod rodzicem, także zdarzenia składowe), pochodzenie w <…>, argumenty i kluczowe
pola wyniku; potem odpowiedź i (opcjonalnie) ocena.
"""

from __future__ import annotations

import json
from typing import Any

from kovalo.agents.trace import Trace

__all__ = ["ARCHITECTURES", "SKIP", "architecture_label", "brief", "format_trace"]

SKIP = {
    "status",
    "currency",
    "note",
    "requested",
    "query",
    "settlement",
    "terms_version",
    "message",
    "supplier",
}

ARCHITECTURES = {
    "single agent": "jeden agent",
    "supervisor + 3 specialists": "agent koordynujący i 3 specjalistów",
    "single agent over MCP": "jeden agent przez MCP",
}


def architecture_label(stored: str) -> str:
    """Polska nazwa architektury; w zapisanym przebiegu zostaje wartość kodu."""
    return ARCHITECTURES.get(stored, stored)


def brief(result: Any) -> str:
    """Kluczowe pola wyniku narzędzia, w jednym wierszu."""
    if not isinstance(result, dict):
        return str(result)[:160]
    if "passages" in result:
        return f"{result['status']}: " + ", ".join(p["chunk_id"] for p in result["passages"])
    if "summary" in result:
        return "podsumowanie: " + str(result["summary"]).replace("\n", " ")[:160]
    fields = {k: v for k, v in result.items() if k not in SKIP and v not in (None, [], "")}
    if "observations" in fields:
        fields["observations"] = ", ".join(
            f"{o['month']}: {o['otif_pct']}" for o in fields["observations"]
        )
    text = f"{result.get('status')}: " + json.dumps(fields, ensure_ascii=False)
    return text if not result.get("message") else f"{text} ({result['message']})"


def format_trace(trace: Trace, score: dict[str, Any] | None = None) -> str:
    """Ścieżka dowodowa w formacie show_trace.py kursu."""
    lines = [
        f"{trace.question_id or ''}  {trace.question}",
        f"status: {trace.status}"
        + (f" ({trace.error})" if trace.error else "")
        + f"; kroki: {trace.steps}, {trace.seconds:.0f} s",
    ]
    depth: dict[str, int] = {}
    for n, event in enumerate(trace.events, start=1):
        depth[event.call_id] = (
            depth.get(event.parent_call_id, -1) + 1 if event.parent_call_id else 0
        )
        indent = "    " * depth[event.call_id]
        args = ", ".join(f"{k}={v!r}" for k, v in event.arguments.items())
        outcome = (
            brief(event.result)
            if event.status == "ok"
            else f"{event.status.upper()}: {(event.error or '')[:160]}"
        )
        who = "" if event.agent == "agent" else f"[{event.agent}] "
        lines.append(f"  {indent}{n:2}. {who}{event.tool}({args})  <{event.origin}>")
        lines.append(f"  {indent}      -> {outcome}")
    lines.append("odpowiedź: " + (trace.answer or "(brak)"))
    if score:
        lines.append("ocena: " + json.dumps(score, ensure_ascii=False))
    return "\n".join(lines)
