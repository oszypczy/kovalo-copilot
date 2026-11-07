# ruff: noqa: RUF001, RUF002  (polska typografia: półpauza w zakresach)
"""Treść `reports/dashboard.md` (kontrakt F6, 8.4): tabele z `Dashboard`, bez daty generowania.

Liczby po polsku przez `kovalo.text_pl.number`; brak wartości to „–”. Linki są względne wobec
katalogu raportów (tam leży `dashboard.md`).
"""

from __future__ import annotations

from collections.abc import Sequence

from kovalo.eval.dashboard.models import Dashboard, StageCost, StageSummary
from kovalo.eval.dashboard.stages import calls_text, cents_text, met_overall
from kovalo.text_pl import number

__all__ = ["render_markdown"]

DASH = "–"
# PINNED w config.yaml (cost.scenario); MonthlyCost przenosi tylko kwoty, opis jest stały.
SCENARIO = "400 pytań dziennie, 21 dni roboczych, 350 tokenów odpowiedzi"


def _cell(value: object) -> str:
    if value is None or value == "":
        return DASH
    return str(value).replace("|", "\\|").replace("\n", " ")


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    return [_row(header), "|" + "---|" * len(header), *(_row(r) for r in rows)]


def _met(value: bool | None) -> str:
    return DASH if value is None else ("tak" if value else "nie")


def _num(value: float | None) -> str:
    return DASH if value is None else number(value)


def _cents(value: float | None) -> str:
    return DASH if value is None else number(round(value, 2))


def _usd(value: float | None) -> str:
    return DASH if value is None else "$" + f"{value:.2f}".replace(".", ",")


def _link(path: str) -> str:
    return f"[{path}]({path})"


def _calls_line(summary: StageSummary) -> str | None:
    calls = summary.calls
    if calls is None:
        return None
    parts = []
    if calls.planned is not None:
        parts.append(f"plan {number(calls.planned)}")
    if calls.network is not None:
        parts.append(f"sieciowo {number(calls.network)}")
    if calls.cached is not None:
        parts.append(f"z cache {number(calls.cached)}")
    if calls.failed is not None:
        parts.append(f"nieudane {number(calls.failed)}")
    if calls.seconds is not None:
        parts.append(f"{number(calls.seconds)} s")
    source = "raport" if calls.source == "report" else "dashboard.yaml"
    origin = f", {calls.origin}" if calls.origin else ""
    run = f"przebieg {calls.run_id}; " if calls.run_id else ""
    return f"- Wywołania LLM-a: {', '.join(parts)} ({run}źródło: {source}{origin})"


def _cost_line(cost: StageCost | None) -> str | None:
    if cost is None:
        return None
    parts = []
    if cost.tokens_in is not None and cost.tokens_out is not None:
        parts.append(f"tokeny {number(cost.tokens_in)} / {number(cost.tokens_out)}")
    if cost.ours_cents is not None:
        parts.append(f"{cents_text(cost.ours_cents)} po cenach płatnego tieru Gemini")
    if cost.reference_cents is not None:
        parts.append(f"{cents_text(cost.reference_cents)} po cenach kursu")
    if cost.course_cents is not None:
        parts.append(f"kurs: {cents_text(cost.course_cents)}")
    line = "- Tokeny i koszt: " + "; ".join(parts)
    if cost.includes_cached:
        line += " (liczą też tury z cache)"
    return line


def _stage_section(summary: StageSummary) -> list[str]:
    lines = [f"## {summary.stage.upper()}: {summary.title}", ""]
    models = [m for m in (summary.model, summary.embedding_model) if m]
    meta = f"Data: {summary.date.isoformat()}"
    if summary.model:
        meta += f" · Model: {summary.model}"
    if summary.embedding_model:
        meta += f" · Embeddingi: {summary.embedding_model}"
    lines += [meta if models else f"Data: {summary.date.isoformat()}", ""]
    lines += [f"{summary.headline}.", ""]
    lines += _table(
        ("Metryka", "Kurs", "Bez LLM-a / odtworzenie", "Na żywo", "Cel", "Spełniony"),
        [(m.name, m.reference, m.replay, m.live, m.target, _met(m.met)) for m in summary.metrics],
    )
    lines.append("")
    for line in (_calls_line(summary), _cost_line(summary.cost)):
        if line:
            lines.append(line)
    report = f"- Raport: {_link(summary.report_md)}"
    if summary.related_md:
        report += "; powiązane: " + ", ".join(_link(p) for p in summary.related_md)
    lines.append(report)
    for note in summary.notes:
        lines.append(f"- Uwaga: {note}")
    lines.append("")
    return lines


def render_markdown(dashboard: Dashboard) -> str:
    lines = [
        "# Dashboard ewaluacji",
        "",
        "Plik generowany poleceniem `uv run kovalo eval dashboard` (sprawdzenie: "
        "`uv run kovalo eval dashboard --check`) z `reports/dashboard.yaml`, JSON-ów raportów "
        "E1–E5 i poprzedniego `reports/dashboard.json`. Nie poprawiaj go ręcznie: zmiany wpisuj "
        "w `dashboard.yaml` albo w raportach etapów. Metryki i cele: "
        "[docs/06-evaluation.md](../docs/06-evaluation.md) (06-evaluation).",
        "",
        f"Stan na: {dashboard.as_of.isoformat()} (data najnowszego raportu)",
        "",
        "## Podsumowanie",
        "",
    ]
    lines += _table(
        ("Etap", "Data", "Model", "Najważniejszy wynik (kurs → nasz)", "Cel spełniony", "Raport"),
        [
            (
                s.stage.upper(),
                s.date.isoformat(),
                s.model or s.embedding_model,
                s.headline,
                met_overall(s.metrics),
                _link(s.report_md),
            )
            for s in dashboard.stages
        ],
    )
    lines.append("")
    for summary in dashboard.stages:
        lines += _stage_section(summary)

    lines += ["## Koszty", ""]
    cost_rows = []
    for s in dashboard.stages:
        calls, cost = s.calls, s.cost
        tokens = (
            f"{number(cost.tokens_in)} / {number(cost.tokens_out)}"
            if cost is not None and cost.tokens_in is not None and cost.tokens_out is not None
            else None
        )
        cost_rows.append(
            (
                s.stage.upper(),
                _num(calls.network) if calls else None,
                _num(calls.cached) if calls else None,
                tokens,
                _cents(cost.ours_cents) if cost else None,
                _cents(cost.reference_cents) if cost else None,
                _cents(cost.course_cents) if cost else None,
            )
        )
    lines += _table(
        (
            "Etap",
            "Wywołania sieciowe",
            "Z cache",
            "Tokeny wej. / wyj.",
            "Nasz koszt (centy, cennik płatny Gemini)",
            "Po cenach kursu (centy)",
            "Koszt kursu (README, centy)",
        ),
        cost_rows,
    )
    lines.append("")
    if any(s.cost is not None and s.cost.includes_cached for s in dashboard.stages):
        cached = ", ".join(
            s.stage.upper() for s in dashboard.stages if s.cost and s.cost.includes_cached
        )
        lines += [f"Tokeny i koszt {cached} liczą też tury z cache.", ""]
    monthly = dashboard.monthly_cost
    if monthly is not None:
        text = (
            f"Scenariusz miesięczny kursu (E1: {SCENARIO}), retrieval / cały korpus: po cenach "
            f"referencyjnych kursu {_usd(monthly.reference_rag_usd)} / "
            f"{_usd(monthly.reference_full_usd)}"
        )
        if monthly.ours_rag_usd is not None and monthly.ours_full_usd is not None:
            text += (
                f", po cenach płatnego tieru {monthly.ours_model or 'Gemini'} "
                f"{_usd(monthly.ours_rag_usd)} / {_usd(monthly.ours_full_usd)}"
            )
        lines += [text + ".", ""]
    lines += [
        "Faktyczny koszt projektu w darmowym tierze: 0 USD (kwoty wyżej to przeliczenie tych "
        "samych tokenów po cennikach płatnych).",
        "",
        "## Zużycie darmowego limitu Gemini",
        "",
    ]
    lines += _table(
        ("Etap", "Wywołania", "Uwagi"),
        [
            (g.stage, _num(g.calls) if g.calls is not None else "brak pomiaru", g.note)
            for g in dashboard.gemini_usage
        ],
    )
    known = [n for n in (g.calls for g in dashboard.gemini_usage) if n is not None]
    if known:
        lines += ["", f"Razem w etapach z pomiarem: {calls_text(sum(known))}."]
    lines += ["", "## Znane ograniczenia", ""]
    notes = [(s.stage.upper(), n) for s in dashboard.stages for n in s.notes]
    lines += [f"- {stage}: {note}" for stage, note in notes] or ["- Brak."]
    return "\n".join(lines).rstrip("\n") + "\n"
