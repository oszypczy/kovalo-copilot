"""Raporty E5: Markdown (dowód etapu), pełny JSON i assistant_run.<tryb>.json (kontrakt F5, 10.5).

FR-6.4. Liczby po polsku (25/25, 4,4 centa), identyfikatory bez zmian. Nazwy modeli pochodzą
z raportu i configu (`reference_llm.model`), nigdy ze stałych. Komentarze przeglądu ręcznego
tylko z `review.yaml` (etykiety ours-replay / ours-live, id tur C1-t1 … D8-t1); wygenerowanego
Markdownu nie edytujemy. Pliki assistant_run.<tryb>.json mają kształt `run_assistant_eval.py`
kursu (`run`, `turns`, `mcp_captures`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kovalo.agents.show import format_trace
from kovalo.agents.trace import trace_from_record
from kovalo.eval.assistant.e5 import MAX_CALLS_SKIPPED, E5Report, E5Run
from kovalo.eval.assistant.reference import COURSE_E5_RUN
from kovalo.eval.assistant.scoring import AGENT_DIMENSIONS, DIMENSIONS
from kovalo.eval.review import REVIEW_FILE, Review

__all__ = ["REVIEW_LABELS", "render_markdown", "turn_id", "write_reports"]

NONE = "\u2013"
COMMENT = "  - Komentarz: (do uzupełnienia w T2)"
REVIEW_LABELS = ("ours-replay", "ours-live")
MESSAGE_CHARS = 44

DIMENSION_NAMES: dict[str, str] = {  # nazwy z run_assistant_eval.py kursu
    "route": "ścieżka (oczekiwany kind)",
    "status": "status końcowy",
    "fields": "pola wyodrębnione przez LLM",
    "pauses": "pauzy (rodzaj, rola, kolejność)",
    "approvers": "osoby zatwierdzające (polityka)",
    "request": "stan wniosku",
    "agent_trace": "zapis przebiegu agenta tylko tam, gdzie oczekiwany",
    "tool_coverage": "pytania o dane: każde potrzebne wywołanie",
    "arguments_correct": "pytania o dane: argumenty obliczenia",
    "answer_facts": "pytania o dane: każdy kluczowy fakt",
    "numbers_backed": "pytania o dane: żadnej liczby bez pokrycia",
    "no_approval_claim": "pytania o dane: brak twierdzenia o akceptacji",
}
LAYER_NAMES = {
    "tool": "błąd narzędzia",
    "protocol": "błąd protokołu",
    "transport": "błąd transportu",
    "validation": "błąd walidacji lokalnej",
}


def turn_id(dialogue: str, n: int) -> str:
    return f"{dialogue}-t{n}"


def _num(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _int(value: int) -> str:
    return f"{value:,}".replace(",", "\u00a0")


def _frac(pair: Sequence[int] | None) -> str:
    if pair is None or not pair[1]:
        return NONE
    return f"{pair[0]}/{pair[1]}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [str(c).replace("|", "\\|").replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _target(dim: str, course: Sequence[int]) -> str:
    passed, assessed = course
    if dim == "route":
        return f"≥ {assessed - 1}/{assessed}"
    if dim == "arguments_correct":
        return f"{passed}/{assessed}"
    return f"≥ {max(passed - 1, 0)}/{assessed}"


def _shown_run(report: E5Report) -> E5Run:
    return report.live if report.live is not None else report.replay


def _all_captures(run: E5Run) -> list[dict[str, Any]]:
    return [c for t in run.turns for c in t["mcp"]]


def _failed(run: E5Run) -> int:
    return sum(run.calls.mcp_failures.values())


# ---------------------------------------------------------------------------
# Sekcje
# ---------------------------------------------------------------------------


def _header(report: E5Report, reference_model: str, settings: Any) -> list[str]:
    replay, live, plan = report.replay, report.live, report.llm_calls_planned
    created = report.created_at.strftime("%Y-%m-%d %H:%M")
    budget = replay.run.get("budget") or {}
    variant = replay.run.get("tools_variant")
    count = 7 if variant == "ours" else 6
    mcp = settings.mcp if settings is not None else None
    limits = (
        f"{mcp.startup_timeout_s:g} / {mcp.call_timeout_s:g} / {mcp.cleanup_timeout_s:g} s"
        if mcp is not None
        else "60 / 120 / 10 s"
    )
    server = (
        "podproces `kovalo-mcp`"
        if report.transport == "stdio"
        else "kontener `mcp-server` (streamable HTTP)"
        if report.transport == "http"
        else "serwer w pamięci procesu"
    )
    lines = [
        "# E5: końcowy asystent (MP5)",
        "",
        f"Data: {created} UTC · Rozmowy: {', '.join(report.conversations)} · Transport MCP: "
        f"{report.transport} ({server}) · Wyszukiwanie: {report.backend} "
        f"({report.embedding_model or NONE})",
        f"Narzędzia: {variant} ({count}) · Prompty: `assistant_understand_system` i "
        f"`agent_system` (PINNED) · Polityka: {replay.run.get('policy_version')} · Budżet agenta: "
        f"{budget.get('max_steps', 24)} kroki / {budget.get('timeout_s', 180)} s (PINNED) · "
        f"Limity MCP: {limits} (PINNED)",
        f"Odtworzenie: przebieg {replay.run['run_id']} · idealne rozumienie ze wzorca, agent ze "
        f"skryptu przez MCP · `understand` {replay.run.get('understand_calls')} razy, agent "
        f"{replay.run.get('agent_calls')} razy · wywołania LLM-a: 0",
    ]
    if live is not None:
        c = live.calls
        models = ", ".join(f"{m}: {n}" for m, n in live.answered_by.items()) or NONE
        limit = str(report.max_calls) if report.max_calls is not None else "bez limitu"
        lines.append(
            f"Na żywo: przebieg {live.run['run_id']} · {live.run['provider']} / "
            f"{live.run['model']} (odpowiedziały: {models}) · wywołania modelu: plan "
            f"{plan['estimate']} (understand {plan['understand']} + agent ≈ "
            f"{plan['agent_estimate']}), górna granica {plan['upper_bound']}, sieciowo "
            f"{c.understand_network + c.agent_network} (understand {c.understand_network}, agent "
            f"{c.agent_network}), z cache {c.understand_cache + c.agent_cache}, błędy {c.errors}, "
            f"limit --max-calls {limit}"
            + (" (przerwano limitem)" if report.stopped_by_max_calls else "")
        )
    else:
        lines.append(
            f"Na żywo: nie uruchomiono (`kovalo eval e5 --live`; plan {plan['estimate']} "
            f"wywołań: understand {plan['understand']} + agent ≈ {plan['agent_estimate']}, górna "
            f"granica {plan['upper_bound']}; limit --max-calls domyślnie 40)"
        )
    lines.append(
        f"Referencja kursu: {reference_model}, przez MCP, przebieg {COURSE_E5_RUN} "
        "(`run_assistant_eval.py`)"
    )
    return lines


def _parity_section(report: E5Report) -> list[str]:
    rows = []
    for row in report.parity:
        args = json.dumps(row.arguments, ensure_ascii=False)
        result = "zgodne" if row.same else ("pominięte" if row.skipped else "RÓŻNICA")
        diffs = "; ".join(f"{d['path']}: {d['kind']}" for d in row.differences) or (
            row.error or NONE
        )
        rows.append([row.tool, args, result, diffs])
    same = sum(1 for r in report.parity if r.same)
    lines = ["## Zgodność MCP z wywołaniem bezpośrednim", ""]
    lines += [
        f"{same}/{len(report.parity)} wywołań daje przez MCP dokładnie ten sam rekord co "
        "bezpośrednio (17 wywołań macierzy kursu + 3 dla `check_termination_condition`; "
        "porównanie `compare_call`, krok 2 kursu).",
        "",
    ]
    lines += _table(["Narzędzie", "Argumenty", "Wynik", "Różnice"], rows)
    return lines


def _checks_section(report: E5Report, reference_model: str) -> list[str]:
    ref = report.reference
    live = report.live
    rows = []
    for dim in DIMENSIONS:
        replay_value = report.replay.totals.get(dim)
        rows.append(
            [
                DIMENSION_NAMES[dim],
                _frac(ref[dim]),
                _frac(replay_value) if dim not in AGENT_DIMENSIONS else "nie dotyczy",
                _frac(live.totals.get(dim)) if live else NONE,
                _target(dim, ref[dim]),
            ]
        )
    rows.append(
        [
            "wywołania MCP (nieudane)",
            f"{ref['mcp_calls']} ({ref['mcp_failures']})",
            f"{report.replay.calls.mcp_calls} ({_failed(report.replay)})",
            f"{live.calls.mcp_calls} ({_failed(live)})" if live else NONE,
            "0 nieudanych",
        ]
    )
    header = ["Sprawdzenie", f"Kurs ({reference_model}, MCP)", "Odtworzenie", "Na żywo", "Cel"]
    lines = ["## Sprawdzenia", ""]
    lines += _table(header, rows)
    lines += [
        "",
        "Odtworzenie: agent ze skryptu, więc sprawdzenia pytań o dane mierzy tylko przebieg na "
        "żywo (i E4). Wynik poniżej celu na żywo nie zmienia kodu wyjścia; opisuje go przegląd.",
    ]
    return lines


def _per_turn_section(report: E5Report) -> list[str]:
    run = _shown_run(report)
    rows = []
    for t in run.turns:
        entry, score = t["entry"], t["score"]
        expected = t.get("expected_kind") or NONE
        got = (entry.get("turn") or {}).get("kind") or NONE
        wrong = ", ".join(k for k, v in score.items() if v is False) or "ok"
        user = (entry.get("user") or "")[:MESSAGE_CHARS]
        rows.append(
            [
                t["dialogue"],
                str(t["n"]),
                expected,
                got,
                entry.get("status") or NONE,
                wrong,
                str(len(t["mcp"])),
                user,
            ]
        )
    label = "na żywo" if report.live is not None else "odtworzenie"
    lines = ["## Tura po turze", "", f"Przebieg: {label} ({run.run['run_id']}).", ""]
    lines += _table(
        [
            "Rozmowa",
            "Nr",
            "Oczekiwany kind",
            "Rozpoznany kind",
            "Status",
            "Niezaliczone",
            "MCP (wywołania)",
            f"Wiadomość ({MESSAGE_CHARS} znaki)",
        ],
        rows,
    )
    return lines


def _data_section(report: E5Report) -> list[str]:
    run = _shown_run(report)
    label = "na żywo" if report.live is not None else "odtworzenie"
    lines = ["## Pytania o dane", "", f"Ścieżka dowodowa każdej tury o dane ({label}).", ""]
    found = False
    for t in run.turns:
        trace = trace_from_record(t["entry"].get("record"))
        if trace is None:
            continue
        found = True
        ids = ", ".join(c["invocation_id"] for c in t["mcp"]) or NONE
        lines += [
            f"### {turn_id(t['dialogue'], t['n'])} (przez MCP; wywołania: {ids})",
            "",
            "```",
            format_trace(trace),
            "```",
            "",
        ]
    if not found:
        lines.append("- Brak tur o dane z zapisem przebiegu.")
    return lines


def _mcp_section(report: E5Report) -> list[str]:
    lines = ["## Wywołania MCP", ""]
    for label, run in (("odtworzenie", report.replay), ("na żywo", report.live)):
        if run is None:
            continue
        failures = run.calls.mcp_failures
        detail = (
            ", ".join(f"{LAYER_NAMES.get(k, k)}: {v}" for k, v in sorted(failures.items()))
            if failures
            else "brak"
        )
        lines.append(
            f"- {label}: {run.calls.mcp_calls} wywołań w turach, nieudane "
            f"{sum(failures.values())} ({detail})"
        )
        for t in run.turns:
            for c in t["mcp"]:
                if c.get("outcome") != "ok":
                    lines.append(
                        f"  - {turn_id(t['dialogue'], t['n'])}: {c['tool']} "
                        f"({c.get('invocation_id')}): {c.get('outcome')}"
                    )
    ref = report.reference
    lines.append(f"- kurs: {ref['mcp_calls']} wywołań, nieudane {ref['mcp_failures']}")
    lines.append(
        f"- zgodność: {len(report.parity)} wywołań macierzy (poza turami), nieudane "
        f"{sum(1 for r in report.parity if r.error)}"
    )
    return lines


def _tokens_section(report: E5Report) -> list[str]:
    ref = report.reference
    lines = ["## Tokeny i koszt", ""]
    lines.append(
        f"- kurs ({report.reference_cost_model}): {_int(ref['tokens_in'])} wejściowych / "
        f"{_int(ref['tokens_out'])} wyjściowych; około {_num(ref['cost_cents'])} centa"
    )
    live = report.live
    if live is not None:
        cost = report.live_cost_cents or {}
        lines.append(
            f"- na żywo (understand + agent): {_int(live.tokens_in)} wejściowych / "
            f"{_int(live.tokens_out)} wyjściowych; po cenach {report.our_cost_model} "
            f"(sprawdzone {report.our_prices_checked_on.isoformat()}): "
            f"{_num(cost.get('ours', 0.0), 2)} centa; po cenach {report.reference_cost_model}: "
            f"{_num(cost.get('reference', 0.0), 2)} centa"
        )
    else:
        lines.append("- na żywo: nie uruchomiono (odtworzenie: 0 tokenów, 0 wywołań LLM-a)")
    return lines


def _flagged(report: E5Report) -> list[tuple[str, str, list[str]]]:
    entries: list[tuple[str, str, list[str]]] = []
    for label, run in (("ours-replay", report.replay), ("ours-live", report.live)):
        if run is None:
            continue
        for t in run.turns:
            tid = turn_id(t["dialogue"], t["n"])
            wrong = [k for k, v in t["score"].items() if v is False]
            entry = t["entry"]
            problems = []
            if wrong:
                problems.append("niezaliczone: " + ", ".join(wrong))
            if entry.get("status") == "skipped":
                problems.append(entry.get("error") or MAX_CALLS_SKIPPED)
            elif entry.get("status") == "error" and entry.get("error"):
                problems.append(f"błąd: {entry['error']}"[:200])
            failed = [c for c in t["mcp"] if c.get("outcome") != "ok"]
            if failed:
                problems.append(f"nieudane wywołania MCP: {len(failed)}")
            if problems:
                entries.append((label, tid, [f"- {label}, {tid}: " + "; ".join(problems)]))
    return entries


def _comment_lines(comment: str) -> list[str]:
    first, *rest = comment.splitlines()
    return [f"  - Komentarz: {first}", *(f"    {ln}" if ln.strip() else "" for ln in rest)]


def _review_section(report: E5Report, review: Review | None) -> list[str]:
    lines = ["## Rozbieżności przejrzane ręcznie", ""]
    flagged = _flagged(report)
    if review is not None:
        if review.intro.strip():
            lines += [review.intro.strip(), ""]
        lines += [f"Komentarze: {REVIEW_FILE}, przegląd {review.reviewed_on.isoformat()}", ""]
    covered: set[tuple[str, str]] = set()
    for label, tid, text in flagged:
        lines += text
        item = review.find(label, tid) if review else None
        if item is None:
            lines.append(COMMENT)
        else:
            covered.add((label, tid))
            lines += _comment_lines(item.comment)
    if not flagged:
        lines.append("- Brak rozbieżności wykrytych automatycznie.")
    if review is not None:
        extra = []
        for item in review.items:
            rest = [v for v in item.variants if (v, item.id) not in covered]
            if rest:
                extra += [f"- {', '.join(rest)}, {item.id}", *_comment_lines(item.comment)]
        if extra:
            lines += ["", "Pozostałe komentarze przeglądu:", "", *extra]
        if review.notes:
            lines += ["", *(f"- {note.strip()}" for note in review.notes)]
    return lines


def render_markdown(report: E5Report, review: Review | None = None) -> str:
    """Markdown raportu w kolejności sekcji z kontraktu (10.5); komentarze z `review`."""
    try:
        from kovalo.config import get_settings

        settings: Any = get_settings()
        reference_model = settings.reference_llm.model
    except Exception:  # raport bez configu: nazwa modelu nieznana
        settings, reference_model = None, "model referencyjny"
    lines = _header(report, reference_model, settings)
    lines.append("")
    for section in (
        _parity_section(report),
        _checks_section(report, reference_model),
        _per_turn_section(report),
        _data_section(report),
        _mcp_section(report),
        _tokens_section(report),
        _review_section(report, review),
    ):
        lines += [*section, ""]
    return "\n".join(lines).rstrip() + "\n"


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _assistant_run(run: E5Run) -> dict[str, Any]:
    """Kształt `run_assistant_eval.py` kursu: run, turns, mcp_captures."""
    captures = _all_captures(run)
    info = {
        **run.run,
        "mcp_calls": len(captures),
        "mcp_failures": sum(1 for c in captures if c.get("outcome") != "ok"),
        "input_tokens": run.tokens_in,
        "output_tokens": run.tokens_out,
    }
    turns = [
        {"dialogue": t["dialogue"], "n": t["n"], "entry": t["entry"], "score": t["score"]}
        for t in run.turns
    ]
    return {"run": info, "turns": turns, "mcp_captures": captures}


def write_reports(report: E5Report, out_dir: Path, *, review: Review | None = None) -> list[Path]:
    """<data>-e5.md, <data>-e5.json (pełny E5Report), assistant_run.replay.json i (z przebiegiem
    na żywo) assistant_run.live.json. Zapis zawsze, także przy błędach."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day = report.created_at.date().isoformat()
    paths = [
        _write(out_dir / f"{day}-e5.md", render_markdown(report, review)),
        _write(
            out_dir / f"{day}-e5.json",
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str),
        ),
        _write(
            out_dir / "assistant_run.replay.json",
            json.dumps(_assistant_run(report.replay), indent=2, ensure_ascii=False, default=str),
        ),
    ]
    if report.live is not None:
        paths.append(
            _write(
                out_dir / "assistant_run.live.json",
                json.dumps(_assistant_run(report.live), indent=2, ensure_ascii=False, default=str),
            )
        )
    return paths
