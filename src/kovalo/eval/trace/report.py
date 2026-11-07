"""Raporty E4: Markdown (dowód etapu), pełny JSON i agent_run.<tryb>.json (kontrakt F4, 11.4).

FR-6.4. Liczby po polsku (8/9, 3,8 centa), identyfikatory bez zmian. Nazwy modeli pochodzą
z raportu (z configu), nigdy ze stałych. Komentarze przeglądu ręcznego tylko z `review.yaml`
(etykiety ours-replay / ours-live / course, id pytań P1…B1); wygenerowanego Markdownu nie
edytujemy. Pliki agent_run.<tryb>.json mają format `save_traces` kursu (czyta je show-trace).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kovalo.agents.show import architecture_label, format_trace
from kovalo.agents.trace import Trace, save_traces
from kovalo.eval.assistant.reference import COURSE_E4_MCP, COURSE_E4_MCP_RUN
from kovalo.eval.review import REVIEW_FILE, Review
from kovalo.eval.trace.e4 import E4Report, E4Run

__all__ = ["REVIEW_LABELS", "render_markdown", "write_reports"]

NONE = "\u2013"
COMMENT = "  - Komentarz: (do uzupełnienia w T2)"
REVIEW_LABELS = ("ours-replay", "ours-live", "course-single", "course-supervisor")
ANSWER_CHARS = 100

TESTS = (
    ("tools", "1 pokrycie narzędzi: każde potrzebne wywołanie", "9/9"),
    ("args", "2 poprawne argumenty (gdzie liczono karę)", "3/3"),
    ("facts", "3 fakty: każdy kluczowy fakt w odpowiedzi", "9/9"),
    ("uncovered", "4 liczby bez źródła (osobno: podane przez użytkownika)", "0"),
    ("approval", "5 twierdzenia o zatwierdzeniu w odpowiedziach", "0/9"),
)
COURSE_QUESTIONS = (
    "Gdzie każda z architektur pomyliła się w T1?",
    "Które podsumowanie specjalisty mówi coś, czego nie zwróciło żadne narzędzie?",
    "Ile kosztował zespół (wywołania, tokeny, czas na pytanie)?",
    "Kiedy zespół się opłaca?",
)


def _num(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _int(value: int) -> str:
    return f"{value:,}".replace(",", "\u00a0")


def _frac(pair: Sequence[int]) -> str:
    return f"{pair[0]}/{pair[1]}"


def _cents(value: float) -> str:
    return f"{_num(value, 2)} centa"


def _pct(value: float | None) -> str:
    return NONE if value is None else f"{round(100 * value)}%"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    cells = (cell.replace("|", "\\|").replace("\n", " ") for row in rows for cell in row)
    flat = list(cells)
    width = len(header)
    lines += ["| " + " | ".join(flat[i : i + width]) + " |" for i in range(0, len(flat), width)]
    return lines


def _test_value(summary: dict[str, Any] | None, key: str) -> str:
    if summary is None:
        return NONE
    if key == "uncovered":
        return f"{summary['uncovered']} ({summary['user_stated']})"
    return _frac(summary[key])


def _code(text: str) -> list[str]:
    return ["```", text, "```"]


# ---------------------------------------------------------------------------
# Sekcje
# ---------------------------------------------------------------------------


def _header(report: E4Report) -> list[str]:
    plan, live, replay = report.llm_calls_planned, report.live, report.replay
    created = report.created_at.strftime("%Y-%m-%d %H:%M")
    budget = replay.run["budget"]
    lines = [
        "# E4: agent z narzędziami (MP4)",
        "",
        f"Data: {created} UTC · Pytania: {', '.join(report.questions)} · Budżet: "
        f"{budget['max_steps']} kroki / {budget['timeout_s']} s na pytanie (PINNED) · "
        "Prompt: `agent_system` (PINNED)",
        f"Odtworzenie: przebieg {replay.run['run_id']} · wywołania przebiegu kursu "
        f"{report.course.run['run_id']} na naszych narzędziach (6 narzędzi kursu) · "
        "wywołania LLM-a: 0",
    ]
    if live is not None:
        variant = live.run.get("tools_variant")
        count = 7 if variant == "ours" else 6
        tools = (
            "z check_termination_condition"
            if variant == "ours"
            else "bez check_termination_condition"
        )
        models = ", ".join(f"{m}: {n}" for m, n in live.answered_by.items()) or NONE
        c = live.calls
        limit = str(report.max_calls) if report.max_calls is not None else "bez limitu"
        lines.append(
            f"Na żywo: przebieg {live.run['run_id']} · {live.run['provider']} / "
            f"{live.run['model']} (odpowiedziały: {models}) · narzędzia: {variant} ({count}, "
            f"{tools}) · wywołania modelu: szacunek {plan['estimate']}, górna granica "
            f"{plan['upper_bound']}, wykonane sieciowo {c.network}, z cache {c.from_cache}, "
            f"błędy {c.errors}, limit --max-calls {limit}"
            + (" (przerwano limitem)" if report.stopped_by_max_calls else "")
            + f" · wyszukiwanie: {report.backend} ({report.embedding_model or NONE})"
        )
    else:
        lines.append(
            f"Na żywo: nie uruchomiono (`kovalo eval e4 --live`; szacunek {plan['estimate']}, "
            f"górna granica {plan['upper_bound']} wywołań modelu; limit --max-calls domyślnie 20; "
            "narzędzia: check_termination_condition w wariancie ours)"
        )
    ref = report.reference
    lines += [
        f"Referencja kursu: {ref['model']} · jeden agent przebieg {report.course.run['run_id']}, "
        f"zespół przebieg {report.course_supervisor.run['run_id']}",
        "",
    ]
    return lines


def _tests_section(report: E4Report) -> list[str]:
    ref = report.reference
    live = report.live.summary if report.live else None
    via_mcp = report.transport != "direct"  # F5: kolumna „Kurs przez MCP” (ADR-0014 pkt 4)
    summaries: list[dict[str, Any] | None] = [
        report.course.summary,
        report.course_supervisor.summary,
        report.replay.summary,
        live,
    ]
    if via_mcp:
        summaries.append(COURSE_E4_MCP)
    rows = [
        [label, *(_test_value(s, key) for s in summaries), target] for key, label, target in TESTS
    ]
    rows.append(
        [
            "wyniki: answered / error / limit",
            *(
                NONE if s is None else "{answered} / {error} / {limit}".format(**s["outcomes"])
                for s in summaries
            ),
            NONE,
        ]
    )
    rows.append(
        [
            "wywołania narzędzi (nieudane)",
            *(NONE if s is None else f"{s['tool_calls']} ({s['failed']})" for s in summaries),
            NONE,
        ]
    )
    header = [
        "Test (co mierzy)",
        "Kurs: jeden agent",
        "Kurs: zespół",
        "Odtworzenie",
        f"Na żywo (przez MCP, {report.transport})" if via_mcp else "Na żywo",
        *([f"Kurs przez MCP ({ref['model']})"] if via_mcp else []),
        "Cel",
    ]
    lines = ["## Testy", ""]
    lines += _table(header, rows)
    lines += [
        "",
        "Kolumny kursu: przeliczenie zapisanych przebiegów naszym `score_trace`; "
        f"README MP4 podaje "
        f"{_frac(ref['single']['tools'])}, {_frac(ref['single']['args'])}, "
        f"{_frac(ref['single']['facts'])}, {ref['single']['uncovered']} "
        f"({ref['single']['user_stated']}), {_frac(ref['single']['approval'])} (jeden agent) "
        f"oraz {_frac(ref['supervisor']['tools'])}, {_frac(ref['supervisor']['args'])}, "
        f"{_frac(ref['supervisor']['facts'])}, {ref['supervisor']['uncovered']} "
        f"({ref['supervisor']['user_stated']}), {_frac(ref['supervisor']['approval'])} (zespół). "
        "Cel na żywo z docs/06-evaluation.md; wynik poniżej celu nie zmienia kodu wyjścia.",
    ]
    if via_mcp:
        lines.append(
            f"Na żywo przez MCP ({report.transport}): narzędzia agenta z serwera MCP Kovalo; "
            f"kolumna „Kurs przez MCP” to README MP5 (`run_agent_mcp.py`, przebieg "
            f"{COURSE_E4_MCP_RUN}, około {_num(COURSE_E4_MCP['cost_cents'])} centa)."
        )
    return lines


def _score_cells(row: dict[str, Any] | None) -> str:
    if row is None:
        return NONE
    s = row["score"]
    args = {None: NONE, True: "ok", False: "ŹLE"}[s["arguments_correct"]]
    unbacked = ", ".join(s["uncovered_numbers"]) or "brak"
    return f"{_pct(s['tool_coverage'])}, {args}, {_pct(s['answer_fact_coverage'])}, {unbacked}"


def _per_question_section(report: E4Report, gold_kinds: dict[str, str]) -> list[str]:
    replay = {r["trace"].question_id: r for r in report.replay.rows}
    live = {r["trace"].question_id: r for r in report.live.rows} if report.live else {}
    rows = []
    for qid in report.questions:
        shown = live.get(qid) or replay.get(qid)
        trace: Trace | None = shown["trace"] if shown else None
        models = ", ".join(sorted({c.model for c in trace.model_calls if c.model})) if trace else ""
        answer = (trace.answer or f"({trace.error})") if trace else NONE
        rows.append(
            [
                qid,
                gold_kinds.get(qid, NONE),
                _score_cells(replay.get(qid)),
                _score_cells(live.get(qid)),
                trace.status if trace else NONE,
                str(trace.steps) if trace else NONE,
                models or NONE,
                answer.replace("\n", " ")[:ANSWER_CHARS],
            ]
        )
    header = [
        "ID",
        "Rodzaj",
        "Odtworzenie: narzędzia, argumenty, fakty, bez źródła",
        "Na żywo: narzędzia, argumenty, fakty, bez źródła",
        "Status",
        "Kroki",
        "Modele",
        f"Odpowiedź ({ANSWER_CHARS} znaków)",
    ]
    note = "Status, kroki, modele i odpowiedź: przebieg na żywo, a bez niego odtworzenie."
    return ["## Pytanie po pytaniu", "", *_table(header, rows), "", note]


def _t1_section(report: E4Report) -> list[str]:
    lines = ["## T1: warunek rozwiązania umowy", ""]
    live = (
        next((r for r in report.live.rows if r["trace"].question_id == "T1"), None)
        if report.live
        else None
    )
    if live is not None:
        lines += [
            "Na żywo (nasz agent):",
            "",
            *_code(format_trace(live["trace"], live["score"])),
            "",
        ]
    else:
        lines += ["Na żywo: nie uruchomiono albo T1 poza wybranymi pytaniami.", ""]
    course = next((r for r in report.course.rows if r["trace"].question_id == "T1"), None)
    if course is not None:
        lines += [
            "Odpowiedź kursu (jeden agent) dla porównania:",
            "",
            *_code(course["trace"].answer or "(brak)"),
        ]
    return lines


def _paths_section(report: E4Report) -> list[str]:
    lines = ["## Ścieżki dowodowe", ""]
    if report.live is None:
        return [*lines, "Bez przebiegu na żywo."]
    shown = 0
    for row in report.live.rows:
        s, trace = row["score"], row["trace"]
        full = all(
            s[k] in (None, 1, 1.0, True)
            for k in ("tool_coverage", "arguments_correct", "answer_fact_coverage")
        )
        if (
            full
            and not s["uncovered_numbers"]
            and not s["approval_claim"]
            and trace.status == "answered"
        ):
            continue
        lines += [*_code(format_trace(trace, s)), ""]
        shown += 1
    if not shown:
        lines.append("Każde pytanie na żywo: wszystkie testy 100%, status answered.")
    return lines


def _per_question_seconds(run: E4Run) -> str:
    traces = [r["trace"] for r in run.rows]
    return _num(sum(t.seconds for t in traces) / len(traces)) + " s" if traces else NONE


def _team_section(report: E4Report, review: Review | None) -> list[str]:
    ref = report.reference
    rows = []
    for label, run, side in (
        (architecture_label("single agent"), report.course, "single"),
        (architecture_label("supervisor + 3 specialists"), report.course_supervisor, "supervisor"),
    ):
        s = run.summary
        rows.append(
            [
                label,
                str(s["tool_calls"]),
                f"{_int(s['tokens_in'])} / {_int(s['tokens_out'])}",
                _cents(ref["cost_cents"][side]),
                _per_question_seconds(run),
            ]
        )
    header = [
        "Architektura (kurs)",
        "Wywołania narzędzi",
        "Tokeny we / wy",
        "Koszt (README)",
        "Czas na pytanie",
    ]
    lines = ["## Jeden agent czy zespół?", "", *_table(header, rows), ""]
    lines.append("Pytania kursu (odpowiedzi z przeglądu ręcznego, review.yaml):")
    lines.append("")
    for n, question in enumerate(COURSE_QUESTIONS, start=1):
        lines.append(f"{n}. {question}")
        item = review.find("course-supervisor", f"Q{n}") if review else None
        lines.append(
            f"   - {item.comment.splitlines()[0]}" if item else "   - (do uzupełnienia w T2)"
        )
    return lines


def _tokens_section(report: E4Report) -> list[str]:
    lines = ["## Tokeny i koszt", ""]
    if report.live is None:
        return [*lines, "Bez przebiegu na żywo (odtworzenie i przeliczenie: 0 tokenów)."]
    s, cost = report.live.summary, report.live_cost_cents or {}
    header = [
        "Przebieg",
        "Tury modelu",
        "Tokeny wejścia",
        "Tokeny wyjścia",
        f"Koszt ({report.our_cost_model})",
        f"Koszt ({report.reference_cost_model})",
    ]
    rows = [
        [
            "na żywo",
            str(report.live.calls.model_turns),
            _int(s["tokens_in"]),
            _int(s["tokens_out"]),
            _cents(cost["ours"]) if cost else NONE,
            _cents(cost["reference"]) if cost else NONE,
        ],
        [
            "kurs, jeden agent",
            NONE,
            _int(report.reference["single"]["tokens_in"]),
            _int(report.reference["single"]["tokens_out"]),
            NONE,
            _cents(report.reference["cost_cents"]["single"]),
        ],
    ]
    lines += _table(header, rows)
    lines += [
        "",
        "Tokeny policzone przez dostawcę (także odpowiedzi z cache); koszt po cenach "
        f"`cost.our_prices_per_1m_tokens` (sprawdzone {report.our_prices_checked_on.isoformat()}) "
        "i po cenach referencyjnych kursu (PINNED). Darmowy tier Gemini: 0 USD.",
    ]
    return lines


def _flagged(report: E4Report) -> list[tuple[str, str, list[str]]]:
    entries = []
    runs = (("ours-replay", report.replay), ("ours-live", report.live))
    for label, run in runs:
        if run is None:
            continue
        for row in run.rows:
            s, trace = row["score"], row["trace"]
            problems = []
            if s["tool_coverage"] not in (None, 1, 1.0):
                problems.append(f"narzędzia {_pct(s['tool_coverage'])}")
            if s["arguments_correct"] is False:
                problems.append("argumenty ŹLE")
            if s["answer_fact_coverage"] not in (None, 1, 1.0):
                problems.append(f"fakty {_pct(s['answer_fact_coverage'])}")
            if s["uncovered_numbers"]:
                problems.append("bez źródła: " + ", ".join(s["uncovered_numbers"]))
            if s["approval_claim"]:
                problems.append("twierdzi, że zatwierdzono")
            if trace.status != "answered":
                problems.append(f"status {trace.status}: {trace.error}")
            if problems:
                entries.append(
                    (
                        label,
                        trace.question_id or "",
                        ["- " + f"{label}, {trace.question_id}: " + "; ".join(problems)],
                    )
                )
    return entries


def _comment_lines(comment: str) -> list[str]:
    first, *rest = comment.splitlines()
    return [f"  - Komentarz: {first}", *(f"    {ln}" if ln.strip() else "" for ln in rest)]


def _review_section(report: E4Report, review: Review | None) -> list[str]:
    lines = ["## Rozbieżności przejrzane ręcznie", ""]
    flagged = _flagged(report)
    if review is not None:
        if review.intro.strip():
            lines += [review.intro.strip(), ""]
        lines += [f"Komentarze: {REVIEW_FILE}, przegląd {review.reviewed_on.isoformat()}", ""]
    covered: set[tuple[str, str]] = set()
    for label, qid, text in flagged:
        lines += text
        item = review.find(label, qid) if review else None
        if item is None:
            lines.append(COMMENT)
        else:
            covered.add((label, qid))
            lines += _comment_lines(item.comment)
    if not flagged:
        lines.append("- Brak rozbieżności wykrytych automatycznie.")
    if review is not None:
        extra = []
        for item in review.items:
            rest = [v for v in item.variants if (v, item.id) not in covered]
            if rest and not item.id.startswith("Q"):
                extra += [f"- {', '.join(rest)}, {item.id}", *_comment_lines(item.comment)]
        if extra:
            lines += ["", "Pozostałe komentarze przeglądu:", "", *extra]
        if review.notes:
            lines += ["", *(f"- {note.strip()}" for note in review.notes)]
    return lines


def render_markdown(report: E4Report, review: Review | None = None) -> str:
    """Markdown raportu w kolejności sekcji z kontraktu (11.4); komentarze z `review`."""
    from kovalo.config import get_settings
    from kovalo.eval.datasets import load_agent_gold

    try:
        kinds = {
            g.id: g.kind for g in load_agent_gold(get_settings().paths.mini_project_4_questions)
        }
    except Exception:  # raport bez rodzajów pytań nadal ma sens
        kinds = {}
    lines = _header(report)
    for section in (
        _tests_section(report),
        _per_question_section(report, kinds),
        _t1_section(report),
        _paths_section(report),
        _team_section(report, review),
        _tokens_section(report),
        _review_section(report, review),
    ):
        lines += [*section, ""]
    return "\n".join(lines).rstrip() + "\n"


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _save_run(path: Path, run: E4Run) -> Path:
    save_traces(path, run.run, [r["trace"] for r in run.rows], [r["score"] for r in run.rows])
    return path


def write_reports(report: E4Report, out_dir: Path, *, review: Review | None = None) -> list[Path]:
    """<data>-e4.md, <data>-e4.json (pełny E4Report), agent_run.replay.json i (z przebiegiem na
    żywo) agent_run.live.json w formacie save_traces kursu. Zapis zawsze, także przy błędach."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day = report.created_at.date().isoformat()
    paths = [
        _write(out_dir / f"{day}-e4.md", render_markdown(report, review)),
        _write(
            out_dir / f"{day}-e4.json",
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
        ),
        _save_run(out_dir / "agent_run.replay.json", report.replay),
    ]
    if report.live is not None:
        paths.append(_save_run(out_dir / "agent_run.live.json", report.live))
    return paths
