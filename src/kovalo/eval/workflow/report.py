"""Raporty E3: Markdown (commitowany dowód etapu), pełny JSON i dialogues.<tryb>.json.

Kontrakt F3, sekcja 10.5 (FR-6.4). Liczby w Markdownie po polsku (przecinek dziesiętny),
identyfikatory bez zmian. Nazwy modeli pochodzą z raportu (z configu), nigdy ze stałych.
Komentarze przeglądu ręcznego tylko z `review.yaml` (etykiety ours-replay / ours-live,
id `<dialog>-t<n>`); wygenerowanego Markdownu nie edytujemy.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kovalo.eval.review import REVIEW_FILE, Review
from kovalo.eval.workflow.e3 import E3Report, E3Run
from kovalo.eval.workflow.scoring import STAGES

__all__ = [
    "REVIEW_LABELS",
    "course_dialogues_report",
    "render_markdown",
    "turn_ids",
    "write_reports",
]

NONE = "\u2013"  # półpauza w pustych komórkach
COMMENT = "  - Komentarz: (do uzupełnienia w T2)"
REVIEW_LABELS = ("ours-replay", "ours-live")
MESSAGE_CHARS = 48

STAGE_LABELS = {
    "route": "1 kierowanie (rodzaj wiadomości)",
    "fields": "2 pola odczytane przez LLM",
    "request": "3 stan wniosku po aktualizacji",
    "approvers": "4 osoby zatwierdzające (polityka)",
    "pauses": "5 pauzy (typ, rola, kolejność)",
    "status": "6 status końcowy",
}
TARGETS = {"route": "odtworzenie 100%; na żywo ≥ 12/13"}
DEFAULT_TARGET = "odtworzenie 100%"


# ---------------------------------------------------------------------------
# Formatowanie
# ---------------------------------------------------------------------------


def _num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _int(value: int) -> str:
    return f"{value:,}".replace(",", "\u00a0")


def _frac(pair: Sequence[int]) -> str:
    return f"{pair[0]}/{pair[1]}"


def _cents(value: float) -> str:
    return f"{_num(value)} centa"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]
    return lines


def turn_ids(transcript: Sequence[Mapping[str, Any]]) -> list[str]:
    """Id tur `<dialog>-t<n>` (n od 1 w obrębie dialogu), w kolejności zapisu."""
    seen: Counter[str] = Counter()
    ids = []
    for entry in transcript:
        dialogue = str(entry["dialogue"])
        seen[dialogue] += 1
        ids.append(f"{dialogue}-t{seen[dialogue]}")
    return ids


def _models(entry: Mapping[str, Any]) -> list[str]:
    return [str(c["model"]) for c in entry.get("calls") or [] if c.get("model")]


def _wrong(detail: Mapping[str, Any]) -> str:
    return ", ".join(detail["wrong"]) or "ok"


def _calls_text(run: E3Run) -> str:
    c = run.calls
    return f"wykonane {c.understand + c.answer}, z cache {c.from_cache}, błędy {c.errors}"


# ---------------------------------------------------------------------------
# Sekcje
# ---------------------------------------------------------------------------


def _header(report: E3Report) -> list[str]:
    ref, plan, replay, live = report.reference, report.llm_calls_planned, report.replay, report.live
    created = report.created_at.strftime("%Y-%m-%d %H:%M")
    lines = [
        "# E3: workflow decyzyjny (MP3)",
        "",
        f"Data: {created} UTC · Polityka: {replay.policy} · Dialogi: {', '.join(report.dialogues)} "
        f"({len(replay.transcript)} tur)",
        f"Odtworzenie: przebieg {replay.run_id} · idealne `understand`, RAG offline (embeddingi "
        "referencyjne kursu, stała odpowiedź) · wywołania LLM-a: 0",
    ]
    if live is not None:
        models = ", ".join(f"{m}: {n}" for m, n in live.answered_by.items()) or NONE
        lines.append(
            f"Na żywo: przebieg {live.run_id} · {live.provider} / {live.model} (odpowiedziały: "
            f"{models}) · wywołania: plan {plan['total']} ({plan['understand']} understand + "
            f"{plan['answer']} odpowiedzi RAG), {_calls_text(live)} · wyszukiwanie: "
            f"{report.backend} ({report.embedding_model or NONE})"
        )
    else:
        lines.append(
            f"Na żywo: nie uruchomiono (`kovalo eval e3 --live`; plan {plan['total']} wywołań: "
            f"{plan['understand']} understand + {plan['answer']} odpowiedzi RAG)"
        )
    calls = ref["calls"]
    lines += [
        f"Referencja kursu: przebieg {ref['run_id']} · {ref['provider']} / {ref['model']} · "
        f"wywołania {calls['understand']} + {calls['answer']} · koszt {_cents(ref['cost_cents'])}",
        "",
    ]
    return lines


def _stages_section(report: E3Report) -> list[str]:
    ref, replay, live = report.reference, report.replay, report.live
    rows = []
    for stage in STAGES:
        rows.append(
            [
                STAGE_LABELS[stage],
                _frac(ref[stage]),
                _frac(replay.score[stage]),
                _frac(live.score[stage]) if live else NONE,
                TARGETS.get(stage, DEFAULT_TARGET),
            ]
        )
    rows.append(
        [
            "błędy (nieudane tury)",
            str(ref["errors"]),
            str(replay.score["errors"]),
            str(live.score["errors"]) if live else NONE,
            "0",
        ]
    )
    lines = ["## Etapy", ""]
    lines += _table(["Etap", "Referencja kursu", "Odtworzenie", "Na żywo", "Cel"], rows)
    lines += [
        "",
        "Trafne/ocenione; etap oceniany tylko w turach, których wzorzec go zawiera, porażki "
        "zostają w mianowniku. Wzorzec (osoby zatwierdzające) wyliczono ręcznie, nie naszym "
        "silnikiem polityki. Referencja kursu: przebieg na żywo z LLM-em referencyjnym "
        "(odtworzenie kursu z idealnym `understand` daje 100% na każdym etapie).",
    ]
    return lines


def _per_turn_section(report: E3Report) -> list[str]:
    replay, live = report.replay, report.live
    ids = turn_ids(replay.transcript)
    rows = []
    for i, entry in enumerate(replay.transcript):
        actual = live.transcript[i] if live else entry
        kind = (actual.get("turn") or {}).get("kind") or NONE
        models = Counter(_models(actual)) if live else Counter()
        rows.append(
            [
                str(entry["dialogue"]),
                ids[i].rsplit("-", 1)[1],
                kind,
                str(actual.get("status") or NONE),
                _wrong(replay.details[i]),
                _wrong(live.details[i]) if live else NONE,
                ", ".join(f"{m} x{n}" if n > 1 else m for m, n in models.items()) or NONE,
                str(entry["user"])[:MESSAGE_CHARS],
            ]
        )
    header = [
        "D",
        "Tura",
        "Rodzaj",
        "Status",
        "Odtworzenie: błędne etapy",
        "Na żywo: błędne etapy",
        "Model",
        f"Wiadomość ({MESSAGE_CHARS} znaków)",
    ]
    lines = ["## Tura po turze", ""]
    lines += _table(header, rows)
    lines += [
        "",
        "Rodzaj i status: przebieg na żywo (bez niego odtworzenie). Model: faktyczny model "
        "z odpowiedzi dostawcy (fallback widoczny).",
    ]
    return lines


def _tokens_section(report: E3Report) -> list[str]:
    live = report.live
    if live is None:
        return []
    usage = live.usage
    cost = report.live_cost_cents
    c = live.calls
    rows = [
        [
            "na żywo",
            str(c.understand + c.answer),
            str(c.from_cache),
            _int(usage.input_tokens) if usage else NONE,
            _int(usage.output_tokens) if usage else NONE,
            _int(usage.reasoning_tokens) if usage and usage.reasoning_tokens is not None else NONE,
            _cents(cost["ours"]) if cost else NONE,
            _cents(cost["reference"]) if cost else NONE,
        ],
        [
            "referencja kursu",
            str(sum(report.reference["calls"].values())),
            NONE,
            NONE,
            NONE,
            NONE,
            NONE,
            _cents(report.reference["cost_cents"]),
        ],
    ]
    header = [
        "Przebieg",
        "Wywołania",
        "Z cache",
        "Tokeny wejścia",
        "Tokeny wyjścia",
        "W tym myślenie",
        f"Koszt ({report.our_cost_model})",
        f"Koszt ({report.reference_cost_model})",
    ]
    lines = ["## Tokeny i koszt", ""]
    lines += _table(header, rows)
    lines += [
        "",
        "Tokeny policzone przez dostawcę (suma wszystkich wywołań, także odpowiedzi z cache: "
        "cache zwraca oryginalne `usage_metadata`); wyjście razem z tokenami myślenia. "
        f"Koszt: te same tokeny po cenach `cost.our_prices_per_1m_tokens` (sprawdzone "
        f"{report.our_prices_checked_on.isoformat()}) i po cenach referencyjnych kursu (PINNED).",
    ]
    return lines


@dataclass(frozen=True)
class _ReviewEntry:
    label: str
    id: str
    text: list[str]
    flagged: bool  # rozbieżność wykryta automatycznie


def _review_entries(report: E3Report) -> list[_ReviewEntry]:
    entries: list[_ReviewEntry] = []
    runs = [("ours-replay", report.replay), ("ours-live", report.live)]
    for label, run in runs:
        if run is None:
            continue
        for tid, entry, detail in zip(
            turn_ids(run.transcript), run.transcript, run.details, strict=True
        ):
            kind = (entry.get("turn") or {}).get("kind") or NONE
            parts = [f"rodzaj {kind}", f"status {entry.get('status') or NONE}"]
            parts.append(f"błędne etapy: {_wrong(detail)}")
            text = [f"- {label}, {tid}: " + "; ".join(parts)]
            if entry.get("error"):
                text.append(f"  - Błąd: {entry['error']}")
            if entry.get("pauses"):
                pauses = ", ".join(f"{p[0]} {p[1]}" for p in entry["pauses"])
                text.append(f"  - Pauzy: {pauses}")
            flagged = bool(detail["wrong"]) or entry.get("status") == "error"
            entries.append(_ReviewEntry(label, tid, text, flagged))
    return entries


def _comment_lines(comment: str) -> list[str]:
    first, *rest = comment.splitlines()
    return [f"  - Komentarz: {first}", *(f"    {ln}" if ln.strip() else "" for ln in rest)]


def _review_section(report: E3Report, review: Review | None) -> list[str]:
    lines = ["## Rozbieżności przejrzane ręcznie", ""]
    entries = _review_entries(report)
    flagged = [e for e in entries if e.flagged]
    if review is None:
        items: list[str] = []
        for e in flagged:
            items += [*e.text, COMMENT]
        return [*lines, *(items or ["- Brak rozbieżności wykrytych automatycznie.", COMMENT])]

    if review.intro.strip():
        lines += [review.intro.strip(), ""]
    lines += [f"Komentarze: {REVIEW_FILE}, przegląd {review.reviewed_on.isoformat()}", ""]
    covered: set[tuple[str, str]] = set()
    for e in flagged:
        lines += e.text
        item = review.find(e.label, e.id)
        if item is None:
            lines.append(COMMENT)
            continue
        covered.add((e.label, e.id))
        lines += _comment_lines(item.comment)
    if not flagged:
        lines += ["- Brak rozbieżności wykrytych automatycznie."]

    current = {(e.label, e.id): e for e in entries}
    orphans: list[str] = []
    for item in review.items:
        rest = [v for v in item.variants if (v, item.id) not in covered]
        present = [v for v in rest if (v, item.id) in current]
        missing = [v for v in rest if (v, item.id) not in current]
        if present:
            lines.append(f"- {', '.join(present)}, {item.id} (dodane w przeglądzie ręcznym)")
            for label in present:
                lines += [f"  {line}" for line in current[(label, item.id)].text]
            lines += _comment_lines(item.comment)
        if missing:
            orphans += [f"- {', '.join(missing)}, {item.id}", *_comment_lines(item.comment)]
    if orphans:
        lines += ["", "Komentarze bez pozycji w tym przebiegu:", "", *orphans]
    if review.notes:
        lines += ["", *(f"- {note.strip()}" for note in review.notes)]
    return lines


def render_markdown(report: E3Report, review: Review | None = None) -> str:
    """Markdown raportu w kolejności sekcji z kontraktu (10.5); komentarze z `review`."""
    lines = _header(report)
    for section in (
        _stages_section(report),
        _per_turn_section(report),
        _tokens_section(report),
        _review_section(report, review),
    ):
        if section:
            lines += [*section, ""]
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Zapis
# ---------------------------------------------------------------------------


def course_dialogues_report(report: E3Report, run: E3Run) -> dict[str, Any]:
    """Kształt dialogues.json z run_dialogues.py kursu (+ nasze pola wpisów: calls, seconds)."""
    return {
        "run_id": run.run_id,
        "date": report.created_at.date().isoformat(),
        "provider": run.provider,
        "model": run.model,
        "policy": run.policy,
        "score": run.score,
        "details": run.details,
        "transcript": run.transcript,
    }


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def write_reports(report: E3Report, out_dir: Path, *, review: Review | None = None) -> list[Path]:
    """<data>-e3.md, <data>-e3.json (pełny E3Report), dialogues.replay.json i (z przebiegiem
    na żywo) dialogues.live.json. Zapis zawsze, także gdy tury na żywo to błędy."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day = report.created_at.date().isoformat()
    paths = [
        _write(out_dir / f"{day}-e3.md", render_markdown(report, review)),
        _write(out_dir / f"{day}-e3.json", report.model_dump_json(indent=2)),
        _write(
            out_dir / "dialogues.replay.json",
            _json(course_dialogues_report(report, report.replay)),
        ),
    ]
    if report.live is not None:
        paths.append(
            _write(
                out_dir / "dialogues.live.json", _json(course_dialogues_report(report, report.live))
            )
        )
    return paths
