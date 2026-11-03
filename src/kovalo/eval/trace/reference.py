"""Przebiegi referencyjne kursu MP4 (kontrakt F4, 11.2; P3).

Zapisane przebiegi `reference_traces/{agent_run,supervisor_run}.json` przeliczamy naszym
`score_trace`: każdy wiersz musi dać dokładnie zapisany przez kurs `score` (inaczej błąd portu,
E4ReferenceMismatch, kod 4 w CLI). Liczby zbiorcze z README MP4 (course_e4) w kształcie
E4Run.summary. Nazwę modelu referencyjnego bierzemy z configu, nigdy z literału.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from kovalo.agents.trace import Trace, load_traces
from kovalo.config import Settings
from kovalo.eval.datasets import load_agent_gold
from kovalo.eval.trace.scoring import score_trace

__all__ = [
    "COURSE_RUNS",
    "CourseRun",
    "E4ReferenceMismatch",
    "course_e4",
    "course_run_path",
    "rescore_course",
]

CourseRun = Literal["single", "supervisor"]

# Ścieżki względne wobec settings.root (ścieżka bezwzględna też działa), czytane w chwili
# wywołania, więc test może podmienić wpis.
COURSE_RUNS: dict[str, str] = {
    "single": "reference/course/mini_project_4/reference_traces/agent_run.json",
    "supervisor": "reference/course/mini_project_4/reference_traces/supervisor_run.json",
}


class E4ReferenceMismatch(RuntimeError):
    """Przeliczenie przebiegu kursu ≠ zapisany 'score'; komunikat '<id>: <klucz>: <nasze> !=
    <kurs>'."""


def course_e4(reference_model: str) -> dict[str, Any]:
    """Liczby z README MP4 (przebiegi 20261006T212905-2d338f6e i 20261006T213238-33cb1f4c)
    w kształcie E4Run.summary."""
    return {
        "single": {
            "tools": [8, 9],
            "args": [3, 3],
            "facts": [8, 9],
            "approval": [0, 9],
            "uncovered": 0,
            "user_stated": 0,
            "outcomes": {"answered": 9, "error": 0, "limit": 0},
            "tool_calls": 21,
            "failed": 0,
            "tokens_in": 39043,
            "tokens_out": 1881,
        },
        "supervisor": {
            "tools": [8, 9],
            "args": [3, 3],
            "facts": [8, 9],
            "approval": [0, 9],
            "uncovered": 3,
            "user_stated": 2,
            "outcomes": {"answered": 9, "error": 0, "limit": 0},
            "tool_calls": 58,
            "failed": 0,
            "tokens_in": 84024,
            "tokens_out": 17109,
        },
        "model": reference_model,
        "cost_cents": {"single": 3.8, "supervisor": 14.0},
    }


def course_run_path(settings: Settings, which: CourseRun) -> Path:
    path = Path(COURSE_RUNS[which])
    return path if path.is_absolute() else Path(settings.root) / path


def _same(ours: Any, theirs: Any) -> bool:
    if isinstance(theirs, float) and isinstance(ours, int | float) and not isinstance(ours, bool):
        return math.isclose(float(ours), theirs, rel_tol=0, abs_tol=1e-9)
    return bool(ours == theirs)


def rescore_course(
    settings: Settings, which: CourseRun
) -> tuple[dict[str, Any], list[Trace], list[dict[str, Any]]]:
    """(run, traces, scores): load_traces + score_trace na wzorcu z data/eval. Każdy wiersz musi
    być równy zapisanemu 'score' kursu (liczby z tolerancją 1e-9); różnica →
    E4ReferenceMismatch z id pytania i kluczem."""
    run, rows = load_traces(course_run_path(settings, which))
    gold = {g.id: g for g in load_agent_gold(settings.paths.mini_project_4_questions)}
    traces: list[Trace] = []
    scores: list[dict[str, Any]] = []
    for row in rows:
        trace: Trace = row["trace"]
        qid = trace.question_id or ""
        if qid not in gold:
            raise E4ReferenceMismatch(f"{qid}: brak pytania we wzorcu")
        ours = score_trace(trace, gold[qid])
        for key, theirs in (row.get("score") or {}).items():
            if key not in ours or not _same(ours[key], theirs):
                raise E4ReferenceMismatch(f"{qid}: {key}: {ours.get(key)!r} != {theirs!r}")
        traces.append(trace)
        scores.append(ours)
    return run, traces, scores
