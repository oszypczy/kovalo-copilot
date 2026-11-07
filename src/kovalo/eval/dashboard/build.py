"""Budowa, zapis i sprawdzanie dashboardu ewaluacji (kontrakt F6, 8.5).

Źródła: `dashboard.yaml` (ręczne dane), JSON-y raportów etapów (poza gitem, gdy są na dysku)
i poprzedni `dashboard.json` (w gicie) dla etapów bez JSON-u. Wynik jest deterministyczny:
bez daty generowania, a `as_of` to data najnowszego raportu.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kovalo.config import ConfigError
from kovalo.eval.dashboard.models import Dashboard, MonthlyCost, StageSummary
from kovalo.eval.dashboard.render import render_markdown
from kovalo.eval.dashboard.spec import DashboardSpec, StageSpec
from kovalo.eval.dashboard.stages import (
    ADAPTERS,
    DashboardDataError,
    calls_from_first_live,
    monthly_cost_e1,
)

__all__ = [
    "JSON_NAME",
    "MARKDOWN_NAME",
    "SPEC_NAME",
    "build_dashboard",
    "check_dashboard",
    "load_previous",
    "render_outputs",
    "write_dashboard",
]

SPEC_NAME = "dashboard.yaml"
MARKDOWN_NAME = "dashboard.md"
JSON_NAME = "dashboard.json"


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


def load_previous(reports_dir: Path) -> Dashboard | None:
    """Poprzedni `dashboard.json` albo None, gdy go nie ma; uszkodzony → `ConfigError`."""
    path = reports_dir / JSON_NAME
    if not path.is_file():
        return None
    try:
        return Dashboard.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ConfigError(f"{path}: niepoprawny poprzedni dashboard:\n{exc}") from exc


def _read_report(spec: StageSpec, path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DashboardDataError(spec.stage, spec.report, "niepoprawny JSON w {path}") from exc
    if not isinstance(data, dict):
        raise DashboardDataError(spec.stage, spec.report, "oczekiwano obiektu JSON w {path}")
    return data


def _from_previous(spec: StageSpec, summary: StageSummary) -> StageSummary:
    """Wpis z poprzedniego dashboardu; ręczne pola (linki, uwagi, first_live) z bieżącego yaml."""
    update: dict[str, Any] = {
        "report_md": spec.markdown,
        "related_md": list(spec.related),
        "notes": list(spec.notes),
    }
    planned = summary.calls.planned if summary.calls is not None else None
    first = calls_from_first_live(spec, planned)
    if first is not None:
        update["calls"] = first
    return summary.model_copy(update=update)


def build_dashboard(
    spec: DashboardSpec,
    reports_dir: Path,
    previous: Dashboard | None,
    *,
    warn: Callable[[str], None] = _stderr,
) -> Dashboard:
    """Dashboard z raportów etapów; etap bez JSON-u z `previous` (komunikat przez `warn`).

    Etap bez JSON-u i bez wpisu w `previous` albo brak pola w JSON-ie → `DashboardDataError`.
    """
    old = {s.stage: s for s in previous.stages} if previous is not None else {}
    stages: list[StageSummary] = []
    monthly: MonthlyCost | None = None
    for stage_spec in spec.stages:
        path = reports_dir / stage_spec.report
        if path.is_file():
            data = _read_report(stage_spec, path)
            stages.append(ADAPTERS[stage_spec.stage](data, stage_spec))
            if stage_spec.stage == "e1":
                monthly = monthly_cost_e1(data)
            continue
        if stage_spec.stage not in old:
            raise DashboardDataError(
                stage_spec.stage,
                stage_spec.report,
                f"brak raportu {{path}} i wpisu w {JSON_NAME}",
            )
        warn(f"{stage_spec.stage.upper()}: brak {path.name}, dane z {JSON_NAME}")
        stages.append(_from_previous(stage_spec, old[stage_spec.stage]))
        if stage_spec.stage == "e1" and previous is not None:
            monthly = previous.monthly_cost
    if not stages:
        raise ConfigError(f"{SPEC_NAME}: brak etapów")
    return Dashboard(
        as_of=max(s.date for s in stages),
        stages=stages,
        monthly_cost=monthly,
        gemini_usage=list(spec.gemini_usage),
    )


def render_outputs(dashboard: Dashboard) -> dict[str, bytes]:
    """Bajty obu plików wyjściowych (UTF-8, LF, nowa linia na końcu)."""
    return {
        MARKDOWN_NAME: render_markdown(dashboard).encode("utf-8"),
        JSON_NAME: (dashboard.model_dump_json(indent=2) + "\n").encode("utf-8"),
    }


def write_dashboard(dashboard: Dashboard, reports_dir: Path) -> None:
    """Zapisuje `dashboard.md` i `dashboard.json` w katalogu raportów."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name, content in render_outputs(dashboard).items():
        (reports_dir / name).write_bytes(content)


def check_dashboard(dashboard: Dashboard, reports_dir: Path) -> list[str]:
    """Nazwy plików, które różnią się od wygenerowanych (brak pliku też); nic nie zapisuje."""
    diffs = []
    for name, content in render_outputs(dashboard).items():
        path = reports_dir / name
        if not path.is_file() or path.read_bytes() != content:
            diffs.append(name)
    return diffs
