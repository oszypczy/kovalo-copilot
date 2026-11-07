"""Pomocniki testów dashboardu (kontrakt F6, 8, 12): specyfikacja z 8.1 i katalog raportów
zbudowany z wycinków `tests/fixtures/dashboard/e*.json`.

Moduł nie importuje kodu produkcyjnego na poziomie modułu, więc importuje się przed implementacją.
Współdzielony przez `tests/unit/eval/dashboard/` i `tests/unit/test_cli.py`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "dashboard"

# Ścieżki raportów względem katalogu raportów, jak w `reports/dashboard.yaml` (kontrakt F6, 8.1).
REPORT_PATHS = {
    "e1": "mp1/2026-11-02-e1",
    "e2": "mp2/2026-11-02-e2",
    "e3": "mp3/2026-11-03-e3",
    "e4": "mp4/2026-11-03-e4",
    "e5": "mp5/2026-11-04-e5",
}
STAGES = tuple(REPORT_PATHS)

SPEC_YAML = """\
version: 1
stages:
  - stage: e1
    report: mp1/2026-11-02-e1.json
    markdown: mp1/2026-11-02-e1.md
  - stage: e2
    report: mp2/2026-11-02-e2.json
    markdown: mp2/2026-11-02-e2.md
  - stage: e3
    report: mp3/2026-11-03-e3.json
    markdown: mp3/2026-11-03-e3.md
  - stage: e4
    report: mp4/2026-11-03-e4.json
    markdown: mp4/2026-11-03-e4.md
    related: [mp4/2026-11-04-e4.md]
    notes:
      - "B1: agent nie wywołał check_discount_policy (testy 1 i 3 niespełnione)."
  - stage: e5
    report: mp5/2026-11-04-e5.json
    markdown: mp5/2026-11-04-e5.md
    first_live:
      run_id: 20261104T092904-0f4dd55b
      network_calls: 30
      cached_calls: 4
      failed_attempts: 1
      seconds: 132
      source: reports/mp5/review.yaml
    notes:
      - "Tokeny i koszt E5 liczą też tury z cache (około +20%, R1-3 z F5)."
gemini_usage:
  - {stage: F3, calls: 18, note: "16 + 2 na przeklikanie D2 w UI (07-roadmap)"}
  - {stage: F4, calls: 47}
  - {stage: F5, calls: 55}
  - {stage: F6, calls: null, note: "uzupełnić po H1 (pokaz H2 osobno)"}
"""


def fixture_json(stage: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{stage}.json").read_text(encoding="utf-8"))


def make_reports_dir(
    root: Path, *, spec_yaml: str = SPEC_YAML, stages: tuple[str, ...] = STAGES
) -> Path:
    """Katalog raportów: dashboard.yaml, JSON-y etapów (z wycinków) i puste raporty Markdown."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "dashboard.yaml").write_text(spec_yaml, encoding="utf-8")
    for stage in stages:
        base = root / REPORT_PATHS[stage]
        base.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURES / f"{stage}.json", base.with_suffix(".json"))
        base.with_suffix(".md").write_text(f"# {stage}\n", encoding="utf-8")
    related = root / "mp4" / "2026-11-04-e4.md"
    related.parent.mkdir(parents=True, exist_ok=True)
    related.write_text("# E4 przez MCP\n", encoding="utf-8")
    return root
