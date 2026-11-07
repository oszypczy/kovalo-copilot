# ruff: noqa: RUF002  (polska typografia: półpauza w zakresach)
"""Dashboard ewaluacji (FR-6.4, FR-6.5; kontrakt F6, 8): `reports/dashboard.md` i `.json`
zestawione z raportów E1–E5 i `reports/dashboard.yaml`, bez wywołań LLM-a i bez sieci.

Zależności (kontrakt F6, 2.2): biblioteka standardowa, pydantic, yaml, kovalo.config,
kovalo.text_pl; bez innych pakietów kovalo.eval.* (dashboard czyta pliki, nie uruchamia ewaluacji).
"""

from kovalo.eval.dashboard.build import build_dashboard, write_dashboard
from kovalo.eval.dashboard.spec import DashboardSpec

__all__ = ["DashboardSpec", "build_dashboard", "write_dashboard"]
