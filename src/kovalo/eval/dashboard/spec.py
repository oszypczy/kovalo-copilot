# ruff: noqa: RUF002  (polska typografia: półpauza w zakresach)
"""`reports/dashboard.yaml`: ręczna część dashboardu ewaluacji (kontrakt F6, 8.1).

Plik wskazuje raporty etapów E1–E5 i dopisuje to, czego nie ma w JSON-ach raportów: liczby
oryginalnego przebiegu na żywo (gdy JSON jest powtórką z cache), uwagi i zużycie darmowego
limitu Gemini na etap projektu. Każdy błąd (brak pliku, zły YAML, nieznany klucz, powtórzony
etap) to `ConfigError`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError, model_validator

from kovalo.config import ConfigError

__all__ = ["DashboardSpec", "FirstLiveRun", "GeminiUsageEntry", "StageSpec", "load_spec"]

StageId = Literal["e1", "e2", "e3", "e4", "e5"]


class FirstLiveRun(BaseModel, extra="forbid"):
    """Liczby oryginalnego przebiegu na żywo, gdy JSON raportu opisuje powtórkę z cache."""

    run_id: str
    network_calls: int
    cached_calls: int = 0
    failed_attempts: int = 0
    seconds: float | None = None
    source: str


class StageSpec(BaseModel, extra="forbid"):
    stage: StageId
    report: str  # JSON raportu, względem katalogu raportów
    markdown: str  # raport Markdown, względem katalogu raportów
    related: list[str] = []
    first_live: FirstLiveRun | None = None
    notes: list[str] = []


class GeminiUsageEntry(BaseModel, extra="forbid"):
    stage: str
    calls: int | None
    note: str | None = None


class DashboardSpec(BaseModel, extra="forbid"):
    version: Literal[1]
    stages: list[StageSpec]
    gemini_usage: list[GeminiUsageEntry] = []

    @model_validator(mode="after")
    def _unique_stages(self) -> DashboardSpec:
        seen: set[str] = set()
        for spec in self.stages:
            if spec.stage in seen:
                raise ValueError(f"etap {spec.stage} powtórzony w dashboard.yaml")
            seen.add(spec.stage)
        return self


def load_spec(path: Path) -> DashboardSpec:
    """Wczytuje `dashboard.yaml`; każdy błąd pliku albo treści → `ConfigError`."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Brak specyfikacji dashboardu {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: niepoprawny YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: oczekiwano mapy z kluczami version, stages, gemini_usage")
    try:
        return DashboardSpec.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path}: niepoprawna specyfikacja dashboardu:\n{exc}") from exc
