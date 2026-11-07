"""Modele dashboardu ewaluacji (kontrakt F6, 8.2): to, co trafia do `reports/dashboard.json`.

Liczby w `MetricRow` są gotowym tekstem w zapisie raportów („8/9”, „0,9”, „$17,78”), bo dashboard
zestawia wyniki, a nie liczy ich od nowa; liczby w `RunCalls`, `StageCost` i `MonthlyCost` zostają
liczbami (render zapisuje je po polsku przez `kovalo.text_pl`).
"""

from __future__ import annotations

import datetime
from typing import Literal

from pydantic import BaseModel

from kovalo.eval.dashboard.spec import GeminiUsageEntry

__all__ = [
    "Dashboard",
    "GeminiUsageEntry",
    "MetricRow",
    "MonthlyCost",
    "RunCalls",
    "StageCost",
    "StageSummary",
]


class MetricRow(BaseModel):
    name: str  # po polsku, jak w raporcie Markdown etapu
    reference: str | None  # kurs (model referencyjny), zapis jak w raporcie: "8/9", "0,9", "$17,78"
    replay: str | None  # bez LLM-a: przeliczenie / odtworzenie
    live: str | None  # na żywo (Gemini)
    target: str | None  # cel z 06-evaluation
    met: bool | None  # None: brak celu albo brak przebiegu


class RunCalls(BaseModel):
    planned: int | None
    network: int | None
    cached: int | None
    failed: int | None
    seconds: float | None
    source: Literal["report", "dashboard.yaml"]
    run_id: str | None = None  # przebieg, którego dotyczą liczby (addytywnie wobec 8.2)
    origin: str | None = None  # plik źródłowy liczb przy source="dashboard.yaml"


class StageCost(BaseModel):
    tokens_in: int | None
    tokens_out: int | None
    ours_cents: float | None
    reference_cents: float | None
    course_cents: float | None
    includes_cached: bool  # True dla E5 (R1-3 z F5)


class StageSummary(BaseModel):
    stage: str
    title: str
    date: datetime.date
    model: str | None
    embedding_model: str | None
    report_md: str
    related_md: list[str]
    headline: str
    metrics: list[MetricRow]
    calls: RunCalls | None
    cost: StageCost | None
    notes: list[str]


class MonthlyCost(BaseModel):  # scenariusz kursu: 400 pytań/dzień, 21 dni (E1)
    reference_rag_usd: float
    reference_full_usd: float
    ours_rag_usd: float | None
    ours_full_usd: float | None
    ours_model: str | None


class Dashboard(BaseModel):
    version: Literal[1] = 1
    as_of: datetime.date  # najnowsza data raportu, nie data generowania (idempotencja)
    stages: list[StageSummary]
    monthly_cost: MonthlyCost | None
    gemini_usage: list[GeminiUsageEntry]
