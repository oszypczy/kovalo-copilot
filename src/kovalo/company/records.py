"""Rekordy wyników narzędzi danych firmy (FR-4.1, FR-4.2, FR-4.7; port helpers/company.py 1:1).

Kontrakt: docs/contracts/f4.md, sekcje 5.3 i 5.5. Zasady: `status` mówi, co się stało ("ok" albo
zwykła odpowiedź jak "no_data"); jednostki w nazwach pól (`_pct`, `_points`, `_grosze`); kwoty
nigdy nie są typu float; identyfikatory kanoniczne, miesiące YYYY-MM.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "ContractTermsResult",
    "InvoiceResult",
    "OtifObservation",
    "OtifResult",
    "PenaltyCalculation",
    "TerminationCheck",
    "TerminationMonth",
]


class OtifObservation(BaseModel):
    month: str  # YYYY-MM
    otif_pct: float


class OtifResult(BaseModel):
    """get_supplier_otif: miesięczny OTIF dostawcy w danym okresie."""

    status: Literal["ok", "no_data", "unknown_supplier", "invalid_request"]
    supplier_id: str | None = None  # kanoniczny identyfikator, gdy dostawca jest znany
    supplier: str | None = None  # jego nazwa
    requested: dict[str, str]  # {"from": "2026-07", "to": "2026-10"}, tak jak w zapytaniu
    observations: list[OtifObservation] = Field(default_factory=list)  # według miesiąca
    missing_months: list[str] = Field(default_factory=list)  # miesiące z zapytania bez danych
    complete: bool = False  # True tylko wtedy, gdy każdy miesiąc z zapytania ma dane
    message: str | None = None  # dlaczego: przy wszystkim poza kompletnym "ok"


class InvoiceResult(BaseModel):
    """get_invoice_value: ile dostawca zafakturował Kovalo w jednym miesiącu."""

    status: Literal["ok", "no_data", "unknown_supplier", "invalid_request"]
    supplier_id: str | None = None
    supplier: str | None = None
    month: str
    invoiced_grosze: int | None = None
    currency: str = "PLN"
    message: str | None = None


class ContractTermsResult(BaseModel):
    """get_contract_terms: warunki umowy z dostawcą dotyczące miesięcznego OTIF."""

    status: Literal["ok", "unknown_supplier", "not_applicable"]
    supplier_id: str | None = None
    supplier: str | None = None
    source_doc_id: str | None = None  # dokument umowy, z którego pochodzą warunki
    otif_target_pct: float | None = None
    penalty_clause: bool | None = None  # False = umowa NIE MA klauzuli o karze umownej
    rate_per_full_point_pct: float | None = None  # None, gdy nie ma klauzuli (nigdy 0)
    cap_pct: float | None = None
    settlement: str | None = None
    terms_version: str | None = None
    message: str | None = None


class PenaltyCalculation(BaseModel):
    """calculate_otif_penalty: arytmetyka na podanych danych wejściowych, z każdą wartością
    pośrednią."""

    status: Literal["ok", "invalid_request"]
    otif_pct: float | None = None  # dane wejściowe, powtórzone
    target_pct: float | None = None
    rate_per_full_point_pct: float | None = None
    cap_pct: float | None = None
    invoice_grosze: int | None = None
    shortfall_points: float | None = None  # cel - OTIF, nigdy poniżej 0
    full_points: int | None = None  # pełne punkty niedoboru
    penalty_pct_uncapped: float | None = None
    penalty_pct: float | None = None  # po zastosowaniu limitu
    penalty_grosze: int | None = None
    currency: str = "PLN"
    note: str = "arytmetyka na podanych danych wejściowych; nie przesądza, czy kara się należy"
    message: str | None = None


class TerminationMonth(BaseModel):
    """Jeden miesiąc okna warunku rozwiązania umowy (NASZE, FR-4.7)."""

    month: str  # YYYY-MM
    otif_pct: float | None = None  # None: brak danych za ten miesiąc
    below_threshold: bool | None = None  # otif_pct < otif_below_pct (Decimal); None bez danych


class TerminationCheck(BaseModel):
    """check_termination_condition: warunek rozwiązania umowy z powodu OTIF, policzony w kodzie."""

    status: Literal["ok", "no_data", "unknown_supplier", "not_applicable", "invalid_request"]
    supplier_id: str | None = None
    supplier: str | None = None
    source_doc_id: str | None = None  # umowa, z której pochodzi warunek (CON-101)
    as_of_month: str  # tak, jak podano
    otif_below_pct: float | None = None  # próg: warunek spełnia OTIF MNIEJSZY niż ta wartość
    consecutive_months: int | None = None
    notice_days: int | None = None
    months: list[TerminationMonth] = Field(default_factory=list)  # okno, rosnąco
    missing_months: list[str] = Field(default_factory=list)
    months_below: int | None = None  # ile miesięcy z danymi jest poniżej progu
    condition_met: bool | None = None  # None: braki danych, a seria nie jest przerwana
    otif: OtifResult | None = None  # rekord odczytu (zdarzenie składowe w trace)
    terms_version: str | None = None  # company.termination_version
    note: str = (
        "sprawdza tylko warunek OTIF z umowy; nie rozwiązuje umowy i nie przesądza o decyzji"
    )
    message: str | None = None
