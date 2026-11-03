"""Warunek rozwiązania umowy z powodu OTIF policzony w kodzie (NASZE, FR-4.7; ADR-0011 pkt 4).

Kontrakt: docs/contracts/f4.md, sekcja 5.5. Próg i liczba miesięcy z `company.termination_terms`
(config.yaml, sprawdzane testem z tekstem CON-101); OTIF z tego samego kodu co narzędzie
get_supplier_otif; porównanie „mniej niż próg” w Decimal (94,0 nie jest mniejsze niż 94).
Funkcja czysta: niczego nie rozwiązuje i nie przesądza o decyzji.
"""

from __future__ import annotations

from decimal import Decimal

from kovalo.company.data import CompanyData, is_month, month_range
from kovalo.company.lookup import get_supplier_otif
from kovalo.company.records import TerminationCheck, TerminationMonth
from kovalo.config import CompanyConfig

__all__ = ["check_termination_condition"]


def _window_start(as_of_month: str, months: int) -> str:
    year, month = int(as_of_month[:4]), int(as_of_month[5:])
    index = year * 12 + (month - 1) - (months - 1)
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def check_termination_condition(
    data: CompanyData, terms: CompanyConfig, supplier: str, as_of_month: str
) -> TerminationCheck:
    """Warunek z umowy: OTIF mniejszy niż próg przez consecutive_months kolejnych miesięcy
    kończących się na as_of_month. condition_met: False, gdy któryś miesiąc z danymi nie jest
    poniżej progu; True, gdy wszystkie mają dane i są poniżej; None przy brakach bez przerwania
    serii."""
    if not is_month(as_of_month):
        return TerminationCheck(
            status="invalid_request",
            as_of_month=as_of_month,
            message=f"as_of_month musi mieć format YYYY-MM, otrzymano {as_of_month!r}",
        )
    other = data.not_otif_contract(supplier)
    if other is not None:
        doc = other.get("source_doc_id")
        return TerminationCheck(
            status="not_applicable",
            source_doc_id=doc,
            as_of_month=as_of_month,
            message=(
                "warunek rozwiązania tej umowy nie jest liczony z miesięcznego OTIF: "
                f"przeczytaj {doc} przez search_documents"
            ),
        )
    sid = data.resolve_supplier(supplier)
    if sid is None:
        return TerminationCheck(
            status="unknown_supplier",
            as_of_month=as_of_month,
            message=f"nie znaleziono dostawcy {supplier!r}",
        )
    name = data.suppliers[sid]
    condition = terms.termination_terms.get(sid)
    if condition is None:
        return TerminationCheck(
            status="not_applicable",
            supplier_id=sid,
            supplier=name,
            as_of_month=as_of_month,
            message=f"w danych nie ma warunku rozwiązania umowy z powodu OTIF dla {name}",
        )

    window = month_range(_window_start(as_of_month, condition.consecutive_months), as_of_month)
    otif = get_supplier_otif(data, sid, window[0], window[-1])
    observed = {o.month: o.otif_pct for o in otif.observations}
    months: list[TerminationMonth] = []
    for month in window:
        value = observed.get(month)
        below = None if value is None else Decimal(str(value)) < condition.otif_below_pct
        months.append(TerminationMonth(month=month, otif_pct=value, below_threshold=below))
    with_data = [m for m in months if m.below_threshold is not None]
    if any(m.below_threshold is False for m in with_data):
        met: bool | None = False
    elif len(with_data) == len(months):
        met = True
    else:
        met = None
    return TerminationCheck(
        status="ok" if with_data else "no_data",
        supplier_id=sid,
        supplier=name,
        source_doc_id=condition.source_doc_id,
        as_of_month=as_of_month,
        otif_below_pct=float(condition.otif_below_pct),
        consecutive_months=condition.consecutive_months,
        notice_days=condition.notice_days,
        months=months,
        missing_months=list(otif.missing_months),
        months_below=sum(1 for m in with_data if m.below_threshold),
        condition_met=met,
        otif=otif,
        terms_version=terms.termination_version,
        message=otif.message,
    )
