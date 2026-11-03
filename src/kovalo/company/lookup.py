"""Odczyty danych firmy: OTIF, faktura, warunki umowy (FR-4.1; krok 1 MP4).

Kontrakt: docs/contracts/f4.md, sekcja 5.4. Każdy wynik jawnie mówi, o co pytano, co znaleziono
i czego brakuje; zwykłe wyniki (brak danych, nieznany dostawca) nigdy nie są wyjątkiem.
"""

from __future__ import annotations

from kovalo.company.data import CompanyData, is_month, month_range
from kovalo.company.records import (
    ContractTermsResult,
    InvoiceResult,
    OtifObservation,
    OtifResult,
)

__all__ = ["get_contract_terms", "get_invoice_value", "get_supplier_otif"]


def _unknown(supplier: str) -> str:
    return f"nie znaleziono dostawcy {supplier!r} (podaj identyfikator, np. SUP-114, albo nazwę)"


def get_supplier_otif(
    data: CompanyData, supplier: str, from_month: str, to_month: str
) -> OtifResult:
    """Miesięczny OTIF dostawcy od from_month do to_month (YYYY-MM, oba włącznie)."""
    requested = {"from": from_month, "to": to_month}
    if not (is_month(from_month) and is_month(to_month)):
        return OtifResult(
            status="invalid_request",
            requested=requested,
            message=(
                f"miesiące muszą mieć format YYYY-MM, otrzymano {from_month!r} i {to_month!r}"
            ),
        )
    if from_month > to_month:
        return OtifResult(
            status="invalid_request",
            requested=requested,
            message=f"początek okresu {from_month} jest późniejszy niż koniec {to_month}",
        )
    sid = data.resolve_supplier(supplier)
    if sid is None:
        return OtifResult(
            status="unknown_supplier", requested=requested, message=_unknown(supplier)
        )
    observations: list[OtifObservation] = []
    missing: list[str] = []
    for month in month_range(from_month, to_month):
        value = data.otif.get((sid, month))
        if value is None:
            missing.append(month)
        else:
            observations.append(OtifObservation(month=month, otif_pct=value))
    message = None
    if missing:
        message = f"brak danych za {', '.join(missing)}"
    return OtifResult(
        status="ok" if observations else "no_data",
        supplier_id=sid,
        supplier=data.suppliers[sid],
        requested=requested,
        observations=observations,
        missing_months=missing,
        complete=not missing,
        message=message,
    )


def get_invoice_value(data: CompanyData, supplier: str, month: str) -> InvoiceResult:
    """Wartość faktury dostawcy za jeden miesiąc (YYYY-MM), w całkowitych groszach."""
    if not is_month(month):
        return InvoiceResult(
            status="invalid_request",
            month=month,
            message=f"miesiąc musi mieć format YYYY-MM, otrzymano {month!r}",
        )
    sid = data.resolve_supplier(supplier)
    if sid is None:
        return InvoiceResult(status="unknown_supplier", month=month, message=_unknown(supplier))
    grosze = data.invoices.get((sid, month))
    if grosze is None:
        return InvoiceResult(
            status="no_data",
            supplier_id=sid,
            supplier=data.suppliers[sid],
            month=month,
            message=f"brak faktury za {month}",
        )
    return InvoiceResult(
        status="ok",
        supplier_id=sid,
        supplier=data.suppliers[sid],
        month=month,
        invoiced_grosze=grosze,
    )


def get_contract_terms(data: CompanyData, supplier: str) -> ContractTermsResult:
    """Warunki miesięcznego OTIF z umowy z dostawcą; brak klauzuli kary to penalty_clause False,
    a stawka i limit None (nigdy 0)."""
    sid = data.resolve_supplier(supplier)
    if sid is not None:
        terms = data.terms.get(sid)
        if terms is None:
            return ContractTermsResult(
                status="not_applicable",
                supplier_id=sid,
                supplier=data.suppliers[sid],
                message=f"w danych nie ma warunków OTIF z umowy z {data.suppliers[sid]}",
            )
        return ContractTermsResult(
            status="ok",
            supplier_id=sid,
            supplier=data.suppliers[sid],
            source_doc_id=terms.get("source_doc_id"),
            otif_target_pct=terms.get("otif_target_pct"),
            penalty_clause=terms.get("penalty_clause"),
            rate_per_full_point_pct=terms.get("rate_per_full_point_pct"),
            cap_pct=terms.get("cap_pct"),
            settlement=terms.get("settlement"),
            terms_version=data.terms_version,
        )
    entry = data.not_otif_contract(supplier)
    if entry is not None:
        return ContractTermsResult(
            status="not_applicable",
            source_doc_id=entry.get("source_doc_id"),
            message=entry.get("reason"),
        )
    return ContractTermsResult(status="unknown_supplier", message=_unknown(supplier))
