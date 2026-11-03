"""Pakiet `company` (MP4): dane firmy, odczyty, kara OTIF, warunek rozwiązania umowy.

Kontrakt: docs/contracts/f4.md, sekcja 5. Bez LangChaina, LangGrapha, kovalo.llm i kovalo.knowledge.
"""

from kovalo.company.data import (
    CompanyData,
    CompanyDataError,
    is_month,
    load_company_data,
    month_range,
)
from kovalo.company.lookup import get_contract_terms, get_invoice_value, get_supplier_otif
from kovalo.company.penalty import calculate_otif_penalty
from kovalo.company.records import (
    ContractTermsResult,
    InvoiceResult,
    OtifObservation,
    OtifResult,
    PenaltyCalculation,
    TerminationCheck,
    TerminationMonth,
)
from kovalo.company.termination import check_termination_condition

__all__ = [
    "CompanyData",
    "CompanyDataError",
    "ContractTermsResult",
    "InvoiceResult",
    "OtifObservation",
    "OtifResult",
    "PenaltyCalculation",
    "TerminationCheck",
    "TerminationMonth",
    "calculate_otif_penalty",
    "check_termination_condition",
    "get_contract_terms",
    "get_invoice_value",
    "get_supplier_otif",
    "is_month",
    "load_company_data",
    "month_range",
]
