"""Specyfikacje narzędzi agenta: jedna dla LangChaina i MCP (FR-4.3; ADR-0011 pkt 2 i 3).

Kontrakt: docs/contracts/f4.md, sekcja 6.3. Każde narzędzie to zwykła funkcja z adnotacjami
`Annotated[typ, Field(description=...)]`, docstringiem (opis dla modelu) i rekordem Pydantic
jako typem wyniku. Zależności (dane, retriever, polityka, top_k) idą przez ToolContext, nigdy
jako argumenty modelu. Żadne narzędzie nie zatwierdza, nie zapisuje i nie obiecuje.
Bez LangChaina: F5 rejestruje te same funkcje w FastMCP.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from kovalo import company
from kovalo.company import (
    CompanyData,
    ContractTermsResult,
    InvoiceResult,
    OtifResult,
    PenaltyCalculation,
    TerminationCheck,
)
from kovalo.config import CompanyConfig, PolicyConfig
from kovalo.policy import check as policy_check
from kovalo.policy.check import PolicyCheck
from kovalo.tools.search import Retriever, SearchResult
from kovalo.tools.search import search_documents as _search

__all__ = [
    "COURSE_TOOLS",
    "ORIGINS",
    "OUR_TOOLS",
    "TOOL_SPECS",
    "Origin",
    "ToolContext",
    "ToolSpec",
    "ToolsVariant",
    "tool_specs",
]

Origin = Literal["data", "calculation"]
ToolsVariant = Literal["course", "ours"]


@dataclass(frozen=True)
class ToolContext:
    """Zależności narzędzi (nigdy argumenty modelu)."""

    data: CompanyData
    policy: PolicyConfig
    company: CompanyConfig  # termination_terms
    retriever: Retriever | None  # None → search_documents zgłasza RuntimeError
    top_k: int  # settings.retrieval.top_k (PINNED 3)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    origin: Origin
    record: type[BaseModel]  # typ wyniku (F5: outputSchema / structuredContent)
    make: Callable[[ToolContext], Callable[..., BaseModel]]


Supplier = Annotated[
    str,
    Field(description="dostawca: identyfikator (np. SUP-114) albo nazwa (np. Mleczna Dolina)"),
]


_MONTH = "miesiąc w formacie YYYY-MM, np. 2026-07"
FromMonth = Annotated[str, Field(description=f"pierwszy miesiąc okresu (włącznie): {_MONTH}")]
ToMonth = Annotated[str, Field(description=f"ostatni miesiąc okresu (włącznie): {_MONTH}")]
Month = Annotated[str, Field(description=f"miesiąc faktury: {_MONTH}")]
AsOfMonth = Annotated[str, Field(description=f"ostatni miesiąc sprawdzanej serii: {_MONTH}")]


def _search_tool(ctx: ToolContext) -> Callable[..., BaseModel]:
    def search_documents(
        question: Annotated[
            str,
            Field(description="pytanie albo fraza do wyszukania w dokumentach, po polsku"),
        ],
    ) -> SearchResult:
        """Przeszukuje dokumenty Kovalo (umowy, polityki, notatki ze spotkań, raporty) i zwraca
        surowe fragmenty od najlepiej pasującego: passages z doc_id (identyfikator dokumentu,
        np. CON-101, do cytowania), chunk_id i text. Status: ok, no_results (nic nie
        znaleziono) albo invalid_request (puste pytanie)."""
        if ctx.retriever is None:
            raise RuntimeError("indeks dokumentów niedostępny: wyszukiwanie nie działa")
        return _search(ctx.retriever, question, ctx.top_k)

    return search_documents


def _otif_tool(ctx: ToolContext) -> Callable[..., BaseModel]:
    def get_supplier_otif(
        supplier: Supplier, from_month: FromMonth, to_month: ToMonth
    ) -> OtifResult:
        """Miesięczny OTIF dostawcy (procent dostaw na czas i w komplecie, pole otif_pct)
        w okresie od from_month do to_month, oba w formacie YYYY-MM. Zwraca observations
        (miesiąc i otif_pct, rosnąco), missing_months (miesiące bez danych) i complete.
        Status: ok (choć jeden miesiąc ma dane; braki są w missing_months), no_data (żaden
        miesiąc nie ma danych), unknown_supplier albo invalid_request (zły format miesiąca)."""
        return company.get_supplier_otif(ctx.data, supplier, from_month, to_month)

    return get_supplier_otif


def _invoice_tool(ctx: ToolContext) -> Callable[..., BaseModel]:
    def get_invoice_value(supplier: Supplier, month: Month) -> InvoiceResult:
        """Wartość faktury dostawcy dla Kovalo za jeden miesiąc (YYYY-MM): invoiced_grosze
        to całkowita liczba groszy (100 groszy = 1 PLN), currency PLN. Status: ok, no_data
        (brak faktury za ten miesiąc), unknown_supplier albo invalid_request."""
        return company.get_invoice_value(ctx.data, supplier, month)

    return get_invoice_value


def _terms_tool(ctx: ToolContext) -> Callable[..., BaseModel]:
    def get_contract_terms(supplier: Supplier) -> ContractTermsResult:
        """Warunki umowy z dostawcą dotyczące miesięcznego OTIF: source_doc_id (umowa),
        otif_target_pct (cel w procentach), penalty_clause (czy umowa ma klauzulę kary umownej;
        false = kara nie przysługuje), rate_per_full_point_pct (procent faktury za każdy pełny
        punkt poniżej celu), cap_pct (limit kary w procentach faktury), settlement. Status: ok,
        not_applicable (umowa nie mierzy miesięcznego OTIF albo brak warunków; zob. message)
        albo unknown_supplier."""
        return company.get_contract_terms(ctx.data, supplier)

    return get_contract_terms


def _penalty_tool(ctx: ToolContext) -> Callable[..., BaseModel]:
    def calculate_otif_penalty(
        otif_pct: Annotated[float, Field(description="OTIF za miesiąc w procentach, np. 94.6")],
        target_pct: Annotated[
            float, Field(description="cel OTIF z umowy w procentach (otif_target_pct)")
        ],
        rate_per_full_point_pct: Annotated[
            float,
            Field(description="procent faktury za każdy pełny punkt procentowy poniżej celu"),
        ],
        cap_pct: Annotated[float, Field(description="limit kary w procentach faktury")],
        invoice_grosze: Annotated[
            int, Field(description="wartość faktury za ten miesiąc: całkowita liczba groszy")
        ],
    ) -> PenaltyCalculation:
        """Liczy karę umowną za niski OTIF dokładnie, z każdą wartością pośrednią:
        shortfall_points (punkty poniżej celu), full_points (pełne punkty), penalty_pct_uncapped,
        penalty_pct (po limicie) i penalty_grosze (kara w całkowitych groszach). Tylko
        arytmetyka: to, czy kara się należy, wynika z get_contract_terms (penalty_clause).
        Status: ok albo invalid_request (zła wartość wejściowa, zob. message)."""
        return company.calculate_otif_penalty(
            otif_pct, target_pct, rate_per_full_point_pct, cap_pct, invoice_grosze
        )

    return calculate_otif_penalty


def _policy_tool(ctx: ToolContext) -> Callable[..., BaseModel]:
    def check_discount_policy(
        discount_pct: Annotated[float, Field(description="rabat w procentach, np. 22")],
        margin_after_pct: Annotated[
            float | None,
            Field(description="marża po rabacie w procentach; null, gdy nieznana"),
        ] = None,
        clearance: Annotated[
            bool, Field(description="czy to wyprzedaż końcówek (clearance)")
        ] = False,
        strategic_account: Annotated[
            bool, Field(description="czy klient jest klientem strategicznym")
        ] = False,
        competitor_offer: Annotated[
            bool, Field(description="czy rabat odpowiada na ofertę konkurencji")
        ] = False,
        contractual_volume_rebate: Annotated[
            bool, Field(description="czy to rabat wolumenowy z umowy z klientem")
        ] = False,
    ) -> PolicyCheck:
        """Ustala według polityki rabatowej POL-001, kto musiałby zaakceptować rabat: role
        w kolejności pytania o zgodę, reasons (uzasadnienie), version, margin_checked i partial
        (true, gdy marża była nieznana). Tylko informacja, niczego nie zatwierdza: wnioski
        przechodzą przez proces akceptacji w Kovalo. Status: ok, out_of_scope (rabat
        wolumenowy z umowy nie podlega tej polityce) albo invalid_request."""
        return policy_check.check_discount_policy(
            ctx.policy,
            discount_pct,
            margin_after_pct,
            clearance=clearance,
            strategic_account=strategic_account,
            competitor_offer=competitor_offer,
            contractual_volume_rebate=contractual_volume_rebate,
        )

    return check_discount_policy


def _termination_tool(ctx: ToolContext) -> Callable[..., BaseModel]:
    def check_termination_condition(supplier: Supplier, as_of_month: AsOfMonth) -> TerminationCheck:
        """Sprawdza w kodzie warunek rozwiązania umowy z powodu OTIF (trzy kolejne miesiące
        kończące się na as_of_month, porównanie „mniej niż próg”), sam pobiera OTIF tych
        miesięcy; zwraca każdy miesiąc (months: month, otif_pct, below_threshold), próg
        otif_below_pct, liczbę miesięcy poniżej progu months_below, notice_days i condition_met
        (null przy brakach danych). Źródło warunku to source_doc_id: jego treść cytuj po
        wyszukaniu przez search_documents. Niczego nie rozwiązuje. Status: ok, no_data,
        not_applicable (umowa bez takiego warunku; zob. message), unknown_supplier albo
        invalid_request."""
        return company.check_termination_condition(ctx.data, ctx.company, supplier, as_of_month)

    return check_termination_condition


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("search_documents", "data", SearchResult, _search_tool),
    ToolSpec("get_supplier_otif", "data", OtifResult, _otif_tool),
    ToolSpec("get_invoice_value", "data", InvoiceResult, _invoice_tool),
    ToolSpec("get_contract_terms", "data", ContractTermsResult, _terms_tool),
    ToolSpec("calculate_otif_penalty", "calculation", PenaltyCalculation, _penalty_tool),
    ToolSpec("check_discount_policy", "calculation", PolicyCheck, _policy_tool),
    ToolSpec("check_termination_condition", "calculation", TerminationCheck, _termination_tool),
)
ORIGINS: dict[str, Origin] = {spec.name: spec.origin for spec in TOOL_SPECS}
OUR_TOOLS: frozenset[str] = frozenset(ORIGINS)
COURSE_TOOLS: frozenset[str] = OUR_TOOLS - {"check_termination_condition"}


def tool_specs(variant: ToolsVariant) -> list[ToolSpec]:
    """course: 6 narzędzi kursu; ours: + check_termination_condition (kolejność TOOL_SPECS)."""
    if variant not in ("course", "ours"):
        raise ValueError(f"Nieznany wariant narzędzi: {variant!r} (dozwolone: course, ours)")
    names = COURSE_TOOLS if variant == "course" else OUR_TOOLS
    return [spec for spec in TOOL_SPECS if spec.name in names]
