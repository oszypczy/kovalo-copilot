"""Dane firmy Kovalo: wczytanie i sprawdzenie data/company/ (FR-4.1; port helpers/company.py 1:1).

Kontrakt: docs/contracts/f4.md, sekcja 5.2. Różnica wobec kursu: `load_company_data` przyjmuje
katalog, a nie Config. Uszkodzony plik to zawsze CompanyDataError, nigdy „brak danych”.
"""

from __future__ import annotations

import csv
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kovalo.config import ConfigError

__all__ = [
    "MONTH",
    "CompanyData",
    "CompanyDataError",
    "is_month",
    "load_company_data",
    "month_range",
]


class CompanyDataError(ConfigError):
    """Brak pliku danych firmy albo plik uszkodzony. Nigdy nie jest zgłaszany jako „brak danych”."""


MONTH = re.compile(r"\d{4}-(0[1-9]|1[0-2])")


def is_month(text: object) -> bool:
    """True dla napisu w formacie YYYY-MM, np. "2026-07"."""
    return isinstance(text, str) and MONTH.fullmatch(text) is not None


def month_range(from_month: str, to_month: str) -> list[str]:
    """Każdy miesiąc od from_month do to_month włącznie: ("2026-11", "2027-01") ->
    ["2026-11", "2026-12", "2027-01"]. Oba w formacie YYYY-MM; pusta lista, gdy
    from_month > to_month."""
    if not (is_month(from_month) and is_month(to_month)):
        raise ValueError(
            f"miesiące muszą mieć format YYYY-MM, otrzymano {from_month!r} i {to_month!r}"
        )
    year, month = int(from_month[:4]), int(from_month[5:])
    months = []
    while f"{year:04d}-{month:02d}" <= to_month:
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def _plain(text: str) -> str:
    """Małe litery, bez polskich znaków (ł -> l), pojedyncze spacje: do dopasowywania nazw."""
    text = unicodedata.normalize("NFKD", text.replace("ł", "l").replace("Ł", "L"))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip().casefold()


@dataclass
class CompanyData:
    """Wczytane i sprawdzone dane firmy. Testy budują małe zestawy bezpośrednio."""

    suppliers: dict[str, str]  # supplier_id -> nazwa
    otif: dict[tuple[str, str], float]  # (supplier_id, month) -> OTIF %
    invoices: dict[tuple[str, str], int]  # (supplier_id, month) -> grosze
    terms: dict[str, dict[str, Any]] = field(default_factory=dict)  # supplier_id -> warunki OTIF
    terms_version: str = ""
    not_otif: list[dict[str, Any]] = field(default_factory=list)  # umowy bez miesięcznego OTIF

    def resolve_supplier(self, text: str) -> str | None:
        """Kanoniczny identyfikator dostawcy na podstawie identyfikatora lub nazwy, bez względu na
        wielkość liter, z polskimi znakami lub bez ("sup-114", "Aqua Krysztal", "Mleczna Dolina");
        None, gdy dostawca jest nieznany albo niejednoznaczny."""
        query = _plain(str(text))
        if not query:
            return None
        for supplier_id, name in self.suppliers.items():
            if query in (_plain(supplier_id), _plain(name)):
                return supplier_id
        hits = [
            sid
            for sid, name in self.suppliers.items()
            if _plain(name) in query or query in _plain(name)
        ]
        return hits[0] if len(hits) == 1 and len(query) >= 4 else None

    def not_otif_contract(self, text: str) -> dict[str, Any] | None:
        """Wpis not_otif dla strony umowy, np. "TransBałt", jeśli `text` ją wymienia."""
        query = _plain(str(text))
        return next(
            (
                entry
                for entry in self.not_otif
                if query and (_plain(entry["name"]) in query or query in _plain(entry["name"]))
            ),
            None,
        )


def _rows(path: Path, columns: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise CompanyDataError(f"Nie znaleziono pliku danych firmy: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != columns:
            raise CompanyDataError(
                f"{path.name}: oczekiwano kolumn {columns}, znaleziono {reader.fieldnames}"
            )
        return list(reader)


def load_company_data(directory: Path) -> CompanyData:
    """Wczytuje data/company/ (settings.paths.company_data_dir) i sprawdza dane. Przy każdym
    błędzie zgłasza CompanyDataError z nazwą pliku i wiersza: zły miesiąc, wartość, która nie
    jest skończonym procentem, kwota, która nie jest całkowitą liczbą groszy, waluta inna niż
    PLN, powtórzona para dostawca-miesiąc, nieznany dostawca."""
    folder = Path(directory)
    suppliers = {
        row["supplier_id"]: row["name"]
        for row in _rows(folder / "suppliers.csv", ["supplier_id", "name", "contract_doc_id"])
    }

    def key(row: dict[str, str], where: str) -> tuple[str, str]:
        if row["supplier_id"] not in suppliers:
            raise CompanyDataError(f"{where}: nieznany dostawca {row['supplier_id']!r}")
        if not is_month(row["month"]):
            raise CompanyDataError(
                f"{where}: miesiąc musi mieć format YYYY-MM, otrzymano {row['month']!r}"
            )
        return row["supplier_id"], row["month"]

    otif: dict[tuple[str, str], float] = {}
    otif_rows = _rows(folder / "supplier_otif.csv", ["supplier_id", "month", "otif_pct", "source"])
    for n, row in enumerate(otif_rows, start=2):
        where = f"supplier_otif.csv, wiersz {n}"
        k = key(row, where)
        try:
            value = float(row["otif_pct"])
        except ValueError:
            raise CompanyDataError(
                f"{where}: otif_pct nie jest liczbą: {row['otif_pct']!r}"
            ) from None
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise CompanyDataError(
                f"{where}: otif_pct musi mieścić się w zakresie od 0 do 100, otrzymano {value}"
            )
        if k in otif:
            raise CompanyDataError(f"{where}: zduplikowany wiersz dla {k[0]} {k[1]}")
        otif[k] = value

    invoices: dict[tuple[str, str], int] = {}
    invoice_rows = _rows(
        folder / "supplier_invoices.csv",
        ["supplier_id", "month", "invoiced_grosze", "currency", "source"],
    )
    for n, row in enumerate(invoice_rows, start=2):
        where = f"supplier_invoices.csv, wiersz {n}"
        k = key(row, where)
        if not row["invoiced_grosze"].isdigit():
            raise CompanyDataError(
                f"{where}: invoiced_grosze musi być całkowitą liczbą groszy, "
                f"otrzymano {row['invoiced_grosze']!r}"
            )
        if row["currency"] != "PLN":
            raise CompanyDataError(f"{where}: waluta musi być PLN, otrzymano {row['currency']!r}")
        if k in invoices:
            raise CompanyDataError(f"{where}: zduplikowany wiersz dla {k[0]} {k[1]}")
        invoices[k] = int(row["invoiced_grosze"])

    terms_path = folder / "contract_terms.yaml"
    if not terms_path.is_file():
        raise CompanyDataError(f"Nie znaleziono pliku danych firmy: {terms_path}")
    try:
        terms = yaml.safe_load(terms_path.read_text(encoding="utf-8"))
        otif_terms = terms["otif_terms"]
        version = terms["version"]
    except (yaml.YAMLError, KeyError, TypeError) as exc:
        raise CompanyDataError(f"contract_terms.yaml: uszkodzony plik ({exc})") from None
    unknown = set(otif_terms) - set(suppliers)
    if unknown:
        raise CompanyDataError(f"contract_terms.yaml: nieznani dostawcy {sorted(unknown)}")
    return CompanyData(
        suppliers=suppliers,
        otif=otif,
        invoices=invoices,
        terms=otif_terms,
        terms_version=version,
        not_otif=terms.get("not_otif", []),
    )
