"""Kanoniczny kontrakt rekordów narzędzi MCP (FR-5.2; port `helpers/mcp_runtime.py`).

Kontrakt: docs/contracts/f5.md, sekcja 8.1. O poprawności wyniku rozstrzyga TA tabela (rekordy
z `tools.specs.TOOL_SPECS`), nigdy schemat ogłaszany przez testowany serwer. Surowy
`structuredContent` jest sprawdzany ZANIM cokolwiek uzupełni wartości domyślne.
NASZE: wpis dla `check_termination_condition` i zejście do pojedynczego zagnieżdżonego rekordu
(`TerminationCheck.otif`), z którego trace robi zdarzenie składowe `get_supplier_otif`.
"""

from __future__ import annotations

import json
import types
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from kovalo.mcp_client.errors import McpCallError
from kovalo.tools.specs import TOOL_SPECS

__all__ = ["CANONICAL", "STATUS_RULES", "STATUS_VALUES", "check_record"]

CANONICAL: dict[str, type[BaseModel]] = {spec.name: spec.record for spec in TOOL_SPECS}

# Pola, które nie mogą być null dla danej pary (narzędzie, status): 5 wpisów kursu 1:1 + nasz.
STATUS_RULES: dict[tuple[str, str], tuple[str, ...]] = {
    ("get_supplier_otif", "ok"): ("supplier_id",),
    ("get_invoice_value", "ok"): ("supplier_id", "invoiced_grosze"),
    ("get_contract_terms", "ok"): (
        "supplier_id",
        "source_doc_id",
        "otif_target_pct",
        "penalty_clause",
        "terms_version",
    ),
    ("calculate_otif_penalty", "ok"): (
        "shortfall_points",
        "full_points",
        "penalty_pct_uncapped",
        "penalty_pct",
        "penalty_grosze",
    ),
    ("check_discount_policy", "ok"): ("version",),
    ("check_termination_condition", "ok"): (  # NASZE
        "supplier_id",
        "source_doc_id",
        "otif_below_pct",
        "consecutive_months",
        "otif",
    ),
}
# Pola, które dla danej pary muszą mieć jedną konkretną wartość: kwoty to całkowite grosze PLN.
STATUS_VALUES: dict[tuple[str, str], dict[str, Any]] = {
    ("get_invoice_value", "ok"): {"currency": "PLN"},
    ("calculate_otif_penalty", "ok"): {"currency": "PLN"},
}


def _record_types(annotation: Any) -> list[type[BaseModel]]:
    """Klasy rekordów w adnotacji `Model | None` (pojedynczy zagnieżdżony rekord)."""
    if get_origin(annotation) not in (Union, types.UnionType):
        candidates: tuple[Any, ...] = (annotation,)
    else:
        candidates = get_args(annotation)
    return [a for a in candidates if isinstance(a, type) and issubclass(a, BaseModel)]


def _missing_and_extra(model: type[BaseModel], raw: Any, path: str = "") -> list[str]:
    """Brakujące pola kanonicznego rekordu i pola, których rekord nie ma. Zagnieżdżone rekordy
    (listy modeli, jak kurs, i NASZE: pojedynczy rekord `Model | None`) element po elemencie."""
    if not isinstance(raw, dict):
        return [f"{path or 'result'}: expected an object, got {type(raw).__name__}"]
    problems = [f"{path}{name}: brak" for name in model.model_fields if name not in raw]
    problems += [f"{path}{name}: spoza kontraktu" for name in raw if name not in model.model_fields]
    for name, field in model.model_fields.items():
        value = raw.get(name)
        inner = get_args(field.annotation)
        if (
            isinstance(value, list)
            and inner
            and isinstance(inner[0], type)
            and issubclass(inner[0], BaseModel)
        ):
            for i, item in enumerate(value):
                problems += _missing_and_extra(inner[0], item, f"{path}{name}[{i}].")
        elif isinstance(value, dict):
            nested = _record_types(field.annotation)
            if len(nested) == 1:
                problems += _missing_and_extra(nested[0], value, f"{path}{name}.")
    return problems


def check_record(tool: str, structured: Any, text: list[str]) -> dict[str, Any]:
    """Wiążący rekord udanego wywołania albo McpCallError("validation").

    Kolejność: rekord w ogóle jest → KAŻDY blok tekstu to równa mu kopia JSON (dodatkowy lub inny
    blok jest odrzucany, nigdy pomijany) → są wszystkie pola kanoniczne i żadne inne (zanim
    jakakolwiek wartość domyślna mogłaby je uzupełnić) → ścisłe typy (żadnego bool ani "123"
    zamiast liczby) → reguły statusu (pola różne od null, konkretne wartości, np. waluta PLN).
    Zwraca surowe dane bez zmian, więc null i brak pola dalej da się rozróżnić.
    """
    model = CANONICAL.get(tool)
    if model is None:
        raise McpCallError("validation", f"{tool}: brak kanonicznego kontraktu dla tego narzędzia")
    if structured is None:
        raise McpCallError("validation", f"{tool}: wynik nie ma structuredContent")
    for block in text:
        try:
            mirror = json.loads(block)
        except ValueError:
            raise McpCallError(
                "validation", f"{tool}: blok tekstu nie jest kopią JSON structuredContent"
            ) from None
        if mirror != structured:
            raise McpCallError(
                "validation", f"{tool}: treść tekstowa i structuredContent się nie zgadzają"
            )
    problems = _missing_and_extra(model, structured)
    if problems:
        raise McpCallError("validation", f"{tool}: " + "; ".join(problems[:6]))
    try:
        model.model_validate_json(json.dumps(structured), strict=True)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(map(str, first["loc"]))
        raise McpCallError("validation", f"{tool}: {location}: {first['msg']}") from None
    status = structured.get("status")
    for name in STATUS_RULES.get((tool, status), ()):
        if structured.get(name) is None:
            raise McpCallError(
                "validation", f"{tool}: {name} nie może być null, gdy status to {status!r}"
            )
    for name, value in STATUS_VALUES.get((tool, status), {}).items():
        if structured.get(name) != value:
            raise McpCallError(
                "validation",
                f"{tool}: {name} musi mieć wartość {value!r}, gdy status to {status!r}, "
                f"otrzymano {structured.get(name)!r}",
            )
    return structured
