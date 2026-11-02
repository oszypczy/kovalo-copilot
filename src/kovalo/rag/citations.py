"""Normalizacja odwołań do źródeł (FR-2.2; kontrakt F2, 7.2; wymagania kroku 2 MP2 kursu).

Naprawiamy tylko formę zapisu (nawiasy, wielkość liter, spacje, przyrostek chunku z kontekstem).
Wszystko inne zostaje bez zmian, więc nieznany identyfikator dalej jest nieprawidłowy.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

__all__ = ["DOC_ID", "normalize_citation", "normalize_citations"]

DOC_ID = r"[A-Z]{2,4}-\d{3}"
_CITATION = re.compile(rf"(?P<doc>{DOC_ID})(?:#CTX\d{{3}})?")


def normalize_citation(raw: str) -> str | None:
    """'[PL-601]' → 'PL-601'; ' pol-001 ' → 'POL-001'; 'CON-101#ctx003' → 'CON-101';
    'CON-101#003' i 'CON-1019' bez zmian (wielkimi literami); pusty napis → None."""
    s = raw.strip().upper()
    if not s:
        return None
    inner = s[1:-1].strip() if s.startswith("[") and s.endswith("]") else s
    match = _CITATION.fullmatch(inner)
    return match.group("doc") if match else s


def normalize_citations(raw: Sequence[str]) -> list[str]:
    """normalize_citation dla każdego elementu, bez None i duplikatów, w kolejności wystąpienia."""
    seen: dict[str, None] = {}
    for item in raw:
        normalized = normalize_citation(item)
        if normalized is not None:
            seen.setdefault(normalized, None)
    return list(seen)
