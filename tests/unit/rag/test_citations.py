"""Normalizacja odwołań (FR-2.2; kontrakt F2, 7.2; port testów kroku 2 MP2)."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("[PL-601]", "PL-601"),
        (" pol-001 ", "POL-001"),
        ("CON-101#ctx003", "CON-101"),
        ("[CON-101#ctx003]", "CON-101"),
        ("con-101#CTX003", "CON-101"),
        ("TST-001", "TST-001"),
        ("CON-101#003", "CON-101#003"),
        ("CON-1019", "CON-1019"),
        ("XTST-001", "XTST-001"),
        ("TST-0019", "TST-0019"),
        ("", None),
        ("   ", None),
    ],
)
def test_normalize_citation(raw, expected):
    from kovalo.rag.citations import normalize_citation

    assert normalize_citation(raw) == expected


def test_normalize_citations_dedup_order_and_none():
    from kovalo.rag.citations import normalize_citations

    assert normalize_citations(["[TST-001]", " tst-002 ", "TST-001#ctx000", ""]) == [
        "TST-001",
        "TST-002",
    ]
    assert normalize_citations([]) == []


def test_unknown_ids_are_not_repaired():
    from kovalo.rag.citations import normalize_citations

    assert normalize_citations(["TST-0019", "XTST-001", "TST-001"]) == [
        "TST-0019",
        "XTST-001",
        "TST-001",
    ]
