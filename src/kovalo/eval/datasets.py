"""Zestawy pytań ewaluacyjnych (port helpers/eval_queries.py kursu; kontrakt F1, 7.3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["EvalQuery", "load_eval_queries", "load_mp2_retrieval_queries"]


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    question: str
    relevant_docs: tuple[str, ...]


def _read_list(path: Path, key: str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku z pytaniami ewaluacyjnymi: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = raw.get(key) if isinstance(raw, dict) else None
    if not items:
        raise ValueError(f"{path}: oczekiwano niepustej listy {key!r}")
    return list(items)


def load_eval_queries(path: Path) -> list[EvalQuery]:
    """Pytania MP1 (klucz "queries")."""
    return [
        EvalQuery(
            query_id=str(q["query_id"]),
            question=str(q["question"]),
            relevant_docs=tuple(str(d) for d in q["relevant_docs"]),
        )
        for q in _read_list(path, "queries")
    ]


def load_mp2_retrieval_queries(path: Path) -> list[EvalQuery]:
    """Pytania MP2 z odpowiedzią w korpusie (answerable: true), id → query_id."""
    return [
        EvalQuery(
            query_id=str(q["id"]),
            question=str(q["question"]),
            relevant_docs=tuple(str(d) for d in q["relevant_docs"]),
        )
        for q in _read_list(path, "questions")
        if q.get("answerable") is True
    ]
