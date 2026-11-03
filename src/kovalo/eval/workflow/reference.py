"""Przebieg referencyjny kursu MP3 (stałe; źródło: docs/course/mp3_README.md, kontrakt F3, 10.4).

`run_dialogues.py` kursu z LLM-em referencyjnym: każdy etap 100%, 0 błędów, 13 wywołań
`understand` + 1 odpowiedź RAG, zmierzony koszt 1,65 centa. Nazwę modelu referencyjnego
bierzemy z configu (`reference_llm.model`, PINNED), nigdy z literału w kodzie.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["COURSE_E3", "COURSE_E3_RUN_ID", "course_e3"]

COURSE_E3_RUN_ID = "20261006T212757-fe6041ef"

if TYPE_CHECKING:
    COURSE_E3: dict[str, Any]  # liczony leniwie w __getattr__ (model z configu)


def course_e3(reference_model: str) -> dict[str, Any]:
    """Wynik referencyjny kursu z nazwą modelu z configu."""
    return {
        "run_id": COURSE_E3_RUN_ID,
        "provider": "openai",
        "model": reference_model,
        "route": [13, 13],
        "fields": [11, 11],
        "request": [8, 8],
        "approvers": [11, 11],
        "pauses": [13, 13],
        "status": [13, 13],
        "errors": 0,
        "cost_cents": 1.65,
        "calls": {"understand": 13, "answer": 1},
    }


def __getattr__(name: str) -> Any:
    """COURSE_E3 liczony przy pierwszym odczycie (model z get_settings(), bez literału)."""
    if name == "COURSE_E3":
        from kovalo.config import get_settings

        return course_e3(get_settings().reference_llm.model)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
