"""Zestawy pytań ewaluacyjnych (port helpers/eval_queries.py i helpers/gold.py kursu).

Kontrakty: F1 sekcja 7.3, F2 sekcja 8.1.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

__all__ = [
    "AgentGold",
    "Dialogue",
    "DialogueTurn",
    "EvalQuery",
    "GoldCalculation",
    "GoldQuestion",
    "Need",
    "load_agent_gold",
    "load_conversations",
    "load_dialogues",
    "load_eval_queries",
    "load_gold",
    "load_mp2_retrieval_queries",
]


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


@dataclass(frozen=True)
class GoldQuestion:
    """Pytanie MP2 z faktami wzorcowymi (port helpers/gold.py; listy jak w kursie)."""

    id: str
    question: str
    answerable: bool
    relevant_docs: list[str]
    requires_all: bool
    evidence_facts: list[list[str]]  # każdy fakt: lista akceptowanych wariantów
    key_facts: list[list[str]]
    reference: str


def load_gold(path: Path) -> list[GoldQuestion]:
    """Pytania MP2 (klucz "questions"). Brak pliku → FileNotFoundError; zły kształt → ValueError."""
    path = Path(path)
    items = _read_list(path, "questions")
    try:
        return [GoldQuestion(**q) for q in items]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: każde pytanie musi mieć pola GoldQuestion ({exc})") from exc


class DialogueTurn(BaseModel):
    """Tura dialogu MP3: wiadomość, idealny Turn, odpowiedzi na pauzy, wzorzec (F3, 10.1)."""

    user: str
    understand: dict[str, Any]  # idealny Turn (walidowany przy wczytaniu)
    resume: list[float | int | str] = Field(default_factory=list)
    expect: dict[str, Any]  # kind, fields, request, approvers, pauses, status (podzbiór)

    @field_validator("understand")
    @classmethod
    def _valid_turn(cls, value: dict[str, Any]) -> dict[str, Any]:
        from kovalo.workflow.schemas import Turn

        Turn.model_validate(value)
        return value


class Dialogue(BaseModel):
    id: str
    title: str
    turns: list[DialogueTurn] = Field(min_length=1)


def load_dialogues(path: Path) -> list[Dialogue]:
    """Dialogi MP3 (klucz "dialogues"). Brak pliku → FileNotFoundError; zły kształt albo
    `understand`, który nie jest poprawnym Turn → ValueError z nazwą pliku."""
    path = Path(path)
    items = _read_list(path, "dialogues")
    try:
        return [Dialogue.model_validate(d) for d in items]
    except ValidationError as exc:
        raise ValueError(f"{path}: niepoprawny dialog ({exc})") from exc


# --- wzorzec agenta MP4 (kontrakt F4, 11.1; port helpers/agent_trace.py 1:1) --------------------


class Need(BaseModel):
    """Udane wywołanie, którego wymaga pytanie: narzędzie i to, co musi obejmować jego wynik.
    Albo, z any_of, kilka alternatywnych dróg: potrzeba jest spełniona, gdy spełniona jest jedna
    z nich."""

    tool: str | None = None
    any_of: list[Need] = Field(default_factory=list)
    supplier_id: str | None = None  # dostawca rozpoznany w wynikach
    months: list[str] = Field(default_factory=list)  # miesiące z danymi we wszystkich wywołaniach
    missing: list[str] = Field(default_factory=list)  # miesiące ZGŁOSZONE jako brakujące
    doc_id: str | None = None  # wyszukiwanie: fragment z tego dokumentu


class GoldCalculation(BaseModel):
    """Obliczenie kary wykonane ręcznie, z dostawcą i miesiącem, do których należą jego dane."""

    supplier_id: str
    month: str
    otif_pct: float
    target_pct: float
    rate_per_full_point_pct: float
    cap_pct: float
    invoice_grosze: int


class AgentGold(BaseModel):
    id: str
    question: str
    kind: str  # penalty, termination, no_clause, policy, document, no_data, boundary
    needs: list[Need] = Field(default_factory=list)
    calculations: list[GoldCalculation] = Field(default_factory=list)  # jedyne poprawne obliczenia
    key_facts: list[list[str]] = Field(default_factory=list)  # każdy: akceptowane warianty
    reference: str = ""  # ręcznie napisana oczekiwana odpowiedź, dla ludzi
    approval_claim_phrases: list[str] = Field(default_factory=list)  # z najwyższego poziomu pliku


def load_agent_gold(path: Path) -> list[AgentGold]:
    """data/eval/mini_project_4_questions.yaml → lista AgentGold; brak pliku → FileNotFoundError."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku z pytaniami agenta: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    phrases = raw["approval_claim_phrases"]
    return [AgentGold(**q, approval_claim_phrases=phrases) for q in raw["questions"]]


# --- rozmowy E5 (kontrakt F5, 10.1; port helpers.assistant_eval.load_conversations) -------------


def load_conversations(mp5_path: Path, mp3_path: Path) -> list[Dialogue]:
    """Rozmowy MP5 (C1-C5), potem dialogi MP3 (D1-D8) z nałożonym `mp3_overlays` (głęboka
    kopia; plik MP3 nietknięty). Wynik: 13 Dialogue, 25 tur. `expect.agent` (needs,
    calculations, key_facts) walidowany przy wczytaniu jako AgentGold bez id. Brak pliku →
    FileNotFoundError; zły kształt → ValueError z nazwą pliku."""
    path = Path(mp5_path)
    if not path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku z rozmowami: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw.get("conversations"):
        raise ValueError(f"{path}: oczekiwano niepustej listy 'conversations'")
    dialogues = copy.deepcopy(_read_list(Path(mp3_path), "dialogues"))
    overlays = raw.get("mp3_overlays") or {}
    try:
        for dialogue in dialogues:
            for number, extra in (overlays.get(dialogue["id"]) or {}).items():
                dialogue["turns"][int(number) - 1]["expect"].update(copy.deepcopy(extra))
        result = [
            Dialogue.model_validate(d) for d in [*copy.deepcopy(raw["conversations"]), *dialogues]
        ]
        for loaded in result:
            for turn in loaded.turns:
                agent = turn.expect.get("agent")
                if agent is not None:
                    AgentGold(id=loaded.id, question=turn.user, kind="mp5", **agent)
    except (ValidationError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"{path}: niepoprawna rozmowa ({exc})") from exc
    return result
