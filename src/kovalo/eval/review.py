"""Komentarze z przeglądu ręcznego: `<out>/review.yaml` (kontrakt F2, 14.3).

Komentarze żyją w pliku wejściowym, a generator raportu wstawia je do sekcji
„Rozbieżności przejrzane ręcznie”. Wygenerowanego Markdownu nie edytujemy ręcznie.
Klucz wpisu: etykieta wariantu `<strona>-<wariant>` + id pytania. Moduł ogólny (E2-E5),
bez LangChaina.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from kovalo.config import ConfigError

__all__ = [
    "REVIEW_FILE",
    "Review",
    "ReviewItem",
    "answer_digest",
    "check_review_keys",
    "load_review",
]

REVIEW_FILE = "review.yaml"

Label = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+-[a-z0-9]+$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{12}$")]


class ReviewItem(BaseModel):
    """Komentarz do jednego pytania w jednym albo kilku wariantach."""

    model_config = ConfigDict(extra="forbid")

    variants: list[Label] = Field(min_length=1)
    id: str = Field(min_length=1)
    comment: str = Field(min_length=1)
    answer_digest: dict[Label, Digest] = {}

    @field_validator("comment")
    @classmethod
    def _comment_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("komentarz nie może być pusty")
        return value.strip("\n").rstrip()

    @model_validator(mode="after")
    def _digest_keys_in_variants(self) -> ReviewItem:
        extra = sorted(set(self.answer_digest) - set(self.variants))
        if extra:
            raise ValueError(f"answer_digest ma etykiety spoza variants: {', '.join(extra)}")
        return self


class Review(BaseModel):
    """Zawartość `review.yaml` (version 1)."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    reviewed_on: date
    intro: str = ""
    items: list[ReviewItem] = []
    notes: list[str] = []

    @model_validator(mode="after")
    def _unique_pairs(self) -> Review:
        seen: set[tuple[str, str]] = set()
        for i, item in enumerate(self.items):
            for label in item.variants:
                pair = (label, item.id)
                if pair in seen:
                    raise ValueError(f"items.{i}: para ({label}, {item.id}) powtórzona")
                seen.add(pair)
        return self

    def find(self, variant: str, id: str) -> ReviewItem | None:
        return next((i for i in self.items if i.id == id and variant in i.variants), None)


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"]) or "(plik)"
        lines.append(f"  {loc}: {error['msg']}")
    return "\n".join(lines)


def load_review(path: Path) -> Review | None:
    """Brak pliku → None. Błąd YAML albo schematu → ConfigError z nazwą pliku i polem."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: błędny YAML: {exc}") from exc
    try:
        return Review.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path}: błędny plik przeglądu:\n{_format_errors(exc)}") from exc


def check_review_keys(review: Review, *, variants: Collection[str], ids: Collection[str]) -> None:
    """Etykieta albo id spoza znanych → ConfigError (literówka nie gubi komentarza po cichu)."""
    known_variants, known_ids = set(variants), set(ids)
    bad_labels = sorted({v for item in review.items for v in item.variants} - known_variants)
    bad_ids = sorted({item.id for item in review.items} - known_ids)
    problems = []
    if bad_labels:
        problems.append(
            f"nieznane etykiety wariantów: {', '.join(bad_labels)} "
            f"(znane: {', '.join(sorted(known_variants))})"
        )
    if bad_ids:
        problems.append(
            f"nieznane id pytań: {', '.join(bad_ids)} (znane: {', '.join(sorted(known_ids))})"
        )
    if problems:
        raise ConfigError(f"{REVIEW_FILE}: " + "; ".join(problems))


def answer_digest(answer: str | None, error: str | None = None) -> str:
    """Skrót odpowiedzi (albo błędu, gdy odpowiedzi brak): 12 znaków hex sha256."""
    text = answer if answer is not None else error or ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
