"""Wczytywanie korpusu (FR-1.1, krok 1 kursu; kontrakt F1, sekcja 5.1).

Zamiast DirectoryLoader/TextLoader (langchain-community) czytamy pliki przez pathlib (ADR-0007);
wynik jest ten sam: Document z tekstem pliku bez zmian i wierszem katalogu w metadanych.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel

from kovalo.config import Settings
from kovalo.cost.tokens import count_tokens

__all__ = ["CorpusError", "CorpusStats", "corpus_stats", "load_corpus", "load_documents"]


class CorpusError(ValueError):
    """Błąd danych korpusu. Komunikat zawsze zawiera nazwę pliku albo ścieżkę katalogu."""


def _read_catalog(catalog_path: Path) -> dict[str, dict[str, str]]:
    """Katalog jako {nazwa pliku: wiersz} (jak helpers.catalog.load_catalog)."""
    if not catalog_path.is_file():
        raise CorpusError(f"Nie znaleziono katalogu korpusu: {catalog_path}")
    with catalog_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    catalog: dict[str, dict[str, str]] = {}
    for row in rows:
        file = (row.get("file") or "").strip()
        if not file:
            raise CorpusError(f"{catalog_path.name}: wiersz bez nazwy pliku (kolumna 'file').")
        if file in catalog:
            raise CorpusError(f"{catalog_path.name}: plik {file} ma więcej niż jeden wiersz.")
        catalog[file] = row
    return catalog


def load_documents(
    corpus_dir: Path, catalog_path: Path, required_metadata: Sequence[str]
) -> list[Document]:
    """Jeden Document na plik *.md w corpus_dir (bez podkatalogów), posortowane po doc_id.

    metadata = pola required_metadata z wiersza katalogu + "source" (nazwa pliku).
    """
    corpus_dir, catalog_path = Path(corpus_dir), Path(catalog_path)
    if not corpus_dir.is_dir():
        raise CorpusError(f"Nie znaleziono katalogu korpusu: {corpus_dir}")
    catalog = _read_catalog(catalog_path)
    files = sorted(p for p in corpus_dir.glob("*.md") if p.is_file())

    docs: list[Document] = []
    seen_ids: dict[str, str] = {}
    for path in files:
        row = catalog.get(path.name)
        if row is None:
            raise CorpusError(f"Plik {path.name} nie ma wiersza w katalogu {catalog_path.name}.")
        metadata: dict[str, str] = {}
        for field in required_metadata:
            value = row.get(field)
            if value is None or value == "":
                raise CorpusError(
                    f"Plik {path.name}: puste pole {field!r} w katalogu {catalog_path.name}."
                )
            metadata[field] = value
        doc_id = metadata.get("doc_id", "")
        if doc_id in seen_ids:
            raise CorpusError(
                f"Powtórzony doc_id {doc_id}: pliki {seen_ids[doc_id]} i {path.name}."
            )
        seen_ids[doc_id] = path.name
        metadata["source"] = path.name
        docs.append(Document(page_content=path.read_text(encoding="utf-8"), metadata=metadata))

    present = {p.name for p in files}
    orphaned = sorted(name for name in catalog if name not in present)
    if orphaned:
        raise CorpusError(
            f"Katalog {catalog_path.name} wskazuje nieistniejące pliki: {', '.join(orphaned)}"
        )
    return sorted(docs, key=lambda d: str(d.metadata.get("doc_id", "")))


def load_corpus(settings: Settings) -> list[Document]:
    """load_documents ze ścieżkami i wymaganymi polami z configu."""
    return load_documents(
        settings.paths.corpus_dir,
        settings.paths.corpus_catalog,
        settings.corpus.required_metadata,
    )


class CorpusStats(BaseModel):
    """Pola i zaokrąglenia jak sekcja "corpus" w report.json kursu."""

    n_documents: int
    total_tokens: int
    tokens_by_doc_type: dict[str, int]
    largest_document: str
    tokens_per_word_by_language: dict[str, float]


def corpus_stats(docs: Sequence[Document], *, encoding: str) -> CorpusStats:
    """Statystyki korpusu liczone jak w run_pipeline.py kursu."""
    if not docs:
        raise ValueError("Korpus jest pusty.")
    tokens = {d.metadata["doc_id"]: count_tokens(d.page_content, encoding) for d in docs}
    by_type: dict[str, int] = {}
    for d in docs:
        doc_type = d.metadata["doc_type"]
        by_type[doc_type] = by_type.get(doc_type, 0) + tokens[d.metadata["doc_id"]]
    per_word: dict[str, float] = {}
    for lang in sorted({d.metadata["language"] for d in docs}):
        text = "\n".join(d.page_content for d in docs if d.metadata["language"] == lang)
        per_word[lang] = round(count_tokens(text, encoding) / len(text.split()), 3)
    return CorpusStats(
        n_documents=len(docs),
        total_tokens=sum(tokens.values()),
        tokens_by_doc_type=by_type,
        largest_document=min(tokens, key=lambda i: (-tokens[i], i)),
        tokens_per_word_by_language=per_word,
    )
