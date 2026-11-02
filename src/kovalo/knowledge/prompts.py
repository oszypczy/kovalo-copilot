"""Układ promptu: port 1:1 z reference/course/helpers/prompts.py."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document

__all__ = ["build_prompt"]


def build_prompt(system: str, context: Sequence[Document], question: str) -> str:
    """Tekst systemowy, kontekst (każdy blok od [doc_id]) i pytanie."""
    blocks = [f"[{doc.metadata['doc_id']}]\n{doc.page_content}" for doc in context]
    body = "\n\n---\n\n".join(blocks)
    return f"{system.strip()}\n\n# Kontekst\n\n{body}\n\n# Pytanie\n\n{question.strip()}"
