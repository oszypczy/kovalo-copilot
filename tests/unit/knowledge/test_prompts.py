"""Układ promptu (port test_provided_prompt kursu; kontrakt F1, 5.2)."""

from __future__ import annotations

from langchain_core.documents import Document


def test_every_context_block_names_its_document():
    from kovalo.knowledge.prompts import build_prompt

    context = [
        Document(page_content="Zasada pierwsza.", metadata={"doc_id": "TST-001"}),
        Document(page_content="Zasada druga.", metadata={"doc_id": "TST-002"}),
    ]
    prompt = build_prompt(
        "Przy każdym fakcie podaj identyfikator dokumentu.", context, "Które zasady obowiązują?"
    )
    assert "[TST-001]\nZasada pierwsza." in prompt and "[TST-002]\nZasada druga." in prompt
    assert prompt.endswith("Które zasady obowiązują?")


def test_exact_layout():
    from kovalo.knowledge.prompts import build_prompt

    context = [
        Document(page_content="A", metadata={"doc_id": "X-1"}),
        Document(page_content="B", metadata={"doc_id": "X-2"}),
    ]
    assert build_prompt("  System.\n", context, " Pytanie? \n") == (
        "System.\n\n# Kontekst\n\n[X-1]\nA\n\n---\n\n[X-2]\nB\n\n# Pytanie\n\nPytanie?"
    )
