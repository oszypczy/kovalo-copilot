"""Ocena bez LLM-a (FR-2.3; kontrakt F2, 8.1; port 9 testów kroku 3 MP2 kursu)."""

from __future__ import annotations

from tests.fakes import toy_gold as gold
from tests.fakes import toy_record as record


def score(rec, g):
    from kovalo.eval.rag.scoring import score_record

    return score_record(rec, g).model_dump()


def test_a_correct_cited_answer():
    s = score(record("CFO zatwierdza wyjazdy zagraniczne [TST-001].", ["TST-001"]), gold())
    assert s["outcome"] == "answered"
    assert s["evidence_phrase_coverage"] == 1.0 and s["answer_coverage"] == 1.0
    assert s["citations_present"] and s["citations_valid"] and s["cites_relevant"]
    assert s["refusal"] is None


def test_facts_match_whole_words_and_numbers_only():
    g = gold(key_facts=[["9%"], ["zarząd"]])
    assert (
        score(record("Minimalna marża to 19%; decyzja należy do zarządu.", ["TST-001"]), g)[
            "answer_coverage"
        ]
        == 0.0
    )
    assert (
        score(record("Minimalna marża to 19,9% albo 1.9% (decydują zarządy).", ["TST-001"]), g)[
            "answer_coverage"
        ]
        == 0.0
    )
    assert (
        score(record("Poniżej  9%.  Decyduje (ZARZĄD).", ["TST-001"]), g)["answer_coverage"] == 1.0
    )
    grouped = gold(key_facts=[["500,000 PLN"]])
    assert score(record("Powyżej 1,500,000 PLN.", ["TST-001"]), grouped)["answer_coverage"] == 0.0
    spaced = gold(key_facts=[["500 000 PLN"]])
    assert score(record("Powyżej 1 500 000 PLN.", ["TST-001"]), spaced)["answer_coverage"] == 0.0
    assert (
        score(record("Powyżej 500 000 PLN rocznie.", ["TST-001"]), spaced)["answer_coverage"] == 1.0
    )


def test_zero_decimals_do_not_hide_a_found_fact():
    """Regresja z przebiegu na żywo (hybrid, M07): prompt kursu każe pisać „7296,00 PLN”,
    więc odpowiedź „500 000,00 PLN” jest poprawna, a nie może dawać pokrycia 0,00."""
    spaced = gold(key_facts=[["500 000 PLN"]])
    for text in ("Powyżej 500 000,00 PLN rocznie.", "Powyżej 500 000.00 PLN rocznie."):
        assert score(record(text, ["TST-001"]), spaced)["answer_coverage"] == 1.0
    assert (
        score(record("Marża 9,00%.", ["TST-001"]), gold(key_facts=[["9%"]]))["answer_coverage"]
        == 1.0
    )
    # nadal tylko cała liczba: 1 500 000,00 i 500 000,50 to nie 500 000 PLN
    for text in ("Powyżej 1 500 000,00 PLN.", "Powyżej 500 000,50 PLN."):
        assert score(record(text, ["TST-001"]), spaced)["answer_coverage"] == 0.0
    assert (
        score(record("Marża 9,000%.", ["TST-001"]), gold(key_facts=[["9%"]]))["answer_coverage"]
        == 0.0
    )


def test_negation_is_a_known_limitation():
    assert score(record("Nie CFO; zatwierdza CEO.", ["TST-001"]), gold())["answer_coverage"] == 1.0


def test_citation_rules():
    assert score(record("CFO.", []), gold())["citations_present"] is False
    assert score(record("CFO.", []), gold())["citations_valid"] is False
    s = score(record("CFO.", ["TST-009"]), gold())
    assert s["citations_valid"] is False and s["cites_relevant"] is False


def test_error_never_earns_answer_or_refusal_credit():
    s = score(
        record(error="OutputParserException: bad JSON"),
        gold(answerable=False, key_facts=[], evidence_facts=[]),
    )
    assert s["outcome"] == "error"
    assert all(v is None for k, v in s.items() if k != "outcome")


def test_evidence_does_not_depend_on_generation():
    ok = score(record("CFO.", ["TST-001"]), gold())
    failed = score(record(error="APIError: timeout"), gold())
    assert failed["evidence_phrase_coverage"] == ok["evidence_phrase_coverage"] == 1.0
    assert failed["answer_coverage"] is None and failed["citations_valid"] is None


def test_refusals():
    unanswerable = gold(answerable=False, relevant_docs=[], key_facts=[], evidence_facts=[])
    refused = record("Brak tej informacji w dokumentach.", answerable=False)
    assert score(refused, unanswerable)["refusal"] == "correct"
    assert score(record("To 9000 PLN.", ["TST-001"]), unanswerable)["refusal"] == "missed"
    no_evidence = record(
        "Brak tej informacji w dokumentach.",
        answerable=False,
        context_text="Zasady floty dla furgonetek.",
    )
    s = score(no_evidence, gold())
    assert s["evidence_phrase_coverage"] == 0.0 and s["refusal"] == "phrases_missing"
    assert score(refused, gold())["refusal"] == "phrases_present"


def test_relationship_phrases_avoid_loose_matches():
    g = gold(evidence_facts=[["powyżej 25 000 PLN | CFO"]], key_facts=[["CFO"]])
    unrelated = record(
        "Brak tej informacji w dokumentach.",
        answerable=False,
        context_text="Podróże powyżej 25 000 PLN rocznie zgłasza się CFO.",
    )
    assert score(unrelated, g)["evidence_phrase_coverage"] == 0.0
    table = record(
        "CFO.",
        ["TST-001"],
        context_text="| 5000 do 25 000 PLN | Kierownik finansowy |\n| powyżej 25 000 PLN | CFO |",
    )
    assert score(table, g)["evidence_phrase_coverage"] == 1.0


def test_alternative_evidence_paths():
    g = gold(
        evidence_facts=[["94,6%", "dwa pełne punkty"], ["za każdy pełny punkt procentowy"]],
        key_facts=[["4%"]],
    )
    text = "2% za każdy pełny punkt procentowy poniżej celu. Lipiec: dwa pełne punkty, 4%."
    rec = record("4% [TST-001]", ["TST-001"], context_text=text)
    assert score(rec, g)["evidence_phrase_coverage"] == 1.0


def test_unanswerable_questions_have_no_evidence_coverage():
    unanswerable = gold(answerable=False, relevant_docs=[], key_facts=[], evidence_facts=[])
    refused = record("Brak tej informacji w dokumentach.", answerable=False)
    assert score(refused, unanswerable)["evidence_phrase_coverage"] is None


def test_record_score_dump_matches_course_keys():
    from kovalo.eval.rag.scoring import RecordScore, score_record

    s = score_record(record("CFO.", ["TST-001"]), gold())
    assert isinstance(s, RecordScore)
    assert list(s.model_dump()) == [
        "outcome",
        "evidence_phrase_coverage",
        "answer_coverage",
        "citations_present",
        "citations_valid",
        "cites_relevant",
        "refusal",
    ]


def test_text_helpers():
    from kovalo.eval.rag.scoring import (
        all_required_found,
        fact_coverage,
        fact_found,
        normalize_text,
        retrieval_hit,
    )

    assert normalize_text("Dyrektor \n  Handlowy") == "dyrektor handlowy"
    assert fact_found(["cfo", "dyrektor finansowy"], "Zatwierdza Dyrektor Finansowy.")
    assert not fact_found(["9%"], "marża 19%")
    assert fact_coverage([], "cokolwiek") is None
    assert fact_coverage([["a"], ["b"]], "a i c") == 0.5

    g = gold(relevant_docs=["CON-101", "MTG-403"], requires_all=True)
    assert retrieval_hit(["MTG-403", "X-1"], g) is True
    assert retrieval_hit(["X-1"], g) is False
    assert all_required_found(["MTG-403", "CON-101"], g) is True
    assert all_required_found(["MTG-403"], g) is False
    assert all_required_found(["MTG-403"], gold(requires_all=False)) is None
