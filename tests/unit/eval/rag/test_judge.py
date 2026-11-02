"""Sędzia LLM (FR-2.4; kontrakt F2, 8.2; port 3 testów kroku 4 MP2 kursu)."""

from __future__ import annotations

import pytest

from tests.fakes import FakeStructuredChatModel
from tests.fakes import toy_gold as gold
from tests.fakes import toy_record as record


def _verdict_model(**kw):
    from kovalo.rag.records import Verdict

    result = kw.pop("result", Verdict(correct=True, grounded=True, reason="Podaje CFO."))
    return FakeStructuredChatModel(results={Verdict: result}, **kw)


def judge(rec, g, fake, **kw):
    from kovalo.eval.rag.judge import judge_record
    from kovalo.rag.records import Verdict

    return judge_record(
        rec, g, fake.with_structured_output(Verdict, **kw), system_prompt="RUBRYKA SĘDZIEGO"
    )


def test_judge_prompt_holds_everything_it_needs():
    from kovalo.rag.records import Verdict

    fake = _verdict_model()
    rec = record("Zatwierdza CFO [TST-001].", ["TST-001"])
    g = gold(reference="Wzorzec: wyjazdy zagraniczne zatwierdza CFO.")
    result = judge(rec, g, fake)
    assert result.course_dict() == {
        "correct": True,
        "grounded": True,
        "reason": "Podaje CFO.",
        "error": None,
    }
    assert fake.schemas == [Verdict]
    prompt = fake.prompts[0]
    for part in (
        rec.question,
        "Wyjazdy zagraniczne wymagają zgody CFO.",
        "Zatwierdza CFO [TST-001].",
        g.reference,
    ):
        assert part in prompt


def test_only_answered_records_are_judged():
    fake = _verdict_model()
    empty = {"correct": None, "grounded": None, "reason": None, "error": None}
    refused = record("Brak tej informacji w dokumentach.", answerable=False)
    assert judge(refused, gold(), fake).course_dict() == empty
    assert judge(record(error="bad JSON"), gold(), fake).course_dict() == empty
    assert fake.prompts == []


def test_a_judge_failure_is_recorded_not_raised():
    v = judge(record("CFO.", ["TST-001"]), gold(), _verdict_model(error=ValueError("bad JSON")))
    assert v.error and v.error.startswith("ValueError") and v.correct is None
    assert len(v.error) <= 300
    v = judge(record("CFO.", ["TST-001"]), gold(), _verdict_model(result=None))
    assert v.error and v.correct is None


def test_judge_parsing_error_is_recorded():
    v = judge(
        record("CFO.", ["TST-001"]), gold(), _verdict_model(parsing_error=True), include_raw=True
    )
    assert v.error and "OutputParserException" in v.error and v.correct is None


def test_judge_include_raw_fills_our_fields():
    v = judge(record("CFO.", ["TST-001"]), gold(), _verdict_model(), include_raw=True)
    assert v.correct is True and v.error is None
    assert v.answered_by == "fake-model"
    assert v.judged_by == "fake-model"  # model, który faktycznie odpowiedział sędziemu
    assert v.model_dump()["judged_by"] == "fake-model"  # trafia do <data>-e2.json
    assert v.usage.input_tokens == 120
    assert v.seconds is not None and v.seconds >= 0
    assert set(v.course_dict()) == {"correct", "grounded", "reason", "error"}


def test_judge_prompt_layout():
    from kovalo.eval.rag.judge import build_judge_prompt
    from kovalo.rag.records import ContextItem

    rec = record("Zatwierdza CFO.", ["TST-001", "TST-002"])
    rec.context.append(
        ContextItem(doc_id="TST-002", chunk_id="TST-002#ctx000", text="Drugi tekst.")
    )
    g = gold(reference="  Wzorzec.  ")
    prompt = build_judge_prompt("  RUBRYKA  ", rec, g)
    assert prompt.startswith("RUBRYKA\n\n# Pytanie\n\n")
    markers = ["# Pytanie", "# Kontekst", "# Oceniana odpowiedź", "# Odpowiedź wzorcowa"]
    positions = [prompt.index(m) for m in markers]
    assert positions == sorted(positions)
    assert "[TST-001] TST-001#ctx000\nWyjazdy zagraniczne wymagają zgody CFO." in prompt
    assert "\n\n---\n\n[TST-002] TST-002#ctx000\nDrugi tekst." in prompt
    assert (
        "Zatwierdza CFO.\n\nOdwołania: TST-001, TST-002\n\n# Odpowiedź wzorcowa\n\nWzorzec."
        in prompt
    )
    assert prompt.endswith("Wzorzec.")


def test_judge_prompt_empty_context_and_citations():
    from kovalo.eval.rag.judge import build_judge_prompt

    rec = record("CFO.", [])
    rec.context.clear()
    prompt = build_judge_prompt("R", rec, gold())
    assert "# Kontekst\n\n(brak)\n\n" in prompt
    assert "Odwołania: -" in prompt


@pytest.mark.parametrize("kind", ["refused", "error"])
def test_judge_returns_empty_result_without_calling_model(kind):
    fake = _verdict_model(error=AssertionError("nie wolno wołać"))
    rec = record("Brak.", answerable=False) if kind == "refused" else record(error="APIError: 500")
    assert judge(rec, gold(), fake).error is None


def test_judged_by_reports_the_fallback_model():
    """Gdy sędziemu odpowiedział fallback, judged_by podaje fallback, nie model z configu."""
    fake = _verdict_model(model_name_in_metadata="fallback-model")
    v = judge(record("CFO.", ["TST-001"]), gold(), fake, include_raw=True)
    assert v.judged_by == "fallback-model"


def test_judged_by_empty_without_a_call_or_metadata():
    fake = _verdict_model()
    refused = record("Brak tej informacji w dokumentach.", answerable=False)
    assert judge(refused, gold(), fake).judged_by is None  # bez wywołania sędziego
    assert judge(record("CFO.", ["TST-001"]), gold(), fake).judged_by is None  # bez include_raw
