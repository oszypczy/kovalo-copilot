"""answer_question (FR-2.2; kontrakt F2, 7.3; port 5 testów kroku 2 MP2)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from tests.fakes import TOY_CHUNKS, TOY_QUESTION, FakeStructuredChatModel

QUESTION = TOY_QUESTION


def _answer(settings, retriever, fake, **kw):
    from kovalo.rag.answer import answer_question
    from kovalo.rag.records import Answer

    return answer_question(
        QUESTION,
        retriever,
        fake.with_structured_output(Answer),
        system_prompt=settings.prompts.answer_system,
        k=kw.pop("k", 2),
        **kw,
    )


def _fake(result=None, **kw):
    from kovalo.rag.records import Answer

    return FakeStructuredChatModel(results={Answer: result}, **kw)


def test_record_holds_the_context_the_model_saw(settings, toy_retriever):
    from kovalo.rag.records import Answer, RunRecord

    fake = _fake(Answer(answer="CFO.", citations=["TST-001"], answerable=True))
    rec = _answer(settings, toy_retriever, fake)
    assert isinstance(rec, RunRecord)
    assert rec.question == QUESTION and rec.error is None
    texts = {c.page_content for c in TOY_CHUNKS}
    assert rec.context and all(item.text in texts for item in rec.context)
    assert {item.chunk_id for item in rec.context} <= {c.metadata["chunk_id"] for c in TOY_CHUNKS}
    assert fake.schemas == [Answer]
    assert rec.retrieval == "vector"


def test_prompt_uses_the_answer_system_and_doc_ids(settings, toy_retriever):
    from kovalo.rag.records import Answer

    fake = _fake(Answer(answer="x", citations=[], answerable=True))
    rec = _answer(settings, toy_retriever, fake)
    prompt = fake.prompts[0]
    assert settings.prompts.answer_system.strip()[:40] in prompt
    assert "[TST-001]" in prompt and "[TST-002]" in prompt
    assert QUESTION in prompt
    assert rec.prompt == prompt


def test_k_limits_the_context(settings, toy_retriever):
    from kovalo.rag.records import Answer

    rec = _answer(
        settings, toy_retriever, _fake(Answer(answer="x", citations=[], answerable=True)), k=1
    )
    assert len(rec.context) == 1
    assert rec.context[0].score is not None


def test_citations_are_normalised(settings, toy_retriever):
    from kovalo.rag.records import Answer

    fake = _fake(
        Answer(answer="x", citations=["[TST-001]", " tst-002 ", "TST-001#ctx000"], answerable=True)
    )
    rec = _answer(settings, toy_retriever, fake)
    assert rec.output.citations == ["TST-001", "TST-002"]


def test_a_model_failure_becomes_an_error_not_an_exception(settings, toy_retriever):
    rec = _answer(settings, toy_retriever, _fake(error=ValueError("bad JSON")))
    assert rec.output is None and rec.error.startswith("ValueError: bad JSON")
    assert rec.context and rec.prompt
    rec = _answer(settings, toy_retriever, _fake(result=None))
    assert rec.output is None and rec.error == "Model nie zwrócił obiektu Answer"


def test_long_error_is_truncated(settings, toy_retriever):
    rec = _answer(settings, toy_retriever, _fake(error=ValueError("x" * 1000)))
    assert len(rec.error) <= 300


def test_unknown_ids_are_not_repaired(settings, toy_retriever):
    from kovalo.rag.records import Answer

    fake = _fake(Answer(answer="x", citations=["TST-0019", "XTST-001", "TST-001"], answerable=True))
    rec = _answer(settings, toy_retriever, fake)
    assert rec.output.citations == ["TST-0019", "XTST-001", "TST-001"]


def test_include_raw_result_fills_usage(settings, toy_retriever):
    from kovalo.rag.answer import answer_question
    from kovalo.rag.records import Answer

    fake = _fake(Answer(answer="CFO.", citations=["TST-001"], answerable=True))
    ticks = iter([10.0, 10.5])
    rec = answer_question(
        QUESTION,
        toy_retriever,
        fake.with_structured_output(Answer, include_raw=True),
        system_prompt=settings.prompts.answer_system,
        k=2,
        chunks="ctx",
        clock=lambda: next(ticks),
    )
    assert rec.error is None and rec.output.answer == "CFO."
    assert rec.usage.input_tokens == 120 and rec.usage.output_tokens == 40
    assert rec.usage.reasoning_tokens == 25
    assert rec.answered_by == "fake-model"
    assert rec.chunks == "ctx" and rec.retrieval == "vector"
    assert rec.prompt and QUESTION in rec.prompt
    assert rec.seconds == pytest.approx(0.5)


def test_parsing_error_is_error_record(settings, toy_retriever):
    from kovalo.rag.answer import answer_question
    from kovalo.rag.records import Answer

    fake = _fake(parsing_error=True)
    rec = answer_question(
        QUESTION,
        toy_retriever,
        fake.with_structured_output(Answer, include_raw=True),
        system_prompt=settings.prompts.answer_system,
        k=2,
    )
    assert rec.output is None
    assert rec.error.startswith("OutputParserException")
    assert rec.context  # kontekst zapisany mimo błędu


def test_parsed_none_without_error_is_error_record(settings, toy_retriever):
    from kovalo.rag.answer import answer_question
    from kovalo.rag.records import Answer

    fake = _fake(result=None)
    rec = answer_question(
        QUESTION,
        toy_retriever,
        fake.with_structured_output(Answer, include_raw=True),
        system_prompt=settings.prompts.answer_system,
        k=2,
    )
    assert rec.output is None and rec.error == "Model nie zwrócił obiektu Answer"


def test_retrieval_error_propagates(settings):
    from kovalo.knowledge.pgvector import IndexUnavailableError
    from kovalo.rag.answer import answer_question
    from kovalo.rag.records import Answer

    class Broken:
        name = "vector"

        def retrieve(self, question, k):
            raise IndexUnavailableError("OperationalError")

    fake = _fake(Answer(answer="x", citations=[], answerable=True))
    with pytest.raises(IndexUnavailableError):
        answer_question(
            QUESTION,
            Broken(),
            fake.with_structured_output(Answer),
            system_prompt=settings.prompts.answer_system,
            k=2,
        )
    assert fake.prompts == []  # model nie był wywołany


def test_run_record_readable_by_course_loader(settings, toy_retriever):
    """RunRecord z naszymi polami waliduje się klasą kursu (kopia helpers/records.py)."""
    from kovalo.rag.answer import answer_question
    from kovalo.rag.records import Answer

    class CourseContextItem(BaseModel):
        doc_id: str
        chunk_id: str
        text: str

    class CourseAnswer(BaseModel):
        answer: str
        citations: list[str]
        answerable: bool

    class CourseRunRecord(BaseModel):
        question: str
        context: list[CourseContextItem]
        output: CourseAnswer | None = None
        error: str | None = None

    fake = _fake(Answer(answer="CFO.", citations=["TST-001"], answerable=True))
    rec = answer_question(
        QUESTION,
        toy_retriever,
        fake.with_structured_output(Answer, include_raw=True),
        system_prompt=settings.prompts.answer_system,
        k=2,
        chunks="ctx",
    )
    dumped = rec.model_dump()
    assert {"chunks", "retrieval", "prompt", "answered_by", "usage", "seconds"} <= set(dumped)
    course = CourseRunRecord.model_validate(dumped)
    assert course.output.citations == ["TST-001"]
    assert len(course.context) == len(rec.context)


def test_answer_and_verdict_schema_descriptions_match_course():
    from kovalo.rag.records import Answer, Verdict

    assert Answer.__doc__.strip() == "Odpowiedź oparta na dokumentach z kontekstu."
    assert Answer.model_fields["answer"].description == (
        "odpowiedź oparta wyłącznie na faktach z dokumentów kontekstu"
    )
    assert Answer.model_fields["answerable"].description == (
        "false, jeśli dokumenty kontekstu nie zawierają odpowiedzi"
    )
    assert Verdict.__doc__.strip() == "Werdykt sędziego (LLM-as-judge) dla jednej odpowiedzi."
    assert (
        Verdict.model_fields["reason"].description == "jedno lub dwa zdania uzasadnienia werdyktu"
    )


def test_token_usage_from_message():
    from langchain_core.messages import AIMessage

    from kovalo.rag.records import TokenUsage

    assert TokenUsage.from_message(None) is None
    assert TokenUsage.from_message(AIMessage(content="x")) is None
    msg = AIMessage(
        content="x",
        usage_metadata={
            "input_tokens": 5,
            "output_tokens": 9,
            "total_tokens": 14,
            "output_token_details": {"reasoning": 4},
        },
    )
    usage = TokenUsage.from_message(msg)
    assert (usage.input_tokens, usage.output_tokens, usage.reasoning_tokens) == (5, 9, 4)
    plain = AIMessage(
        content="x", usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    )
    assert TokenUsage.from_message(plain).reasoning_tokens is None
