"""Atrapy i dane pomocnicze testów F2 (kontrakt: docs/contracts/f2.md, 11.1).

Tylko `langchain_core`: moduł nie importuje kodu produkcyjnego `kovalo`, więc
importuje się nawet przed implementacją (brak kodu = czerwone testy, nie błąd zbierania).
Port `FakeStructuredLLM` i helperów z `reference/course/mini_project_2/tests/conftest.py`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

DEFAULT_USAGE: dict[str, Any] = {
    "input_tokens": 120,
    "output_tokens": 40,
    "total_tokens": 160,
    "output_token_details": {"reasoning": 25},
}


def _prompt_text(prompt: Any) -> str:
    return prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)


class FakeStructuredChatModel(BaseChatModel):
    """Odpowiednik `FakeStructuredLLM` kursu jako podklasa `BaseChatModel`.

    `with_structured_output(schema, include_raw=...)` zwraca `RunnableLambda`, który zapisuje
    prompt i schemat oraz zwraca `results[schema]` (gotowy obiekt albo funkcja `prompt -> obiekt`).
    `error` zgłasza wyjątek, `parsing_error` zwraca słownik z `parsed=None`.
    `_generate` zgłasza NotImplementedError (model nie służy do zwykłych wywołań).
    """

    results: dict[Any, Any] = Field(default_factory=dict)
    error: Any = None
    parsing_error: bool = False
    usage: dict[str, Any] | None = Field(default_factory=lambda: dict(DEFAULT_USAGE))
    model_name_in_metadata: str = "fake-model"
    prompts: list[str] = Field(default_factory=list)
    schemas: list[Any] = Field(default_factory=list)
    include_raw_flags: list[bool] = Field(default_factory=list)
    calls: list[tuple[Any, str]] = Field(default_factory=list)  # (schema, prompt)

    @property
    def _llm_type(self) -> str:
        return "fake-structured"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("FakeStructuredChatModel obsługuje tylko with_structured_output")

    def with_structured_output(  # type: ignore[override]
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[Any, Any]:
        self.schemas.append(schema)
        self.include_raw_flags.append(include_raw)

        def raw_message() -> AIMessage:
            return AIMessage(
                content="",
                usage_metadata=self.usage,  # type: ignore[arg-type]
                response_metadata={"model_name": self.model_name_in_metadata},
            )

        def respond(prompt: Any) -> Any:
            text = _prompt_text(prompt)
            self.prompts.append(text)
            self.calls.append((schema, text))
            if self.error is not None:
                raise self.error
            if self.parsing_error:
                bad = OutputParserException("zły JSON od modelu")
                return {"raw": raw_message(), "parsed": None, "parsing_error": bad}
            result = self.results.get(schema)
            if callable(result) and not isinstance(result, BaseModel):
                result = result(text)
            if include_raw:
                return {"raw": raw_message(), "parsed": result, "parsing_error": None}
            return result

        return RunnableLambda(respond)


class CachingFakeChatModel(BaseChatModel):
    """Prawdziwy `BaseChatModel` (działa z `cache=`), zwraca JSON schematu; liczy wywołania.

    Służy do testów cache LLM-a przez `run_e2`: `with_structured_output(schema, include_raw=True)`
    zwraca `self | parser`, więc wywołanie przechodzi przez `_generate_with_cache`.
    """

    reply: Any = None  # Callable[[str], str]: prompt -> JSON
    sent: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "caching-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        self.sent.append(prompt)
        message = AIMessage(
            content=self.reply(prompt),
            usage_metadata={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},  # type: ignore[arg-type]
            response_metadata={"model_name": "caching-fake"},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(  # type: ignore[override]
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[Any, Any]:
        def parse(message: AIMessage) -> Any:
            parsed = schema.model_validate(json.loads(str(message.content)))
            if include_raw:
                return {"raw": message, "parsed": parsed, "parsing_error": None}
            return parsed

        return self | RunnableLambda(parse)


class ExplodingModel(Runnable[Any, Any]):
    """Każde użycie to AssertionError: dowodzi, że model nie został wywołany."""

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        raise AssertionError("model nie powinien być wywołany (przebieg bez LLM-a)")


TOY_CHUNKS = [
    Document(
        page_content="Polityka podróży > Akceptacje\n\nWyjazdy zagraniczne wymagają zgody CFO.",
        metadata={"doc_id": "TST-001", "chunk_id": "TST-001#ctx000", "title": "Polityka podróży"},
    ),
    Document(
        page_content=(
            "Zasady floty > Kierowcy\n\n"
            "Furgonetki mogą prowadzić tylko kierowcy z ważnym prawem jazdy."
        ),
        metadata={"doc_id": "TST-002", "chunk_id": "TST-002#ctx000", "title": "Zasady floty"},
    ),
]

TOY_QUESTION = "Kto zatwierdza wyjazdy zagraniczne?"


def toy_gold(**fields: Any) -> Any:
    from kovalo.eval.datasets import GoldQuestion

    base: dict[str, Any] = {
        "id": "T01",
        "question": TOY_QUESTION,
        "answerable": True,
        "relevant_docs": ["TST-001"],
        "requires_all": False,
        "evidence_facts": [["CFO"]],
        "key_facts": [["CFO", "dyrektor finansowy"]],
        "reference": "Wyjazdy zagraniczne wymagają zgody CFO.",
    }
    base.update(fields)
    return GoldQuestion(**base)


def toy_record(
    answer: str | None = None,
    citations: tuple[str, ...] | list[str] = (),
    answerable: bool = True,
    error: str | None = None,
    context_text: str = "Wyjazdy zagraniczne wymagają zgody CFO.",
    context_doc: str = "TST-001",
    question: str = TOY_QUESTION,
) -> Any:
    from kovalo.rag.records import Answer, ContextItem, RunRecord

    context = [ContextItem(doc_id=context_doc, chunk_id=f"{context_doc}#ctx000", text=context_text)]
    output = (
        None
        if answer is None
        else Answer(answer=answer, citations=list(citations), answerable=answerable)
    )
    return RunRecord(question=question, context=context, output=output, error=error)


FakeReply = Callable[[str], str]
