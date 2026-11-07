"""Atrapy i pomocniki testów F4 (kontrakt: docs/contracts/f4.md, 15.1).

Port `conftest.py` Miniprojektu 4 (`reference/course/mini_project_4/tests/`): `small_data`,
`ScriptedModel`, `call`, `answer`, `event`, `make_trace`, `TERMS_114`, `TERMS_207`, `otif_result`,
`invoice_result`, `penalty_result`, `july_events`, plus `CountingRetriever`
(zamiast `CountingStore`),
`BrokenRetriever`, `tool_context`, `deps_for`, `agent_for`, `agent_answerer`.

Moduł nie importuje kodu produkcyjnego `kovalo` na poziomie modułu (tylko wewnątrz funkcji), więc
importuje się przed implementacją: brak kodu = czerwone testy, nie błąd zbierania. Współdzielony
przez testy `agents/`, `tools/`, `company/`, `eval/`, `workflow/`, `api/` i `test_cli.py`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

_ids = itertools.count()


# --- model z wywołaniami narzędzi ---------------------------------------------------------------


class ScriptedModel(BaseChatModel):
    """Zastępuje model czatu wywołujący narzędzia. Każde wywołanie bierze następny element
    scenariusza: AIMessage do zwrócenia, wyjątek do zgłoszenia albo funkcję wiadomości."""

    script: list[Any]
    calls: int = 0
    seen: list[Any] = Field(default_factory=list)
    bound: list[str] = Field(default_factory=list)
    label: str = "scripted-for-tests"

    @property
    def _llm_type(self) -> str:
        return self.label

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:  # type: ignore[override]
        self.bound = [getattr(t, "name", str(t)) for t in tools]
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.seen.append(list(messages))
        if self.calls >= len(self.script):
            raise AssertionError("modelowi ze scenariusza skończyły się odpowiedzi")
        item = self.script[self.calls]
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        message = item(messages) if callable(item) else item
        return ChatResult(generations=[ChatGeneration(message=message)])


def call(name: str, **args: Any) -> AIMessage:
    """Tura modelu, która wywołuje jedno narzędzie."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}]
    )


def answer(
    text: str, *, model: str | None = None, usage: tuple[int, int] | None = None
) -> AIMessage:
    """Ostateczna odpowiedź modelu; `model` trafia do response_metadata (fallback widoczny)."""
    kwargs: dict[str, Any] = {}
    if model is not None:
        kwargs["response_metadata"] = {"model_name": model}
    if usage is not None:
        kwargs["usage_metadata"] = {
            "input_tokens": usage[0],
            "output_tokens": usage[1],
            "total_tokens": sum(usage),
        }
    return AIMessage(content=text, **kwargs)


def parallel_calls(name: str, arguments: Sequence[dict[str, Any]], ids: Sequence[str]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"id": i, "name": name, "args": dict(a)} for i, a in zip(ids, arguments, strict=True)
        ],
    )


# --- dane i retrievery --------------------------------------------------------------------------


def make_small_data() -> Any:
    """Dane zrobione ręcznie: wiersze celowo nie po kolei, luka w lipcu, dostawca bez warunków."""
    from kovalo.company.data import CompanyData

    return CompanyData(
        suppliers={
            "SUP-114": "Mleczna Dolina",
            "SUP-207": "Aqua Kryształ",
            "SUP-300": "Test Foods",
        },
        otif={
            ("SUP-114", "2026-09"): 93.5,
            ("SUP-114", "2026-06"): 95.2,
            ("SUP-114", "2026-08"): 94.0,
            ("SUP-114", "2026-05"): 96.4,
            ("SUP-207", "2026-07"): 95.1,
        },
        invoices={("SUP-114", "2026-06"): 18120000, ("SUP-207", "2026-07"): 9630000},
        terms={
            "SUP-114": {
                "source_doc_id": "CON-101",
                "otif_target_pct": 97,
                "penalty_clause": True,
                "rate_per_full_point_pct": 2,
                "cap_pct": 10,
                "settlement": "credit note",
            },
            "SUP-207": {
                "source_doc_id": "CON-102",
                "otif_target_pct": 96,
                "penalty_clause": False,
                "rate_per_full_point_pct": None,
                "cap_pct": None,
                "settlement": "quarterly review",
            },
        },
        terms_version="terms-test",
        not_otif=[
            {
                "name": "TransBałt",
                "source_doc_id": "CON-103",
                "reason": "terminowe dostawy palet, nie OTIF",
            }
        ],
    )


COURSE_PASSAGES = [
    Document(
        page_content="TransBałt > Poziom obsługi i odpowiedzialność\n\n"
        "TransBałt musi dostarczać 98% palet w uzgodnionym oknie czasowym.",
        metadata={"doc_id": "CON-103", "chunk_id": "CON-103#ctx003", "doc_type": "contract"},
    ),
    Document(
        page_content="Mleczna Dolina > Rozwiązanie umowy\n\nKovalo może rozwiązać umowę z "
        "60-dniowym okresem wypowiedzenia, jeśli OTIF przez trzy kolejne miesiące wynosi "
        "mniej niż 94%.",
        metadata={"doc_id": "CON-101", "chunk_id": "CON-101#ctx004", "doc_type": "contract"},
    ),
    Document(
        page_content="Podróże służbowe > Noclegi\n\nKoszty hotelu są zwracane do 450 PLN za noc.",
        metadata={"doc_id": "POL-004", "chunk_id": "POL-004#ctx001", "doc_type": "policy"},
    ),
]


class CountingRetriever:
    """Malutki retriever (VectorRetriever na trzech chunkach kursu), który zapisuje, o jakie k go
    poproszono (odpowiednik `CountingStore` kursu)."""

    name = "counting"

    def __init__(self) -> None:
        from kovalo.knowledge.retrieval import build_vector_store
        from kovalo.knowledge.retrievers import VectorRetriever

        store = build_vector_store(COURSE_PASSAGES, DeterministicFakeEmbedding(size=16))
        self._inner = VectorRetriever(store)
        self.ks: list[int] = []
        self.questions: list[str] = []

    def retrieve(self, question: str, k: int) -> list[Any]:
        self.ks.append(k)
        self.questions.append(question)
        return self._inner.retrieve(question, k)


class BrokenRetriever:
    """Indeks wyszukiwania, który zawodzi, jak retriever, gdy usługa embeddingów nie działa."""

    name = "broken"

    def retrieve(self, question: str, k: int) -> list[Any]:
        raise RuntimeError("embedding service unavailable")


def tool_context(settings: Any, data: Any, retriever: Any = None) -> Any:
    from kovalo.tools.specs import ToolContext

    return ToolContext(
        data=data,
        policy=settings.policy,
        company=settings.company,
        retriever=retriever,
        top_k=settings.retrieval.top_k,
    )


def deps_for_script(
    settings: Any, data: Any, retriever: Any, script: Sequence[Any]
) -> tuple[ScriptedModel, Any]:
    """(atrapa modelu ze scenariuszem, ToolContext) - port `deps_for(script)` kursu."""
    return ScriptedModel(script=list(script)), tool_context(settings, data, retriever)


def agent_runner(
    settings: Any,
    data: Any,
    retriever: Any,
    script: Sequence[Any],
    *,
    variant: str = "course",
    extra_middleware: Sequence[Any] = (),
    model: BaseChatModel | None = None,
    fallbacks: Sequence[BaseChatModel] = (),
    fallback_exceptions: tuple[type[BaseException], ...] | None = None,
) -> tuple[Any, ScriptedModel]:
    """(AgentRunner z atrapą modelu, atrapa) - `agent_for(script, variant)` z kontraktu.

    `fallback_exceptions` None → llm.FALLBACK_EXCEPTIONS (na żywo);
    offline: llm.fallback_exceptions(True)."""
    from kovalo.agents.agent import AgentRunner, build_agent
    from kovalo.llm import FALLBACK_EXCEPTIONS
    from kovalo.tools.langchain import langchain_tools

    scripted = model or ScriptedModel(script=list(script))
    ctx = tool_context(settings, data, retriever)
    agent = build_agent(
        scripted,
        langchain_tools(ctx, variant),  # type: ignore[arg-type]
        system_prompt=settings.prompts.agent_system,
        fallbacks=fallbacks,
        fallback_exceptions=FALLBACK_EXCEPTIONS
        if fallback_exceptions is None
        else fallback_exceptions,
        extra_middleware=extra_middleware,
    )
    return AgentRunner(agent, settings.agent), scripted  # type: ignore[return-value]


# --- ręcznie zrobione zapisy przebiegu do oceny -------------------------------------------------


def event(
    tool: str,
    result: Any = None,
    origin: str | None = None,
    status: str = "ok",
    call_id: str | None = None,
    parent: str | None = None,
    **arguments: Any,
) -> Any:
    from kovalo.agents.trace import ORIGINS, TraceEvent

    return TraceEvent(
        call_id=call_id or f"e{next(_ids)}",
        parent_call_id=parent,
        tool=tool,
        arguments=arguments,
        origin=origin or ORIGINS.get(tool, "unknown"),  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        result=result,
    )


def make_trace(
    question: str, answer_text: str | None, events: list[Any], status: str | None = None
) -> Any:
    from kovalo.agents.trace import Trace

    return Trace(
        question=question,
        answer=answer_text,
        events=events,
        status=status or ("answered" if answer_text is not None else "error"),  # type: ignore[arg-type]
    )


TERMS_114 = {
    "status": "ok",
    "supplier_id": "SUP-114",
    "supplier": "Mleczna Dolina",
    "source_doc_id": "CON-101",
    "otif_target_pct": 97.0,
    "penalty_clause": True,
    "rate_per_full_point_pct": 2.0,
    "cap_pct": 10.0,
}
TERMS_207 = {
    "status": "ok",
    "supplier_id": "SUP-207",
    "supplier": "Aqua Kryształ",
    "source_doc_id": "CON-102",
    "otif_target_pct": 96.0,
    "penalty_clause": False,
    "rate_per_full_point_pct": None,
    "cap_pct": None,
}


def otif_result(
    supplier_id: str, observations: dict[str, float], missing: Sequence[str] = ()
) -> dict:
    return {
        "status": "ok" if observations else "no_data",
        "supplier_id": supplier_id,
        "observations": [{"month": m, "otif_pct": v} for m, v in sorted(observations.items())],
        "missing_months": list(missing),
        "complete": not missing,
    }


def invoice_result(supplier_id: str, month: str, grosze: int | None) -> dict:
    return {
        "status": "ok" if grosze is not None else "no_data",
        "supplier_id": supplier_id,
        "month": month,
        "invoiced_grosze": grosze,
        "currency": "PLN",
    }


def penalty_result(otif, target, rate, cap, invoice, full_points, pct, grosze) -> dict:  # type: ignore[no-untyped-def]
    return {
        "status": "ok",
        "otif_pct": otif,
        "target_pct": target,
        "rate_per_full_point_pct": rate,
        "cap_pct": cap,
        "invoice_grosze": invoice,
        "shortfall_points": round(target - otif, 4),
        "full_points": full_points,
        "penalty_pct_uncapped": pct,
        "penalty_pct": pct,
        "penalty_grosze": grosze,
        "currency": "PLN",
    }


def july_events() -> list[Any]:
    """Poprawna ścieżka dowodów dla lipcowej kary Mlecznej Doliny."""
    return [
        event("get_contract_terms", TERMS_114, supplier="Mleczna Dolina"),
        event(
            "get_supplier_otif",
            otif_result("SUP-114", {"2026-07": 94.6}),
            supplier="SUP-114",
            from_month="2026-07",
            to_month="2026-07",
        ),
        event(
            "get_invoice_value",
            invoice_result("SUP-114", "2026-07", 18240000),
            supplier="SUP-114",
            month="2026-07",
        ),
        event(
            "calculate_otif_penalty",
            penalty_result(94.6, 97.0, 2.0, 10.0, 18240000, 2, 4.0, 729600),
            otif_pct=94.6,
            target_pct=97,
            rate_per_full_point_pct=2,
            cap_pct=10,
            invoice_grosze=18240000,
        ),
    ]


JULY = (
    "Ile wynosi kara, którą Mleczna Dolina jest winna Kovalo za lipiec 2026 r.? Podaj kwotę w PLN."
)
T1_QUESTION = (
    "Czy na koniec września 2026 r. Kovalo może rozwiązać umowę z Mleczną Doliną z powodu jej OTIF?"
)


def p1_script() -> list[AIMessage]:
    """Idealny skrypt P1 (4 narzędzia + odpowiedź), jak w przebiegu kursu."""
    return [
        call("get_contract_terms", supplier="Mleczna Dolina"),
        call("get_supplier_otif", supplier="SUP-114", from_month="2026-07", to_month="2026-07"),
        call("get_invoice_value", supplier="SUP-114", month="2026-07"),
        call(
            "calculate_otif_penalty",
            otif_pct=94.6,
            target_pct=97,
            rate_per_full_point_pct=2,
            cap_pct=10,
            invoice_grosze=18240000,
        ),
        answer(
            "Kara za lipiec wynosi 7296,00 PLN (CON-101): OTIF 94,6% jest o 2 pełne punkty "
            "niższy od celu 97%, więc kara to 4% faktury na 182 400 PLN.",
            model="scripted-model",
            usage=(100, 20),
        ),
    ]


def t1_script() -> list[AIMessage]:
    """Idealny skrypt T1 z narzędziem FR-4.7 (search_documents, check_termination_condition)."""
    return [
        call("search_documents", question="warunki rozwiązania umowy z Mleczną Doliną OTIF"),
        call("check_termination_condition", supplier="Mleczna Dolina", as_of_month="2026-09"),
        answer(
            "Nie, warunek nie jest spełniony (CON-101): w lipcu 2026 r. OTIF wyniósł 94,6%, "
            "czyli nie był poniżej progu 94%; w sierpniu 93,8% i we wrześniu 93,5%. "
            "Seria trzech kolejnych miesięcy poniżej 94% została przerwana.",
            model="scripted-model",
            usage=(100, 20),
        ),
    ]


def agent_answerer(trace_or_script: Any, *, calls: list[str] | None = None) -> Callable[..., Any]:
    """Atrapa `AgentAnswerer`: zwraca podany `Trace` (albo kolejne z listy) i woła `on_event`
    ('start' i 'end' dla każdego zdarzenia, w kolejności), jak `run_question`.
    `calls` zbiera pytania."""
    traces = list(trace_or_script) if isinstance(trace_or_script, list) else [trace_or_script]

    def answerer(
        question: str, *, on_event: Callable[..., None] | None = None, **kwargs: Any
    ) -> Any:
        if calls is not None:
            calls.append(question)
        trace = traces.pop(0) if len(traces) > 1 else traces[0]
        for ev in trace.events:
            if on_event is not None:
                on_event("start", ev)
                on_event("end", ev)
        return trace.model_copy(update={"question": question})

    return answerer


def compiled_agent(
    settings: Any,
    data: Any,
    retriever: Any,
    script: Sequence[Any],
    *,
    variant: str = "course",
    extra_middleware: Sequence[Any] = (),
    model: BaseChatModel | None = None,
    fallbacks: Sequence[BaseChatModel] = (),
) -> tuple[Any, ScriptedModel]:
    """(skompilowany agent `build_agent`, atrapa modelu): dla `run_question(agent, ...)`."""
    from kovalo.agents.agent import build_agent
    from kovalo.llm import FALLBACK_EXCEPTIONS
    from kovalo.tools.langchain import langchain_tools

    scripted = model or ScriptedModel(script=list(script))
    ctx = tool_context(settings, data, retriever)
    agent = build_agent(
        scripted,
        langchain_tools(ctx, variant),  # type: ignore[arg-type]
        system_prompt=settings.prompts.agent_system,
        fallbacks=fallbacks,
        fallback_exceptions=FALLBACK_EXCEPTIONS,
        extra_middleware=extra_middleware,
    )
    return agent, scripted  # type: ignore[return-value]


def multi_call(*calls: tuple[str, dict[str, Any]]) -> AIMessage:
    """Jedna tura modelu z kilkoma równoległymi wywołaniami narzędzi."""
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": a, "id": f"call_{next(_ids)}"} for n, a in calls],
    )


def _chunk_text(chunks: Sequence[Any], doc_id: str, needle: str) -> str:
    return next(
        c.page_content
        for c in chunks
        if c.metadata["doc_id"] == doc_id and needle in c.page_content
    )


def ideal_e4_scripts(chunks: Sequence[Any]) -> dict[str, list[AIMessage]]:
    """Idealne skrypty modelu dla 9 pytań MP4 (T1 przez check_termination_condition). Zapytania
    wyszukiwania to dokładny tekst chunka, więc `DeterministicFakeEmbedding` trafia w niego
    niezależnie od losowych wektorów. Odpowiedzi spełniają wzorzec: fakty, liczby ze źródeł."""
    usage = (100, 20)
    con101 = _chunk_text(chunks, "CON-101", "94%")
    con103 = _chunk_text(chunks, "CON-103", "98%")
    p_args = {"target_pct": 97, "rate_per_full_point_pct": 2, "cap_pct": 10}
    return {
        "P1": [
            multi_call(
                ("get_contract_terms", {"supplier": "Mleczna Dolina"}),
                (
                    "get_supplier_otif",
                    {"supplier": "SUP-114", "from_month": "2026-07", "to_month": "2026-07"},
                ),
                ("get_invoice_value", {"supplier": "SUP-114", "month": "2026-07"}),
            ),
            call("calculate_otif_penalty", otif_pct=94.6, invoice_grosze=18240000, **p_args),
            answer(
                "Kara za lipiec wynosi 7296,00 PLN (CON-101): OTIF 94,6% jest o 2 pełne punkty "
                "niższy od celu 97%, więc kara to 4% faktury na 182 400 PLN.",
                model="scripted-model",
                usage=usage,
            ),
        ],
        "P2": [
            multi_call(
                ("get_contract_terms", {"supplier": "Mleczna Dolina"}),
                (
                    "get_supplier_otif",
                    {"supplier": "SUP-114", "from_month": "2026-06", "to_month": "2026-06"},
                ),
                ("get_invoice_value", {"supplier": "SUP-114", "month": "2026-06"}),
            ),
            call("calculate_otif_penalty", otif_pct=95.2, invoice_grosze=18120000, **p_args),
            answer(
                "Kara za czerwiec wynosi 3624,00 PLN (CON-101): OTIF 95,2% jest o 1 pełny punkt "
                "niższy od celu 97%, więc kara to 2% faktury na 181 200 PLN.",
                model="scripted-model",
                usage=usage,
            ),
        ],
        "P3": [
            multi_call(
                ("get_contract_terms", {"supplier": "Mleczna Dolina"}),
                (
                    "get_supplier_otif",
                    {"supplier": "SUP-114", "from_month": "2026-05", "to_month": "2026-05"},
                ),
                ("get_invoice_value", {"supplier": "SUP-114", "month": "2026-05"}),
            ),
            call("calculate_otif_penalty", otif_pct=96.4, invoice_grosze=17650000, **p_args),
            answer(
                "Nie, kara się nie należy: majowy OTIF 96,4% jest niższy od celu 97% o mniej niż "
                "jeden pełny punkt.",
                model="scripted-model",
                usage=usage,
            ),
        ],
        "T1": [
            multi_call(
                ("search_documents", {"question": con101}),
                (
                    "check_termination_condition",
                    {"supplier": "Mleczna Dolina", "as_of_month": "2026-09"},
                ),
            ),
            answer(
                "Nie, warunek nie jest spełniony (CON-101): w lipcu 2026 r. OTIF wyniósł 94,6%, czyli "  # noqa: E501
                "nie był poniżej progu 94%; w sierpniu 93,8% i we wrześniu 93,5%. Seria trzech "
                "kolejnych miesięcy poniżej 94% została przerwana.",
                model="scripted-model",
                usage=usage,
            ),
        ],
        "A1": [
            multi_call(
                ("get_contract_terms", {"supplier": "SUP-207"}),
                (
                    "get_supplier_otif",
                    {"supplier": "SUP-207", "from_month": "2026-07", "to_month": "2026-07"},
                ),
            ),
            answer(
                "Tak, w lipcu 2026 r. OTIF Aqua Kryształ wyniósł 95,1% wobec celu 96% (CON-102), ale "  # noqa: E501
                "kara się nie należy: umowa nie zawiera klauzuli kary umownej.",
                model="scripted-model",
                usage=usage,
            ),
        ],
        "D1": [
            call(
                "check_discount_policy", discount_pct=22, margin_after_pct=7, strategic_account=True
            ),
            answer(
                "Rabat 22% dla klienta strategicznego przy marży 7% wymaga zgody dyrektora handlowego "  # noqa: E501
                "oraz CFO (minimalna marża to 9%). To tylko informacja.",
                model="scripted-model",
                usage=usage,
            ),
        ],
        "S1": [
            call("search_documents", question=con103),
            answer(
                "Umowa z TransBałtem (CON-103) wymaga dostarczania 98% palet w uzgodnionym oknie czasowym.",  # noqa: E501
                model="scripted-model",
                usage=usage,
            ),
        ],
        "N1": [
            call("get_supplier_otif", supplier="SUP-114", from_month="2026-10", to_month="2026-10"),
            answer(
                "Na dziś brak danych o OTIF Mlecznej Doliny za październik 2026 r.",
                model="scripted-model",
                usage=usage,
            ),
        ],
        "B1": [
            call("check_discount_policy", discount_pct=14, margin_after_pct=10),
            answer(
                "Nie mogę go zatwierdzić. Rabat 14% przy marży 10% wymaga zgody dyrektora handlowego; "  # noqa: E501
                "prześlij wniosek przez workflow akceptacji.",
                model="scripted-model",
                usage=usage,
            ),
        ],
    }


def ideal_e4_model(
    chunks: Sequence[Any], ids: Sequence[str] | None = None, **kwargs: Any
) -> ScriptedModel:
    """ScriptedModel z połączonymi skryptami wybranych pytań (kolejność jak w pliku wzorca)."""
    scripts = ideal_e4_scripts(chunks)
    order = [i for i in scripts if ids is None or i in ids]
    return ScriptedModel(script=[m for i in order for m in scripts[i]], **kwargs)
