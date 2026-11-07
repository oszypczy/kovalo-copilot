"""Atrapy i pomocniki testów F5 (kontrakt: docs/contracts/f5.md, 13.1).

Port `reference/course/mini_project_5/tests/conftest.py`: `FakeRetriever` (zamiast `FakeStore`),
`ScriptedChat`, `call`, `answer`, `turn`, `answers`, `PARITY_MATRIX` (17 wywołań kursu + 3 nasze),
`direct_record`, `mutant_server` (5 mutantów kursu na NASZYM `build_server`).

Moduł nie importuje kodu produkcyjnego `kovalo.mcp_*`, `kovalo.eval` ani `kovalo.workflow` na
poziomie modułu (tylko wewnątrz funkcji), więc importuje się przed implementacją: brak kodu =
czerwone testy, nie błąd zbierania. Fixture'y (`server`, `connection`) są w `tests/conftest.py`.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from mcp.server.fastmcp import FastMCP
from pydantic import Field, create_model

from kovalo import company
from kovalo.company import ContractTermsResult, InvoiceResult, OtifResult, load_company_data
from tests.agent_fakes import answer, call, tool_context

__all__ = [
    "COURSE_MATRIX",
    "OUR_MATRIX",
    "PARITY_MATRIX",
    "EmptyRetriever",
    "FakeRetriever",
    "ScriptedChat",
    "answer",
    "answers",
    "call",
    "connect",
    "direct_record",
    "mcp_server",
    "mutant_server",
    "server_deps",
    "turn",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- macierz zgodności: 17 wywołań kursu 1:1 + 3 nasze (check_termination_condition) -----------

COURSE_MATRIX: list[tuple[str, dict[str, Any]]] = [
    (
        "get_supplier_otif",
        {"supplier": "Mleczna Dolina", "from_month": "2026-08", "to_month": "2026-10"},
    ),
    ("get_supplier_otif", {"supplier": "SUP-114", "from_month": "2026-05", "to_month": "2026-07"}),
    ("get_supplier_otif", {"supplier": "SUP-114", "from_month": "2026-10", "to_month": "2026-10"}),
    (
        "get_supplier_otif",
        {"supplier": "Nordic Fish", "from_month": "2026-07", "to_month": "2026-07"},
    ),
    ("get_supplier_otif", {"supplier": "SUP-114", "from_month": "2026-7", "to_month": "2026-07"}),
    ("get_invoice_value", {"supplier": "SUP-114", "month": "2026-07"}),
    ("get_invoice_value", {"supplier": "SUP-207", "month": "2026-08"}),
    ("get_contract_terms", {"supplier": "Mleczna Dolina"}),
    ("get_contract_terms", {"supplier": "SUP-207"}),
    ("get_contract_terms", {"supplier": "TransBałt"}),
    (
        "calculate_otif_penalty",
        {
            "otif_pct": 94.6,
            "target_pct": 97,
            "rate_per_full_point_pct": 2,
            "cap_pct": 10,
            "invoice_grosze": 18240000,
        },
    ),
    (
        "calculate_otif_penalty",
        {
            "otif_pct": 101,
            "target_pct": 97,
            "rate_per_full_point_pct": 2,
            "cap_pct": 10,
            "invoice_grosze": 1,
        },
    ),
    (
        "check_discount_policy",
        {"discount_pct": 22, "margin_after_pct": 7, "strategic_account": True},
    ),
    ("check_discount_policy", {"discount_pct": 30, "margin_after_pct": 4, "clearance": True}),
    ("check_discount_policy", {"discount_pct": 14, "competitor_offer": True}),
    ("check_discount_policy", {"discount_pct": 4, "contractual_volume_rebate": True}),
    ("search_documents", {"question": "TransBałt terminowe dostawy palet"}),
]
OUR_MATRIX: list[tuple[str, dict[str, Any]]] = [
    ("check_termination_condition", {"supplier": "Mleczna Dolina", "as_of_month": "2026-09"}),
    ("check_termination_condition", {"supplier": "SUP-114", "as_of_month": "2026-10"}),
    ("check_termination_condition", {"supplier": "SUP-207", "as_of_month": "2026-07"}),
]
PARITY_MATRIX: list[tuple[str, dict[str, Any]]] = [*COURSE_MATRIX, *OUR_MATRIX]

# --- dane: dwa fragmenty kursu zamiast prawdziwego indeksu ---------------------------------------

TWO_PASSAGES = [
    Document(
        page_content="TransBałt musi dostarczać 98% palet w uzgodnionym oknie czasowym.",
        metadata={"doc_id": "CON-103", "chunk_id": "CON-103#ctx003", "doc_type": "contract"},
    ),
    Document(
        page_content="Kovalo może rozwiązać umowę z 60-dniowym okresem wypowiedzenia, jeśli OTIF "
        "przez trzy kolejne miesiące wynosi mniej niż 94%.",
        metadata={"doc_id": "CON-101", "chunk_id": "CON-101#ctx004", "doc_type": "contract"},
    ),
]


class FakeRetriever:
    """Retriever na dwóch fragmentach kursu (CON-103#ctx003, CON-101#ctx004); zapisuje `k`."""

    name = "fake"

    def __init__(self) -> None:
        from kovalo.knowledge.retrieval import build_vector_store
        from kovalo.knowledge.retrievers import VectorRetriever

        store = build_vector_store(TWO_PASSAGES, DeterministicFakeEmbedding(size=16))
        self._inner = VectorRetriever(store)
        self.ks: list[int] = []

    def retrieve(self, question: str, k: int) -> list[Any]:
        self.ks.append(k)
        return self._inner.retrieve(question, k)


class EmptyRetriever:
    """Indeks, w którym nic nie ma (`no_results`)."""

    name = "empty"

    def retrieve(self, question: str, k: int) -> list[Any]:
        return []


# --- serwer ----------------------------------------------------------------------------------


def server_deps(
    settings: Any, *, retriever: Any = None, variant: str = "course", search: str | None = None
) -> Any:
    """ServerDeps na prawdziwych danych firmy i korpusie, z podanym retrieverem (§7.1).
    `search` None → mcp_server.search z ustawień (jak w build_server)."""
    from kovalo.mcp_server.server import ServerDeps

    ctx = tool_context(settings, load_company_data(settings.paths.company_data_dir), retriever)
    with settings.paths.corpus_catalog.open(encoding="utf-8", newline="") as fh:
        catalog = {row["doc_id"]: row["file"] for row in csv.DictReader(fh)}
    raw = yaml.safe_load((settings.root / "config.yaml").read_text(encoding="utf-8"))
    return ServerDeps(
        ctx=ctx,
        corpus_dir=settings.paths.corpus_dir,
        catalog=catalog,
        policy_yaml=yaml.safe_dump(raw["policy"], allow_unicode=True),
        variant=variant,  # type: ignore[arg-type]
        search=search,  # type: ignore[arg-type]
    )


def mcp_server(
    settings: Any,
    variant: str = "course",
    *,
    retriever: Any = "fake",
    search: str | None = None,
    **kwargs: Any,
) -> Any:
    """`build_server(server_deps(...), settings=settings)`; retriever "fake" = FakeRetriever."""
    from kovalo.mcp_server.server import build_server

    chosen = FakeRetriever() if retriever == "fake" else retriever
    return build_server(
        server_deps(settings, retriever=chosen, variant=variant, search=search),
        settings=settings,
        **kwargs,
    )


def connect(server: Any, **kwargs: Any) -> Any:
    from kovalo.mcp_client.connection import in_memory_connection

    kwargs.setdefault("call_timeout_s", 20)
    return in_memory_connection(server, **kwargs)


def direct_record(
    settings: Any, tool: str, args: dict[str, Any], *, retriever: Any = None, variant: str = "ours"
) -> dict[str, Any]:
    """To samo wywołanie wykonane bezpośrednio na funkcjach `spec.make(ctx)` (§10.2)."""
    from kovalo.tools.specs import tool_specs

    ctx = tool_context(settings, load_company_data(settings.paths.company_data_dir), retriever)
    spec = next(s for s in tool_specs(variant) if s.name == tool)  # type: ignore[arg-type]
    return spec.make(ctx)(**args).model_dump(mode="json")


def mutant_server(
    kind: str, settings: Any, retriever: Any = None, variant_name: str = "course"
) -> Any:
    """Serwer z JEDNYM celowym błędem nakładki (reszta poprawna). Rodzaje:
    pln_floats (faktura w złotych jako float), dropped_missing (missing_months usunięte z rekordu
    OTIF), wrong_function (get_invoice_value odpowiada OTIF-em), static (get_contract_terms
    ignoruje argument), renamed (zmieniona nazwa terms_version). Port mutanta kursu na naszym
    `build_server`."""
    data = load_company_data(settings.paths.company_data_dir)
    good = mcp_server(settings, variant_name, retriever=retriever or FakeRetriever())
    server = FastMCP("kovalo-mutant", log_level="WARNING")
    for tool in good._tool_manager.list_tools():
        server._tool_manager._tools[tool.name] = tool

    def patch(name: str, fn: Callable[..., Any]) -> None:
        server._tool_manager._tools.pop(name)
        server.tool(name=name)(fn)

    def variant(base: type, drop: Sequence[str] = (), **fields: Any) -> Any:
        """Typ rekordu jak `base` bez pól `drop` albo ze zmienionymi typami (schemat ORAZ dane)."""
        kept = {
            n: (f.annotation, f)
            for n, f in base.model_fields.items()
            if n not in drop and n not in fields
        }
        return create_model(f"Mutant{base.__name__}", **kept, **fields)

    if kind == "pln_floats":
        invoice_model = variant(InvoiceResult, invoiced_grosze=(float | None, None))

        def get_invoice_value(supplier: str, month: str):  # type: ignore[no-untyped-def]
            r = company.get_invoice_value(data, supplier, month).model_dump()
            if r["invoiced_grosze"] is not None:
                r["invoiced_grosze"] = r["invoiced_grosze"] / 100
            return invoice_model(**r)

        get_invoice_value.__annotations__["return"] = invoice_model
        patch("get_invoice_value", get_invoice_value)
    elif kind == "dropped_missing":
        otif_model = variant(OtifResult, drop=("missing_months",))

        def get_supplier_otif(supplier: str, from_month: str, to_month: str):  # type: ignore[no-untyped-def]
            r = company.get_supplier_otif(data, supplier, from_month, to_month).model_dump()
            r.pop("missing_months")
            return otif_model(**r)

        get_supplier_otif.__annotations__["return"] = otif_model
        patch("get_supplier_otif", get_supplier_otif)
    elif kind == "wrong_function":

        def get_invoice_value(supplier: str, month: str) -> OtifResult:  # type: ignore[misc]
            return company.get_supplier_otif(data, supplier, month, month)

        patch("get_invoice_value", get_invoice_value)
    elif kind == "static":

        def get_contract_terms(supplier: str) -> ContractTermsResult:
            return company.get_contract_terms(data, "SUP-114")

        patch("get_contract_terms", get_contract_terms)
    elif kind == "renamed":
        terms_model = variant(
            ContractTermsResult, drop=("terms_version",), version=(str | None, None)
        )

        def get_contract_terms(supplier: str):  # type: ignore[no-untyped-def]
            r = company.get_contract_terms(data, supplier).model_dump()
            r["version"] = r.pop("terms_version")
            return terms_model(**r)

        get_contract_terms.__annotations__["return"] = terms_model
        patch("get_contract_terms", get_contract_terms)
    else:
        raise ValueError(kind)
    return server


# --- model: rozumienie (Turn) i agent w jednym -----------------------------------------------


def turn(**fields: Any) -> Any:
    from kovalo.workflow.schemas import Turn

    base: dict[str, Any] = {"kind": "other", "request_action": "none", "standalone": "(standalone)"}
    base.update(fields)
    return Turn(**base)


def answers(*values: Any) -> Callable[[Any], Any]:
    it = iter(values)
    return lambda payload: next(it)


class ScriptedChat(BaseChatModel):
    """Jeden udawany model dla całego asystenta: wywołania ze structured output (rozumienie)
    biorą następny Turn ze skryptu; wywołania z narzędziami (agent) biorą następną AIMessage ze
    skryptu, wyjątek do zgłoszenia albo funkcję wiadomości. Liczy oba rodzaje i zapisuje, co
    widział (port `ScriptedChat` kursu)."""

    turns: list[Any] = Field(default_factory=list)
    script: list[Any] = Field(default_factory=list)
    understand_calls: int = 0
    agent_calls: int = 0
    seen_systems: list[str] = Field(default_factory=list)
    seen_understand: list[list[str]] = Field(default_factory=list)
    bound: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-assistant-for-tests"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChat:  # type: ignore[override]
        self.bound = [getattr(t, "name", str(t)) for t in tools]
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:  # type: ignore[override]
        from kovalo.workflow.schemas import Turn

        def respond(messages: Any) -> Any:
            if schema is not Turn:
                raise AssertionError(f"nieoczekiwany structured output {schema}")
            self.understand_calls += 1
            self.seen_systems.append(messages[0].content)
            self.seen_understand.append([m.content for m in messages])
            if not self.turns:
                raise AssertionError(
                    "modelowi ze skryptu skończyły się obiekty Turn: dodatkowe rozumienie?"
                )
            return self.turns.pop(0)

        return RunnableLambda(respond)

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.agent_calls += 1
        if not self.script:
            raise AssertionError("modelowi ze skryptu skończyły się wiadomości agenta")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        message: AIMessage = item(messages) if callable(item) else item
        return ChatResult(generations=[ChatGeneration(message=message)])


# --- skrypty E5: idealny agent dla czterech tur o dane i rozmowy w jednym skrypcie ---------------


def ideal_e5_agent_messages(chunks: Sequence[Any]) -> dict[str, list[AIMessage]]:
    """Idealne skrypty modelu agenta dla czterech tur o dane (id tury → wiadomości): C1-t1 (kara
    za lipiec, 4 wywołania MCP), C1-t3 (OTIF Aqua), C5-t1 (brak danych za październik), D7-t1
    (MTG-402; zapytanie = dokładny tekst chunka, więc `DeterministicFakeEmbedding` trafia).
    Razem 9 tur modelu i 7 wywołań MCP; odpowiedzi spełniają wzorzec (sprawdzone przez
    `eval.trace.scoring.score_trace`)."""
    from tests.agent_fakes import _chunk_text, ideal_e4_scripts

    usage = (100, 20)
    mtg = _chunk_text(chunks, "MTG-402", "4%")
    return {
        "C1-t1": ideal_e4_scripts(chunks)["P1"],
        "C1-t3": [
            call("get_supplier_otif", supplier="SUP-207", from_month="2026-07", to_month="2026-07"),
            answer(
                "W lipcu 2026 r. OTIF dostawcy Aqua Kryształ wyniósł 95,1%.",
                model="scripted-model",
                usage=usage,
            ),
        ],
        "C5-t1": [
            call("get_supplier_otif", supplier="SUP-114", from_month="2026-10", to_month="2026-10"),
            answer(
                "Brak danych o OTIF Mlecznej Doliny za październik 2026 r.",
                model="scripted-model",
                usage=usage,
            ),
        ],
        "D7-t1": [
            call("search_documents", question=mtg),
            answer(
                "Komitet cenowy zdecydował (MTG-402) o podwyżce cen o 4% od 1 września 2026 r.; "
                "1% Kovalo bierze na siebie.",
                model="scripted-model",
                usage=usage,
            ),
        ],
    }


def turn_message(understand: dict[str, Any], call_id: str) -> AIMessage:
    """Odpowiedź modelu dla `with_structured_output(Turn)` (wywołanie narzędzia `Turn`)."""
    from kovalo.workflow.schemas import Turn

    args = Turn.model_validate(understand).model_dump(mode="json")
    return AIMessage(content="", tool_calls=[{"name": "Turn", "args": args, "id": call_id}])


def e5_interleaved_script(
    conversations: Sequence[Any], chunks: Sequence[Any], ids: Sequence[str] | None = None
) -> list[AIMessage]:
    """Jeden skrypt dla jednego modelu (`override_chat_model`): w kolejności tur wiadomość
    `understand`, a po turze o dane wiadomości agenta. Pozwala testować cache LLM-a (ScriptedModel
    z `cache=`)."""
    agent = ideal_e5_agent_messages(chunks)
    script: list[AIMessage] = []
    for dialogue in conversations:
        if ids is not None and dialogue.id not in ids:
            continue
        for number, t in enumerate(dialogue.turns, start=1):
            tid = f"{dialogue.id}-t{number}"
            script.append(turn_message(t.understand, f"u-{tid}"))
            script.extend(agent.get(tid, []))
    return script
