"""McpConnection (kontrakt F5, 8.2 i 13.2): port `test_provided_mcp_runtime.py` i
`test_provided_mp4_guards_over_mcp.py` kursu na naszych modułach (serwer w pamięci, model ze
skryptem, ocena E4) plus nasze: narzędzia z artefaktem, filtrowanie nazw, zgodność z narzędziami
bezpośrednimi (warunek trafień w cache F4, U9), limit zapisów."""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from mcp.server.fastmcp import FastMCP

from tests.mcp_fakes import (
    PARITY_MATRIX,
    EmptyRetriever,
    FakeRetriever,
    ScriptedChat,
    answer,
    call,
    connect,
    mcp_server,
    mutant_server,
)

INVOICE = {
    "status": "ok",
    "supplier_id": "SUP-114",
    "supplier": "Mleczna Dolina",
    "month": "2026-07",
    "invoiced_grosze": 18240000,
    "currency": "PLN",
    "message": None,
}
INVOICE_ARGS = {"supplier": "SUP-114", "month": "2026-07"}


def error_class():
    from kovalo.mcp_client.connection import McpCallError

    return McpCallError


def agent_over(settings, tools, script):
    """build_agent z atrapą modelu i promptem agent_system (jak AgentRunner w aplikacji)."""
    from kovalo.agents.agent import build_agent

    model = ScriptedChat(script=list(script))
    return model, build_agent(model, tools, system_prompt=settings.prompts.agent_system)


def run(agent, question, steps=10):
    from kovalo.agents.trace import Budget, run_question

    return run_question(agent, question, Budget(steps, 60))


# --- mutanty: opakowanie nie może zdefiniować swojego błędu -------------------------------------


@pytest.mark.parametrize(
    ("kind", "tool", "args", "says"),
    [
        (
            "dropped_missing",
            "get_supplier_otif",
            {"supplier": "SUP-114", "from_month": "2026-08", "to_month": "2026-10"},
            "missing_months: brak",
        ),
        ("renamed", "get_contract_terms", {"supplier": "SUP-114"}, "terms_version: brak"),
        ("pln_floats", "get_invoice_value", INVOICE_ARGS, "invoiced_grosze"),
    ],
)
def test_a_wrapper_cannot_define_away_its_own_defect(settings, kind, tool, args, says):
    # mutant usuwa pole albo zmienia jego typ ZARÓWNO w ogłaszanym schemacie, jak i w danych
    with connect(mutant_server(kind, settings)) as conn:
        with pytest.raises(error_class(), match=says):
            conn.call(tool, args)
        capture = conn.captures[-1]
    assert capture["outcome"] == "validation" and capture["structured"] is not None, (
        "surowe dane nadal można obejrzeć"
    )


def test_a_rejected_record_is_never_evidence(settings):
    with connect(mutant_server("pln_floats", settings)) as conn:
        _, agent = agent_over(
            settings,
            conn.tools(),
            [call("get_invoice_value", **INVOICE_ARGS), answer("Faktura wyniosła 182 400 PLN.")],
        )
        trace = run(agent, "faktura?")
    event = trace.events[0]
    assert event.status == "error" and event.result is None
    assert "[błąd walidacji lokalnej]" in event.error


# --- cztery warstwy błędów ----------------------------------------------------------------------


def test_unknown_tool_is_a_tool_error(connection):
    with pytest.raises(error_class(), match=r"\[błąd narzędzia\]") as info:
        connection.call("approve_discount", {"discount_pct": 14})
    assert info.value.layer == "tool"
    assert connection.captures[-1]["outcome"] in {"tool", "protocol"}


def test_a_closed_connection_is_a_transport_error(server):
    conn = connect(server)
    conn.close()
    conn.close()  # idempotentne
    with pytest.raises(error_class(), match=r"\[błąd transportu\]"):
        conn.call("get_invoice_value", INVOICE_ARGS)
    assert not conn.alive


def test_mcp_call_error_is_a_tool_exception_with_layer_prefixes():
    from langchain_core.tools import ToolException

    McpCallError = error_class()
    assert issubclass(McpCallError, ToolException)
    prefixes = {
        "tool": "[błąd narzędzia]",
        "protocol": "[błąd protokołu]",
        "transport": "[błąd transportu]",
        "validation": "[błąd walidacji lokalnej]",
    }
    for layer, prefix in prefixes.items():
        exc = McpCallError(layer, "x", "abc123")
        assert str(exc) == f"{prefix} x (wywołanie abc123)" and exc.layer == layer
        assert exc.invocation_id == "abc123"


def test_startup_error_is_a_config_error():
    from kovalo.config import ConfigError
    from kovalo.mcp_client.connection import McpStartupError

    assert issubclass(McpStartupError, ConfigError)


# --- tożsamość, wywołania równoległe, limity czasu ----------------------------------------------


def test_parallel_identical_calls_stay_distinct(connection):
    with ThreadPoolExecutor(4) as pool:
        results = list(
            pool.map(lambda _: connection.call("get_invoice_value", INVOICE_ARGS), range(4))
        )
    ids = [provenance["invocation_id"] for _, provenance in results]
    assert len(set(ids)) == 4
    assert sorted(ids) == sorted(c["invocation_id"] for c in connection.captures)


def test_call_returns_record_and_provenance(connection):
    connection.context.update(thread_id="T1", turn=3)
    record, provenance = connection.call("get_invoice_value", INVOICE_ARGS)
    assert record == INVOICE
    assert set(provenance) >= {
        "transport",
        "invocation_id",
        "server",
        "thread_id",
        "turn",
        "seconds",
    }
    assert (provenance["transport"], provenance["server"]) == ("mcp", "kovalo")
    assert (provenance["thread_id"], provenance["turn"]) == ("T1", 3)
    assert provenance["seconds"] >= 0
    assert connection.captures[0]["invocation_id"] == provenance["invocation_id"]
    assert connection.captures[0]["outcome"] == "ok"


def slow_server(seconds: float):
    from kovalo.company import InvoiceResult

    server = FastMCP("slow", log_level="WARNING")

    @server.tool()
    async def get_invoice_value(supplier: str, month: str) -> InvoiceResult:
        """Wolna faktura (dubler testowy)."""
        await asyncio.sleep(seconds)
        return InvoiceResult(
            status="ok", supplier_id="SUP-114", supplier="M", month=month, invoiced_grosze=1
        )

    return server


def test_a_timed_out_call_is_cancelled_and_never_comes_back():
    conn = connect(slow_server(1.0), call_timeout_s=0.2, cleanup_timeout_s=3)
    started = time.monotonic()
    with pytest.raises(error_class(), match=r"\[błąd transportu\].*limit czasu"):
        conn.call("get_invoice_value", INVOICE_ARGS)
    assert time.monotonic() - started < 3, "czeka tylko do terminu wywołania (i sprzątania)"
    time.sleep(1.3)  # serwer zdążyłby już odpowiedzieć
    assert conn.captures[0]["structured"] is None and conn.captures[0]["outcome"] == "transport", (
        "żadnych spóźnionych dowodów"
    )
    with pytest.raises(error_class(), match="nie nadaje się do użycia"):
        conn.call("get_invoice_value", INVOICE_ARGS)
    assert conn.unusable
    conn.close()
    assert not conn.alive


def test_close_while_a_call_is_pending_is_bounded():
    conn = connect(slow_server(30), call_timeout_s=60, cleanup_timeout_s=1)
    errors: list = []

    def pending():
        with pytest.raises(error_class()) as info:
            conn.call("get_invoice_value", {"supplier": "S", "month": "2026-07"})
        errors.append(info.value)

    worker = threading.Thread(target=pending)
    worker.start()
    time.sleep(0.3)
    started = time.monotonic()
    conn.close()
    assert time.monotonic() - started < 8 and not conn.alive
    worker.join(timeout=10)
    assert not worker.is_alive(), "oczekujący wywołujący zostaje zwolniony"
    assert errors and errors[0].layer == "transport"


# --- dwa bloki tekstu i zabezpieczenia MP4 przez MCP --------------------------------------------


def two_block_server(currency="PLN", second='{"invoiced_grosze": 1}'):
    """Prawdziwy serwer, którego narzędzie odpowiada structuredContent i DWOMA blokami tekstu."""
    from mcp.types import CallToolResult, TextContent

    server = FastMCP("two-blocks", log_level="WARNING")
    record = {**INVOICE, "currency": currency}

    @server.tool()
    def get_invoice_value(supplier: str, month: str) -> CallToolResult:
        """Faktura z dwoma blokami tekstu (dubler testowy)."""
        return CallToolResult(
            content=[
                TextContent(type="text", text=json.dumps(record)),
                TextContent(type="text", text=second),
            ],
            structuredContent=record,
        )

    return server


@pytest.mark.parametrize("server_args", [{"second": '{"invoiced_grosze": 1}'}, {"currency": "USD"}])
def test_conflicting_or_foreign_records_are_rejected_over_mcp_and_back_no_numbers(
    settings, server_args
):
    from kovalo.eval.datasets import AgentGold
    from kovalo.eval.trace.scoring import score_trace

    second = server_args.get("second") or json.dumps({**INVOICE, "currency": "USD"})
    server = two_block_server(currency=server_args.get("currency", "PLN"), second=second)
    with connect(server) as conn:
        with pytest.raises(error_class(), match=r"\[błąd walidacji lokalnej\]"):
            conn.call("get_invoice_value", INVOICE_ARGS)
        assert conn.captures[-1]["outcome"] == "validation"
        assert conn.captures[-1]["structured"] is not None
        _, agent = agent_over(
            settings,
            conn.tools(),
            [call("get_invoice_value", **INVOICE_ARGS), answer("Faktura wyniosła 182 400 PLN.")],
        )
        trace = run(agent, "faktura?")
    assert trace.events[0].status == "error"
    gold = AgentGold(id="x", question="faktura?", kind="x")
    assert score_trace(trace, gold)["uncovered_numbers"] == ["182 400"], (
        "odrzucony rekord nigdy nie stanowi pokrycia dla liczby"
    )


def agent_for_connection(settings, connection, script):
    return agent_over(settings, connection.tools(), script)


def test_an_invented_search_query_is_not_evidence_over_mcp(settings):
    from kovalo.eval.datasets import AgentGold
    from kovalo.eval.trace.scoring import score_trace

    with connect(mcp_server(settings, "course", retriever=EmptyRetriever())) as conn:
        _, agent = agent_for_connection(
            settings,
            conn,
            [
                call("search_documents", question="Czy kara wynosi 123456 PLN?"),
                answer("Kara wynosi 123456 PLN."),
            ],
        )
        trace = run(agent, "Jaka kara?")
    assert trace.events[0].transport == "mcp" and trace.events[0].result["passages"] == []
    gold = AgentGold(id="mcp", question="Jaka kara?", kind="mcp")
    assert score_trace(trace, gold)["uncovered_numbers"] == ["123456"]


def test_grosze_from_an_mcp_calculation_back_pln_only(settings, connection):
    from kovalo.eval.datasets import AgentGold
    from kovalo.eval.trace.scoring import score_trace

    question = "Ile wynosi kara za lipiec?"
    script = [
        call(
            "calculate_otif_penalty",
            otif_pct=94.6,
            target_pct=97,
            rate_per_full_point_pct=2,
            cap_pct=10,
            invoice_grosze=18240000,
        ),
        answer("Kara wynosi 7296 PLN, a nie 729 600 PLN."),
    ]
    _, agent = agent_for_connection(settings, connection, script)
    trace = run(agent, question)
    assert trace.events[0].result["penalty_grosze"] == 729600
    gold = AgentGold(id="mcp", question=question, kind="mcp")
    assert score_trace(trace, gold)["uncovered_numbers"] == ["729 600"]


def test_parallel_mcp_results_crossing_the_limit_are_all_kept(settings, connection):
    batch = AIMessage(
        content="",
        tool_calls=[
            {"id": sid, "name": "get_invoice_value", "args": {"supplier": "SUP-114", "month": m}}
            for sid, m in [("a", "2026-05"), ("b", "2026-06"), ("c", "2026-07")]
        ],
    )
    model, agent = agent_for_connection(settings, connection, [batch, answer("gotowe")])
    trace = run(agent, "faktury?", steps=2)
    assert trace.status == "limit" and model.agent_calls == 1
    assert sorted((e.call_id, e.status, e.transport) for e in trace.events) == [
        ("a", "ok", "mcp"),
        ("b", "ok", "mcp"),
        ("c", "ok", "mcp"),
    ]
    assert len({e.invocation_id for e in trace.events}) == 3


def test_a_missing_month_needs_the_reported_absence_over_mcp(settings, connection):
    from kovalo.eval.datasets import AgentGold, Need
    from kovalo.eval.trace.scoring import score_trace

    question = "OTIF za październik?"
    _, agent = agent_for_connection(
        settings,
        connection,
        [
            call("get_supplier_otif", supplier="SUP-114", from_month="2026-10", to_month="2026-10"),
            answer("Brak danych za październik."),
        ],
    )
    trace = run(agent, question)
    need = Need(tool="get_supplier_otif", supplier_id="SUP-114", missing=["2026-10"])
    gold = AgentGold(id="mcp", question=question, kind="mcp", needs=[need])
    assert score_trace(trace, gold)["tool_coverage"] == 1.0


# --- NASZE: tools(), nazwy, zgodność z narzędziami bezpośrednimi, zapisy ------------------------


def tool_call(tool, args, call_id="c1"):
    return {"type": "tool_call", "id": call_id, "name": tool.name, "args": args}


def test_tools_return_record_and_artifact(connection):
    tools = {t.name: t for t in connection.tools()}
    assert tools["get_invoice_value"].response_format == "content_and_artifact"
    message = tools["get_invoice_value"].invoke(tool_call(tools["get_invoice_value"], INVOICE_ARGS))
    assert isinstance(message, ToolMessage) and message.status == "success"
    assert json.loads(message.content) == INVOICE, "model widzi sam rekord, bez pochodzenia"
    assert "invocation_id" not in message.content and "__mcp__" not in message.content
    capture = connection.captures[-1]
    assert message.artifact["invocation_id"] == capture["invocation_id"]
    assert message.artifact["transport"] == "mcp"


def test_a_tool_error_is_a_readable_tool_message(connection):
    """handle_tool_error=True: McpCallError (ToolException) staje się ToolMessage z błędem, który
    model może przeczytać; trace zapisuje go jako zdarzenie z błędem."""
    tools = {t.name: t for t in connection.tools()}
    tool = tools["calculate_otif_penalty"]
    assert tool.handle_tool_error is True
    message = tool.invoke(
        tool_call(
            tool,
            {
                "otif_pct": "high",
                "target_pct": 97,
                "rate_per_full_point_pct": 2,
                "cap_pct": 10,
                "invoice_grosze": 1,
            },
        )
    )
    assert message.status == "error" and "[błąd narzędzia]" in message.content


def test_tools_filter_by_names(ours_connection):
    from kovalo.mcp_client.connection import McpStartupError
    from kovalo.tools.specs import COURSE_TOOLS, OUR_TOOLS

    assert {t.name for t in ours_connection.tools()} == set(OUR_TOOLS)
    assert {t.name for t in ours_connection.tools(names=COURSE_TOOLS)} == set(COURSE_TOOLS)
    assert len(ours_connection.tools(names=COURSE_TOOLS)) == 6
    with pytest.raises(McpStartupError, match="nie_ma_takiego_narzedzia"):
        ours_connection.tools(names={"get_invoice_value", "nie_ma_takiego_narzedzia"})


def test_tools_names_missing_on_the_server_are_listed(connection):
    from kovalo.mcp_client.connection import McpStartupError

    with pytest.raises(McpStartupError, match="check_termination_condition"):
        connection.tools(names={"get_invoice_value", "check_termination_condition"})


def test_mcp_tools_look_like_direct_tools(settings):
    """Warunek trafień w cache F4 (U9): model widzi te same definicje i tę samą treść wyników,
    niezależnie od transportu. `convert_to_openai_tool` równe dla każdego narzędzia; ta sama treść
    `ToolMessage` dla tych samych argumentów (macierz 20 wywołań). Różnica tylko w `title` byłaby
    usuwana przez klienta; test wymaga pełnej równości."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from kovalo.company import load_company_data
    from kovalo.tools.langchain import langchain_tools
    from tests.agent_fakes import tool_context

    retriever = FakeRetriever()
    server = mcp_server(settings, "ours", retriever=retriever)
    ctx = tool_context(settings, load_company_data(settings.paths.company_data_dir), retriever)
    direct = {t.name: t for t in langchain_tools(ctx, "ours")}
    with connect(server) as conn:
        remote = {t.name: t for t in conn.tools()}
        assert set(remote) == set(direct) and len(direct) == 7
        for name, tool in direct.items():
            assert convert_to_openai_tool(remote[name]) == convert_to_openai_tool(tool), name
        for index, (name, args) in enumerate(PARITY_MATRIX):
            via_mcp = remote[name].invoke(tool_call(remote[name], args, f"m{index}"))
            via_direct = direct[name].invoke(tool_call(direct[name], args, f"m{index}"))
            assert via_mcp.content == via_direct.content, (name, args)
            assert via_mcp.status == via_direct.status == "success"
            assert via_direct.artifact is None and via_mcp.artifact["transport"] == "mcp"


def test_capture_limit(settings):
    """`capture_limit=3`: po 5 wywołaniach zostają 3 ostatnie (api: 100, E5: bez limitu)."""
    with connect(mcp_server(settings, "course"), capture_limit=3) as conn:
        ids = [conn.call("get_invoice_value", INVOICE_ARGS)[1]["invocation_id"] for _ in range(5)]
        assert [c["invocation_id"] for c in conn.captures] == ids[-3:]


def test_captures_are_unlimited_by_default(connection):
    for _ in range(5):
        connection.call("get_invoice_value", INVOICE_ARGS)
    assert len(connection.captures) == 5
