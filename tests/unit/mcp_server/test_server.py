"""Serwer MCP Kovalo (kontrakt F5, 7 i 13.2): port testów kroku 1 kursu
(`test_step1_build_server.py`) na wariancie `course` plus testy naszych elementów. Serwer
w pamięci, FakeRetriever, bez sieci."""

from __future__ import annotations

import inspect
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from tests.mcp_fakes import (
    PARITY_MATRIX,
    EmptyRetriever,
    FakeRetriever,
    connect,
    mcp_server,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN_WORDS = ("approve", "zatwierdź", "zapisz", "send", "prześlij")


def listed(connection):
    return {t.name: t for t in connection.run(lambda s: s.list_tools()).tools}


def records():
    from kovalo.tools.specs import TOOL_SPECS

    return {spec.name: spec.record for spec in TOOL_SPECS}


# --- port kroku 1 kursu (wariant course: 6 narzędzi) --------------------------------------------


def test_exactly_the_six_read_only_tools(connection):
    from kovalo.tools.specs import COURSE_TOOLS

    tools = listed(connection)
    assert set(tools) == set(COURSE_TOOLS) and len(tools) == 6
    for name, t in tools.items():
        assert t.annotations and t.annotations.readOnlyHint is True, f"{name}: tylko do odczytu"
        assert t.annotations.idempotentHint is True and t.annotations.openWorldHint is False, name
        assert t.description and len(t.description) > 40, f"{name}: docstring dla modelu"


def test_every_tool_publishes_its_records_output_schema(connection):
    canonical = records()
    for name, t in listed(connection).items():
        assert t.outputSchema, f"{name}: zwracaj typ rekordu, nie zwykły dict"
        assert set(t.outputSchema["properties"]) == set(canonical[name].model_fields), name


def test_input_arguments_match_mini_project_4(connection):
    tools = listed(connection)
    assert set(tools["get_supplier_otif"].inputSchema["properties"]) == {
        "supplier",
        "from_month",
        "to_month",
    }
    assert set(tools["calculate_otif_penalty"].inputSchema["properties"]) == {
        "otif_pct",
        "target_pct",
        "rate_per_full_point_pct",
        "cap_pct",
        "invoice_grosze",
    }
    assert "config" not in str(tools["check_discount_policy"].inputSchema), (
        "zależności nigdy nie są argumentami"
    )


@pytest.mark.parametrize(("tool", "args"), PARITY_MATRIX)
def test_every_outcome_arrives_as_a_valid_structured_record(ours_connection, tool, args):
    """Macierz 17 wywołań kursu + 3 nasze (check_termination_condition)."""
    record, provenance = ours_connection.call(tool, args)  # wyjątek, gdy rekord łamie kontrakt
    assert set(record) == set(records()[tool].model_fields)
    assert provenance["transport"] == "mcp" and provenance["invocation_id"]


def test_a_partial_period_keeps_its_missing_months(connection):
    record, _ = connection.call(
        "get_supplier_otif", {"supplier": "SUP-114", "from_month": "2026-08", "to_month": "2026-10"}
    )
    assert record["missing_months"] == ["2026-10"] and record["complete"] is False


def test_search_uses_the_pinned_top_k(settings):
    retriever = FakeRetriever()
    with connect(mcp_server(settings, "course", retriever=retriever)) as conn:
        record, _ = conn.call("search_documents", {"question": "pallets"})
    assert settings.retrieval.top_k == 3, "PINNED"
    assert retriever.ks == [3]
    assert record["status"] == "ok" and len(record["passages"]) == 2  # atrapa ma dwa fragmenty
    assert record["passages"][0]["doc_id"] in {"CON-103", "CON-101"}


def test_a_bad_argument_is_a_tool_error(connection):
    from kovalo.mcp_client.connection import McpCallError

    with pytest.raises(McpCallError, match=r"\[błąd narzędzia\]"):
        connection.call(
            "calculate_otif_penalty",
            {
                "otif_pct": "high",
                "target_pct": 97,
                "rate_per_full_point_pct": 2,
                "cap_pct": 10,
                "invoice_grosze": 1,
            },
        )


def test_documents_by_exact_catalogue_id(connection):
    text = connection.run(lambda s: s.read_resource("kovalo://documents/CON-103")).contents[0].text
    assert "98% palet" in text


@pytest.mark.parametrize(
    "doc_id",
    [
        "CON-10[3]",
        "CON-10*",
        "XYZ-999",
        "CON-1033",
        "CON-101_supplier_agreement_mleczna_dolina.md",
        "..",
        "con-103",
    ],
)
def test_anything_but_an_exact_id_is_rejected(connection, doc_id):
    from mcp.shared.exceptions import McpError

    with pytest.raises(McpError):
        connection.run(lambda s: s.read_resource(f"kovalo://documents/{doc_id}"))


def test_a_path_is_never_a_document(connection):
    from mcp.shared.exceptions import McpError

    with pytest.raises(McpError):
        connection.run(lambda s: s.read_resource("kovalo://documents/../../config.yaml"))


def test_the_policy_resource(connection, settings):
    assert settings.policy.version == "course-v1"
    text = connection.run(lambda s: s.read_resource("kovalo://policy/course-v1")).contents[0].text
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert yaml.safe_load(text) == raw["policy"]


def test_the_prompt(connection):
    prompt = connection.run(
        lambda s: s.get_prompt("supplier_review", {"supplier": "Mleczna Dolina"})
    )
    assert len(prompt.messages) == 1 and prompt.messages[0].role == "user"
    assert prompt.messages[0].content.text == (
        "Podaj warunki umowy z dostawcą Mleczna Dolina, jego ostatni miesięczny OTIF i "
        "ewentualną należną karę umowną. Przy każdej liczbie podaj źródło (narzędzie albo "
        "dokument)."
    )


def test_nothing_is_printed_to_stdout(settings, capfd):
    with connect(mcp_server(settings, "course")) as conn:
        conn.call("get_invoice_value", {"supplier": "SUP-114", "month": "2026-07"})
    assert capfd.readouterr().out == "", (
        "przy stdio stdout przenosi wyłącznie protokół: loguj na stderr"
    )


# --- NASZE: wariant ours, schematy, FR-5.5, wątki, błędy, /health, main -------------------------


def test_ours_variant_has_seven_tools(ours_connection):
    from kovalo.tools.specs import OUR_TOOLS

    tools = listed(ours_connection)
    assert set(tools) == set(OUR_TOOLS) and len(tools) == 7
    termination = tools["check_termination_condition"]
    assert termination.outputSchema and "otif" in termination.outputSchema["properties"]
    assert termination.annotations.readOnlyHint is True


def _strip_titles(node):
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


def _expected_input_schema(function):
    from pydantic import create_model

    hints = inspect.get_annotations(function, eval_str=True)
    fields = {}
    for param in inspect.signature(function).parameters.values():
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (hints[param.name], default)
    return _strip_titles(create_model("Args", **fields).model_json_schema())


def test_tool_schemas_match_specs(settings, ours_connection):
    """inputSchema i outputSchema każdego narzędzia pochodzą z sygnatury `spec.make(ctx)` i
    `spec.record`: nazwy, typy, opisy, wartości domyślne."""
    from kovalo.company import load_company_data
    from kovalo.tools.specs import TOOL_SPECS
    from tests.agent_fakes import tool_context

    ctx = tool_context(settings, load_company_data(settings.paths.company_data_dir), None)
    tools = listed(ours_connection)
    assert set(tools) == {s.name for s in TOOL_SPECS}
    for spec in TOOL_SPECS:
        function = spec.make(ctx)
        tool = tools[spec.name]
        assert tool.description.strip() == (function.__doc__ or "").strip(), spec.name
        want = _expected_input_schema(function)
        got = _strip_titles(tool.inputSchema)
        assert set(got["properties"]) == set(want["properties"]), spec.name
        assert set(got.get("required", [])) == set(want.get("required", [])), spec.name
        for name, prop in want["properties"].items():
            assert got["properties"][name].get("description") == prop.get("description"), (
                spec.name,
                name,
            )
            assert got["properties"][name].get("default") == prop.get("default"), (spec.name, name)
            assert got["properties"][name].get("type") == prop.get("type"), (spec.name, name)
            assert got["properties"][name].get("anyOf") == prop.get("anyOf"), (spec.name, name)
        out_want = _strip_titles(spec.record.model_json_schema())
        out_got = _strip_titles(tool.outputSchema)
        assert set(out_got["properties"]) == set(out_want["properties"]), spec.name
        assert set(out_got.get("required", [])) == set(out_want.get("required", [])), spec.name
        for name, prop in out_want["properties"].items():
            assert out_got["properties"][name].get("description") == prop.get("description"), (
                spec.name,
                name,
            )


def test_nothing_approves_or_writes(ours_connection):
    """FR-5.5: żadna nazwa ani opis narzędzia, zasobu i promptu nie sugeruje zatwierdzania,
    zapisu ani wysyłki."""
    tools = ours_connection.run(lambda s: s.list_tools()).tools
    templates = ours_connection.run(lambda s: s.list_resource_templates()).resourceTemplates
    resources = ours_connection.run(lambda s: s.list_resources()).resources
    prompts = ours_connection.run(lambda s: s.list_prompts()).prompts
    assert len(tools) == 7 and [p.name for p in prompts] == ["supplier_review"]
    texts = [(t.name, t.description) for t in tools]
    texts += [(t.name, t.description) for t in templates]
    texts += [(r.name, r.description) for r in resources]
    texts += [(p.name, p.description) for p in prompts]
    for name, description in texts:
        blob = f"{name} {description or ''}".lower()
        assert not [w for w in FORBIDDEN_WORDS if w in blob], (name, description)


def test_documents_template_and_policy_resource_are_published(ours_connection):
    """Szablon `kovalo://documents/{doc_id}` i zasób polityki `kovalo://policy/course-v1`
    (tabela 7.3 mówi: szablon + zasób; kryterium 4 mówi „2 szablony zasobów”: rozbieżność
    zgłoszona w raporcie T1, test dopuszcza politykę jako zasób albo szablon)."""
    templates = ours_connection.run(lambda s: s.list_resource_templates()).resourceTemplates
    resources = ours_connection.run(lambda s: s.list_resources()).resources
    assert "kovalo://documents/{doc_id}" in [str(t.uriTemplate) for t in templates]
    policy = [str(t.uriTemplate) for t in templates] + [str(r.uri) for r in resources]
    assert any(u.startswith("kovalo://policy/") for u in policy)


class SleepingRetriever:
    name = "sleeping"

    def retrieve(self, question, k):
        time.sleep(0.5)
        return []


def test_tools_run_off_the_event_loop(settings):
    """Wyszukiwanie śpiące 0,5 s nie blokuje pętli serwera: równoległe get_invoice_value wraca
    przed nim (async nakładka z anyio.to_thread, §7.2)."""
    done: dict[str, float] = {}
    with connect(mcp_server(settings, "course", retriever=SleepingRetriever())) as conn:

        def search():
            conn.call("search_documents", {"question": "cokolwiek"})
            done["search"] = time.monotonic()

        def invoice():
            time.sleep(0.15)
            conn.call("get_invoice_value", {"supplier": "SUP-114", "month": "2026-07"})
            done["invoice"] = time.monotonic()

        with ThreadPoolExecutor(2) as pool:
            for future in [pool.submit(search), pool.submit(invoice)]:
                future.result(timeout=30)
    assert done["invoice"] < done["search"], "drugie wywołanie wróciło przed pierwszym"


def test_search_error_is_a_tool_error(settings):
    from kovalo.knowledge.pgvector import IndexUnavailableError
    from kovalo.mcp_client.connection import McpCallError

    class Broken:
        name = "broken"

        def retrieve(self, question, k):
            raise IndexUnavailableError("indeks dokumentów niedostępny")

    with connect(mcp_server(settings, "course", retriever=Broken())) as conn:
        with pytest.raises(McpCallError) as info:
            conn.call("search_documents", {"question": "kara"})
        assert info.value.layer == "tool" and conn.captures[-1]["is_error"] is True
        record, _ = conn.call("get_invoice_value", {"supplier": "SUP-114", "month": "2026-07"})
        assert record["invoiced_grosze"] == 18240000, "inne narzędzia działają"


def test_no_results_is_a_record_not_an_error(settings):
    with connect(mcp_server(settings, "course", retriever=EmptyRetriever())) as conn:
        record, _ = conn.call("search_documents", {"question": "nic"})
    assert record["status"] == "no_results" and record["passages"] == []


def test_missing_retriever_is_a_tool_error(settings):
    from kovalo.mcp_client.connection import McpCallError

    with (
        connect(mcp_server(settings, "course", retriever=None)) as conn,
        pytest.raises(McpCallError, match="indeks dokumentów niedostępny"),
    ):
        conn.call("search_documents", {"question": "kara"})


def test_each_call_is_logged(settings, caplog):
    """Log INFO z narzędziem i statusem rekordu (NFR-5); długie argumenty obcięte do 200 znaków."""
    import logging

    caplog.set_level(logging.INFO)
    long_question = "pytanie " * 100
    with connect(mcp_server(settings, "course")) as conn:
        conn.call("get_invoice_value", {"supplier": "SUP-114", "month": "2026-07"})
        conn.call("search_documents", {"question": long_question})
    lines = [r.getMessage() for r in caplog.records if "get_invoice_value" in r.getMessage()]
    assert lines and any("ok" in line for line in lines)
    assert not any(long_question in r.getMessage() for r in caplog.records)


# --- /health ------------------------------------------------------------------------------------


def _health(server):
    from starlette.testclient import TestClient

    with TestClient(server.streamable_http_app()) as client:
        return client.get("/health")


def test_health_reports_index_memory(settings):
    response = _health(mcp_server(settings, "ours"))
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "mcp-server" and body["status"] == "ok"
    assert body["checks"]["company_data"]["status"] == "ok"
    assert body["checks"]["company_data"]["critical"] is True
    assert (
        body["checks"]["index"]["status"] == "ok" and body["checks"]["index"]["critical"] is False
    )


def test_health_index_missing_is_degraded(monkeypatch):
    """pgvector, a Postgres nieosiągalny (port 1): indeks `error` (niekrytyczny) → `degraded`,
    kod 200: sześć narzędzi poza search_documents działa."""
    from kovalo.config import load_settings

    monkeypatch.setenv("KOVALO_MCP_SERVER__SEARCH", "pgvector")
    monkeypatch.setenv("KOVALO_DB__HOST", "127.0.0.1")
    monkeypatch.setenv("KOVALO_DB__PORT", "1")
    monkeypatch.setenv("KOVALO_DB__CONNECT_TIMEOUT_S", "1")
    configured = load_settings()
    assert configured.mcp_server.search == "pgvector"
    response = _health(mcp_server(configured, "ours"))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["company_data"]["status"] == "ok"
    assert body["checks"]["index"]["status"] == "error" and not body["checks"]["index"]["critical"]


# --- open_server_deps i main --------------------------------------------------------------------


def test_open_server_deps_needs_no_llm_key(monkeypatch):
    """NFR-7: serwer nie buduje modelu czatu i nie potrzebuje klucza; indeks leniwy."""
    from kovalo.config import load_settings
    from kovalo.llm import override_chat_model
    from kovalo.mcp_server.server import open_server_deps
    from tests.workflow_fakes import ExplodingChatModel

    settings = load_settings()  # bez GOOGLE_API_KEY
    exploding = ExplodingChatModel()
    with override_chat_model(exploding):
        deps = open_server_deps(settings, search="memory")
    assert exploding.calls == []
    assert deps.variant == settings.tools.variant == "ours"
    assert deps.corpus_dir == settings.paths.corpus_dir
    assert deps.catalog["CON-103"].startswith("CON-103")
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert yaml.safe_load(deps.policy_yaml) == raw["policy"]
    assert deps.ctx.top_k == 3 and deps.ctx.retriever is not None


def test_main_config_error_exit_2(tmp_path, monkeypatch, capsys):
    """Uszkodzone dane firmy → kod 2, jedna linia na stderr, stdout pusty (nic przed protokołem)."""
    import shutil

    from kovalo.config import get_settings
    from kovalo.mcp_server.server import main

    folder = tmp_path / "company"
    shutil.copytree(REPO_ROOT / "data" / "company", folder)
    otif = folder / "supplier_otif.csv"
    otif.write_text(otif.read_text(encoding="utf-8").replace("93.8", "n/a", 1), encoding="utf-8")
    monkeypatch.setenv("KOVALO_PATHS__COMPANY_DATA_DIR", str(folder))
    get_settings.cache_clear()
    assert main(["--transport", "stdio", "--search", "memory"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.strip().splitlines()) == 1


def test_main_argument_surface():
    """`--transport stdio|streamable-http`, `--search pgvector|memory`, `--host`, `--port`."""
    from kovalo.mcp_server.server import _parser

    args = _parser().parse_args([])
    assert (args.transport, args.search) == ("stdio", None)
    args = _parser().parse_args(
        ["--transport", "streamable-http", "--search", "memory", "--host", "0.0.0.0", "--port", "9"]
    )
    assert (args.transport, args.search, args.host, args.port) == (
        "streamable-http",
        "memory",
        "0.0.0.0",
        9,
    )
    with pytest.raises(SystemExit):
        _parser().parse_args(["--search", "bm25"])


def test_python_dash_m_entry_exists():
    assert (REPO_ROOT / "src" / "kovalo" / "mcp_server" / "__main__.py").is_file()
    import importlib.util

    assert importlib.util.find_spec("kovalo.mcp_server.__main__") is not None


def test_registry_module_names():
    from kovalo.mcp_server import registry

    assert registry.DOCUMENT_ID.fullmatch("CON-103") and not registry.DOCUMENT_ID.fullmatch(
        "con-103"
    )
    assert isinstance(registry.DOCUMENT_ID, re.Pattern)
    for name in ("register_tools", "register_resources", "register_prompts"):
        assert callable(getattr(registry, name))


def test_structured_content_is_json(connection):
    """Wynik ma structuredContent i dokładnie jeden blok tekstu będący jego kopią JSON."""
    connection.call("get_invoice_value", {"supplier": "SUP-114", "month": "2026-07"})
    capture = connection.captures[-1]
    assert capture["outcome"] == "ok" and len(capture["text"]) == 1
    assert json.loads(capture["text"][0]) == capture["structured"]


def test_parallel_calls_from_threads_share_one_connection(connection):
    errors: list[Exception] = []

    def worker():
        try:
            connection.call("get_invoice_value", {"supplier": "SUP-207", "month": "2026-07"})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert errors == [] and len(connection.captures) == 4
