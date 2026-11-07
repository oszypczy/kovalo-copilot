"""MCP przez HTTP w docker compose (kontrakt F5, 13.2; marker integration). Wymaga `make up`
(`mcp-server` na localhost:8765 z indeksem `kovalo_ctx`); bez LLM-a."""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def http_conn(pg_settings):
    from kovalo.mcp_client.connection import http_connection

    conn = http_connection(
        pg_settings.services.mcp_url,
        startup_timeout_s=pg_settings.mcp.startup_timeout_s,
        call_timeout_s=pg_settings.mcp.call_timeout_s,
        cleanup_timeout_s=pg_settings.mcp.cleanup_timeout_s,
    )
    yield conn
    conn.close()


def test_mcp_over_http_in_compose(pg_settings, http_conn):
    from kovalo.tools.specs import OUR_TOOLS

    listed = http_conn.run(lambda s: s.list_tools()).tools
    assert {t.name for t in listed} == set(OUR_TOOLS) and len(listed) == 7
    assert all(t.outputSchema and t.annotations.readOnlyHint for t in listed)

    record, provenance = http_conn.call(
        "get_supplier_otif", {"supplier": "SUP-114", "from_month": "2026-07", "to_month": "2026-07"}
    )
    assert record["observations"][0]["otif_pct"] == 94.6 and provenance["transport"] == "mcp"

    search, _ = http_conn.call(
        "search_documents", {"question": "TransBałt terminowe dostawy palet"}
    )
    assert search["status"] == "ok" and search["passages"], "fragmenty z indeksu kovalo_ctx"
    assert any(p["doc_id"] == "CON-103" for p in search["passages"])

    health = httpx.get(pg_settings.services.mcp_url.rsplit("/", 1)[0] + "/health", timeout=5)
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok" and body["checks"]["index"]["status"] == "ok"


def test_resources_and_prompt_in_compose(http_conn):
    text = http_conn.run(lambda s: s.read_resource("kovalo://documents/CON-103")).contents[0].text
    assert "98% palet" in text
    prompt = http_conn.run(
        lambda s: s.get_prompt("supplier_review", {"supplier": "Mleczna Dolina"})
    )
    assert "Mleczna Dolina" in prompt.messages[0].content.text


def test_parity_over_http(pg_settings, http_conn):
    """`run_parity` przez HTTP: wiersze bez `search_documents` (inny retriever: pgvector w
    kontenerze, pamięć lokalnie) są wszystkie `same`."""
    from kovalo.company import load_company_data
    from kovalo.eval.assistant.parity import run_parity
    from kovalo.tools.specs import ToolContext

    ctx = ToolContext(
        data=load_company_data(pg_settings.paths.company_data_dir),
        policy=pg_settings.policy,
        company=pg_settings.company,
        retriever=None,
        top_k=pg_settings.retrieval.top_k,
    )
    rows = [r for r in run_parity(ctx, http_conn) if r.tool != "search_documents"]
    assert len(rows) == 19
    assert all(r.same for r in rows), [
        (r.tool, r.arguments, r.differences) for r in rows if not r.same
    ]
