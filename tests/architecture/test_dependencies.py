"""Zasady zależności modułów (kontrakt F0, sekcja 2.3). Analiza AST, bez importowania kodu."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "kovalo"


def _py_files(*relative: str) -> list[Path]:
    files: list[Path] = []
    for rel in relative:
        path = SRC / rel
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        else:
            files.append(path)
    return files


def _imports(path: Path) -> set[str]:
    """Pełne nazwy importowanych modułów (dla `from a import b` także `a.b`)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _violations(files: list[Path], forbidden: tuple[str, ...]) -> list[str]:
    found = []
    for f in files:
        for name in sorted(_imports(f)):
            if any(name == p or name.startswith(p + ".") for p in forbidden):
                found.append(f"{f.relative_to(SRC)}: {name}")
    return found


def test_core_modules_do_not_import_llm_stack():
    for required in ("config.py", "health.py"):
        assert (SRC / required).is_file(), f"brak src/kovalo/{required}"
    files = _py_files("policy", "company", "cost", "config.py", "health.py")
    forbidden = ("langchain", "langchain_core", "langchain_google_genai", "langchain_openai")
    forbidden += ("langgraph", "kovalo.llm")
    assert _violations(files, forbidden) == []


def test_only_llm_module_imports_providers():
    assert (SRC / "llm.py").is_file(), "brak src/kovalo/llm.py"
    forbidden = ("langchain_google_genai", "langchain_openai")
    others = [f for f in SRC.rglob("*.py") if f != SRC / "llm.py"]
    assert _violations(others, forbidden) == []
    init_chat = [
        str(f.relative_to(SRC))
        for f in others
        if re.search(r"\binit_chat_model\b", f.read_text(encoding="utf-8"))
    ]
    assert init_chat == []


def test_ui_does_not_import_backend_logic():
    forbidden = (
        "kovalo.llm",
        "kovalo.workflow",
        "kovalo.agents",
        "kovalo.tools",
        "kovalo.knowledge",
        "kovalo.rag",
        "kovalo.cost",
        "kovalo.eval",
    )
    assert _violations(_py_files("ui"), forbidden) == []


def test_workflow_does_not_import_mcp():
    """F5 (kontrakt 2.3): od F5 wyjątek TYLKO `workflow/factory.py` (kovalo.mcp_client); graf,
    serwis i reszta nadal bez `mcp`, `kovalo.mcp_*`. Zmiana świadoma (F0-F4 zakazywały całego
    pakietu)."""
    forbidden = ("mcp", "kovalo.mcp_client", "kovalo.mcp_server")
    others = [f for f in _py_files("workflow") if f.name != "factory.py"]
    assert _violations(others, forbidden) == []
    factory = SRC / "workflow" / "factory.py"
    assert _violations([factory], ("mcp", "kovalo.mcp_server")) == [], "factory: tylko mcp_client"


@pytest.mark.parametrize(
    "literal", ["gemini-", "gpt-", "intfloat/", "jinaai/", "sentence-transformers/"]
)
def test_no_model_names_in_src(literal: str):
    offenders = [
        str(f.relative_to(SRC))
        for f in SRC.rglob("*.py")
        if literal in f.read_text(encoding="utf-8")
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# F1 (kontrakt: docs/contracts/f1.md, sekcja 2.3)
# ---------------------------------------------------------------------------


def _module_level_imports(path: Path) -> set[str]:
    """Importy na najwyższym poziomie modułu (leniwe, wewnątrz funkcji, nie liczą się)."""
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize(
    ("package", "allowed"),
    [
        ("fastembed", "knowledge/embeddings.py"),
        ("langchain_postgres", "knowledge/pgvector.py"),
        ("tiktoken", "cost/tokens.py"),
    ],
)
def test_fastembed_postgres_tiktoken_isolated(package: str, allowed: str):
    assert (SRC / allowed).is_file(), f"brak src/kovalo/{allowed}"
    others = [f for f in SRC.rglob("*.py") if f != SRC / allowed]
    assert _violations(others, (package,)) == []
    assert package in {n.split(".")[0] for n in _imports(SRC / allowed)}


def test_fastembed_imported_lazily():
    path = SRC / "knowledge" / "embeddings.py"
    assert path.is_file(), "brak src/kovalo/knowledge/embeddings.py"
    top = _module_level_imports(path)
    assert not any(n == "fastembed" or n.startswith("fastembed.") for n in top)


def test_importing_knowledge_does_not_load_fastembed():
    import subprocess
    import sys

    code = "import sys, kovalo.knowledge; assert 'fastembed' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-800:]


def test_knowledge_and_cost_do_not_import_llm():
    for required in (
        "corpus",
        "chunking",
        "prompts",
        "embeddings",
        "retrieval",
        "pgvector",
        "ingest",
    ):
        assert (SRC / "knowledge" / f"{required}.py").is_file(), f"brak knowledge/{required}.py"
    for required in ("tokens", "pricing"):
        assert (SRC / "cost" / f"{required}.py").is_file(), f"brak cost/{required}.py"
    forbidden = ("kovalo.llm", "kovalo.eval", "kovalo.cli")
    assert _violations(_py_files("knowledge", "cost"), forbidden) == []


def test_cost_has_no_langchain():
    forbidden = ("langchain", "langchain_core", "langchain_google_genai", "langchain_openai")
    forbidden += ("langgraph",)
    assert _violations(_py_files("cost"), forbidden) == []


def test_eval_does_not_import_entrypoints():
    assert (SRC / "eval" / "retrieval" / "e1.py").is_file(), "brak eval/retrieval/e1.py"
    forbidden = ("kovalo.cli", "kovalo.api", "kovalo.ui")
    assert _violations(_py_files("eval"), forbidden) == []


# ---------------------------------------------------------------------------
# F2 (kontrakt: docs/contracts/f2.md, sekcja 2.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("package", "allowed"),
    [
        ("rank_bm25", "knowledge/lexical.py"),
        ("langchain_text_splitters", "knowledge/chunking.py"),
    ],
)
def test_bm25_and_splitters_isolated(package: str, allowed: str):
    assert (SRC / allowed).is_file(), f"brak src/kovalo/{allowed}"
    others = [f for f in SRC.rglob("*.py") if f != SRC / allowed]
    assert _violations(others, (package,)) == []
    assert package in {n.split(".")[0] for n in _imports(SRC / allowed)}


def test_no_langchain_community():
    # ADR-0007/0008: BM25 i fuzja bez BM25Retriever/EnsembleRetriever
    forbidden = ("langchain_community", "langchain_classic.retrievers")
    assert _violations(sorted(SRC.rglob("*.py")), forbidden) == []


def test_f2_modules_exist():
    for rel in (
        "knowledge/lexical.py",
        "knowledge/retrievers.py",
        "rag/records.py",
        "rag/citations.py",
        "rag/answer.py",
        "rag/factory.py",
        "eval/llm_cache.py",
        "eval/rag/scoring.py",
        "eval/rag/judge.py",
        "eval/rag/summary.py",
        "eval/rag/reference.py",
        "eval/rag/e2.py",
        "eval/rag/report.py",
    ):
        assert (SRC / rel).is_file(), f"brak src/kovalo/{rel}"


def test_rag_layering():
    rag_files = _py_files("rag")
    factory = SRC / "rag" / "factory.py"
    assert factory.is_file(), "brak src/kovalo/rag/factory.py"
    forbidden = (
        "kovalo.eval",
        "kovalo.cli",
        "kovalo.api",
        "kovalo.ui",
        "kovalo.workflow",
        "kovalo.agents",
        "kovalo.tools",
        "kovalo.mcp_client",
        "kovalo.mcp_server",
        "kovalo.cost",
    )
    assert _violations(rag_files, forbidden) == []
    # kovalo.llm importuje tylko rag/factory.py; answer.py dostaje model przez parametr
    assert _violations([f for f in rag_files if f != factory], ("kovalo.llm",)) == []
    assert any(n == "kovalo.llm" or n.startswith("kovalo.llm.") for n in _imports(factory))
    # knowledge/ jest warstwą niżej niż rag/
    assert _violations(_py_files("knowledge"), ("kovalo.rag", "kovalo.llm")) == []


def test_llm_does_not_import_eval():
    assert (SRC / "llm.py").is_file()
    assert _violations([SRC / "llm.py"], ("kovalo.eval",)) == []


def test_eval_may_use_rag_but_not_entrypoints():
    # eval/ może importować rag, knowledge, cost, llm; nadal nie importuje cli/api/ui
    assert _violations(_py_files("eval"), ("kovalo.cli", "kovalo.api", "kovalo.ui")) == []


# ---------------------------------------------------------------------------
# F3 (kontrakt: docs/contracts/f3.md, sekcja 2.3). Listy DOZWOLONYCH importów (N2).
# ---------------------------------------------------------------------------

import sys  # noqa: E402

STDLIB = frozenset(sys.stdlib_module_names)


def _import_targets(path: Path) -> set[str]:
    """Moduły importowane w pliku (także leniwe). `from kovalo import x` -> `kovalo.x`."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module == "kovalo":
                names.update(f"kovalo.{alias.name}" for alias in node.names)
            else:
                names.add(node.module)
    return names


def _allowlist_violations(
    files: list[Path],
    allowed: tuple[str, ...],
    *,
    exact: tuple[str, ...] = (),
    extra: dict[str, tuple[str, ...]] | None = None,
) -> list[str]:
    """Importy spoza: biblioteki standardowej, `allowed` (nazwa albo jej podmoduł), `exact`
    (dokładnie ta nazwa) i `extra[plik względem kovalo/]` (prefiksy tylko dla tego pliku)."""
    extra = extra or {}
    found: list[str] = []
    for f in files:
        rel = str(f.relative_to(SRC))
        prefixes = (*allowed, *extra.get(rel, ()), "kovalo.__version__")
        for name in sorted(_import_targets(f)):
            if name.split(".")[0] in STDLIB or name in exact:
                continue
            if any(name == p or name.startswith(p + ".") for p in prefixes):
                continue
            found.append(f"{rel}: {name}")
    return found


def test_text_pl_allowlist():
    path = SRC / "text_pl.py"
    assert path.is_file(), "brak src/kovalo/text_pl.py"
    assert _allowlist_violations([path], ()) == []


def test_policy_allowlist():
    files = _py_files("policy")
    assert (SRC / "policy" / "engine.py").is_file(), "brak policy/engine.py"
    assert (SRC / "policy" / "models.py").is_file(), "brak policy/models.py"
    allowed = ("pydantic", "kovalo.config", "kovalo.text_pl", "kovalo.policy")
    assert _allowlist_violations(files, allowed) == []


def test_workflow_allowlist():
    """F3 + F4 (kontrakt F4, 2.3). R1-5: `kovalo.rag.citations` zniknęło z listy dozwolonych."""
    files = _py_files("workflow")
    for module in (
        "schemas",
        "understand",
        "state",
        "replies",
        "graph",
        "checkpoint",
        "service",
        "factory",
    ):
        assert (SRC / "workflow" / f"{module}.py").is_file(), f"brak workflow/{module}.py"
    allowed = (
        "pydantic",
        "langchain_core",
        "langgraph",
        "kovalo.config",
        "kovalo.text_pl",
        "kovalo.policy",
        "kovalo.workflow",
        "kovalo.rag.records",
        "kovalo.rag.answer",
        "kovalo.agents.trace",
        "kovalo.agents.show",
        "kovalo.agents.agent",
    )
    extra = {
        # jedyny moduł workflow/ z modelem, RAG, indeksem, narzędziami i danymi firmy
        "workflow/factory.py": (
            "kovalo.llm",
            "kovalo.rag",
            "kovalo.knowledge",
            "kovalo.tools",
            "kovalo.company",
            "kovalo.agents",
            "kovalo.mcp_client",  # F5: open_tool_source, McpStartupError
        ),
        # jedyny moduł z Postgresem
        "workflow/checkpoint.py": ("psycopg", "psycopg_pool", "langgraph.checkpoint.postgres"),
    }
    exact = ("kovalo.rag", "kovalo.agents")
    assert _allowlist_violations(files, allowed, exact=exact, extra=extra) == []
    assert not any("kovalo.rag.citations" in _import_targets(f) for f in files), "R1-5"


def test_rag_allowlist():
    """N2: rag/ sprawdzany listą dozwolonych, nie tylko zakazów (test_rag_layering zostaje)."""
    files = _py_files("rag")
    allowed = ("pydantic", "langchain_core", "kovalo.config", "kovalo.knowledge", "kovalo.rag")
    extra = {"rag/factory.py": ("kovalo.llm",)}
    assert _allowlist_violations(files, allowed, extra=extra) == []


def test_api_allowlist():
    files = _py_files("api")
    for module in ("schemas", "threads", "deps", "main"):
        assert (SRC / "api" / f"{module}.py").is_file(), f"brak api/{module}.py"
    # uvicorn: `kovalo-api` (run()) już w F0; starlette: typy wymagane przez FastAPI (Request, ...)
    allowed = (
        "fastapi",
        "starlette",
        "uvicorn",
        "pydantic",
        "kovalo.config",
        "kovalo.health",
        "kovalo.workflow",
        "kovalo.api",
    )
    assert _allowlist_violations(files, allowed) == []


def test_ui_allowlist():
    files = _py_files("ui")
    for module in ("client", "render", "app", "server"):
        assert (SRC / "ui" / f"{module}.py").is_file(), f"brak ui/{module}.py"
    allowed = (
        "chainlit",
        "httpx",
        "httpx_sse",
        "pydantic",
        "kovalo.config",
        "kovalo.health",
        "kovalo.ui",
    )
    extra = {"ui/server.py": ("fastapi", "uvicorn")}
    assert _allowlist_violations(files, allowed, extra=extra) == []


def test_interrupt_and_command_isolated():
    graph = SRC / "workflow" / "graph.py"
    service = SRC / "workflow" / "service.py"
    assert graph.is_file() and service.is_file()
    all_files = sorted(SRC.rglob("*.py"))
    assert _violations([f for f in all_files if f != graph], ("langgraph.types.interrupt",)) == []
    assert "langgraph.types.interrupt" in _imports(graph)
    assert (
        _violations(
            [f for f in all_files if f not in (graph, service)], ("langgraph.types.Command",)
        )
        == []
    )
    # Command: dozwolony w graph.py i service.py; wymagany tylko w serwisie (wznowienie pauzy)
    assert "langgraph.types.Command" in _imports(service)


def test_checkpoint_postgres_isolated():
    checkpoint = SRC / "workflow" / "checkpoint.py"
    assert checkpoint.is_file(), "brak workflow/checkpoint.py"
    others = [f for f in SRC.rglob("*.py") if f != checkpoint]
    assert _violations(others, ("langgraph.checkpoint.postgres", "psycopg_pool")) == []
    imported = {n.split(".")[0] for n in _imports(checkpoint)}
    assert {"psycopg_pool", "langgraph"} <= imported
    # psycopg: checkpoint.py, knowledge/pgvector.py i health.py (sonda Postgresa z F0)
    allowed_psycopg = {checkpoint, SRC / "knowledge" / "pgvector.py", SRC / "health.py"}
    assert (
        _violations([f for f in SRC.rglob("*.py") if f not in allowed_psycopg], ("psycopg",)) == []
    )


def test_chainlit_and_fastapi_isolated():
    ui_files = set(_py_files("ui"))
    api_files = set(_py_files("api"))
    server = SRC / "ui" / "server.py"
    assert server.is_file()
    everything = sorted(SRC.rglob("*.py"))
    assert _violations([f for f in everything if f not in ui_files], ("chainlit",)) == []
    assert _violations([f for f in everything if f not in api_files | {server}], ("fastapi",)) == []


def test_read_path_does_not_import_ingest():
    """N1: knowledge/retrievers.py (odczyt indeksu) nie importuje modułu zapisu, także leniwie."""
    retrievers = SRC / "knowledge" / "retrievers.py"
    assert _violations([retrievers], ("kovalo.knowledge.ingest",)) == []
    pgvector = SRC / "knowledge" / "pgvector.py"
    text = pgvector.read_text(encoding="utf-8")
    assert "def index_table" in text and "def configured_embedding_model" in text, (
        "index_table i configured_embedding_model mieszkają w knowledge/pgvector.py"
    )
    ingest = SRC / "knowledge" / "ingest.py"
    assert {"kovalo.knowledge.pgvector"} & _imports(ingest), "ingest.py importuje je z pgvector.py"


def test_workflow_and_policy_do_not_import_ui_api_eval():
    # F4: workflow/ może importować kovalo.agents (i factory: kovalo.tools, kovalo.company), policy/ nie  # noqa: E501
    forbidden = ("kovalo.api", "kovalo.ui", "kovalo.eval", "kovalo.cli")
    assert _violations(_py_files("workflow", "policy"), forbidden) == []
    assert _violations(_py_files("policy"), ("kovalo.agents", "kovalo.tools")) == []
    assert _violations(_py_files("workflow"), ("fastapi", "chainlit", "httpx", "mcp")) == []


def test_eval_may_use_workflow_and_policy():
    # eval/ może importować kovalo.workflow (w tym factory) i kovalo.policy, nadal nie cli/api/ui
    assert _violations(_py_files("eval"), ("kovalo.cli", "kovalo.api", "kovalo.ui")) == []


def test_f3_modules_exist():
    for rel in (
        "text_pl.py",
        "policy/models.py",
        "policy/engine.py",
        "workflow/schemas.py",
        "workflow/understand.py",
        "workflow/state.py",
        "workflow/replies.py",
        "workflow/graph.py",
        "workflow/checkpoint.py",
        "workflow/service.py",
        "workflow/factory.py",
        "api/schemas.py",
        "api/threads.py",
        "api/deps.py",
        "ui/client.py",
        "ui/render.py",
        "eval/workflow/__init__.py",
        "eval/workflow/runner.py",
        "eval/workflow/scoring.py",
        "eval/workflow/reference.py",
        "eval/workflow/e3.py",
        "eval/workflow/report.py",
    ):
        assert (SRC / rel).is_file(), f"brak src/kovalo/{rel}"


# ---------------------------------------------------------------------------
# F4 (kontrakt: docs/contracts/f4.md, sekcja 2.3). Listy DOZWOLONYCH importów.
# ---------------------------------------------------------------------------


def test_f4_modules_exist():
    for rel in (
        "quantities.py",
        "company/__init__.py",
        "company/data.py",
        "company/records.py",
        "company/lookup.py",
        "company/penalty.py",
        "company/termination.py",
        "policy/check.py",
        "tools/__init__.py",
        "tools/search.py",
        "tools/specs.py",
        "tools/langchain.py",
        "agents/__init__.py",
        "agents/trace.py",
        "agents/middleware.py",
        "agents/agent.py",
        "agents/show.py",
        "api/sse.py",
        "eval/trace/__init__.py",
        "eval/trace/scoring.py",
        "eval/trace/reference.py",
        "eval/trace/replay.py",
        "eval/trace/e4.py",
        "eval/trace/report.py",
    ):
        assert (SRC / rel).is_file(), f"brak src/kovalo/{rel}"


def test_quantities_allowlist():
    path = SRC / "quantities.py"
    assert path.is_file(), "brak src/kovalo/quantities.py"
    assert _allowlist_violations([path], ()) == [], "tylko biblioteka standardowa"


def test_company_allowlist():
    files = _py_files("company")
    for module in ("data", "records", "lookup", "penalty", "termination"):
        assert (SRC / "company" / f"{module}.py").is_file(), f"brak company/{module}.py"
    allowed = ("pydantic", "yaml", "kovalo.config", "kovalo.quantities", "kovalo.company")
    assert _allowlist_violations(files, allowed) == []
    forbidden = ("langchain", "langchain_core", "langgraph", "kovalo.llm", "kovalo.knowledge")
    assert _violations(files, forbidden) == []
    assert _violations(files, ("kovalo.text_pl",)) == [], "komunikaty formatuje company/ sam"


def test_tools_allowlist():
    files = _py_files("tools")
    for module in ("search", "specs", "langchain"):
        assert (SRC / "tools" / f"{module}.py").is_file(), f"brak tools/{module}.py"
    allowed = ("pydantic", "kovalo.config", "kovalo.company", "kovalo.policy", "kovalo.tools")
    extra = {
        "tools/langchain.py": ("langchain_core",),
        "tools/search.py": ("kovalo.knowledge",),  # tylko typy Retriever / ScoredChunk
    }
    assert _allowlist_violations(files, allowed, extra=extra) == []
    forbidden = ("kovalo.llm", "kovalo.agents", "kovalo.workflow", "kovalo.eval", "mcp")
    assert _violations(files, (*forbidden, "fastapi", "chainlit")) == []
    plain = [SRC / "tools" / "specs.py", SRC / "tools" / "search.py"]
    assert _violations(plain, ("langchain", "langchain_core", "langgraph")) == [], (
        "specs.py i search.py bez LangChaina (F5 rejestruje te same funkcje w FastMCP)"
    )


def test_agents_allowlist():
    files = _py_files("agents")
    for module in ("trace", "middleware", "agent", "show"):
        assert (SRC / "agents" / f"{module}.py").is_file(), f"brak agents/{module}.py"
    allowed = (
        "pydantic",
        "langchain.agents",
        "langchain_core",
        "langgraph",
        "kovalo.config",
        "kovalo.quantities",
        "kovalo.agents",
    )
    assert _allowlist_violations(files, allowed) == []
    forbidden = (
        "kovalo.tools",
        "kovalo.llm",
        "kovalo.knowledge",
        "kovalo.company",
        "kovalo.workflow",
        "mcp",
    )
    assert _violations(files, forbidden) == [], "narzędzia i model przez parametr"


def test_create_agent_isolated():
    """`langchain.agents` (create_agent, middleware) importuje tylko agents/."""
    agents = set(_py_files("agents"))
    assert agents, "brak src/kovalo/agents/"
    others = [f for f in sorted(SRC.rglob("*.py")) if f not in agents]
    assert _violations(others, ("langchain.agents",)) == []
    assert any(n.startswith("langchain.agents") for f in agents for n in _import_targets(f)), (
        "agents/ używa create_agent"
    )


def test_stream_writer_isolated():
    """`get_stream_writer` wywołuje tylko workflow/graph.py."""
    graph = SRC / "workflow" / "graph.py"
    assert graph.is_file()
    others = [f for f in sorted(SRC.rglob("*.py")) if f != graph]
    assert _violations(others, ("langgraph.config",)) == []
    assert [
        str(f.relative_to(SRC))
        for f in others
        if "get_stream_writer" in f.read_text(encoding="utf-8")
    ] == []
    assert "get_stream_writer" in graph.read_text(encoding="utf-8")


def test_tools_only_in_factory():
    """`kovalo.tools`, `kovalo.company` i `agents.agent.build_agent` w workflow/ tylko w factory.py."""  # noqa: E501
    factory = SRC / "workflow" / "factory.py"
    others = [f for f in _py_files("workflow") if f != factory]
    assert _violations(others, ("kovalo.tools", "kovalo.company", "kovalo.agents.middleware")) == []
    assert _violations(others, ("kovalo.llm", "kovalo.knowledge")) == []
    imported = _import_targets(factory)
    assert any(n == "kovalo.tools" or n.startswith("kovalo.tools.") for n in imported)
    assert any(n == "kovalo.company" or n.startswith("kovalo.company.") for n in imported)


def test_api_does_not_import_agents_tools_company():
    """Trace i ścieżka dowodowa przychodzą z CopilotService (ADR-0012)."""
    forbidden = ("kovalo.agents", "kovalo.tools", "kovalo.company", "kovalo.llm", "kovalo.eval")
    assert _violations(_py_files("api"), forbidden) == []
    assert (SRC / "api" / "sse.py").is_file()


def test_eval_may_use_agents_tools_company_but_not_entrypoints():
    # eval/ może importować kovalo.agents, kovalo.tools, kovalo.company, kovalo.quantities
    assert _violations(_py_files("eval"), ("kovalo.cli", "kovalo.api", "kovalo.ui")) == []
    assert (SRC / "eval" / "trace" / "e4.py").is_file()


# ---------------------------------------------------------------------------
# F5 (kontrakt: docs/contracts/f5.md, sekcja 2.3)
# ---------------------------------------------------------------------------


def test_f5_modules_exist():
    for rel in (
        "mcp_server/__init__.py",
        "mcp_server/__main__.py",
        "mcp_server/server.py",
        "mcp_server/registry.py",
        "mcp_client/__init__.py",
        "mcp_client/contract.py",
        "mcp_client/connection.py",
        "mcp_client/source.py",
        "eval/assistant/__init__.py",
        "eval/assistant/scoring.py",
        "eval/assistant/parity.py",
        "eval/assistant/reference.py",
        "eval/assistant/e5.py",
        "eval/assistant/report.py",
    ):
        assert (SRC / rel).is_file(), f"brak src/kovalo/{rel}"


def test_mcp_server_allowlist():
    files = _py_files("mcp_server")
    assert (SRC / "mcp_server" / "server.py").is_file() and (
        SRC / "mcp_server" / "registry.py"
    ).is_file()
    allowed = (
        "pydantic",
        "yaml",
        "anyio",
        "mcp",
        "kovalo.config",
        "kovalo.tools",
        "kovalo.company",
        "kovalo.knowledge",
        "kovalo.mcp_server",
    )
    extra = {
        # /health (starlette), raport zdrowia i embeddingi zapytań: tylko server.py
        "mcp_server/server.py": ("starlette", "kovalo.health", "kovalo.llm"),
    }
    assert _allowlist_violations(files, allowed, extra=extra) == []
    forbidden = (
        "langchain_core.tools",
        "kovalo.agents",
        "kovalo.workflow",
        "kovalo.mcp_client",
        "kovalo.eval",
        "kovalo.api",
        "kovalo.ui",
        "fastapi",
    )
    assert _violations(files, forbidden) == []


def test_mcp_server_uses_only_embeddings_from_llm():
    """`kovalo.llm` w mcp_server/ tylko `get_embeddings` (żadnego modelu czatu; NFR-7)."""
    for f in _py_files("mcp_server"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == "kovalo.llm":
                assert {a.name for a in node.names} == {"get_embeddings"}, f.name
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("kovalo.llm") for a in node.names), f.name


def test_mcp_client_allowlist():
    files = _py_files("mcp_client")
    for module in ("contract", "connection", "source"):
        assert (SRC / "mcp_client" / f"{module}.py").is_file(), f"brak mcp_client/{module}.py"
    allowed = (
        "pydantic",
        "anyio",
        "mcp",
        "langchain_core.tools",
        "kovalo.config",
        "kovalo.tools.specs",
        "kovalo.mcp_client",
    )
    assert _allowlist_violations(files, allowed) == []
    forbidden = (
        "kovalo.mcp_server",
        "kovalo.agents",
        "kovalo.workflow",
        "kovalo.llm",
        "kovalo.eval",
        "langchain.agents",
        "langchain_mcp_adapters",
        "langchain.mcp",
    )
    assert _violations(files, forbidden) == []


def test_mcp_package_isolated():
    """Pakiet `mcp` importują tylko mcp_server/ i mcp_client/ (oraz testy)."""
    own = set(_py_files("mcp_server", "mcp_client"))
    assert own, "brak mcp_server/ i mcp_client/"
    others = [f for f in sorted(SRC.rglob("*.py")) if f not in own]
    assert _violations(others, ("mcp",)) == []
    assert any(n == "mcp" or n.startswith("mcp.") for f in own for n in _import_targets(f))


def test_no_langchain_mcp_adapters():
    """ADR-0013: własny klient na MCP SDK; ani `langchain-mcp-adapters`, ani `langchain.mcp`."""
    assert _violations(sorted(SRC.rglob("*.py")), ("langchain_mcp_adapters", "langchain.mcp")) == []
    pyproject = (SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert "langchain-mcp-adapters" not in pyproject and "langchain[mcp]" not in pyproject


def test_mcp_client_only_in_factory():
    """`kovalo.mcp_client` w workflow/ tylko w factory.py (open_tool_source, McpStartupError);
    api/, ui/, agents/, tools/, knowledge/, policy/, company/ bez `kovalo.mcp_*`."""
    factory = SRC / "workflow" / "factory.py"
    imported = _import_targets(factory)
    assert any(n == "kovalo.mcp_client" or n.startswith("kovalo.mcp_client.") for n in imported)
    names = {
        alias.name
        for node in ast.walk(ast.parse(factory.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("kovalo.mcp_client")
        for alias in node.names
    }
    assert {"open_tool_source", "McpStartupError"} <= names
    others = [f for f in _py_files("workflow") if f != factory]
    assert _violations(others, ("kovalo.mcp_client", "kovalo.mcp_server")) == []
    rest = _py_files("api", "ui", "agents", "tools", "knowledge", "policy", "company", "rag")
    assert _violations(rest, ("kovalo.mcp_client", "kovalo.mcp_server", "mcp")) == []


def test_lazy_retriever_lives_in_knowledge():
    """`LazyRetriever` przeniesiony z workflow/factory.py do knowledge/retrievers.py."""
    factory = (SRC / "workflow" / "factory.py").read_text(encoding="utf-8")
    assert "class LazyRetriever" not in factory
    retrievers = (SRC / "knowledge" / "retrievers.py").read_text(encoding="utf-8")
    assert "class LazyRetriever" in retrievers
    assert "kovalo.llm" not in _import_targets(SRC / "knowledge" / "retrievers.py")


def test_eval_may_use_mcp_client_and_server_but_not_mcp_or_entrypoints():
    eval_files = _py_files("eval")
    assert _violations(eval_files, ("kovalo.cli", "kovalo.api", "kovalo.ui")) == []
    assert _violations(eval_files, ("mcp", "langchain_mcp_adapters")) == [], (
        "SDK tylko przez klienta"
    )
    e5 = SRC / "eval" / "assistant" / "e5.py"
    imported = _import_targets(e5)
    assert any(n.startswith("kovalo.mcp_client") for n in imported)
    assert any(n.startswith("kovalo.mcp_server") for n in imported), (
        "build_server dla serwera w pamięci"
    )


def test_api_and_ui_do_not_learn_about_mcp():
    """Transport narzędzi przychodzi jako pole `tool_end.transport` (listy z F4 bez zmian)."""
    assert (
        _violations(_py_files("api", "ui"), ("mcp", "kovalo.mcp_client", "kovalo.mcp_server")) == []
    )
