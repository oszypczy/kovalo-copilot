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
    forbidden = ("mcp", "kovalo.mcp_client", "kovalo.mcp_server")
    assert _violations(_py_files("workflow"), forbidden) == []


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
