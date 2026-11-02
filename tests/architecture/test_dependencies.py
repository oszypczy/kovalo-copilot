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
    )
    assert _violations(_py_files("ui"), forbidden) == []


def test_workflow_does_not_import_mcp():
    forbidden = ("mcp", "kovalo.mcp_client", "kovalo.mcp_server")
    assert _violations(_py_files("workflow"), forbidden) == []


@pytest.mark.parametrize("literal", ["gemini-", "gpt-"])
def test_no_model_names_in_src(literal: str):
    offenders = [
        str(f.relative_to(SRC))
        for f in SRC.rglob("*.py")
        if literal in f.read_text(encoding="utf-8")
    ]
    assert offenders == []
