"""Zgodność docker/Dockerfile z config.yaml (kontrakt F1, sekcje 9.2 i 10.2). Analiza tekstu."""

from __future__ import annotations

import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[2] / "docker" / "Dockerfile"


def _stages(text: str) -> dict[str, list[str]]:
    stages: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        m = re.match(r"^\s*FROM\s+\S+(?:\s+AS\s+(\S+))?\s*$", line, re.IGNORECASE)
        if m:
            current = stages.setdefault(m.group(1) or f"stage{len(stages)}", [])
        elif current is not None:
            current.append(line)
    return stages


def test_dockerfile_model_matches_config(settings):
    text = DOCKERFILE.read_text(encoding="utf-8")
    m = re.search(r"^ARG\s+FASTEMBED_MODEL=(\S+)\s*$", text, re.MULTILINE)
    assert m, "brak ARG FASTEMBED_MODEL=<model> w docker/Dockerfile"
    assert m.group(1) == settings.embeddings.local.model


def test_ui_image_has_no_models():
    stages = _stages(DOCKERFILE.read_text(encoding="utf-8"))
    assert "ui" in stages, "brak etapu ui"
    body = "\n".join(stages["ui"])
    assert "--from=models" not in body
    assert "fastembed" not in body.lower()
    assert "models" in stages and "app" in stages
    assert "--from=models" in "\n".join(stages["app"])
