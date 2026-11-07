"""`reports/dashboard.yaml`: wczytanie i błędy (kontrakt F6, 8.1)."""

from __future__ import annotations

import pytest

from tests.dashboard_fakes import SPEC_YAML


def write(tmp_path, text):
    path = tmp_path / "dashboard.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_spec(tmp_path):
    from kovalo.eval.dashboard.spec import DashboardSpec, load_spec

    spec = load_spec(write(tmp_path, SPEC_YAML))
    assert isinstance(spec, DashboardSpec) and spec.version == 1
    assert [s.stage for s in spec.stages] == ["e1", "e2", "e3", "e4", "e5"]
    e4, e5 = spec.stages[3], spec.stages[4]
    assert e4.related == ["mp4/2026-11-04-e4.md"] and e4.first_live is None
    assert e4.report == "mp4/2026-11-03-e4.json" and e4.markdown == "mp4/2026-11-03-e4.md"
    assert e5.first_live.run_id == "20261104T092904-0f4dd55b"
    assert (e5.first_live.network_calls, e5.first_live.cached_calls) == (30, 4)
    assert (e5.first_live.failed_attempts, e5.first_live.seconds) == (1, 132)
    assert e5.first_live.source == "reports/mp5/review.yaml"
    assert e5.notes and "R1-3" in e5.notes[0]
    assert [g.stage for g in spec.gemini_usage] == ["F3", "F4", "F5", "F6"]
    assert spec.gemini_usage[1].calls == 47 and spec.gemini_usage[3].calls is None
    assert spec.gemini_usage[3].note.startswith("uzupełnić")


def test_load_spec_minimal_defaults(tmp_path):
    from kovalo.eval.dashboard.spec import load_spec

    spec = load_spec(
        write(tmp_path, "version: 1\nstages:\n  - {stage: e1, report: a.json, markdown: a.md}\n")
    )
    assert spec.gemini_usage == []
    only = spec.stages[0]
    assert only.related == [] and only.notes == [] and only.first_live is None


@pytest.mark.parametrize(
    "text",
    [
        "version: 1\nstages:\n  - {stage: e6, report: a.json, markdown: a.md}\n",  # nieznany etap
        "version: 1\nstages:\n"
        "  - {stage: e1, report: a.json, markdown: a.md}\n"
        "  - {stage: e1, report: b.json, markdown: b.md}\n",  # powtórzony etap
        "version: 1\nstages: []\nnieznany_klucz: 1\n",  # nieznany klucz główny
        "version: 1\nstages:\n  - {stage: e1, report: a.json, markdown: a.md, extra: 1}\n",
        "version: 2\nstages: []\n",  # nieobsługiwana wersja
        "version: 1\nstages:\n  - {stage: e1, markdown: a.md}\n",  # brak report
        "version: [\nstages: ]]\n",  # zły YAML
        "- to\n- lista\n",  # nie mapa
    ],
    ids=[
        "unknown-stage",
        "duplicate-stage",
        "unknown-top-key",
        "unknown-stage-key",
        "bad-version",
        "missing-field",
        "bad-yaml",
        "not-a-mapping",
    ],
)
def test_spec_errors(tmp_path, text):
    from kovalo.config import ConfigError
    from kovalo.eval.dashboard.spec import load_spec

    with pytest.raises(ConfigError):
        load_spec(write(tmp_path, text))


def test_spec_missing_file(tmp_path):
    from kovalo.config import ConfigError
    from kovalo.eval.dashboard.spec import load_spec

    with pytest.raises(ConfigError):
        load_spec(tmp_path / "nie-ma.yaml")


def test_first_live_rejects_unknown_keys(tmp_path):
    from kovalo.config import ConfigError
    from kovalo.eval.dashboard.spec import load_spec

    text = SPEC_YAML.replace("      seconds: 132\n", "      seconds: 132\n      sekundy: 1\n")
    with pytest.raises(ConfigError):
        load_spec(write(tmp_path, text))
