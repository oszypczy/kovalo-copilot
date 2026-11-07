"""Fixture'y testów dashboardu (kontrakt F6, 12). Pomocniki: tests/dashboard_fakes.py."""

from __future__ import annotations

import pytest

from tests.dashboard_fakes import make_reports_dir


@pytest.fixture
def reports_dir(tmp_path):
    """Katalog raportów z dashboard.yaml z kontraktu 8.1 i JSON-ami wszystkich pięciu etapów."""
    return make_reports_dir(tmp_path / "reports")
