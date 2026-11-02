"""Fixture'y integracyjne F1: Postgres z docker compose. Brak bazy to błąd testu."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest


@pytest.fixture
def pg_settings():  # type: ignore[no-untyped-def]
    from kovalo.config import load_settings

    return load_settings(google_api_key="test-key")


@pytest.fixture
def pg_table(pg_settings) -> Iterator[str]:  # type: ignore[no-untyped-def]
    import psycopg

    table = f"test_mp1_{uuid.uuid4().hex[:8]}"
    yield table
    with psycopg.connect(pg_settings.postgres_dsn()) as conn:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.execute(f'DROP TABLE IF EXISTS "{table}__new"')
        exists = conn.execute(
            "SELECT to_regclass(%s)", (pg_settings.db.index_meta_table,)
        ).fetchone()
        if exists and exists[0]:
            conn.execute(
                f'DELETE FROM "{pg_settings.db.index_meta_table}" WHERE table_name = %s', (table,)
            )
