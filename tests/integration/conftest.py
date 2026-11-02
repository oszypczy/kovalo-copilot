"""Fixture'y integracyjne F1: Postgres z docker compose. Brak bazy to błąd testu."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest


@pytest.fixture
def pg_settings():  # type: ignore[no-untyped-def]
    from kovalo.config import load_settings

    return load_settings(google_api_key="test-key")


def _drop_table(pg_settings, table: str) -> None:  # type: ignore[no-untyped-def]
    import psycopg

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


@pytest.fixture
def pg_table(pg_settings) -> Iterator[str]:  # type: ignore[no-untyped-def]
    table = f"test_mp1_{uuid.uuid4().hex[:8]}"
    yield table
    _drop_table(pg_settings, table)


@pytest.fixture
def pg_ctx_table(pg_settings) -> Iterator[str]:  # type: ignore[no-untyped-def]
    """Tabela testowa dla chunków z kontekstem (F2): test_ctx_<uuid8>."""
    table = f"test_ctx_{uuid.uuid4().hex[:8]}"
    yield table
    _drop_table(pg_settings, table)
