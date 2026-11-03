"""Checkpointer grafu: serializer z typami workflow, pamięć i Postgres (FR-3.4).

Kontrakt: docs/contracts/f3.md, sekcja 6.6; decyzja: ADR-0009 pkt 6. Jedyny moduł z
langgraph.checkpoint.postgres i psycopg_pool. Typy Pydantic w stanie (Turn, DiscountRequest,
Decision) są rejestrowane w serializerze jak w new_checkpointer() kursu: LangGraph ostrzega
przy nieznanych typach, a przyszła wersja ma je odrzucać.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from kovalo.config import Settings

__all__ = [
    "CHECKPOINT_TYPES",
    "CONNECTION_ERRORS",
    "CheckpointerUnavailableError",
    "memory_checkpointer",
    "open_checkpointer",
    "postgres_checkpointer",
    "serializer",
]

CHECKPOINT_TYPES: list[tuple[str, str]] = [
    ("kovalo.workflow.schemas", "Turn"),
    ("kovalo.policy.models", "DiscountRequest"),
    ("kovalo.policy.models", "Decision"),
]


class CheckpointerUnavailableError(RuntimeError):
    """Postgres niedostępny przy otwieraniu puli, setup() albo w trakcie pracy (odczyt stanu,
    zapis checkpointu). Komunikat bez DSN i hasła."""


def serializer() -> JsonPlusSerializer:
    """JsonPlusSerializer z zarejestrowanymi typami workflow (jak new_checkpointer() kursu)."""
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


def memory_checkpointer() -> InMemorySaver:
    """Checkpointer w pamięci: testy, E3, workflow.checkpointer = memory."""
    return InMemorySaver(serde=serializer())


# Utrata połączenia z Postgresem przy otwieraniu i w trakcie pracy (R1-2, ADR-0012 pkt 7).
CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
    psycopg.OperationalError,
    psycopg.InterfaceError,
    PoolTimeout,
    OSError,
    TimeoutError,
)


@contextmanager
def postgres_checkpointer(settings: Settings, *, setup: bool = True) -> Iterator[PostgresSaver]:
    """PostgresSaver na ConnectionPool (autocommit, prepare_threshold=0, dict_row).

    setup=True → saver.setup() (idempotentne migracje tabel checkpoints*). Błąd połączenia przy
    otwieraniu puli albo setup() → CheckpointerUnavailableError. Na wyjściu pool.close().
    """
    db = settings.db
    pool: ConnectionPool[Any] = ConnectionPool(
        settings.postgres_dsn(),
        min_size=db.pool_min_size,
        max_size=db.pool_max_size,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "connect_timeout": max(1, int(db.connect_timeout_s)),
        },
        open=False,
        name="kovalo-checkpoints",
    )
    try:
        try:
            pool.open(wait=True, timeout=db.connect_timeout_s)
            saver = PostgresSaver(cast(Any, pool), serde=serializer())
            if setup:
                saver.setup()
        except CONNECTION_ERRORS as exc:
            # komunikaty sterownika mogą zawierać DSN, więc zostaje tylko klasa błędu
            raise CheckpointerUnavailableError(
                f"Postgres niedostępny dla checkpointów ({type(exc).__name__}). "
                "Uruchom: docker compose up -d postgres"
            ) from None
        yield saver
    finally:
        pool.close()


@contextmanager
def open_checkpointer(settings: Settings) -> Iterator[BaseCheckpointSaver[Any]]:
    """Według workflow.checkpointer: postgres → postgres_checkpointer, memory → w pamięci."""
    if settings.workflow.checkpointer == "memory":
        yield memory_checkpointer()
        return
    with postgres_checkpointer(settings) as saver:
        yield saver
