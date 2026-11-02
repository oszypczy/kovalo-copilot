"""Indeks pgvector: PGVectorStore, podmiana tabeli, metadane indeksu (FR-1.4, FR-6.3).

Kontrakt F1, sekcja 5.6; decyzja: ADR-0007 pkt 1. Jedyny moduł, który importuje
langchain_postgres, i to leniwie, żeby import `kovalo.knowledge` nie ciągnął SQLAlchemy.
Nazwy tabel pochodzą wyłącznie z configu (walidowane regexem), a do SQL trafiają
przez psycopg.sql.Identifier.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import psycopg
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from psycopg import sql
from pydantic import BaseModel

from kovalo.config import ConfigError, Settings

if TYPE_CHECKING:
    from langchain_postgres import PGEngine, PGVectorStore

__all__ = [
    "ID_COLUMN",
    "METADATA_COLUMNS",
    "IndexMeta",
    "IndexMismatchError",
    "IndexUnavailableError",
    "check_index",
    "is_cosine_distance",
    "is_pg_store",
    "open_store",
    "pg_engine",
    "read_index_meta",
    "replace_table",
]

METADATA_COLUMNS: tuple[tuple[str, str], ...] = (
    ("doc_id", "TEXT"),
    ("doc_type", "TEXT"),
    ("chunk_index", "INTEGER"),
)
ID_COLUMN = ("langchain_id", "TEXT")  # wartość = chunk_id
_NEW_SUFFIX = "__new"


class IndexMeta(BaseModel):
    table: str
    embedding_provider: Literal["fastembed", "openai_cache"]
    embedding_model: str
    dimensions: int
    n_chunks: int
    updated_at: datetime  # UTC


class IndexMismatchError(ConfigError):
    """Tabela zbudowana innym modelem/wymiarem niż w configu albo brak tabeli/metadanych."""


class IndexUnavailableError(RuntimeError):
    """Postgres niedostępny (połączenie, timeout). Komunikat bez hasła i bez pełnego DSN."""


def _unavailable(exc: BaseException) -> IndexUnavailableError:
    # Komunikaty sterowników mogą zawierać DSN, więc zostaje tylko klasa błędu.
    return IndexUnavailableError(type(exc).__name__)


_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
    psycopg.OperationalError,
    OSError,
    TimeoutError,
)


def _connection_errors() -> tuple[type[BaseException], ...]:
    from sqlalchemy.exc import InterfaceError, OperationalError

    return (*_CONNECTION_ERRORS, OperationalError, InterfaceError)


def _connect(settings: Settings) -> psycopg.Connection[Any]:
    try:
        return psycopg.connect(
            settings.postgres_dsn(), connect_timeout=max(1, int(settings.db.connect_timeout_s))
        )
    except _CONNECTION_ERRORS as exc:
        raise _unavailable(exc) from None


def pg_engine(settings: Settings) -> PGEngine:
    """PGEngine.from_connection_string(settings.sqlalchemy_url()) z limitem czasu połączenia."""
    from langchain_postgres import PGEngine

    return PGEngine.from_connection_string(
        settings.sqlalchemy_url(),
        connect_args={"connect_timeout": max(1, int(settings.db.connect_timeout_s))},
    )


def is_pg_store(store: object) -> bool:
    from langchain_postgres import PGVectorStore

    return isinstance(store, PGVectorStore)


def is_cosine_distance(store: PGVectorStore) -> bool:
    """Czy PGVectorStore liczy odległość kosinusową (strategia jest w prywatnym polu)."""
    from langchain_postgres.v2.indexes import DistanceStrategy

    inner = getattr(store, "_PGVectorStore__vs", None)
    return getattr(inner, "distance_strategy", None) == DistanceStrategy.COSINE_DISTANCE


def open_store(engine: PGEngine, embeddings: Embeddings, *, table: str) -> PGVectorStore:
    """Istniejąca tabela jako PGVectorStore (id TEXT, kolumny metadanych, odległość kosinusowa)."""
    from langchain_postgres import PGVectorStore
    from langchain_postgres.v2.indexes import DistanceStrategy

    try:
        return PGVectorStore.create_sync(
            engine,
            embedding_service=embeddings,
            table_name=table,
            id_column=ID_COLUMN[0],
            metadata_columns=[name for name, _ in METADATA_COLUMNS],
            distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        )
    except _connection_errors() as exc:
        raise _unavailable(exc) from None


def _ensure_meta_table(conn: psycopg.Connection[Any], meta_table: str) -> None:
    conn.execute(
        sql.SQL(
            "CREATE TABLE IF NOT EXISTS {} ("
            "table_name TEXT PRIMARY KEY, embedding_provider TEXT NOT NULL, "
            "embedding_model TEXT NOT NULL, dimensions INTEGER NOT NULL, "
            "n_chunks INTEGER NOT NULL, updated_at TIMESTAMPTZ NOT NULL)"
        ).format(sql.Identifier(meta_table))
    )


def _table_exists(conn: psycopg.Connection[Any], table: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s)", (f'public."{table}"',)).fetchone()
    return bool(row and row[0])


def _count(conn: psycopg.Connection[Any], table: str) -> int:
    row = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()
    return int(row[0]) if row else 0


def replace_table(
    settings: Settings,
    engine: PGEngine,
    embeddings: Embeddings,
    chunks: Sequence[Document],
    meta: IndexMeta,
    *,
    table: str,
) -> tuple[int, int]:
    """Buduje <table>__new, wypełnia ją i w jednej transakcji podmienia <table> + metadane.

    Zwraca (liczba wierszy po podmianie, liczba wierszy w poprzedniej tabeli albo 0).
    """
    from langchain_postgres import Column

    new_table = f"{table}{_NEW_SUFFIX}"
    try:
        engine.init_vectorstore_table(
            new_table,
            meta.dimensions,
            id_column=Column(ID_COLUMN[0], ID_COLUMN[1], nullable=False),
            metadata_columns=[Column(name, data_type) for name, data_type in METADATA_COLUMNS],
            overwrite_existing=True,
        )
        store = open_store(engine, embeddings, table=new_table)
        store.add_documents(list(chunks), ids=[c.metadata["chunk_id"] for c in chunks])
    except _connection_errors() as exc:
        raise _unavailable(exc) from None

    meta_table = settings.db.index_meta_table
    with _connect(settings) as conn, conn.transaction():
        rows_before = _count(conn, table) if _table_exists(conn, table) else 0
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table)))
        conn.execute(
            sql.SQL("ALTER TABLE {} RENAME TO {}").format(
                sql.Identifier(new_table), sql.Identifier(table)
            )
        )
        # Indeks klucza głównego zachowuje nazwę <table>__new_pkey; bez zmiany nazwy kolejne
        # init_vectorstore_table('<table>__new') kończyłoby się konfliktem nazw relacji.
        conn.execute(
            sql.SQL("ALTER INDEX IF EXISTS {} RENAME TO {}").format(
                sql.Identifier(f"{new_table}_pkey"), sql.Identifier(f"{table}_pkey")
            )
        )
        _ensure_meta_table(conn, meta_table)
        rows_after = _count(conn, table)
        conn.execute(
            sql.SQL(
                "INSERT INTO {} (table_name, embedding_provider, embedding_model, dimensions, "
                "n_chunks, updated_at) VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (table_name) DO UPDATE SET "
                "embedding_provider = EXCLUDED.embedding_provider, "
                "embedding_model = EXCLUDED.embedding_model, dimensions = EXCLUDED.dimensions, "
                "n_chunks = EXCLUDED.n_chunks, updated_at = EXCLUDED.updated_at"
            ).format(sql.Identifier(meta_table)),
            (
                table,
                meta.embedding_provider,
                meta.embedding_model,
                meta.dimensions,
                rows_after,
                meta.updated_at,
            ),
        )
    return rows_after, rows_before


def read_index_meta(settings: Settings, table: str) -> IndexMeta | None:
    """Metadane tabeli indeksu albo None (brak tabeli metadanych lub wiersza)."""
    meta_table = settings.db.index_meta_table
    with _connect(settings) as conn:
        if not _table_exists(conn, meta_table):
            return None
        row = conn.execute(
            sql.SQL(
                "SELECT table_name, embedding_provider, embedding_model, dimensions, n_chunks, "
                "updated_at FROM {} WHERE table_name = %s"
            ).format(sql.Identifier(meta_table)),
            (table,),
        ).fetchone()
    if row is None:
        return None
    return IndexMeta(
        table=row[0],
        embedding_provider=row[1],
        embedding_model=row[2],
        dimensions=row[3],
        n_chunks=row[4],
        updated_at=row[5],
    )


def check_index(
    settings: Settings, table: str, *, provider: str, model: str, dimensions: int
) -> IndexMeta:
    """Meta zgodna z configiem → IndexMeta; brak tabeli/metadanych albo niezgodność → błąd."""
    with _connect(settings) as conn:
        exists = _table_exists(conn, table)
    meta = read_index_meta(settings, table) if exists else None
    if meta is None:
        raise IndexMismatchError(f"Brak indeksu {table} albo jego metadanych. Uruchom make ingest.")
    expected = (provider, model, dimensions)
    actual = (meta.embedding_provider, meta.embedding_model, meta.dimensions)
    if actual != expected:
        raise IndexMismatchError(
            f"Indeks {table} zbudowano modelem {actual[0]} / {actual[1]} ({actual[2]}), "
            f"a config wskazuje {provider} / {model} ({dimensions}). Uruchom make ingest."
        )
    return meta
