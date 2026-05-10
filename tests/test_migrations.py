"""Tests for storage migration runner."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from ollama_mcp.storage.db import get_connection, migrate


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    connection = get_connection()
    yield connection
    connection.close()


def _write_migration_package(
    root: Path,
    package_name: str,
    files: dict[str, str],
) -> str:
    package_dir = root / package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    for filename, content in files.items():
        (package_dir / filename).write_text(content, encoding="utf-8")
    return package_name


def test_migrate_applies_initial_schema(conn: sqlite3.Connection) -> None:
    migrate(conn)

    version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert version == 1


def test_migrate_is_idempotent(conn: sqlite3.Connection) -> None:
    migrate(conn)
    migrate(conn)

    versions = conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
    assert [row[0] for row in versions] == [1]


def test_migrate_creates_tables_and_indexes(conn: sqlite3.Connection) -> None:
    migrate(conn)

    table_rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    table_names = {row[0] for row in table_rows}
    assert {"schema_version", "evals", "routing_history"}.issubset(table_names)

    index_rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
    index_names = {row[0] for row in index_rows}
    assert {"idx_evals_task_type", "idx_evals_winner", "idx_evals_prompt_hash"}.issubset(
        index_names
    )


def test_migrate_applies_multiple_migrations_in_order(
    conn: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_name = _write_migration_package(
        tmp_path,
        "test_migrations_pkg",
        {
            "001_create_table.sql": "CREATE TABLE sample (value INTEGER);",
            "002_insert_first.sql": "INSERT INTO sample (value) VALUES (2);",
            "999_test_migration.sql": "INSERT INTO sample (value) VALUES (999);",
        },
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr("ollama_mcp.storage.db._MIGRATIONS_PACKAGE", package_name)

    migrate(conn)

    values = conn.execute("SELECT value FROM sample ORDER BY rowid").fetchall()
    assert [row[0] for row in values] == [2, 999]
    assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 999


def test_migrate_rejects_malformed_migration_filename(
    conn: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_name = _write_migration_package(
        tmp_path,
        "bad_migrations_pkg",
        {
            "001_ok.sql": "CREATE TABLE sample (value INTEGER);",
            "bad_name.sql": "SELECT 1;",
        },
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr("ollama_mcp.storage.db._MIGRATIONS_PACKAGE", package_name)

    with pytest.raises(ValueError, match="Malformed migration filename"):
        migrate(conn)


def test_migrate_rolls_back_failed_migration_transaction(
    conn: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_name = _write_migration_package(
        tmp_path,
        "broken_migrations_pkg",
        {
            "001_create_table.sql": "CREATE TABLE sample (value INTEGER);",
            "002_broken.sql": "INSERT INTO missing_table(value) VALUES (1);",
        },
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr("ollama_mcp.storage.db._MIGRATIONS_PACKAGE", package_name)

    with pytest.raises(sqlite3.DatabaseError):
        migrate(conn)

    assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 1
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sample'"
    ).fetchone()
    assert table_exists is not None
