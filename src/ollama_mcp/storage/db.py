"""SQLite connection and migration utilities."""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from importlib import resources

from ollama_mcp.logging import get_logger
from ollama_mcp.paths import create_data_file
from ollama_mcp.storage.evals_repo import EvalsRepo

_MIGRATIONS_PACKAGE = "ollama_mcp.storage.migrations"
_MIGRATION_PATTERN = re.compile(r"^(?P<version>[0-9]+)_.+\.sql$")

_log = get_logger("ollama_mcp.storage.db")


@dataclass(frozen=True)
class _Migration:
    version: int
    name: str
    sql: str


def _discover_migrations() -> list[_Migration]:
    migrations: list[_Migration] = []
    seen_versions: set[int] = set()

    for resource in resources.files(_MIGRATIONS_PACKAGE).iterdir():
        if not resource.is_file() or not resource.name.endswith(".sql"):
            continue

        match = _MIGRATION_PATTERN.match(resource.name)
        if match is None:
            raise ValueError(f"Malformed migration filename: {resource.name}")

        version = int(match.group("version"))
        if version in seen_versions:
            raise ValueError(f"Duplicate migration version: {version}")
        seen_versions.add(version)

        migrations.append(
            _Migration(
                version=version,
                name=resource.name,
                sql=resource.read_text(encoding="utf-8"),
            )
        )

    return sorted(migrations, key=lambda migration: migration.version)


def get_connection() -> sqlite3.Connection:
    """Open a hardened SQLite connection for eval storage."""
    db_filename = os.environ.get("DB_PATH", "evals.db")
    db_path = create_data_file(db_filename)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    """Apply all pending migrations in version order."""
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
              version INTEGER PRIMARY KEY,
              applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    row = conn.execute("SELECT COALESCE(MAX(version), 0) AS version FROM schema_version").fetchone()
    current_version = int(row["version"]) if row is not None else 0

    for migration in _discover_migrations():
        if migration.version <= current_version:
            continue

        with conn:
            conn.executescript(migration.sql)
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (migration.version,),
            )

        _log.info("Applied migration %s (version=%s)", migration.name, migration.version)
        current_version = migration.version


def get_repo() -> EvalsRepo:
    """Open the database, run migrations, and return an EvalsRepo."""
    conn = get_connection()
    migrate(conn)
    return EvalsRepo(conn)
