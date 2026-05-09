"""Tests for path hardening utilities."""

import os
import stat
from pathlib import Path

import pytest

from ollama_mcp.paths import PathError, create_data_file, get_data_dir, resolve_data_path


def test_get_data_dir_uses_env(tmp_path: Path) -> None:
    # tmp_data_dir autouse fixture sets DATA_DIR to a temp directory
    data_dir = get_data_dir()
    assert data_dir.is_dir()
    assert str(data_dir).startswith(str(tmp_path))


def test_resolve_data_path_happy(tmp_path: Path) -> None:
    path = resolve_data_path("evals.db")
    assert path.parent == get_data_dir()
    assert path.name == "evals.db"


def test_resolve_data_path_rejects_traversal() -> None:
    with pytest.raises(PathError, match="resolves outside DATA_DIR"):
        resolve_data_path("../escape.txt")


def test_resolve_data_path_rejects_absolute() -> None:
    with pytest.raises(PathError, match="resolves outside DATA_DIR"):
        resolve_data_path("/etc/passwd")


def test_resolve_data_path_rejects_symlink_outside(tmp_path: Path) -> None:
    data_dir = get_data_dir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret")
    # Create a symlink inside DATA_DIR pointing outside
    link = data_dir / "link.txt"
    link.symlink_to(outside_file)

    with pytest.raises(PathError, match="points outside DATA_DIR"):
        resolve_data_path("link.txt")


def test_resolve_data_path_allows_symlink_inside(tmp_path: Path) -> None:
    data_dir = get_data_dir()
    real_file = data_dir / "real.txt"
    real_file.write_text("data")
    link = data_dir / "link.txt"
    link.symlink_to(real_file)

    result = resolve_data_path("link.txt")
    assert result == real_file.resolve()


def test_create_data_file_returns_path() -> None:
    path = create_data_file("test.db")
    assert path.exists()


def test_create_data_file_permissions() -> None:
    path = create_data_file("secure.db")
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600


def test_create_data_file_idempotent() -> None:
    """Calling create_data_file twice must not raise."""
    create_data_file("idempotent.db")
    create_data_file("idempotent.db")  # should not raise


def test_create_data_file_rejects_traversal() -> None:
    with pytest.raises(PathError):
        create_data_file("../outside.db")
