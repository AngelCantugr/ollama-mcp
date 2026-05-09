"""Path hardening utilities.

All data files must live under DATA_DIR. Symlinks pointing outside are rejected
to prevent directory-traversal attacks on the SQLite DB or routing config.
"""

import os
from pathlib import Path


class PathError(Exception):
    pass


def get_data_dir() -> Path:
    """Return the resolved DATA_DIR, creating it if necessary."""
    data_dir = Path(os.environ.get("DATA_DIR", "~/.ollama-mcp")).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def resolve_data_path(filename: str) -> Path:
    """Resolve *filename* under DATA_DIR.

    Raises PathError if the resolved path escapes DATA_DIR or if a symlink in
    the raw path points outside DATA_DIR.
    """
    data_dir = get_data_dir()
    raw = data_dir / filename

    # Check symlinks first — before resolve() silently follows them —
    # so callers get the "points outside" message rather than "resolves outside".
    if raw.exists() and raw.is_symlink():
        target = raw.resolve()
        try:
            target.relative_to(data_dir)
        except ValueError as exc:
            raise PathError(f"Symlink {filename!r} points outside DATA_DIR") from exc

    candidate = raw.resolve()

    # Reject if the resolved path is outside data_dir
    try:
        candidate.relative_to(data_dir)
    except ValueError as exc:
        raise PathError(f"Path {filename!r} resolves outside DATA_DIR") from exc

    return candidate


def create_data_file(filename: str) -> Path:
    """Create *filename* under DATA_DIR with 0600 permissions and return its path.

    The file is created only if it does not already exist; existing files are
    left untouched so callers can open-or-create safely.
    """
    path = resolve_data_path(filename)
    # O_CREAT | O_EXCL would race on existing files; open with mode 'a' is
    # safer and achieves "create if missing" semantics.
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(fd)
    return path
