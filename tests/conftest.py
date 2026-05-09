"""Shared pytest fixtures.

ollama_mock: intercepts httpx calls to localhost:11434 via respx.
tmp_data_dir: sets DATA_DIR to a fresh tmp directory per test (autouse).
"""

from collections.abc import Generator, Iterator
from pathlib import Path

import pytest
import respx
from respx import MockRouter


@pytest.fixture
def ollama_mock() -> Generator[MockRouter, None, None]:
    """Intercept httpx traffic to the default Ollama endpoint."""
    with respx.mock(base_url="http://localhost:11434", assert_all_called=False) as mock:
        mock.get("/api/tags").respond(
            200,
            json={
                "models": [
                    {
                        "name": "llama3:latest",
                        "modified_at": "2024-01-01",
                        "size": 4_000_000_000,
                    }
                ]
            },
        )
        mock.post("/api/generate").respond(
            200,
            json={"model": "llama3", "response": "Hello!", "done": True},
        )
        yield mock


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point DATA_DIR at a fresh tmp directory for each test."""
    data_dir = tmp_path / "ollama-mcp-data"
    data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    yield
