"""Tests for discovery tools."""

from pathlib import Path

import httpx
import pytest
from respx import MockRouter

from ollama_mcp.tools import discovery, get_registry


async def test_list_models_happy_path(ollama_mock: MockRouter) -> None:
    result = await discovery.list_models({})

    assert "models" in result
    models = result["models"]
    assert isinstance(models, list)
    assert len(models) == 1
    first = models[0]
    assert first["name"] == "llama3:latest"
    assert "digest" in first
    assert "family" in first
    assert "size" in first
    assert "modified" in first


async def test_list_models_ollama_unreachable(ollama_mock: MockRouter) -> None:
    ollama_mock.routes.clear()
    ollama_mock.get("/api/tags").mock(side_effect=httpx.ConnectError("cannot connect"))

    result = await discovery.list_models({})

    assert result["error"]["code"] == "OLLAMA_UNREACHABLE"


async def test_list_models_formats_size_mapping(ollama_mock: MockRouter) -> None:
    ollama_mock.routes.clear()
    ollama_mock.get("/api/tags").respond(
        200,
        json={
            "models": [
                {
                    "name": "llama3",
                    "digest": "sha256:abc",
                    "details": {"family": "llama"},
                    "size": 4_700_000_000,
                    "modified_at": "2026-05-01",
                }
            ]
        },
    )

    result = await discovery.list_models({})

    assert result["models"][0]["size"] == "4.7GB"


async def test_health_happy_path(ollama_mock: MockRouter) -> None:
    result = await discovery.health({})

    assert result["ollama"] == "ok"
    assert result["db"] == "ok"
    assert Path(result["data_dir"]).exists()


async def test_health_ollama_unreachable(ollama_mock: MockRouter) -> None:
    ollama_mock.routes.clear()
    ollama_mock.get("/api/tags").mock(side_effect=httpx.ConnectError("cannot connect"))

    result = await discovery.health({})

    assert result["ollama"] == "unreachable"
    assert result["db"] == "ok"
    assert "data_dir" in result


async def test_health_data_dir_unwritable(
    monkeypatch: pytest.MonkeyPatch, ollama_mock: MockRouter
) -> None:
    def _raise_os_error(filename: str) -> Path:
        raise OSError(f"permission denied: {filename}")

    monkeypatch.setattr("ollama_mcp.tools.discovery.paths.create_data_file", _raise_os_error)

    result = await discovery.health({})

    assert result["ollama"] == "ok"
    assert result["db"] == "unwritable"


def test_discovery_tools_register_on_import() -> None:
    # Keep a module reference so static analysis does not prune the import.
    assert discovery is not None
    registry = get_registry()
    assert "list_models" in registry
    assert "health" in registry
