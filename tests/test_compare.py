"""Tests for the compare Learn-Mode tool."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from respx import MockRouter

from ollama_mcp.storage import get_repo
from ollama_mcp.tools import get_registry
from ollama_mcp.tools.compare import compare


async def test_compare_happy_path_multi_model(ollama_mock: MockRouter) -> None:
    result = await compare({"prompt": "hi", "models": ["llama3", "mistral"]})

    eval_id = result["eval_id"]
    assert isinstance(eval_id, str)
    assert eval_id
    assert result["prompt"] == "hi"
    assert result["task_type"] is None
    assert len(result["results"]) == 2
    assert [item["model"] for item in result["results"]] == ["llama3", "mistral"]

    for item in result["results"]:
        assert item["status"] == "ok"
        assert item["response"].startswith(
            f'<ollama_output model="{item["model"]}" untrusted="true">'
        )
        assert item["response"].endswith("</ollama_output>")

    row = get_repo().get(eval_id)
    assert row is not None


async def test_compare_partial_success_timeout(ollama_mock: MockRouter) -> None:
    def _side_effect(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        if payload["model"] == "mistral":
            raise httpx.TimeoutException("Simulated timeout for mistral")
        return httpx.Response(200, json={"response": f"Hello from {payload['model']}"})

    ollama_mock.post("/api/generate").mock(side_effect=_side_effect)

    result = await compare({"prompt": "hi", "models": ["llama3", "mistral"]})

    assert "eval_id" in result
    assert result["results"][0]["status"] == "ok"
    assert result["results"][1]["status"] == "timeout"
    assert isinstance(result["results"][1]["error"], str)


async def test_compare_all_models_fail_returns_success_shape(ollama_mock: MockRouter) -> None:
    ollama_mock.post("/api/generate").mock(side_effect=httpx.ConnectError("cannot connect"))

    result = await compare({"prompt": "hi", "models": ["llama3", "mistral"]})

    assert isinstance(result["eval_id"], str)
    assert result["eval_id"]
    assert len(result["results"]) == 2
    assert all(item["status"] == "error" for item in result["results"])


async def test_compare_sequential_by_default_no_overlap(ollama_mock: MockRouter) -> None:
    active = 0
    max_active = 0

    async def _side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.03)
        active -= 1
        payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"response": payload["model"]})

    ollama_mock.post("/api/generate").mock(side_effect=_side_effect)

    result = await compare({"prompt": "hi", "models": ["llama3", "mistral", "phi4"]})

    assert len(result["results"]) == 3
    assert max_active == 1


async def test_compare_max_concurrency_two_allows_overlap(ollama_mock: MockRouter) -> None:
    active = 0
    max_active = 0

    async def _side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.03)
        active -= 1
        payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"response": payload["model"]})

    ollama_mock.post("/api/generate").mock(side_effect=_side_effect)

    result = await compare(
        {"prompt": "hi", "models": ["llama3", "mistral", "phi4"], "max_concurrency": 2}
    )

    assert len(result["results"]) == 3
    assert max_active >= 2


async def test_compare_preserves_input_order_with_mixed_completion_times(
    ollama_mock: MockRouter,
) -> None:
    async def _side_effect(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        model = payload["model"]
        if model == "llama3":
            await asyncio.sleep(0.005)
        else:
            await asyncio.sleep(0.03)
        return httpx.Response(200, json={"response": f"from-{model}"})

    ollama_mock.post("/api/generate").mock(side_effect=_side_effect)

    result = await compare({"prompt": "hi", "models": ["llama3", "mistral"], "max_concurrency": 2})
    assert [item["model"] for item in result["results"]] == ["llama3", "mistral"]


async def test_compare_invalid_input_models(ollama_mock: MockRouter) -> None:
    invalid_cases: list[dict[str, Any]] = [
        {"prompt": "hi"},
        {"prompt": "hi", "models": []},
        {"prompt": "hi", "models": ["llama3", 3]},
    ]

    for case in invalid_cases:
        result = await compare(case)
        assert result["error"]["code"] == "INVALID_INPUT"


async def test_compare_invalid_task_type(ollama_mock: MockRouter) -> None:
    result = await compare({"prompt": "hi", "models": ["llama3"], "task_type": "debugging"})
    assert result["error"]["code"] == "INVALID_INPUT"


async def test_compare_persists_task_type_and_tags(ollama_mock: MockRouter) -> None:
    result = await compare(
        {
            "prompt": "hi",
            "models": ["llama3"],
            "task_type": "explanation",
            "tags": ["learn", "bench"],
        }
    )

    row = get_repo().get(result["eval_id"])
    assert row is not None
    assert row["task_type"] == "explanation"
    assert row["tags"] == ["learn", "bench"]


def test_compare_registers_on_import() -> None:
    import ollama_mcp.tools.compare  # noqa: F401

    registry = get_registry()
    assert "compare" in registry
