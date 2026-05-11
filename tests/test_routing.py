"""Tests for routing tools."""

from __future__ import annotations

import json
from importlib import resources

import httpx
import pytest
from respx import MockRouter

from ollama_mcp import paths
from ollama_mcp.storage import get_repo
from ollama_mcp.tools import get_registry, routing


@pytest.fixture(autouse=True)
def _clear_routing_cache() -> None:
    routing._invalidate_cache()


def _bundled_defaults() -> dict[str, object]:
    parsed = json.loads(resources.files("ollama_mcp.config").joinpath("routing.json").read_text())
    assert isinstance(parsed, dict)
    return parsed


async def test_get_routing_config_bootstraps_defaults() -> None:
    result = await routing.get_routing_config({})

    assert result["rules"]["code"] == "codellama"
    assert result["rules"]["summary"] == "mistral"
    assert result["rules"]["explanation"] == "mistral"
    assert result["rules"]["general"] == "llama3"
    assert result["tier_overrides"]["fast"] == "phi4"
    assert result["default"] == "llama3"


async def test_get_routing_config_reflects_updated_rule() -> None:
    await routing.update_routing_rule({"task": "code", "model": "mistral", "reason": "test"})

    result = await routing.get_routing_config({})
    assert result["rules"]["code"] == "mistral"


async def test_route_task_type_uses_matching_rule(ollama_mock: MockRouter) -> None:
    result = await routing.route({"prompt": "Review this", "task_type": "code"})

    assert result["status"] == "ok"
    assert result["model"] == "codellama"
    assert result["matched_rule"] == "code"
    assert '<ollama_output model="codellama" untrusted="true">' in result["response"]


async def test_route_tier_override_wins(ollama_mock: MockRouter) -> None:
    result = await routing.route({"prompt": "Quick answer", "tier": "fast"})

    assert result["status"] == "ok"
    assert result["model"] == "phi4"
    assert result["matched_rule"] == "fast"


async def test_route_tier_wins_over_task_type(ollama_mock: MockRouter) -> None:
    result = await routing.route(
        {"prompt": "Quick code review", "task_type": "code", "tier": "fast"}
    )

    assert result["status"] == "ok"
    assert result["model"] == "phi4"
    assert result["matched_rule"] == "fast"


async def test_route_without_match_falls_back_to_default(ollama_mock: MockRouter) -> None:
    result = await routing.route({"prompt": "hello"})

    assert result["status"] == "ok"
    assert result["model"] == "llama3"
    assert result["matched_rule"] == "fallback"


async def test_route_model_not_found_falls_back_to_default(ollama_mock: MockRouter) -> None:
    call_models: list[str] = []

    def _generate_side_effect(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        call_models.append(model)
        if model == "codellama":
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json={"model": model, "response": f"response from {model}"})

    ollama_mock.routes.clear()
    ollama_mock.post("/api/generate").mock(side_effect=_generate_side_effect)

    result = await routing.route({"prompt": "check", "task_type": "code"})

    assert result["status"] == "ok"
    assert result["model"] == "llama3"
    assert result["matched_rule"] == "fallback"
    assert call_models == ["codellama", "llama3"]


async def test_route_ollama_unreachable_returns_error(ollama_mock: MockRouter) -> None:
    ollama_mock.routes.clear()
    ollama_mock.post("/api/generate").mock(side_effect=httpx.ConnectError("cannot connect"))

    result = await routing.route({"prompt": "check", "task_type": "code"})

    assert result["error"]["code"] == "OLLAMA_UNREACHABLE"


async def test_update_routing_rule_task_happy_path() -> None:
    result = await routing.update_routing_rule({"task": "code", "model": "mistral"})

    assert result == {"updated": True, "previous_model": "codellama", "new_model": "mistral"}
    rows = get_repo().list_routing_history()
    assert len(rows) == 1
    assert rows[0]["task"] == "code"
    assert rows[0]["old_model"] == "codellama"
    assert rows[0]["new_model"] == "mistral"


async def test_update_routing_rule_tier_override_happy_path() -> None:
    result = await routing.update_routing_rule({"task": "fast", "model": "llama3"})

    assert result == {"updated": True, "previous_model": "phi4", "new_model": "llama3"}

    config = await routing.get_routing_config({})
    assert config["tier_overrides"]["fast"] == "llama3"
    assert config["rules"]["code"] == "codellama"


async def test_update_routing_rule_invalid_task_returns_invalid_input() -> None:
    result = await routing.update_routing_rule({"task": "unknown", "model": "phi4"})

    assert result["error"]["code"] == "INVALID_INPUT"


async def test_update_routing_rule_empty_model_returns_invalid_input() -> None:
    result = await routing.update_routing_rule({"task": "code", "model": ""})

    assert result["error"]["code"] == "INVALID_INPUT"


async def test_update_routing_rule_atomic_write_failure_keeps_existing_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await routing.get_routing_config({})
    config_path = paths.resolve_data_path("routing.json")
    before = config_path.read_text(encoding="utf-8")

    def _raise_replace(_: object, __: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("ollama_mcp.tools.routing.os.replace", _raise_replace)

    result = await routing.update_routing_rule({"task": "code", "model": "mistral"})

    assert result["error"]["code"] == "DB_ERROR"
    after = config_path.read_text(encoding="utf-8")
    assert before == after


async def test_suggest_routing_updates_empty_history_returns_no_suggestions() -> None:
    result = await routing.suggest_routing_updates({})

    assert result == {"suggestions": []}


async def test_suggest_routing_updates_proposes_swap_when_better_model_exists() -> None:
    repo = get_repo()
    for i in range(3):
        repo.insert_complete(
            prompt=f"summary-{i}",
            models=["phi4", "mistral"],
            winner="phi4",
            scores={"phi4": 9, "mistral": 6},
            criteria=["quality"],
            task_type="summary",
        )

    result = await routing.suggest_routing_updates({})

    summary_suggestion = next(item for item in result["suggestions"] if item["task"] == "summary")
    assert summary_suggestion["current_model"] == "mistral"
    assert summary_suggestion["proposed_model"] == "phi4"
    assert "win rate" in summary_suggestion["rationale"]
    assert summary_suggestion["supporting_eval_ids"]


async def test_suggest_routing_updates_respects_threshold_filters() -> None:
    repo = get_repo()
    for i in range(4):
        repo.insert_complete(
            prompt=f"summary-{i}",
            models=["phi4", "mistral"],
            winner="phi4",
            scores={"phi4": 9, "mistral": 6},
            criteria=["quality"],
            task_type="summary",
        )

    result = await routing.suggest_routing_updates({"min_evals": 5, "win_rate_threshold": 0.8})

    assert result == {"suggestions": []}


async def test_reset_routing_restores_bundled_defaults() -> None:
    await routing.update_routing_rule({"task": "summary", "model": "phi4"})

    result = await routing.reset_routing({"confirm": True})

    assert result["reset"] is True
    config = await routing.get_routing_config({})
    assert config == _bundled_defaults()


async def test_reset_routing_requires_confirm_true() -> None:
    false_result = await routing.reset_routing({"confirm": False})
    missing_result = await routing.reset_routing({})

    assert false_result["error"]["code"] == "INVALID_INPUT"
    assert missing_result["error"]["code"] == "INVALID_INPUT"


async def test_reset_routing_preserves_routing_history() -> None:
    await routing.update_routing_rule({"task": "summary", "model": "phi4"})
    before = get_repo().list_routing_history()

    await routing.reset_routing({"confirm": True})

    after = get_repo().list_routing_history()
    assert before
    assert after
    assert after[0]["task"] == "summary"


async def test_end_to_end_self_improving_loop(ollama_mock: MockRouter) -> None:
    repo = get_repo()
    for i in range(4):
        repo.insert_complete(
            prompt=f"summary-{i}",
            models=["phi4", "mistral"],
            winner="phi4",
            scores={"phi4": 9, "mistral": 6},
            criteria=["quality"],
            task_type="summary",
        )

    suggestions = await routing.suggest_routing_updates({})
    suggestion = next(item for item in suggestions["suggestions"] if item["task"] == "summary")
    assert suggestion["current_model"] == "mistral"
    assert suggestion["proposed_model"] == "phi4"

    update = await routing.update_routing_rule(
        {"task": "summary", "model": "phi4", "reason": "wins in evals"}
    )
    assert update["updated"] is True

    ollama_mock.routes.clear()
    ollama_mock.post("/api/generate").mock(
        side_effect=lambda request: httpx.Response(
            200,
            json={
                "model": json.loads(request.content)["model"],
                "response": "summary response",
                "done": True,
            },
        )
    )

    routed = await routing.route({"prompt": "summarize X", "task_type": "summary"})
    assert routed["status"] == "ok"
    assert routed["model"] == "phi4"
    assert routed["matched_rule"] == "summary"

    history = get_repo().list_routing_history()
    assert any(
        row["task"] == "summary" and row["old_model"] == "mistral" and row["new_model"] == "phi4"
        for row in history
    )


def test_routing_tools_register_on_import() -> None:
    registry = get_registry()
    assert "route" in registry
    assert "get_routing_config" in registry
    assert "update_routing_rule" in registry
    assert "suggest_routing_updates" in registry
    assert "reset_routing" in registry
