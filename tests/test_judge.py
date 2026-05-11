"""Tests for score_comparison and judge_with_model tools."""

from __future__ import annotations

import json

import httpx
import pytest
from respx import MockRouter

from ollama_mcp.storage import get_repo
from ollama_mcp.tools import get_registry
from ollama_mcp.tools.judge import judge_with_model, score_comparison

# ---------------------------------------------------------------------------
# Registry presence
# ---------------------------------------------------------------------------


def test_judge_tools_register_on_import() -> None:
    registry = get_registry()
    assert "score_comparison" in registry
    assert "judge_with_model" in registry


# ---------------------------------------------------------------------------
# score_comparison — happy path
# ---------------------------------------------------------------------------


async def test_score_comparison_happy_path() -> None:
    repo = get_repo()
    eval_id = repo.insert_partial(prompt="Explain recursion", models=["llama3", "mistral"])

    result = await score_comparison(
        {
            "eval_id": eval_id,
            "criteria": ["clarity", "conciseness"],
            "scores": {
                "llama3": {"score": 7, "reasoning": "Accurate but verbose"},
                "mistral": {"score": 9, "reasoning": "Clear and concise"},
            },
            "winner": "mistral",
            "notes": "phi4 timed out, excluded",
        }
    )

    assert result["logged"] is True
    assert result["eval_id"] == eval_id

    # Verify the row was actually updated in storage.
    row = repo.get(eval_id)
    assert row is not None
    assert row["winner"] == "mistral"
    assert row["judge_model"] is None  # Claude-as-judge path sets no judge_model
    assert row["notes"] == "phi4 timed out, excluded"
    assert row["criteria"] == ["clarity", "conciseness"]
    scores = row["scores"]
    assert isinstance(scores, dict)
    assert isinstance(scores["mistral"], dict)
    assert scores["mistral"]["score"] == 9


async def test_score_comparison_happy_path_no_notes() -> None:
    repo = get_repo()
    eval_id = repo.insert_partial(prompt="Hello", models=["llama3", "mistral"])

    result = await score_comparison(
        {
            "eval_id": eval_id,
            "criteria": ["quality"],
            "scores": {"llama3": {"score": 8, "reasoning": "Good"}},
            "winner": "llama3",
        }
    )

    assert result["logged"] is True
    assert result["eval_id"] == eval_id


# ---------------------------------------------------------------------------
# score_comparison — input validation failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments,field",
    [
        (
            {"criteria": ["c"], "scores": {"m": {"score": 1, "reasoning": "r"}}, "winner": "m"},
            "eval_id",
        ),
        (
            {
                "eval_id": "",
                "criteria": ["c"],
                "scores": {"m": {"score": 1, "reasoning": "r"}},
                "winner": "m",
            },
            "eval_id",
        ),
        (
            {"eval_id": "x", "scores": {"m": {"score": 1, "reasoning": "r"}}, "winner": "m"},
            "criteria",
        ),
        (
            {
                "eval_id": "x",
                "criteria": [],
                "scores": {"m": {"score": 1, "reasoning": "r"}},
                "winner": "m",
            },
            "criteria",
        ),
        (
            {"eval_id": "x", "criteria": ["c"], "winner": "m"},
            "scores",
        ),
        (
            {"eval_id": "x", "criteria": ["c"], "scores": {}, "winner": "m"},
            "scores",
        ),
        (
            {
                "eval_id": "x",
                "criteria": ["c"],
                "scores": {"m": {"score": 1, "reasoning": "r"}},
                "winner": "",
            },
            "winner",
        ),
    ],
)
async def test_score_comparison_invalid_input(arguments: dict[str, object], field: str) -> None:
    result = await score_comparison(arguments)
    assert result["error"]["code"] == "INVALID_INPUT"
    assert "duration_ms" in result


async def test_score_comparison_score_entry_not_dict() -> None:
    result = await score_comparison(
        {
            "eval_id": "x",
            "criteria": ["c"],
            "scores": {"model1": "not-a-dict"},
            "winner": "model1",
        }
    )
    assert result["error"]["code"] == "INVALID_INPUT"


async def test_score_comparison_score_entry_missing_score() -> None:
    result = await score_comparison(
        {
            "eval_id": "x",
            "criteria": ["c"],
            "scores": {"model1": {"reasoning": "good"}},
            "winner": "model1",
        }
    )
    assert result["error"]["code"] == "INVALID_INPUT"


async def test_score_comparison_score_entry_missing_reasoning() -> None:
    result = await score_comparison(
        {
            "eval_id": "x",
            "criteria": ["c"],
            "scores": {"model1": {"score": 8}},
            "winner": "model1",
        }
    )
    assert result["error"]["code"] == "INVALID_INPUT"


async def test_score_comparison_notes_not_string() -> None:
    repo = get_repo()
    eval_id = repo.insert_partial(prompt="Hi", models=["llama3", "mistral"])

    result = await score_comparison(
        {
            "eval_id": eval_id,
            "criteria": ["c"],
            "scores": {"llama3": {"score": 8, "reasoning": "good"}},
            "winner": "llama3",
            "notes": 42,  # not a string
        }
    )
    assert result["error"]["code"] == "INVALID_INPUT"


# ---------------------------------------------------------------------------
# score_comparison — unknown eval_id
# ---------------------------------------------------------------------------


async def test_score_comparison_unknown_eval_id() -> None:
    result = await score_comparison(
        {
            "eval_id": "nonexistent-eval-id",
            "criteria": ["clarity"],
            "scores": {"llama3": {"score": 7, "reasoning": "ok"}},
            "winner": "llama3",
        }
    )

    assert result["error"]["code"] == "INVALID_INPUT"
    assert "nonexistent-eval-id" in result["error"]["message"]


# ---------------------------------------------------------------------------
# judge_with_model — happy path
# ---------------------------------------------------------------------------


async def test_judge_with_model_happy_path(ollama_mock: MockRouter) -> None:
    # Stub out the comparison models
    call_count = 0

    def _generate_side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        body = json.loads(request.content)
        call_count += 1
        # First two calls are comparison models; third is the judge
        model = body.get("model", "unknown")
        if call_count <= 2:
            return httpx.Response(
                200,
                json={"model": model, "response": f"Response from {model}", "done": True},
            )
        # Judge call — return valid JSON
        judge_json = json.dumps(
            {
                "scores": {
                    "llama3": {"score": 7, "reasoning": "Decent"},
                    "mistral": {"score": 9, "reasoning": "Excellent"},
                },
                "winner": "mistral",
            }
        )
        return httpx.Response(200, json={"model": model, "response": judge_json, "done": True})

    ollama_mock.post("/api/generate").mock(side_effect=_generate_side_effect)

    result = await judge_with_model(
        {
            "prompt": "Explain recursion simply",
            "models": ["llama3", "mistral"],
            "judge_model": "llama3",
            "criteria": ["clarity"],
            "task_type": "explanation",
        }
    )

    assert result.get("logged") is True
    assert result["winner"] == "mistral"
    assert result["judge_model"] == "llama3"
    assert result["criteria"] == ["clarity"]
    assert "eval_id" in result
    assert isinstance(result["scores"], dict)
    assert result["scores"]["mistral"]["score"] == 9

    # Verify DB row
    repo = get_repo()
    row = repo.get(result["eval_id"])
    assert row is not None
    assert row["winner"] == "mistral"
    assert row["judge_model"] == "llama3"


async def test_judge_with_model_winner_computed_from_scores_when_absent(
    ollama_mock: MockRouter,
) -> None:
    """If judge omits 'winner', server picks model with highest score."""
    call_count = 0

    def _generate(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        body = json.loads(request.content)
        call_count += 1
        model = body.get("model", "unknown")
        if call_count <= 2:
            return httpx.Response(
                200, json={"model": model, "response": "some response", "done": True}
            )
        # Judge: omit winner
        judge_json = json.dumps(
            {
                "scores": {
                    "llama3": {"score": 5, "reasoning": "ok"},
                    "mistral": {"score": 9, "reasoning": "great"},
                }
            }
        )
        return httpx.Response(200, json={"model": model, "response": judge_json, "done": True})

    ollama_mock.post("/api/generate").mock(side_effect=_generate)

    result = await judge_with_model(
        {
            "prompt": "Hello",
            "models": ["llama3", "mistral"],
            "judge_model": "llama3",
            "criteria": ["quality"],
        }
    )

    assert result.get("logged") is True
    assert result["winner"] == "mistral"  # highest score wins


async def test_judge_with_model_judge_output_in_markdown_fence(
    ollama_mock: MockRouter,
) -> None:
    """Judge model wrapping its JSON in ```json code fences should still parse."""
    call_count = 0

    def _generate(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        body = json.loads(request.content)
        call_count += 1
        model = body.get("model", "unknown")
        if call_count <= 2:
            return httpx.Response(200, json={"model": model, "response": "answer", "done": True})
        judge_json = (
            "```json\n"
            + json.dumps(
                {
                    "scores": {
                        "llama3": {"score": 6, "reasoning": "ok"},
                        "mistral": {"score": 8, "reasoning": "good"},
                    },
                    "winner": "mistral",
                }
            )
            + "\n```"
        )
        return httpx.Response(200, json={"model": model, "response": judge_json, "done": True})

    ollama_mock.post("/api/generate").mock(side_effect=_generate)

    result = await judge_with_model(
        {
            "prompt": "Hi",
            "models": ["llama3", "mistral"],
            "judge_model": "llama3",
            "criteria": ["clarity"],
        }
    )
    assert result.get("logged") is True
    assert result["winner"] == "mistral"


# ---------------------------------------------------------------------------
# judge_with_model — input validation failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments,expected_field",
    [
        ({"models": ["a", "b"], "judge_model": "j", "criteria": ["c"]}, "prompt"),
        ({"prompt": "", "models": ["a", "b"], "judge_model": "j", "criteria": ["c"]}, "prompt"),
        ({"prompt": "p", "judge_model": "j", "criteria": ["c"]}, "models"),
        ({"prompt": "p", "models": ["a"], "judge_model": "j", "criteria": ["c"]}, "models"),
        ({"prompt": "p", "models": ["a", "b"], "criteria": ["c"]}, "judge_model"),
        (
            {"prompt": "p", "models": ["a", "b"], "judge_model": "", "criteria": ["c"]},
            "judge_model",
        ),
        ({"prompt": "p", "models": ["a", "b"], "judge_model": "j"}, "criteria"),
        ({"prompt": "p", "models": ["a", "b"], "judge_model": "j", "criteria": []}, "criteria"),
    ],
)
async def test_judge_with_model_invalid_input(
    arguments: dict[str, object], expected_field: str
) -> None:
    result = await judge_with_model(arguments)
    assert result["error"]["code"] == "INVALID_INPUT"
    assert "duration_ms" in result


async def test_judge_with_model_invalid_task_type() -> None:
    result = await judge_with_model(
        {
            "prompt": "p",
            "models": ["a", "b"],
            "judge_model": "j",
            "criteria": ["c"],
            "task_type": "debugging",  # not in the enum
        }
    )
    assert result["error"]["code"] == "INVALID_INPUT"


async def test_judge_with_model_invalid_timeout_ms() -> None:
    result = await judge_with_model(
        {
            "prompt": "p",
            "models": ["a", "b"],
            "judge_model": "j",
            "criteria": ["c"],
            "timeout_ms": 0,
        }
    )
    assert result["error"]["code"] == "INVALID_INPUT"


async def test_judge_with_model_invalid_tags() -> None:
    result = await judge_with_model(
        {
            "prompt": "p",
            "models": ["a", "b"],
            "judge_model": "j",
            "criteria": ["c"],
            "tags": [1, 2],  # not strings
        }
    )
    assert result["error"]["code"] == "INVALID_INPUT"


# ---------------------------------------------------------------------------
# judge_with_model — all comparison models fail
# ---------------------------------------------------------------------------


async def test_judge_with_model_all_models_fail(ollama_mock: MockRouter) -> None:
    ollama_mock.post("/api/generate").mock(side_effect=httpx.TimeoutException("simulated timeout"))

    result = await judge_with_model(
        {
            "prompt": "Hello",
            "models": ["llama3", "mistral"],
            "judge_model": "llama3",
            "criteria": ["quality"],
        }
    )

    assert "error" in result
    assert result["error"]["code"] == "MODEL_TIMEOUT"


# ---------------------------------------------------------------------------
# judge_with_model — judge model failures
# ---------------------------------------------------------------------------


async def test_judge_with_model_judge_timeout(ollama_mock: MockRouter) -> None:
    call_count = 0

    def _generate(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        body = json.loads(request.content)
        call_count += 1
        model = body.get("model", "unknown")
        if call_count <= 2:
            return httpx.Response(
                200, json={"model": model, "response": "response text", "done": True}
            )
        raise httpx.TimeoutException("judge timed out")

    ollama_mock.post("/api/generate").mock(side_effect=_generate)

    result = await judge_with_model(
        {
            "prompt": "Hello",
            "models": ["llama3", "mistral"],
            "judge_model": "llama3",
            "criteria": ["quality"],
        }
    )

    assert result["error"]["code"] == "MODEL_TIMEOUT"


async def test_judge_with_model_judge_unparseable_output(ollama_mock: MockRouter) -> None:
    call_count = 0

    def _generate(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        body = json.loads(request.content)
        call_count += 1
        model = body.get("model", "unknown")
        if call_count <= 2:
            return httpx.Response(
                200, json={"model": model, "response": "response text", "done": True}
            )
        # Judge returns gibberish — not valid JSON
        return httpx.Response(
            200, json={"model": model, "response": "I cannot decide.", "done": True}
        )

    ollama_mock.post("/api/generate").mock(side_effect=_generate)

    result = await judge_with_model(
        {
            "prompt": "Hello",
            "models": ["llama3", "mistral"],
            "judge_model": "llama3",
            "criteria": ["quality"],
        }
    )

    assert result["error"]["code"] == "INVALID_INPUT"
    assert "unparseable" in result["error"]["message"]


async def test_judge_with_model_judge_unreachable(ollama_mock: MockRouter) -> None:
    call_count = 0

    def _generate(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        body = json.loads(request.content)
        call_count += 1
        model = body.get("model", "unknown")
        if call_count <= 2:
            return httpx.Response(200, json={"model": model, "response": "answer", "done": True})
        raise httpx.ConnectError("cannot connect")

    ollama_mock.post("/api/generate").mock(side_effect=_generate)

    result = await judge_with_model(
        {
            "prompt": "Hello",
            "models": ["llama3", "mistral"],
            "judge_model": "llama3",
            "criteria": ["quality"],
        }
    )

    assert result["error"]["code"] == "OLLAMA_UNREACHABLE"


# ---------------------------------------------------------------------------
# judge_with_model — one comparison model fails (partial success)
# ---------------------------------------------------------------------------


async def test_judge_with_model_one_model_fails(ollama_mock: MockRouter) -> None:
    """Partial success: only one comparison model succeeds; judging still works."""
    call_count = 0

    def _generate(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        body = json.loads(request.content)
        call_count += 1
        model = body.get("model", "unknown")
        if call_count == 1:
            # First comparison model fails
            raise httpx.TimeoutException("model1 timed out")
        if call_count == 2:
            return httpx.Response(
                200, json={"model": model, "response": "great answer", "done": True}
            )
        # Judge call
        judge_json = json.dumps(
            {
                "scores": {"mistral": {"score": 8, "reasoning": "good"}},
                "winner": "mistral",
            }
        )
        return httpx.Response(200, json={"model": model, "response": judge_json, "done": True})

    ollama_mock.post("/api/generate").mock(side_effect=_generate)

    result = await judge_with_model(
        {
            "prompt": "Hello",
            "models": ["llama3", "mistral"],
            "judge_model": "llama3",
            "criteria": ["quality"],
        }
    )

    assert result.get("logged") is True
    assert result["winner"] == "mistral"
