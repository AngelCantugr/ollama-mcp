"""Tests for knowledge tools."""

from __future__ import annotations

import csv
import json
import os
import stat
from pathlib import Path

from ollama_mcp import paths
from ollama_mcp.storage import get_repo
from ollama_mcp.tools import get_registry, knowledge


async def test_log_eval_happy_path() -> None:
    result = await knowledge.log_eval(
        {
            "prompt": "Explain recursion",
            "models": ["llama3", "mistral"],
            "winner": "mistral",
            "criteria": ["clarity"],
            "scores": {"llama3": 7, "mistral": 9},
            "task_type": "explanation",
            "tags": ["teaching"],
            "notes": "solid",
        }
    )

    assert result["logged"] is True
    eval_id = result["eval_id"]
    assert isinstance(eval_id, str)

    row = get_repo().get(eval_id)
    assert row is not None
    assert row["winner"] == "mistral"
    assert row["scores"] == {"llama3": 7, "mistral": 9}


async def test_log_eval_invalid_task_type_returns_invalid_input() -> None:
    result = await knowledge.log_eval(
        {
            "prompt": "Explain recursion",
            "models": ["llama3", "mistral"],
            "winner": "mistral",
            "criteria": ["clarity"],
            "scores": {"llama3": 7, "mistral": 9},
            "task_type": "debugging",
        }
    )

    assert result["error"]["code"] == "INVALID_INPUT"


async def test_log_eval_missing_winner_returns_invalid_input() -> None:
    result = await knowledge.log_eval(
        {
            "prompt": "Explain recursion",
            "models": ["llama3", "mistral"],
            "criteria": ["clarity"],
            "scores": {"llama3": 7, "mistral": 9},
        }
    )

    assert result["error"]["code"] == "INVALID_INPUT"
    assert "winner" in result["error"]["message"]


async def test_delete_eval_happy_path() -> None:
    repo = get_repo()
    eval_id = repo.insert_partial(prompt="delete me", models=["llama3"])

    result = await knowledge.delete_eval({"eval_id": eval_id})

    assert result == {"deleted": True}
    assert repo.get(eval_id) is None


async def test_delete_eval_missing_id_returns_false() -> None:
    result = await knowledge.delete_eval({"eval_id": "does-not-exist"})

    assert result == {"deleted": False}


async def test_export_evals_jsonl_happy_path() -> None:
    repo = get_repo()
    repo.insert_complete(
        prompt="p1",
        models=["llama3", "mistral"],
        winner="mistral",
        scores={"llama3": {"score": 6}, "mistral": {"score": 9}},
        criteria=["quality"],
        task_type="summary",
        tags=["tag1"],
    )
    repo.insert_complete(
        prompt="p2",
        models=["llama3", "mistral"],
        winner="llama3",
        scores={"llama3": 8, "mistral": 7},
        criteria=["quality"],
        task_type="code",
        tags=["tag2"],
    )

    result = await knowledge.export_evals({"format": "jsonl"})

    assert result["count"] == 2
    export_path = Path(result["path"])
    assert export_path.exists()

    lines = export_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    decoded = [json.loads(line) for line in lines]
    assert isinstance(decoded[0]["models"], list)
    assert isinstance(decoded[0]["scores"], dict)


async def test_export_evals_csv_happy_path() -> None:
    repo = get_repo()
    repo.insert_complete(
        prompt="csv",
        models=["llama3", "mistral"],
        winner="mistral",
        scores={"llama3": 5, "mistral": 8},
        criteria=["clarity"],
        task_type="summary",
        tags=["one"],
    )

    result = await knowledge.export_evals({"format": "csv"})

    export_path = Path(result["path"])
    assert export_path.exists()

    with export_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
        assert row is not None
        assert "models" in row
        assert json.loads(row["models"]) == ["llama3", "mistral"]
        assert json.loads(row["scores"]) == {"llama3": 5, "mistral": 8}


async def test_export_evals_since_filter() -> None:
    repo = get_repo()
    eval_id = repo.insert_partial(prompt="second", models=["mistral"])
    since = repo.get(eval_id)
    assert since is not None

    result = await knowledge.export_evals({"format": "jsonl", "since": since["created_at"]})

    export_path = Path(result["path"])
    exported_ids = [
        json.loads(line)["id"] for line in export_path.read_text(encoding="utf-8").splitlines()
    ]
    assert exported_ids == [eval_id]


async def test_export_evals_empty_db_outputs_files() -> None:
    jsonl_result = await knowledge.export_evals({"format": "jsonl"})
    csv_result = await knowledge.export_evals({"format": "csv"})

    jsonl_path = Path(jsonl_result["path"])
    csv_path = Path(csv_result["path"])

    assert jsonl_result["count"] == 0
    assert csv_result["count"] == 0
    assert jsonl_path.read_text(encoding="utf-8") == ""
    csv_rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(csv_rows) == 1
    assert "id" in csv_rows[0]


async def test_export_evals_writes_under_exports_with_0600_permissions() -> None:
    get_repo().insert_partial(prompt="for-export", models=["llama3"])

    result = await knowledge.export_evals({"format": "jsonl"})

    export_path = Path(result["path"])
    assert export_path.parent.name == "exports"
    export_path.relative_to(paths.get_data_dir())
    assert stat.S_IMODE(os.stat(export_path).st_mode) == 0o600


async def test_get_model_insights_empty_db() -> None:
    result = await knowledge.get_model_insights({})

    assert result == {"insights": []}


async def test_get_model_insights_best_at_code() -> None:
    repo = get_repo()
    for i in range(3):
        repo.insert_complete(
            prompt=f"code-{i}",
            models=["m1", "m2"],
            winner="m1",
            scores={"m1": 9, "m2": 7},
            criteria=["quality"],
            task_type="code",
        )

    result = await knowledge.get_model_insights({})

    m1 = next(insight for insight in result["insights"] if insight["model"] == "m1")
    assert m1["best_at"] == ["code"]


async def test_get_model_insights_threshold_filtering() -> None:
    repo = get_repo()
    repo.insert_complete(
        prompt="a",
        models=["m1", "m2"],
        winner="m1",
        scores={"m1": 9, "m2": 7},
        criteria=["quality"],
        task_type="code",
    )
    repo.insert_complete(
        prompt="b",
        models=["m1", "m2"],
        winner="m1",
        scores={"m1": 8, "m2": 6},
        criteria=["quality"],
        task_type="code",
    )
    repo.insert_complete(
        prompt="c",
        models=["m1", "m2"],
        winner="m2",
        scores={"m1": 6, "m2": 8},
        criteria=["quality"],
        task_type="code",
    )

    result = await knowledge.get_model_insights({"min_evals": 3, "win_rate_threshold": 0.8})

    m1 = next(insight for insight in result["insights"] if insight["model"] == "m1")
    assert m1["best_at"] == []


async def test_get_model_insights_custom_thresholds() -> None:
    repo = get_repo()
    repo.insert_complete(
        prompt="a",
        models=["m1", "m2"],
        winner="m1",
        scores={"m1": 9, "m2": 7},
        criteria=["quality"],
        task_type="code",
    )
    repo.insert_complete(
        prompt="b",
        models=["m1", "m2"],
        winner="m2",
        scores={"m1": 7, "m2": 8},
        criteria=["quality"],
        task_type="code",
    )

    result = await knowledge.get_model_insights({"min_evals": 1, "win_rate_threshold": 0.5})

    m1 = next(insight for insight in result["insights"] if insight["model"] == "m1")
    assert "code" in m1["best_at"]


async def test_classify_prompt_code() -> None:
    result = await knowledge.classify_prompt({"prompt": "Refactor this function for clarity"})

    assert result["task_type"] == "code"
    assert result["confidence"] > 0


async def test_classify_prompt_translation() -> None:
    result = await knowledge.classify_prompt({"prompt": "Translate this to Spanish"})

    assert result["task_type"] == "translation"


async def test_classify_prompt_no_match_defaults_to_general() -> None:
    result = await knowledge.classify_prompt({"prompt": "abc xyz"})

    assert result == {"task_type": "general", "confidence": 0.5, "alternatives": []}


async def test_classify_prompt_alternatives_sorted() -> None:
    result = await knowledge.classify_prompt(
        {"prompt": "Explain and summarize this function and translate it to Spanish"}
    )

    alternatives = result["alternatives"]
    assert len(alternatives) >= 1
    confidences = [alt["confidence"] for alt in alternatives]
    assert confidences == sorted(confidences, reverse=True)


def test_knowledge_tools_register_on_import() -> None:
    assert knowledge is not None

    registry = get_registry()
    assert "log_eval" in registry
    assert "delete_eval" in registry
    assert "export_evals" in registry
    assert "get_model_insights" in registry
    assert "classify_prompt" in registry
