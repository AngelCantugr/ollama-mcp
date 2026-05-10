"""Tests for storage repository helpers."""

from __future__ import annotations

import hashlib

import pytest

from ollama_mcp.storage import get_repo
from ollama_mcp.storage.evals_repo import TASK_TYPE_ORDER, EvalsRepo


@pytest.fixture
def repo() -> EvalsRepo:
    return get_repo()


def test_insert_partial_happy_path(repo: EvalsRepo) -> None:
    eval_id = repo.insert_partial(
        prompt="Explain recursion",
        models=["llama3", "mistral"],
        task_type="explanation",
        tags=["teaching"],
    )

    row = repo.get(eval_id)
    assert row is not None
    assert row["id"] == eval_id
    assert row["winner"] is None
    assert row["scores"] is None
    assert row["criteria"] is None
    assert row["judge_model"] is None
    assert row["notes"] is None


def test_insert_partial_rejects_invalid_task_type(repo: EvalsRepo) -> None:
    with pytest.raises(ValueError, match="Invalid task_type"):
        repo.insert_partial(prompt="hi", models=["llama3"], task_type="debugging")


@pytest.mark.parametrize("task_type", TASK_TYPE_ORDER)
def test_insert_partial_accepts_all_valid_task_types(repo: EvalsRepo, task_type: str) -> None:
    eval_id = repo.insert_partial(
        prompt=f"prompt-{task_type}", models=["llama3"], task_type=task_type
    )
    row = repo.get(eval_id)
    assert row is not None
    assert row["task_type"] == task_type


def test_insert_partial_computes_prompt_hash(repo: EvalsRepo) -> None:
    prompt = "hash me"
    eval_id = repo.insert_partial(prompt=prompt, models=["llama3"])

    row = repo.get(eval_id)
    assert row is not None
    assert row["prompt_hash"] == hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def test_insert_complete_happy_path(repo: EvalsRepo) -> None:
    eval_id = repo.insert_complete(
        prompt="Explain recursion",
        models=["llama3", "mistral"],
        winner="mistral",
        scores={"llama3": 7, "mistral": 9},
        criteria=["clarity"],
        judge_model="llama3",
        task_type="explanation",
        tags=["recursion"],
        notes="good output",
    )

    row = repo.get(eval_id)
    assert row is not None
    assert row["winner"] == "mistral"
    assert row["scores"] == {"llama3": 7, "mistral": 9}
    assert row["criteria"] == ["clarity"]
    assert row["judge_model"] == "llama3"
    assert row["task_type"] == "explanation"
    assert row["tags"] == ["recursion"]
    assert row["notes"] == "good output"


def test_insert_complete_rejects_invalid_task_type(repo: EvalsRepo) -> None:
    with pytest.raises(ValueError, match="Invalid task_type"):
        repo.insert_complete(
            prompt="Explain recursion",
            models=["llama3", "mistral"],
            winner="mistral",
            scores={"llama3": 7, "mistral": 9},
            criteria=["clarity"],
            task_type="debugging",
        )


def test_update_scores_happy_path(repo: EvalsRepo) -> None:
    eval_id = repo.insert_partial(prompt="Explain recursion", models=["llama3", "mistral"])

    repo.update_scores(
        eval_id=eval_id,
        winner="mistral",
        scores={"llama3": 7, "mistral": 9},
        criteria=["clarity"],
        judge_model=None,
        notes="updated",
    )

    row = repo.get(eval_id)
    assert row is not None
    assert row["winner"] == "mistral"
    assert row["scores"] == {"llama3": 7, "mistral": 9}
    assert row["criteria"] == ["clarity"]
    assert row["judge_model"] is None
    assert row["notes"] == "updated"


def test_update_scores_raises_for_missing_eval(repo: EvalsRepo) -> None:
    with pytest.raises(KeyError, match="missing-id"):
        repo.update_scores(
            eval_id="missing-id",
            winner="mistral",
            scores={"mistral": 9},
            criteria=["clarity"],
            judge_model=None,
        )


def test_get_returns_decoded_row_and_none_for_missing(repo: EvalsRepo) -> None:
    eval_id = repo.insert_complete(
        prompt="Explain recursion",
        models=["llama3", "mistral"],
        winner="mistral",
        scores={"llama3": {"score": 7}, "mistral": {"score": 9}},
        criteria=["clarity"],
        task_type="explanation",
        tags=["recursion"],
    )

    row = repo.get(eval_id)
    assert row is not None
    assert row["models"] == ["llama3", "mistral"]
    assert row["tags"] == ["recursion"]
    assert row["criteria"] == ["clarity"]
    assert row["scores"] == {"llama3": {"score": 7}, "mistral": {"score": 9}}

    assert repo.get("does-not-exist") is None


def test_list_by_task_type_filters_rows(repo: EvalsRepo) -> None:
    first_id = repo.insert_partial(prompt="Explain", models=["llama3"], task_type="explanation")
    repo.insert_partial(prompt="Summarize", models=["mistral"], task_type="summary")

    rows = repo.list_by_task_type("explanation")
    assert [row["id"] for row in rows] == [first_id]


def test_list_since_with_and_without_filter(repo: EvalsRepo) -> None:
    eval_id = repo.insert_partial(prompt="one", models=["llama3"])
    row = repo.get(eval_id)
    assert row is not None

    all_rows = repo.list_since()
    assert len(all_rows) == 1

    filtered_rows = repo.list_since(row["created_at"])
    assert len(filtered_rows) == 1
    assert filtered_rows[0]["id"] == eval_id

    assert repo.list_since("9999-01-01T00:00:00Z") == []


def test_get_insights_happy_path(repo: EvalsRepo) -> None:
    repo.insert_complete(
        prompt="p1",
        models=["model-a", "model-b"],
        winner="model-a",
        scores={"model-a": 9, "model-b": 7},
        criteria=["quality"],
        task_type="code",
    )
    repo.insert_complete(
        prompt="p2",
        models=["model-a", "model-b"],
        winner="model-a",
        scores={"model-a": 8, "model-b": 6},
        criteria=["quality"],
        task_type="code",
    )
    repo.insert_complete(
        prompt="p3",
        models=["model-a", "model-b"],
        winner="model-b",
        scores={"model-a": 6, "model-b": 8},
        criteria=["quality"],
        task_type="summary",
    )

    insights = repo.get_insights(min_evals=1, win_rate_threshold=0.5)

    assert insights[0]["model"] == "model-a"
    assert insights[0]["win_rate"] == pytest.approx(2 / 3)
    assert "code" in insights[0]["best_at"]
    assert insights[0]["total_evals"] == 3
    assert insights[0]["median_duration_ms"] is None


def test_get_insights_applies_min_evals_and_threshold(repo: EvalsRepo) -> None:
    repo.insert_complete(
        prompt="p1",
        models=["model-a", "model-b"],
        winner="model-a",
        scores={"model-a": 9, "model-b": 7},
        criteria=["quality"],
        task_type="code",
    )
    repo.insert_complete(
        prompt="p2",
        models=["model-a", "model-b"],
        winner="model-a",
        scores={"model-a": 8, "model-b": 6},
        criteria=["quality"],
        task_type="code",
    )
    repo.insert_complete(
        prompt="p3",
        models=["model-a", "model-b"],
        winner="model-b",
        scores={"model-a": 6, "model-b": 8},
        criteria=["quality"],
        task_type="code",
    )

    insights = repo.get_insights(min_evals=2, win_rate_threshold=0.8)
    model_a = next(insight for insight in insights if insight["model"] == "model-a")

    assert model_a["best_at"] == []


def test_get_insights_includes_model_with_no_wins(repo: EvalsRepo) -> None:
    repo.insert_complete(
        prompt="p1",
        models=["model-a", "model-b"],
        winner="model-a",
        scores={"model-a": 9, "model-b": 7},
        criteria=["quality"],
        task_type="code",
    )

    insights = repo.get_insights(min_evals=1, win_rate_threshold=0.5)
    model_b = next(insight for insight in insights if insight["model"] == "model-b")

    assert model_b["win_rate"] == 0.0
    assert model_b["best_at"] == []


def test_delete_returns_true_and_false(repo: EvalsRepo) -> None:
    eval_id = repo.insert_partial(prompt="delete-me", models=["llama3"])

    assert repo.delete(eval_id) is True
    assert repo.delete(eval_id) is False


def test_insert_routing_history_happy_path(repo: EvalsRepo) -> None:
    history_id = repo.insert_routing_history(
        task="summary",
        old_model="mistral",
        new_model="phi4",
        reason="faster",
    )

    rows = repo.list_routing_history()
    assert len(rows) == 1
    assert rows[0]["id"] == history_id
    assert rows[0]["task"] == "summary"
    assert rows[0]["old_model"] == "mistral"
    assert rows[0]["new_model"] == "phi4"
    assert rows[0]["reason"] == "faster"


def test_list_routing_history_returns_desc_order(repo: EvalsRepo) -> None:
    repo.insert_routing_history(task="summary", old_model="m1", new_model="m2", reason="r1")
    repo.insert_routing_history(task="summary", old_model="m2", new_model="m3", reason="r2")

    rows = repo.list_routing_history()
    changed_ats = [row["changed_at"] for row in rows]
    assert changed_ats == sorted(changed_ats, reverse=True)
