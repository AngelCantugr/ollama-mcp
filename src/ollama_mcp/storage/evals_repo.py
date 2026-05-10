"""Repository for eval and routing-history persistence.

This is the only module in the runtime codebase that issues SQL queries.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from typing import TypedDict, cast

import ulid

TASK_TYPES = {
    "code",
    "summary",
    "explanation",
    "reasoning",
    "extraction",
    "translation",
    "general",
    "other",
}

TASK_TYPE_ORDER = [
    "code",
    "summary",
    "explanation",
    "reasoning",
    "extraction",
    "translation",
    "general",
    "other",
]


class EvalRow(TypedDict):
    id: str
    schema_version: int
    created_at: str
    prompt: str
    prompt_hash: str
    models: list[str]
    task_type: str | None
    tags: list[str] | None
    winner: str | None
    criteria: list[str] | None
    scores: dict[str, object] | None
    judge_model: str | None
    notes: str | None


class ModelInsight(TypedDict):
    model: str
    win_rate: float
    best_at: list[str]
    total_evals: int
    median_duration_ms: int | None


class RoutingHistoryRow(TypedDict):
    id: str
    changed_at: str
    task: str
    old_model: str | None
    new_model: str
    reason: str | None


class EvalsRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert_partial(
        self,
        *,
        prompt: str,
        models: list[str],
        task_type: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        self._validate_task_type(task_type)

        eval_id = ulid.new().str
        schema_version = self._current_schema_version()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO evals (
                    id,
                    schema_version,
                    prompt,
                    prompt_hash,
                    models,
                    task_type,
                    tags,
                    winner,
                    criteria,
                    scores,
                    judge_model,
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
                """,
                (
                    eval_id,
                    schema_version,
                    prompt,
                    hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    json.dumps(models),
                    task_type,
                    json.dumps(tags) if tags is not None else None,
                ),
            )
        return eval_id

    def update_scores(
        self,
        *,
        eval_id: str,
        winner: str,
        scores: dict[str, object],
        criteria: list[str],
        judge_model: str | None,
        notes: str | None = None,
    ) -> None:
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE evals
                SET winner = ?,
                    scores = ?,
                    criteria = ?,
                    judge_model = ?,
                    notes = ?
                WHERE id = ?
                """,
                (
                    winner,
                    json.dumps(scores),
                    json.dumps(criteria),
                    judge_model,
                    notes,
                    eval_id,
                ),
            )

        if cursor.rowcount == 0:
            raise KeyError(eval_id)

    def insert_complete(
        self,
        *,
        prompt: str,
        models: list[str],
        winner: str,
        scores: dict[str, object],
        criteria: list[str],
        judge_model: str | None = None,
        task_type: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> str:
        self._validate_task_type(task_type)

        eval_id = ulid.new().str
        schema_version = self._current_schema_version()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO evals (
                    id,
                    schema_version,
                    prompt,
                    prompt_hash,
                    models,
                    task_type,
                    tags,
                    winner,
                    criteria,
                    scores,
                    judge_model,
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eval_id,
                    schema_version,
                    prompt,
                    hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    json.dumps(models),
                    task_type,
                    json.dumps(tags) if tags is not None else None,
                    winner,
                    json.dumps(criteria),
                    json.dumps(scores),
                    judge_model,
                    notes,
                ),
            )
        return eval_id

    def get(self, eval_id: str) -> EvalRow | None:
        row = self._conn.execute("SELECT * FROM evals WHERE id = ?", (eval_id,)).fetchone()
        if row is None:
            return None
        return self._to_eval_row(row)

    def list_by_task_type(self, task_type: str) -> list[EvalRow]:
        self._validate_task_type(task_type)
        rows = self._conn.execute(
            "SELECT * FROM evals WHERE task_type = ? ORDER BY created_at DESC",
            (task_type,),
        ).fetchall()
        return [self._to_eval_row(row) for row in rows]

    def list_since(self, since_iso: str | None = None) -> list[EvalRow]:
        if since_iso is None:
            rows = self._conn.execute("SELECT * FROM evals ORDER BY created_at DESC").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM evals WHERE created_at >= ? ORDER BY created_at DESC",
                (since_iso,),
            ).fetchall()
        return [self._to_eval_row(row) for row in rows]

    def get_insights(
        self,
        *,
        min_evals: int = 3,
        win_rate_threshold: float = 0.6,
    ) -> list[ModelInsight]:
        rows = self._conn.execute("SELECT models, winner, task_type FROM evals").fetchall()

        total_by_model: dict[str, int] = defaultdict(int)
        wins_by_model: dict[str, int] = defaultdict(int)
        total_by_model_task: dict[tuple[str, str], int] = defaultdict(int)
        wins_by_model_task: dict[tuple[str, str], int] = defaultdict(int)

        for row in rows:
            models = self._decode_required_str_list(cast(str, row["models"]))
            winner = cast(str | None, row["winner"])
            task_type = cast(str | None, row["task_type"])

            for model in models:
                total_by_model[model] += 1
                if task_type is not None:
                    total_by_model_task[(model, task_type)] += 1

            if winner is not None:
                wins_by_model[winner] += 1
                if task_type is not None:
                    wins_by_model_task[(winner, task_type)] += 1

        insights: list[ModelInsight] = []
        for model, total_evals in total_by_model.items():
            wins = wins_by_model.get(model, 0)
            win_rate = wins / total_evals if total_evals > 0 else 0.0

            best_at: list[str] = []
            for task_type in TASK_TYPE_ORDER:
                wins_for_task = wins_by_model_task.get((model, task_type), 0)
                total_for_task = total_by_model_task.get((model, task_type), 0)
                if wins_for_task < min_evals or total_for_task == 0:
                    continue

                task_win_rate = wins_for_task / total_for_task
                if task_win_rate >= win_rate_threshold:
                    best_at.append(task_type)

            insights.append(
                ModelInsight(
                    model=model,
                    win_rate=win_rate,
                    best_at=best_at,
                    total_evals=total_evals,
                    # We do not store per-model duration yet, so median is unavailable.
                    median_duration_ms=None,
                )
            )

        return sorted(insights, key=lambda insight: (-insight["win_rate"], insight["model"]))

    def delete(self, eval_id: str) -> bool:
        with self._conn:
            cursor = self._conn.execute("DELETE FROM evals WHERE id = ?", (eval_id,))
        return cursor.rowcount > 0

    def insert_routing_history(
        self,
        *,
        task: str,
        old_model: str | None,
        new_model: str,
        reason: str | None,
    ) -> str:
        history_id = ulid.new().str
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO routing_history (id, task, old_model, new_model, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (history_id, task, old_model, new_model, reason),
            )
        return history_id

    def list_routing_history(self) -> list[RoutingHistoryRow]:
        rows = self._conn.execute(
            """
            SELECT id, changed_at, task, old_model, new_model, reason
            FROM routing_history
            ORDER BY changed_at DESC, id DESC
            """
        ).fetchall()
        return [
            RoutingHistoryRow(
                id=cast(str, row["id"]),
                changed_at=cast(str, row["changed_at"]),
                task=cast(str, row["task"]),
                old_model=cast(str | None, row["old_model"]),
                new_model=cast(str, row["new_model"]),
                reason=cast(str | None, row["reason"]),
            )
            for row in rows
        ]

    def _current_schema_version(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM schema_version"
        ).fetchone()
        return int(row["version"]) if row is not None else 0

    def _validate_task_type(self, task_type: str | None) -> None:
        if task_type is not None and task_type not in TASK_TYPES:
            raise ValueError(f"Invalid task_type: {task_type}")

    def _to_eval_row(self, row: sqlite3.Row) -> EvalRow:
        return EvalRow(
            id=cast(str, row["id"]),
            schema_version=cast(int, row["schema_version"]),
            created_at=cast(str, row["created_at"]),
            prompt=cast(str, row["prompt"]),
            prompt_hash=cast(str, row["prompt_hash"]),
            models=self._decode_required_str_list(cast(str, row["models"])),
            task_type=cast(str | None, row["task_type"]),
            tags=self._decode_optional_str_list(cast(str | None, row["tags"])),
            winner=cast(str | None, row["winner"]),
            criteria=self._decode_optional_str_list(cast(str | None, row["criteria"])),
            scores=self._decode_optional_scores(cast(str | None, row["scores"])),
            judge_model=cast(str | None, row["judge_model"]),
            notes=cast(str | None, row["notes"]),
        )

    def _decode_required_str_list(self, value: str) -> list[str]:
        parsed = json.loads(value)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("Expected JSON list[str]")
        return cast(list[str], parsed)

    def _decode_optional_str_list(self, value: str | None) -> list[str] | None:
        if value is None:
            return None
        return self._decode_required_str_list(value)

    def _decode_optional_scores(self, value: str | None) -> dict[str, object] | None:
        if value is None:
            return None
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")
        return cast(dict[str, object], parsed)
