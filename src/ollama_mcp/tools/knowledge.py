"""Knowledge-base tool handlers for eval persistence and analysis."""

from __future__ import annotations

import csv
import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ollama_mcp import paths
from ollama_mcp.errors import ErrorCode, make_error
from ollama_mcp.logging import log_tool_call
from ollama_mcp.storage import get_repo
from ollama_mcp.storage.evals_repo import TASK_TYPE_ORDER, EvalRow
from ollama_mcp.tools import register_tool

_TASK_TYPE_ENUM = TASK_TYPE_ORDER
_ALTERNATIVE_LIMIT = 3

_CLASSIFICATION_RULES: dict[str, list[str]] = {
    "code": [
        "function",
        "class",
        "method",
        "bug",
        "compile",
        "syntax",
        "stack trace",
        "refactor",
        "implement",
    ],
    "summary": [
        "summarize",
        "tldr",
        "tl;dr",
        "in 3 bullets",
        "brief overview",
        "executive summary",
    ],
    "explanation": ["explain", "what is", "how does", "why does", "intuitively"],
    "reasoning": ["prove", "derive", "why", "logic", "step by step", "reasoning"],
    "extraction": ["extract", "pull out", "list all", "find every"],
    "translation": ["translate", "convert to", "in french", "in spanish"],
}

_CSV_FIELDS = [
    "id",
    "schema_version",
    "created_at",
    "prompt",
    "prompt_hash",
    "models",
    "task_type",
    "tags",
    "winner",
    "criteria",
    "scores",
    "judge_model",
    "notes",
]


@register_tool(
    name="log_eval",
    description=(
        "Manually log a fully-populated eval row. Used for hand-curated entries; "
        "score_comparison and judge_with_model are the typical write paths."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "models": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "winner": {"type": "string"},
            "criteria": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "scores": {"type": "object"},
            "task_type": {"type": "string", "enum": _TASK_TYPE_ENUM},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "notes": {"type": "string"},
            "judge_model": {
                "type": "string",
                "description": "Optional. Set if a model judged this entry.",
            },
        },
        "required": ["prompt", "models", "winner", "criteria", "scores"],
    },
)
async def log_eval(arguments: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()

    required_error = _require_fields(
        arguments, ["prompt", "models", "winner", "criteria", "scores"]
    )
    if required_error is not None:
        return _error_result("log_eval", start, ErrorCode.INVALID_INPUT, required_error)

    prompt = arguments["prompt"]
    models = arguments["models"]
    winner = arguments["winner"]
    criteria = arguments["criteria"]
    scores = arguments["scores"]
    task_type = arguments.get("task_type")
    tags = arguments.get("tags")
    notes = arguments.get("notes")
    judge_model = arguments.get("judge_model")

    validation_error = _validate_log_eval_input(
        prompt=prompt,
        models=models,
        winner=winner,
        criteria=criteria,
        scores=scores,
        task_type=task_type,
        tags=tags,
        notes=notes,
        judge_model=judge_model,
    )
    if validation_error is not None:
        return _error_result("log_eval", start, ErrorCode.INVALID_INPUT, validation_error)

    try:
        repo = get_repo()
        eval_id = repo.insert_complete(
            prompt=prompt,
            models=models,
            winner=winner,
            scores=scores,
            criteria=criteria,
            judge_model=judge_model,
            task_type=task_type,
            tags=tags,
            notes=notes,
        )
    except ValueError as exc:
        return _error_result("log_eval", start, ErrorCode.INVALID_INPUT, str(exc))
    except sqlite3.DatabaseError as exc:
        return _error_result("log_eval", start, ErrorCode.DB_ERROR, str(exc))

    log_tool_call(
        tool="log_eval",
        duration_ms=(time.perf_counter() - start) * 1000,
        model=None,
        status="ok",
        error_code=None,
        eval_id=eval_id,
    )
    return {"logged": True, "eval_id": eval_id}


@register_tool(
    name="delete_eval",
    description="Hard-delete an eval row. Used for cleaning up bad data.",
    input_schema={
        "type": "object",
        "properties": {"eval_id": {"type": "string"}},
        "required": ["eval_id"],
    },
)
async def delete_eval(arguments: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()

    eval_id = arguments.get("eval_id")
    if not isinstance(eval_id, str) or not eval_id:
        return _error_result(
            "delete_eval",
            start,
            ErrorCode.INVALID_INPUT,
            "Field 'eval_id' must be a non-empty string.",
        )

    try:
        repo = get_repo()
        deleted = repo.delete(eval_id)
    except sqlite3.DatabaseError as exc:
        return _error_result("delete_eval", start, ErrorCode.DB_ERROR, str(exc))

    log_tool_call(
        tool="delete_eval",
        duration_ms=(time.perf_counter() - start) * 1000,
        model=None,
        status="ok",
        error_code=None,
        eval_id=eval_id,
    )
    return {"deleted": deleted}


@register_tool(
    name="export_evals",
    description=(
        "Export all eval rows as JSONL or CSV. Files are written under DATA_DIR/exports/. "
        "If `since` is provided as ISO8601, only evals with created_at >= since are included."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "format": {"type": "string", "enum": ["jsonl", "csv"]},
            "since": {"type": "string", "description": "ISO8601 datetime; optional."},
        },
        "required": ["format"],
    },
)
async def export_evals(arguments: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()

    export_format = arguments.get("format")
    since = arguments.get("since")

    if export_format not in {"jsonl", "csv"}:
        return _error_result(
            "export_evals",
            start,
            ErrorCode.INVALID_INPUT,
            "Field 'format' must be one of: jsonl, csv.",
        )
    if since is not None and not isinstance(since, str):
        return _error_result(
            "export_evals",
            start,
            ErrorCode.INVALID_INPUT,
            "Field 'since' must be a string when provided.",
        )
    if isinstance(since, str) and not _looks_like_iso8601(since):
        return _error_result(
            "export_evals",
            start,
            ErrorCode.INVALID_INPUT,
            "Field 'since' must be an ISO8601 datetime string.",
        )

    try:
        repo = get_repo()
        rows = repo.list_since(since)

        timestamp = _export_timestamp()
        relative_path = f"exports/evals_{timestamp}.{export_format}"
        export_path = paths.resolve_data_path(relative_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        paths.create_data_file(relative_path)

        if export_format == "jsonl":
            _write_jsonl(export_path, rows)
        else:
            _write_csv(export_path, rows)
    except (paths.PathError, OSError) as exc:
        return _error_result("export_evals", start, ErrorCode.INVALID_INPUT, str(exc))
    except sqlite3.DatabaseError as exc:
        return _error_result("export_evals", start, ErrorCode.DB_ERROR, str(exc))

    log_tool_call(
        tool="export_evals",
        duration_ms=(time.perf_counter() - start) * 1000,
        model=None,
        status="ok",
        error_code=None,
        eval_id=None,
    )
    return {"path": str(export_path), "count": len(rows)}


@register_tool(
    name="get_model_insights",
    description=(
        "Return aggregated learnings from stored eval history: per-model win rate, "
        "best_at task types (≥3 evals, ≥0.6 win rate by default), and total evals."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "min_evals": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Minimum evals in a task_type for it to count toward best_at. Default 3."
                ),
            },
            "win_rate_threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Minimum win rate within a task_type. Default 0.6.",
            },
        },
        "required": [],
    },
)
async def get_model_insights(arguments: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()

    min_evals_arg = arguments.get("min_evals", 3)
    threshold_arg = arguments.get("win_rate_threshold", 0.6)

    if not isinstance(min_evals_arg, int) or min_evals_arg < 1:
        return _error_result(
            "get_model_insights",
            start,
            ErrorCode.INVALID_INPUT,
            "Field 'min_evals' must be an integer >= 1.",
        )
    if not isinstance(threshold_arg, (int, float)):
        return _error_result(
            "get_model_insights",
            start,
            ErrorCode.INVALID_INPUT,
            "Field 'win_rate_threshold' must be a number between 0.0 and 1.0.",
        )
    if not 0 <= float(threshold_arg) <= 1:
        return _error_result(
            "get_model_insights",
            start,
            ErrorCode.INVALID_INPUT,
            "Field 'win_rate_threshold' must be a number between 0.0 and 1.0.",
        )

    try:
        repo = get_repo()
        insights = repo.get_insights(
            min_evals=min_evals_arg,
            win_rate_threshold=float(threshold_arg),
        )
    except sqlite3.DatabaseError as exc:
        return _error_result("get_model_insights", start, ErrorCode.DB_ERROR, str(exc))

    log_tool_call(
        tool="get_model_insights",
        duration_ms=(time.perf_counter() - start) * 1000,
        model=None,
        status="ok",
        error_code=None,
        eval_id=None,
    )
    return {"insights": insights}


@register_tool(
    name="classify_prompt",
    description=(
        "Suggest a task_type for a prompt. Used for tagging evals at log time so "
        "get_model_insights can compute best_at by task_type. Returns the most likely "
        "task_type plus alternatives."
    ),
    input_schema={
        "type": "object",
        "properties": {"prompt": {"type": "string"}},
        "required": ["prompt"],
    },
)
async def classify_prompt(arguments: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()

    prompt = arguments.get("prompt")
    if not isinstance(prompt, str):
        return _error_result(
            "classify_prompt", start, ErrorCode.INVALID_INPUT, "Field 'prompt' must be a string."
        )

    scores = _score_prompt(prompt)

    if not scores:
        result = {"task_type": "general", "confidence": 0.5, "alternatives": []}
    else:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        task_type, confidence = ranked[0]
        alternatives = [
            {"task_type": candidate, "confidence": score}
            for candidate, score in ranked[1 : _ALTERNATIVE_LIMIT + 1]
        ]
        result = {
            "task_type": task_type,
            "confidence": min(1.0, confidence),
            "alternatives": alternatives,
        }

    log_tool_call(
        tool="classify_prompt",
        duration_ms=(time.perf_counter() - start) * 1000,
        model=None,
        status="ok",
        error_code=None,
        eval_id=None,
    )
    return result


def _require_fields(arguments: dict[str, Any], fields: list[str]) -> str | None:
    for field in fields:
        if field not in arguments:
            return f"Field '{field}' is required."
    return None


def _validate_log_eval_input(
    *,
    prompt: object,
    models: object,
    winner: object,
    criteria: object,
    scores: object,
    task_type: object,
    tags: object,
    notes: object,
    judge_model: object,
) -> str | None:
    if not isinstance(prompt, str):
        return "Field 'prompt' must be a string."
    if (
        not isinstance(models, list)
        or not models
        or not all(isinstance(item, str) for item in models)
    ):
        return "Field 'models' must be a non-empty list of strings."
    if not isinstance(winner, str):
        return "Field 'winner' must be a string."
    if (
        not isinstance(criteria, list)
        or not criteria
        or not all(isinstance(item, str) for item in criteria)
    ):
        return "Field 'criteria' must be a non-empty list of strings."
    if not isinstance(scores, dict):
        return "Field 'scores' must be an object."
    if task_type is not None and not isinstance(task_type, str):
        return "Field 'task_type' must be a string when provided."
    if tags is not None and (
        not isinstance(tags, list) or not tags or not all(isinstance(item, str) for item in tags)
    ):
        return "Field 'tags' must be a non-empty list of strings when provided."
    if notes is not None and not isinstance(notes, str):
        return "Field 'notes' must be a string when provided."
    if judge_model is not None and not isinstance(judge_model, str):
        return "Field 'judge_model' must be a string when provided."
    return None


def _looks_like_iso8601(raw: str) -> bool:
    candidate = raw.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return True


def _write_jsonl(path: Path, rows: list[EvalRow]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[EvalRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.get("id"),
                    "schema_version": row.get("schema_version"),
                    "created_at": row.get("created_at"),
                    "prompt": row.get("prompt"),
                    "prompt_hash": row.get("prompt_hash"),
                    "models": json.dumps(row.get("models")),
                    "task_type": row.get("task_type"),
                    "tags": json.dumps(row.get("tags")),
                    "winner": row.get("winner"),
                    "criteria": json.dumps(row.get("criteria")),
                    "scores": json.dumps(row.get("scores")),
                    "judge_model": row.get("judge_model"),
                    "notes": row.get("notes"),
                }
            )


def _score_prompt(prompt: str) -> dict[str, float]:
    lowered = prompt.lower()
    scores: dict[str, float] = {}
    for task_type, keywords in _CLASSIFICATION_RULES.items():
        hits = sum(1 for keyword in keywords if keyword in lowered)
        if hits > 0:
            scores[task_type] = hits / len(keywords)
    return scores


def _export_timestamp() -> str:
    now = datetime.now(tz=UTC)
    return now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z"


def _error_result(tool: str, start: float, code: ErrorCode, message: str) -> dict[str, Any]:
    log_tool_call(
        tool=tool,
        duration_ms=(time.perf_counter() - start) * 1000,
        model=None,
        status="error",
        error_code=code.value,
        eval_id=None,
    )
    payload = make_error(code, message)
    return {"error": payload["error"]}
