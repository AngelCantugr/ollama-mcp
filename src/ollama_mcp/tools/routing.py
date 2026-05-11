"""Routing tool handlers."""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import time
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]

from ollama_mcp import client, paths
from ollama_mcp.envelope import wrap_untrusted
from ollama_mcp.errors import ErrorCode, make_error
from ollama_mcp.logging import log_tool_call
from ollama_mcp.storage import get_repo
from ollama_mcp.storage.evals_repo import TASK_TYPE_ORDER, EvalRow
from ollama_mcp.tools import register_tool

_DEFAULT_TIMEOUT_MS = 120_000
_DEFAULT_ROUTING_CONFIG = "routing.json"
_ROUTING_CONFIG_ENV = "ROUTING_CONFIG"
_TASK_TYPES = frozenset(TASK_TYPE_ORDER)

_CACHED_CONFIG: dict[str, Any] | None = None
_CACHED_SCHEMA: dict[str, Any] | None = None


@register_tool(
    name="route",
    description=(
        "Auto-pick a model for a prompt based on routing.json, then run it. Falls back "
        "to the configured default model on rule miss and surfaces matched_rule='fallback'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "task_type": {
                "type": "string",
                "enum": [
                    "code",
                    "summary",
                    "explanation",
                    "reasoning",
                    "extraction",
                    "translation",
                    "general",
                    "other",
                ],
            },
            "tier": {
                "type": "string",
                "description": "Optional tier override (e.g. 'fast'). Wins over task_type if set.",
            },
            "timeout_ms": {"type": "integer", "minimum": 1},
        },
        "required": ["prompt"],
    },
)
async def route(arguments: dict[str, Any]) -> dict[str, Any]:
    start = time.monotonic()

    prompt = arguments.get("prompt")
    task_type = arguments.get("task_type")
    tier = arguments.get("tier")
    timeout_arg = arguments.get("timeout_ms")

    if not isinstance(prompt, str) or not prompt:
        return _error_result(
            tool="route",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message="Field 'prompt' must be a non-empty string.",
            model=None,
            include_duration=True,
        )
    if task_type is not None and (not isinstance(task_type, str) or task_type not in _TASK_TYPES):
        return _error_result(
            tool="route",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message=(
                "Field 'task_type' must be one of: code, summary, explanation, reasoning, "
                "extraction, translation, general, other."
            ),
            model=None,
            include_duration=True,
        )
    if tier is not None and (not isinstance(tier, str) or not tier):
        return _error_result(
            tool="route",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message="Field 'tier' must be a non-empty string when provided.",
            model=None,
            include_duration=True,
        )
    if timeout_arg is not None and (not isinstance(timeout_arg, int) or timeout_arg < 1):
        return _error_result(
            tool="route",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message="Field 'timeout_ms' must be an integer >= 1 when provided.",
            model=None,
            include_duration=True,
        )

    timeout_ms = timeout_arg if isinstance(timeout_arg, int) else _resolve_timeout_ms()

    try:
        config = _load_routing_config()
    except ValueError as exc:
        return _error_result(
            tool="route",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message=str(exc),
            model=None,
            include_duration=True,
        )
    except (paths.PathError, OSError) as exc:
        return _error_result(
            tool="route",
            start=start,
            code=ErrorCode.DB_ERROR,
            message=str(exc),
            model=None,
            include_duration=True,
        )

    rules = _as_string_map(config.get("rules"))
    tier_overrides = _as_string_map(config.get("tier_overrides"))
    default_model = str(config["default"])

    resolved_model = default_model
    matched_rule = "fallback"
    if isinstance(tier, str) and tier in tier_overrides:
        resolved_model = tier_overrides[tier]
        matched_rule = tier
    elif isinstance(task_type, str) and task_type in rules:
        resolved_model = rules[task_type]
        matched_rule = task_type

    initial = await client.generate(model=resolved_model, prompt=prompt, timeout_ms=timeout_ms)
    initial_error = _extract_error(initial)
    if initial_error is not None:
        initial_code = initial_error["code"]
        initial_message = initial_error["message"]

        should_fallback = (
            initial_code == ErrorCode.MODEL_NOT_FOUND.value and resolved_model != default_model
        )
        if should_fallback:
            fallback = await client.generate(
                model=default_model, prompt=prompt, timeout_ms=timeout_ms
            )
            fallback_error = _extract_error(fallback)
            if fallback_error is not None:
                mapped = _error_code_from_value(fallback_error["code"])
                return _error_result(
                    tool="route",
                    start=start,
                    code=mapped,
                    message=fallback_error["message"],
                    model=default_model,
                    include_duration=True,
                )

            fallback_response = fallback.get("response")
            if not isinstance(fallback_response, str):
                return _error_result(
                    tool="route",
                    start=start,
                    code=ErrorCode.OLLAMA_UNREACHABLE,
                    message="Malformed Ollama response payload",
                    model=default_model,
                    include_duration=True,
                )

            duration_ms = _duration_ms(start)
            log_tool_call(
                tool="route",
                duration_ms=duration_ms,
                model=default_model,
                status="ok",
                error_code=None,
                eval_id=None,
            )
            return {
                "model": default_model,
                "matched_rule": "fallback",
                "response": wrap_untrusted(default_model, fallback_response),
                "duration_ms": duration_ms,
                "status": "ok",
            }

        mapped = _error_code_from_value(initial_code)
        return _error_result(
            tool="route",
            start=start,
            code=mapped,
            message=initial_message,
            model=resolved_model,
            include_duration=True,
        )

    raw_response = initial.get("response")
    if not isinstance(raw_response, str):
        return _error_result(
            tool="route",
            start=start,
            code=ErrorCode.OLLAMA_UNREACHABLE,
            message="Malformed Ollama response payload",
            model=resolved_model,
            include_duration=True,
        )

    duration_ms = _duration_ms(start)
    log_tool_call(
        tool="route",
        duration_ms=duration_ms,
        model=resolved_model,
        status="ok",
        error_code=None,
        eval_id=None,
    )
    return {
        "model": resolved_model,
        "matched_rule": matched_rule,
        "response": wrap_untrusted(resolved_model, raw_response),
        "duration_ms": duration_ms,
        "status": "ok",
    }


@register_tool(
    name="get_routing_config",
    description="Return the active routing configuration from routing.json.",
    input_schema={"type": "object", "properties": {}, "required": []},
)
async def get_routing_config(arguments: dict[str, Any]) -> dict[str, Any]:
    start = time.monotonic()
    if not isinstance(arguments, dict):
        return _error_result(
            tool="get_routing_config",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message="arguments must be an object",
            model=None,
        )

    try:
        config = copy.deepcopy(_load_routing_config())
    except ValueError as exc:
        return _error_result(
            tool="get_routing_config",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message=str(exc),
            model=None,
        )
    except (paths.PathError, OSError) as exc:
        return _error_result(
            tool="get_routing_config",
            start=start,
            code=ErrorCode.DB_ERROR,
            message=str(exc),
            model=None,
        )

    log_tool_call(
        tool="get_routing_config",
        duration_ms=_duration_ms(start),
        model=None,
        status="ok",
        error_code=None,
        eval_id=None,
    )
    return config


@register_tool(
    name="update_routing_rule",
    description=(
        "Update a single routing rule. Writes the change to routing.json (atomic) and "
        "records the transition in routing_history."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "model": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["task", "model"],
    },
)
async def update_routing_rule(arguments: dict[str, Any]) -> dict[str, Any]:
    start = time.monotonic()

    task = arguments.get("task")
    model = arguments.get("model")
    reason = arguments.get("reason")

    if not isinstance(task, str) or not task:
        return _error_result(
            tool="update_routing_rule",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message="Field 'task' must be a non-empty string.",
            model=None,
        )
    if not isinstance(model, str) or not model:
        return _error_result(
            tool="update_routing_rule",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message="Field 'model' must be a non-empty string.",
            model=None,
        )
    if reason is not None and not isinstance(reason, str):
        return _error_result(
            tool="update_routing_rule",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message="Field 'reason' must be a string when provided.",
            model=None,
        )

    try:
        config = _load_routing_config()
    except ValueError as exc:
        return _error_result(
            tool="update_routing_rule",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message=str(exc),
            model=None,
        )
    except (paths.PathError, OSError) as exc:
        return _error_result(
            tool="update_routing_rule",
            start=start,
            code=ErrorCode.DB_ERROR,
            message=str(exc),
            model=None,
        )

    rules = _as_string_map(config.get("rules"))
    tier_overrides = _as_string_map(config.get("tier_overrides"))

    section: str
    previous_model: str | None
    if task in _TASK_TYPES:
        section = "rules"
        previous_model = rules.get(task)
    elif task in tier_overrides:
        section = "tier_overrides"
        previous_model = tier_overrides.get(task)
    else:
        return _error_result(
            tool="update_routing_rule",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message=(
                "Field 'task' must be one of the known task types or an existing tier override key."
            ),
            model=None,
        )

    next_config = copy.deepcopy(config)
    next_section = next_config.get(section)
    if not isinstance(next_section, dict):
        return _error_result(
            tool="update_routing_rule",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message=f"Routing config section {section!r} must be an object.",
            model=None,
        )
    next_section[task] = model

    try:
        _validate_routing_config(next_config)
    except ValueError as exc:
        return _error_result(
            tool="update_routing_rule",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message=str(exc),
            model=None,
        )

    try:
        _atomic_write_routing_config(next_config)
        repo = get_repo()
        repo.insert_routing_history(
            task=task, old_model=previous_model, new_model=model, reason=reason
        )
    except sqlite3.DatabaseError as exc:
        return _error_result(
            tool="update_routing_rule",
            start=start,
            code=ErrorCode.DB_ERROR,
            message=str(exc),
            model=None,
        )
    except (paths.PathError, OSError) as exc:
        return _error_result(
            tool="update_routing_rule",
            start=start,
            code=ErrorCode.DB_ERROR,
            message=str(exc),
            model=None,
        )

    _invalidate_cache()
    log_tool_call(
        tool="update_routing_rule",
        duration_ms=_duration_ms(start),
        model=model,
        status="ok",
        error_code=None,
        eval_id=None,
    )
    return {"updated": True, "previous_model": previous_model, "new_model": model}


@register_tool(
    name="suggest_routing_updates",
    description=(
        "Propose routing.json diffs based on get_model_insights. Does NOT apply them — "
        "call update_routing_rule to accept a suggestion."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "min_evals": {"type": "integer", "minimum": 1, "default": 3},
            "win_rate_threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.6,
            },
        },
        "required": [],
    },
)
async def suggest_routing_updates(arguments: dict[str, Any]) -> dict[str, Any]:
    start = time.monotonic()

    min_evals = arguments.get("min_evals", 3)
    win_rate_threshold = arguments.get("win_rate_threshold", 0.6)

    if not isinstance(min_evals, int) or min_evals < 1:
        return _error_result(
            tool="suggest_routing_updates",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message="Field 'min_evals' must be an integer >= 1.",
            model=None,
        )
    if not isinstance(win_rate_threshold, (int, float)):
        return _error_result(
            tool="suggest_routing_updates",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message="Field 'win_rate_threshold' must be a number between 0.0 and 1.0.",
            model=None,
        )
    threshold = float(win_rate_threshold)
    if not 0 <= threshold <= 1:
        return _error_result(
            tool="suggest_routing_updates",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message="Field 'win_rate_threshold' must be a number between 0.0 and 1.0.",
            model=None,
        )

    try:
        config = _load_routing_config()
        repo = get_repo()
        insights = repo.get_insights(min_evals=min_evals, win_rate_threshold=threshold)
    except ValueError as exc:
        return _error_result(
            tool="suggest_routing_updates",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message=str(exc),
            model=None,
        )
    except sqlite3.DatabaseError as exc:
        return _error_result(
            tool="suggest_routing_updates",
            start=start,
            code=ErrorCode.DB_ERROR,
            message=str(exc),
            model=None,
        )
    except (paths.PathError, OSError) as exc:
        return _error_result(
            tool="suggest_routing_updates",
            start=start,
            code=ErrorCode.DB_ERROR,
            message=str(exc),
            model=None,
        )

    rules = _as_string_map(config.get("rules"))
    default_model = str(config["default"])

    suggestions: list[dict[str, Any]] = []
    for task in TASK_TYPE_ORDER:
        current_model = rules.get(task, default_model)
        rows = repo.list_by_task_type(task)

        candidate: dict[str, Any] | None = None
        for insight in insights:
            model_name = insight.get("model")
            best_at = insight.get("best_at")
            if not isinstance(model_name, str) or not isinstance(best_at, list):
                continue
            if task not in best_at:
                continue
            if model_name == current_model:
                continue

            proposed_win_rate, total_task_evals = _task_win_rate(rows, model_name)
            if total_task_evals == 0:
                continue

            candidate_row = {
                "model": model_name,
                "win_rate": proposed_win_rate,
                "total_task_evals": total_task_evals,
            }
            if candidate is None or _is_better_candidate(candidate_row, candidate):
                candidate = candidate_row

        if candidate is None:
            continue

        current_win_rate, _ = _task_win_rate(rows, current_model)
        supporting_eval_ids = [
            row["id"]
            for row in rows
            if isinstance(row.get("winner"), str) and row["winner"] == candidate["model"]
        ][:5]

        rationale = (
            f"{candidate['model']} has {candidate['win_rate']:.2f} win rate vs "
            f"{current_model} {current_win_rate:.2f} on {task} task_type over last "
            f"{candidate['total_task_evals']} evals"
        )
        suggestions.append(
            {
                "task": task,
                "current_model": current_model,
                "proposed_model": candidate["model"],
                "rationale": rationale,
                "supporting_eval_ids": supporting_eval_ids,
            }
        )

    log_tool_call(
        tool="suggest_routing_updates",
        duration_ms=_duration_ms(start),
        model=None,
        status="ok",
        error_code=None,
        eval_id=None,
    )
    return {"suggestions": suggestions}


@register_tool(
    name="reset_routing",
    description="Revert routing.json to bundled defaults. Requires confirm=True.",
    input_schema={
        "type": "object",
        "properties": {"confirm": {"type": "boolean"}},
        "required": ["confirm"],
    },
)
async def reset_routing(arguments: dict[str, Any]) -> dict[str, Any]:
    start = time.monotonic()

    confirm = arguments.get("confirm")
    if confirm is not True:
        return _error_result(
            tool="reset_routing",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message="Field 'confirm' must be true.",
            model=None,
        )

    try:
        previous = copy.deepcopy(_load_routing_config())
        defaults = _load_bundled_default_config()
        _atomic_write_routing_config(defaults)
    except ValueError as exc:
        return _error_result(
            tool="reset_routing",
            start=start,
            code=ErrorCode.INVALID_INPUT,
            message=str(exc),
            model=None,
        )
    except (paths.PathError, OSError) as exc:
        return _error_result(
            tool="reset_routing",
            start=start,
            code=ErrorCode.DB_ERROR,
            message=str(exc),
            model=None,
        )

    _invalidate_cache()
    log_tool_call(
        tool="reset_routing",
        duration_ms=_duration_ms(start),
        model=None,
        status="ok",
        error_code=None,
        eval_id=None,
    )
    return {"reset": True, "previous_rules": previous}


def _load_routing_config() -> dict[str, Any]:
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None:
        return _CACHED_CONFIG

    config_path = _routing_config_path()
    if not config_path.exists():
        _bootstrap_user_routing_config(config_path)

    raw_text = config_path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid routing JSON: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Invalid routing config: top-level JSON value must be an object.")

    _validate_routing_config(parsed)
    _CACHED_CONFIG = parsed
    return parsed


def _load_schema() -> dict[str, Any]:
    global _CACHED_SCHEMA
    if _CACHED_SCHEMA is not None:
        return _CACHED_SCHEMA

    schema_text = (
        resources.files("ollama_mcp.config")
        .joinpath("routing.schema.json")
        .read_text(encoding="utf-8")
    )
    parsed = json.loads(schema_text)
    if not isinstance(parsed, dict):
        raise ValueError("Bundled routing schema must be a JSON object.")

    _CACHED_SCHEMA = parsed
    return parsed


def _load_bundled_default_config() -> dict[str, Any]:
    bundled = (
        resources.files("ollama_mcp.config").joinpath("routing.json").read_text(encoding="utf-8")
    )
    parsed = json.loads(bundled)
    if not isinstance(parsed, dict):
        raise ValueError("Bundled routing config must be a JSON object.")

    _validate_routing_config(parsed)
    return parsed


def _validate_routing_config(config: dict[str, Any]) -> None:
    schema = _load_schema()
    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.exceptions.ValidationError as exc:
        raise ValueError(f"Invalid routing config: {exc.message}") from exc
    except jsonschema.exceptions.SchemaError as exc:
        raise ValueError(f"Invalid routing schema: {exc.message}") from exc


def _bootstrap_user_routing_config(config_path: Path) -> None:
    config_name = _routing_config_name()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    paths.create_data_file(config_name)
    bundled_defaults = _load_bundled_default_config()
    config_path.write_text(json.dumps(bundled_defaults, indent=2) + "\n", encoding="utf-8")


def _atomic_write_routing_config(config: dict[str, Any]) -> None:
    _validate_routing_config(config)

    config_path = _routing_config_path()
    config_name = _routing_config_name()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    temp_name = f".{Path(config_name).name}.{time.time_ns()}.tmp"
    temp_path = paths.create_data_file(temp_name)

    try:
        temp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_path, config_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _routing_config_name() -> str:
    raw = os.environ.get(_ROUTING_CONFIG_ENV, _DEFAULT_ROUTING_CONFIG)
    return raw if raw else _DEFAULT_ROUTING_CONFIG


def _routing_config_path() -> Path:
    return paths.resolve_data_path(_routing_config_name())


def _as_string_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, str):
            result[key] = item
    return result


def _task_win_rate(rows: list[EvalRow], model: str) -> tuple[float, int]:
    total = 0
    wins = 0
    for row in rows:
        models = row.get("models")
        if not isinstance(models, list) or model not in models:
            continue

        total += 1
        winner = row.get("winner")
        if isinstance(winner, str) and winner == model:
            wins += 1

    if total == 0:
        return 0.0, 0
    return wins / total, total


def _is_better_candidate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_rate = float(left["win_rate"])
    right_rate = float(right["win_rate"])
    if left_rate != right_rate:
        return left_rate > right_rate

    left_total = int(left["total_task_evals"])
    right_total = int(right["total_task_evals"])
    if left_total != right_total:
        return left_total > right_total

    return str(left["model"]) < str(right["model"])


def _resolve_timeout_ms() -> int:
    raw = os.environ.get("OLLAMA_TIMEOUT_MS", str(_DEFAULT_TIMEOUT_MS))
    try:
        timeout_ms = int(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT_MS
    if timeout_ms < 1:
        return _DEFAULT_TIMEOUT_MS
    return timeout_ms


def _extract_error(payload: dict[str, Any]) -> dict[str, str] | None:
    raw = payload.get("error")
    if not isinstance(raw, dict):
        return None

    code = raw.get("code")
    message = raw.get("message")
    if not isinstance(code, str) or not isinstance(message, str):
        return {
            "code": ErrorCode.OLLAMA_UNREACHABLE.value,
            "message": "Unexpected error format from Ollama client",
        }

    return {"code": code, "message": message}


def _error_code_from_value(value: str) -> ErrorCode:
    for code in ErrorCode:
        if code.value == value:
            return code
    return ErrorCode.OLLAMA_UNREACHABLE


def _error_result(
    *,
    tool: str,
    start: float,
    code: ErrorCode,
    message: str,
    model: str | None,
    include_duration: bool = False,
) -> dict[str, Any]:
    duration_ms = _duration_ms(start)
    log_tool_call(
        tool=tool,
        duration_ms=duration_ms,
        model=model,
        status="error",
        error_code=code.value,
        eval_id=None,
    )

    payload: dict[str, Any] = {"error": make_error(code, message)["error"]}
    if include_duration:
        payload["duration_ms"] = duration_ms
    return payload


def _duration_ms(start: float) -> int:
    return max(1, int(round((time.monotonic() - start) * 1000)))


def _invalidate_cache() -> None:
    global _CACHED_CONFIG
    _CACHED_CONFIG = None
