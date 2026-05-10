"""Learn mode compare tool handler."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from typing import Any

from ollama_mcp import client
from ollama_mcp.envelope import wrap_untrusted
from ollama_mcp.errors import ErrorCode, make_error
from ollama_mcp.logging import get_logger, log_tool_call
from ollama_mcp.storage import get_repo
from ollama_mcp.tools import register_tool

_DEFAULT_TIMEOUT_MS = 120_000
_DEFAULT_MAX_COMPARE_CONCURRENCY = 1
_TASK_TYPES = {
    "code",
    "summary",
    "explanation",
    "reasoning",
    "extraction",
    "translation",
    "general",
    "other",
}
_log = get_logger("ollama_mcp.tools.compare")


@register_tool(
    name="compare",
    description=(
        "Run the same prompt through multiple local Ollama models with per-model "
        "timeout and partial-success semantics. Returns an eval_id that score_comparison "
        "or judge_with_model can later reference."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "models": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": (
                    "Ordered list of local model names. Each is called with the same prompt."
                ),
            },
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
                "description": (
                    "Optional taxonomy tag stored on the eval row for later aggregation."
                ),
            },
            "timeout_ms": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Per-model timeout. Defaults to OLLAMA_TIMEOUT_MS env var or 120000."
                ),
            },
            "max_concurrency": {
                "type": "integer",
                "minimum": 1,
                "description": "Override MAX_COMPARE_CONCURRENCY env var (default 1).",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional free-form tags stored on the eval row.",
            },
        },
        "required": ["prompt", "models"],
    },
)
async def compare(arguments: dict[str, Any]) -> dict[str, Any]:
    start = time.monotonic()

    validation_error = _validate_arguments(arguments)
    if validation_error is not None:
        log_tool_call(
            tool="compare",
            duration_ms=_duration_ms(start),
            model=None,
            status="error",
            error_code=validation_error["error"]["code"],
            eval_id=None,
        )
        return validation_error

    prompt = arguments["prompt"]
    models = arguments["models"]
    task_type = arguments.get("task_type")
    tags = arguments.get("tags")
    timeout_ms = _resolve_timeout_ms(arguments.get("timeout_ms"))
    max_concurrency = _resolve_max_concurrency(arguments.get("max_concurrency"))

    repo = get_repo()
    try:
        eval_id = repo.insert_partial(
            prompt=prompt,
            models=models,
            task_type=task_type,
            tags=tags,
        )
    except ValueError as exc:
        payload = _error_payload(ErrorCode.INVALID_INPUT, str(exc))
        log_tool_call(
            tool="compare",
            duration_ms=_duration_ms(start),
            model=None,
            status="error",
            error_code=payload["error"]["code"],
            eval_id=None,
        )
        return payload
    except sqlite3.DatabaseError as exc:
        payload = _error_payload(
            ErrorCode.DB_ERROR, f"Database error while creating eval row: {exc}"
        )
        log_tool_call(
            tool="compare",
            duration_ms=_duration_ms(start),
            model=None,
            status="error",
            error_code=payload["error"]["code"],
            eval_id=None,
        )
        return payload

    sem = asyncio.Semaphore(max_concurrency)
    task_results = await asyncio.gather(
        *[
            _run_one(
                sem=sem,
                model=model,
                prompt=prompt,
                timeout_ms=timeout_ms,
            )
            for model in models
        ],
        return_exceptions=True,
    )

    results: list[dict[str, Any]] = []
    for model, outcome in zip(models, task_results, strict=False):
        if isinstance(outcome, asyncio.CancelledError):
            raise outcome
        if isinstance(outcome, Exception):
            error_message = str(outcome) or "Unexpected internal error in compare worker"
            results.append(
                {
                    "model": model,
                    "response": "",
                    "duration_ms": 0,
                    "status": "error",
                    "error": error_message,
                }
            )
            continue
        if isinstance(outcome, BaseException):
            results.append(
                {
                    "model": model,
                    "response": "",
                    "duration_ms": 0,
                    "status": "error",
                    "error": str(outcome) or "Unexpected base exception in compare worker",
                }
            )
            continue
        results.append(outcome)

    log_tool_call(
        tool="compare",
        duration_ms=_duration_ms(start),
        model=None,
        status="ok",
        error_code=None,
        eval_id=eval_id,
    )
    return {"eval_id": eval_id, "prompt": prompt, "task_type": task_type, "results": results}


async def _run_one(
    *, sem: asyncio.Semaphore, model: str, prompt: str, timeout_ms: int
) -> dict[str, Any]:
    async with sem:
        start = time.monotonic()
        result = await client.generate(model=model, prompt=prompt, timeout_ms=timeout_ms)
        duration_ms = _duration_ms(start)

        if "error" in result:
            error = result.get("error")
            if not isinstance(error, dict):
                error = make_error(
                    ErrorCode.OLLAMA_UNREACHABLE, "Unexpected error format from Ollama client"
                )["error"]

            code = error.get("code")
            message = error.get("message")
            code_str = code if isinstance(code, str) else ErrorCode.OLLAMA_UNREACHABLE.value
            message_str = (
                message
                if isinstance(message, str)
                else "Unexpected error format from Ollama client"
            )
            return {
                "model": model,
                "response": "",
                "duration_ms": duration_ms,
                "status": _status_from_code(code_str),
                "error": message_str,
            }

        raw_response = result.get("response")
        if not isinstance(raw_response, str):
            return {
                "model": model,
                "response": "",
                "duration_ms": duration_ms,
                "status": "error",
                "error": "Malformed Ollama response payload",
            }

        return {
            "model": model,
            "response": wrap_untrusted(model, raw_response),
            "duration_ms": duration_ms,
            "status": "ok",
        }


def _resolve_timeout_ms(timeout_arg: object) -> int:
    if isinstance(timeout_arg, int) and timeout_arg >= 1:
        return timeout_arg

    raw_timeout_ms = os.environ.get("OLLAMA_TIMEOUT_MS", str(_DEFAULT_TIMEOUT_MS))
    try:
        timeout_ms = int(raw_timeout_ms)
    except ValueError:
        _log.warning(
            "Invalid OLLAMA_TIMEOUT_MS value %r; using default %d",
            raw_timeout_ms,
            _DEFAULT_TIMEOUT_MS,
        )
        return _DEFAULT_TIMEOUT_MS
    if timeout_ms <= 0:
        _log.warning(
            "Non-positive OLLAMA_TIMEOUT_MS value %r; using default %d",
            raw_timeout_ms,
            _DEFAULT_TIMEOUT_MS,
        )
        return _DEFAULT_TIMEOUT_MS
    return timeout_ms


def _resolve_max_concurrency(max_concurrency_arg: object) -> int:
    if isinstance(max_concurrency_arg, int) and max_concurrency_arg >= 1:
        return max_concurrency_arg

    raw_value = os.environ.get(
        "MAX_COMPARE_CONCURRENCY",
        str(_DEFAULT_MAX_COMPARE_CONCURRENCY),
    )
    try:
        parsed = int(raw_value)
    except ValueError:
        _log.warning(
            "Invalid MAX_COMPARE_CONCURRENCY value %r; using default %d",
            raw_value,
            _DEFAULT_MAX_COMPARE_CONCURRENCY,
        )
        return _DEFAULT_MAX_COMPARE_CONCURRENCY
    if parsed <= 0:
        _log.warning(
            "Non-positive MAX_COMPARE_CONCURRENCY value %r; using default %d",
            raw_value,
            _DEFAULT_MAX_COMPARE_CONCURRENCY,
        )
        return _DEFAULT_MAX_COMPARE_CONCURRENCY
    return parsed


def _validate_arguments(arguments: dict[str, Any]) -> dict[str, Any] | None:
    prompt = arguments.get("prompt")
    if not isinstance(prompt, str) or prompt == "":
        return _error_payload(ErrorCode.INVALID_INPUT, "Field 'prompt' must be a non-empty string.")

    models = arguments.get("models")
    if not isinstance(models, list) or len(models) == 0:
        return _error_payload(
            ErrorCode.INVALID_INPUT, "Field 'models' must be a non-empty array of strings."
        )
    if not all(isinstance(model, str) for model in models):
        return _error_payload(
            ErrorCode.INVALID_INPUT, "Field 'models' must be a non-empty array of strings."
        )

    task_type = arguments.get("task_type")
    if task_type is not None and (not isinstance(task_type, str) or task_type not in _TASK_TYPES):
        return _error_payload(
            ErrorCode.INVALID_INPUT,
            "Field 'task_type' must be one of: code, summary, explanation, reasoning, "
            "extraction, translation, general, other.",
        )

    timeout_ms = arguments.get("timeout_ms")
    if timeout_ms is not None and (not isinstance(timeout_ms, int) or timeout_ms < 1):
        return _error_payload(
            ErrorCode.INVALID_INPUT, "Field 'timeout_ms' must be an integer >= 1 when provided."
        )

    max_concurrency = arguments.get("max_concurrency")
    if max_concurrency is not None and (
        not isinstance(max_concurrency, int) or max_concurrency < 1
    ):
        return _error_payload(
            ErrorCode.INVALID_INPUT,
            "Field 'max_concurrency' must be an integer >= 1 when provided.",
        )

    tags = arguments.get("tags")
    if tags is not None and (
        not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags)
    ):
        return _error_payload(
            ErrorCode.INVALID_INPUT, "Field 'tags' must be an array of strings when provided."
        )

    return None


def _status_from_code(error_code: str) -> str:
    if error_code == ErrorCode.MODEL_TIMEOUT.value:
        return "timeout"
    return "error"


def _duration_ms(start: float) -> int:
    return max(1, int(round((time.monotonic() - start) * 1000)))


def _error_payload(code: ErrorCode, message: str) -> dict[str, Any]:
    envelope = make_error(code, message)
    return {"error": dict(envelope["error"])}
