"""Discovery tools: list local Ollama models and report service health."""

from __future__ import annotations

import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from ollama_mcp import client, paths
from ollama_mcp.errors import ErrorCode, make_error
from ollama_mcp.logging import log_tool_call
from ollama_mcp.tools import register_tool

_EMPTY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
_HEALTH_PROBE_PREFIX = ".health-probe-"


def _validate_empty_object(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return dict(make_error(ErrorCode.INVALID_INPUT, "arguments must be an object"))
    return {}


def _error_code(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    if not isinstance(code, str):
        return None
    return code


def _format_size(size: object) -> str:
    if isinstance(size, str):
        return size
    if isinstance(size, (int, float)):
        value = float(size)
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        idx = 0
        while value >= 1000 and idx < len(units) - 1:
            value /= 1000
            idx += 1
        formatted = f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{formatted}{units[idx]}"
    return ""


def _extract_family(item: dict[str, Any]) -> str:
    details = item.get("details")
    if isinstance(details, dict):
        detail_family = details.get("family")
        if isinstance(detail_family, str):
            return detail_family
    top_family = item.get("family")
    if isinstance(top_family, str):
        return top_family
    return ""


@register_tool(
    name="list_models",
    description="List all local Ollama models with digest and family metadata.",
    input_schema=_EMPTY_OBJECT_SCHEMA,
)
async def list_models(arguments: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    response: dict[str, Any]
    status = "ok"
    error_code: str | None = None

    validation_error = _validate_empty_object(arguments)
    if validation_error:
        error_code = _error_code(validation_error)
        status = error_code or "INVALID_INPUT"
        log_tool_call(
            tool="list_models",
            duration_ms=(time.perf_counter() - start) * 1000,
            model=None,
            status=status,
            error_code=error_code,
            eval_id=None,
        )
        return validation_error

    payload = await client.list_tags()
    payload_error_code = _error_code(payload)
    if payload_error_code is not None:
        response = payload
        error_code = payload_error_code
        status = payload_error_code
    else:
        raw_models = payload.get("models", [])
        models: list[dict[str, str]] = []
        if isinstance(raw_models, list):
            for item in raw_models:
                if not isinstance(item, dict):
                    continue
                models.append(
                    {
                        "name": str(item.get("name", "")),
                        "digest": str(item.get("digest", "")),
                        "family": _extract_family(item),
                        "size": _format_size(item.get("size")),
                        "modified": str(item.get("modified_at", "")),
                    }
                )
        response = {"models": models}

    log_tool_call(
        tool="list_models",
        duration_ms=(time.perf_counter() - start) * 1000,
        model=None,
        status=status,
        error_code=error_code,
        eval_id=None,
    )
    return response


@register_tool(
    name="health",
    description="Probe Ollama reachability and local data-dir writability.",
    input_schema=_EMPTY_OBJECT_SCHEMA,
)
async def health(arguments: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    status = "ok"
    error_code: str | None = None

    validation_error = _validate_empty_object(arguments)
    if validation_error:
        error_code = _error_code(validation_error)
        status = error_code or "INVALID_INPUT"
        log_tool_call(
            tool="health",
            duration_ms=(time.perf_counter() - start) * 1000,
            model=None,
            status=status,
            error_code=error_code,
            eval_id=None,
        )
        return validation_error

    data_dir = paths.get_data_dir()

    ollama_status = "ok"
    tags_response = await client.list_tags()
    tags_error_code = _error_code(tags_response)
    if tags_error_code == ErrorCode.OLLAMA_UNREACHABLE.value:
        ollama_status = "unreachable"
    elif tags_error_code == ErrorCode.MODEL_TIMEOUT.value:
        ollama_status = "timeout"

    db_status = "ok"
    probe_filename = f"{_HEALTH_PROBE_PREFIX}{uuid.uuid4().hex}"
    probe_path: Path | None = None
    try:
        probe_path = paths.create_data_file(probe_filename)
    except (paths.PathError, OSError):
        db_status = "unwritable"
    finally:
        if probe_path is not None:
            with suppress(OSError):
                probe_path.unlink(missing_ok=True)

    response = {"ollama": ollama_status, "db": db_status, "data_dir": str(data_dir)}
    log_tool_call(
        tool="health",
        duration_ms=(time.perf_counter() - start) * 1000,
        model=None,
        status=status,
        error_code=error_code,
        eval_id=None,
    )
    return response
