"""Work mode tool handlers."""

import os
import time
from typing import Any

import httpx

from ollama_mcp import client
from ollama_mcp.envelope import wrap_untrusted
from ollama_mcp.errors import ErrorCode, make_error
from ollama_mcp.logging import get_logger, log_tool_call
from ollama_mcp.tools import register_tool

_DEFAULT_TIMEOUT_MS = 120_000
_log = get_logger("ollama_mcp.tools.runner")


@register_tool(
    name="run",
    description=(
        "Run a prompt through a specific local Ollama model. Returns the model's response "
        "wrapped in an untrusted-output envelope."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "model": {"type": "string", "description": "Name of a model already pulled in Ollama."},
            "prompt": {"type": "string", "description": "Prompt to send to the model."},
            "timeout_ms": {
                "type": "integer",
                "minimum": 1,
                "description": "Per-call timeout. Defaults to OLLAMA_TIMEOUT_MS env var or 120000.",
            },
        },
        "required": ["model", "prompt"],
    },
)
async def run(arguments: dict[str, Any]) -> dict[str, Any]:
    """Run a prompt against one local model and return wrapped output."""
    start = time.monotonic()

    model = arguments.get("model")
    prompt = arguments.get("prompt")
    timeout_arg = arguments.get("timeout_ms")

    if not isinstance(model, str):
        return _invalid_input(start, "Field 'model' must be a string.")
    if not isinstance(prompt, str):
        return _invalid_input(start, "Field 'prompt' must be a string.", model=model)

    timeout_ms: int
    if timeout_arg is None:
        timeout_ms = _resolve_timeout_ms()
    elif isinstance(timeout_arg, int) and timeout_arg >= 1:
        timeout_ms = timeout_arg
    else:
        return _invalid_input(
            start, "Field 'timeout_ms' must be an integer >= 1 when provided.", model=model
        )

    try:
        result = await client.generate(model=model, prompt=prompt, timeout_ms=timeout_ms)
    except httpx.TimeoutException:
        return _error_result(
            start, ErrorCode.MODEL_TIMEOUT, f"Ollama timed out for model {model!r}", model=model
        )
    except httpx.ConnectError:
        return _error_result(
            start, ErrorCode.OLLAMA_UNREACHABLE, "Cannot reach Ollama", model=model
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return _error_result(
                start, ErrorCode.MODEL_NOT_FOUND, f"Model {model!r} not found", model=model
            )
        return _error_result(
            start,
            ErrorCode.OLLAMA_UNREACHABLE,
            f"Ollama HTTP error: {exc.response.status_code}",
            model=model,
        )
    except httpx.HTTPError as exc:
        return _error_result(start, ErrorCode.OLLAMA_UNREACHABLE, str(exc), model=model)
    except Exception as exc:  # pragma: no cover - defensive guard
        # Keep unexpected exceptions inside the envelope; this code set has no generic
        # internal-error value, so unknown failures are surfaced as unreachable.
        return _error_result(start, ErrorCode.OLLAMA_UNREACHABLE, str(exc), model=model)

    duration_ms = _duration_ms(start)
    if "error" in result:
        error_data = result.get("error")
        error_detail: dict[str, Any]
        if isinstance(error_data, dict):
            error_detail = error_data
        else:
            error_detail = dict(
                make_error(
                    ErrorCode.OLLAMA_UNREACHABLE, "Unexpected error format from Ollama client"
                )["error"]
            )
        error_code = error_detail.get("code")
        log_tool_call(
            tool="run",
            duration_ms=duration_ms,
            model=model,
            status="error",
            error_code=error_code if isinstance(error_code, str) else None,
            eval_id=None,
        )
        return {"error": error_detail, "duration_ms": duration_ms}

    raw_response = result.get("response")
    if not isinstance(raw_response, str):
        return _error_result(
            start, ErrorCode.OLLAMA_UNREACHABLE, "Malformed Ollama response payload", model=model
        )

    response = wrap_untrusted(model, raw_response)
    log_tool_call(
        tool="run",
        duration_ms=duration_ms,
        model=model,
        status="ok",
        error_code=None,
        eval_id=None,
    )
    return {"model": model, "response": response, "duration_ms": duration_ms, "status": "ok"}


def _resolve_timeout_ms() -> int:
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


def _duration_ms(start: float) -> int:
    return max(1, int(round((time.monotonic() - start) * 1000)))


def _invalid_input(start: float, message: str, model: str | None = None) -> dict[str, Any]:
    return _error_result(start, ErrorCode.INVALID_INPUT, message, model=model)


def _error_result(
    start: float, code: ErrorCode, message: str, model: str | None = None
) -> dict[str, Any]:
    duration_ms = _duration_ms(start)
    log_tool_call(
        tool="run",
        duration_ms=duration_ms,
        model=model,
        status="error",
        error_code=code.value,
        eval_id=None,
    )
    payload = make_error(code, message)
    return {"error": payload["error"], "duration_ms": duration_ms}
