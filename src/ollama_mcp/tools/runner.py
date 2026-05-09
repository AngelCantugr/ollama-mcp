"""Work mode tool handlers."""

import os
import time
from typing import Any

from ollama_mcp import client
from ollama_mcp.envelope import wrap_untrusted
from ollama_mcp.errors import ErrorCode, make_error
from ollama_mcp.logging import log_tool_call
from ollama_mcp.tools import register_tool

_DEFAULT_TIMEOUT_MS = 120_000


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
    except Exception as exc:  # pragma: no cover - defensive guard
        duration_ms = _duration_ms(start)
        log_tool_call(
            tool="run",
            duration_ms=duration_ms,
            model=model,
            status="error",
            error_code=ErrorCode.OLLAMA_UNREACHABLE.value,
            eval_id=None,
        )
        payload = make_error(ErrorCode.OLLAMA_UNREACHABLE, str(exc))
        return {"error": payload["error"], "duration_ms": duration_ms}

    duration_ms = _duration_ms(start)
    if "error" in result:
        error = result.get("error")
        if not isinstance(error, dict):
            error = make_error(ErrorCode.OLLAMA_UNREACHABLE, "Malformed error response")["error"]
        error_code = error.get("code")
        log_tool_call(
            tool="run",
            duration_ms=duration_ms,
            model=model,
            status="error",
            error_code=error_code if isinstance(error_code, str) else None,
            eval_id=None,
        )
        return {"error": error, "duration_ms": duration_ms}

    raw_response = result.get("response")
    if not isinstance(raw_response, str):
        log_tool_call(
            tool="run",
            duration_ms=duration_ms,
            model=model,
            status="error",
            error_code=ErrorCode.OLLAMA_UNREACHABLE.value,
            eval_id=None,
        )
        payload = make_error(ErrorCode.OLLAMA_UNREACHABLE, "Malformed Ollama response payload")
        return {"error": payload["error"], "duration_ms": duration_ms}

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
        return _DEFAULT_TIMEOUT_MS
    return timeout_ms if timeout_ms > 0 else _DEFAULT_TIMEOUT_MS


def _duration_ms(start: float) -> int:
    return max(1, int(round((time.monotonic() - start) * 1000)))


def _invalid_input(start: float, message: str, model: str | None = None) -> dict[str, Any]:
    duration_ms = _duration_ms(start)
    log_tool_call(
        tool="run",
        duration_ms=duration_ms,
        model=model,
        status="error",
        error_code=ErrorCode.INVALID_INPUT.value,
        eval_id=None,
    )
    payload = make_error(ErrorCode.INVALID_INPUT, message)
    return {"error": payload["error"], "duration_ms": duration_ms}
