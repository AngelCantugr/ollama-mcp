"""Structured JSON logging to stderr.

Stdout is reserved for the MCP protocol wire format — never log there.
"""

import json
import logging
import os
import sys
import time
from io import TextIOWrapper
from typing import Any


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object on stderr."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class _LiveStderrHandler(logging.StreamHandler[TextIOWrapper]):
    """StreamHandler that resolves sys.stderr at emit time.

    pytest replaces sys.stderr per test (capsys). If we capture the reference
    at handler-creation time, subsequent tests won't see log output in capsys.
    Resolving sys.stderr lazily keeps each test's capture honest.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Re-bind to current sys.stderr before every emit so capsys works.
        self.stream = sys.stderr  # type: ignore[assignment]
        super().emit(record)


def _level_from_env() -> int:
    raw = os.environ.get("LOG_LEVEL", "info").lower()
    mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    return mapping.get(raw, logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes JSON to stderr, honouring LOG_LEVEL."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = _LiveStderrHandler()
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
    logger.setLevel(_level_from_env())
    # Prevent propagation to the root logger which may write to stdout via MCP
    logger.propagate = False
    return logger


def log_tool_call(
    tool: str,
    duration_ms: float,
    model: str | None,
    status: str,
    error_code: str | None,
    eval_id: str | None,
) -> None:
    """Emit a structured log line for every MCP tool invocation."""
    logger = get_logger("ollama_mcp.tool_calls")
    payload: dict[str, Any] = {
        "tool": tool,
        "duration_ms": round(duration_ms, 2),
        "status": status,
    }
    if model is not None:
        payload["model"] = model
    if error_code is not None:
        payload["error_code"] = error_code
    if eval_id is not None:
        payload["eval_id"] = eval_id
    logger.info(json.dumps(payload))
