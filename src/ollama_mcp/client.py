"""Async Ollama HTTP client.

A single module owns all httpx traffic to Ollama so transport configuration,
timeout policy, and error translation stay in one place.
"""

import os
from typing import Any

import httpx

from ollama_mcp.errors import ErrorCode, ErrorEnvelope, make_error

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_TIMEOUT_MS = 120_000


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", _DEFAULT_HOST).rstrip("/")


def _default_timeout_ms() -> int:
    raw = os.environ.get("OLLAMA_TIMEOUT_MS", str(_DEFAULT_TIMEOUT_MS))
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT_MS


def get_client() -> httpx.AsyncClient:
    """Return a configured async httpx client pointed at OLLAMA_HOST."""
    timeout_s = _default_timeout_ms() / 1000.0
    return httpx.AsyncClient(
        base_url=_ollama_host(),
        timeout=httpx.Timeout(timeout_s),
    )


async def generate(
    model: str,
    prompt: str,
    timeout_ms: int | None = None,
) -> dict[str, Any]:
    """Call /api/generate and return the parsed JSON body.

    Translates httpx transport errors into the error envelope so callers
    receive a structured value rather than a raw exception.
    """
    timeout_s = (timeout_ms if timeout_ms is not None else _default_timeout_ms()) / 1000.0
    try:
        async with httpx.AsyncClient(
            base_url=_ollama_host(),
            timeout=httpx.Timeout(timeout_s),
        ) as client:
            resp = await client.post(
                "/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result
    except httpx.TimeoutException:
        return _to_dict(
            make_error(ErrorCode.MODEL_TIMEOUT, f"Ollama timed out for model {model!r}")
        )
    except httpx.ConnectError:
        return _to_dict(
            make_error(ErrorCode.OLLAMA_UNREACHABLE, f"Cannot reach Ollama at {_ollama_host()}")
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return _to_dict(make_error(ErrorCode.MODEL_NOT_FOUND, f"Model {model!r} not found"))
        return _to_dict(
            make_error(
                ErrorCode.OLLAMA_UNREACHABLE, f"Ollama HTTP error: {exc.response.status_code}"
            )
        )
    except httpx.HTTPError as exc:
        return _to_dict(make_error(ErrorCode.OLLAMA_UNREACHABLE, str(exc)))


async def list_tags() -> dict[str, Any]:
    """Call /api/tags and return the parsed JSON body."""
    try:
        async with httpx.AsyncClient(
            base_url=_ollama_host(),
            timeout=httpx.Timeout(_default_timeout_ms() / 1000.0),
        ) as client:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result
    except httpx.TimeoutException:
        return _to_dict(make_error(ErrorCode.MODEL_TIMEOUT, "Ollama /api/tags timed out"))
    except httpx.ConnectError:
        return _to_dict(
            make_error(ErrorCode.OLLAMA_UNREACHABLE, f"Cannot reach Ollama at {_ollama_host()}")
        )
    except httpx.HTTPError as exc:
        return _to_dict(make_error(ErrorCode.OLLAMA_UNREACHABLE, str(exc)))


def _to_dict(envelope: ErrorEnvelope) -> dict[str, Any]:
    # TypedDict → plain dict for uniform return type
    return dict(envelope)
