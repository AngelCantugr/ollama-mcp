"""Tool registry.

Each MCP tool module decorates its handler with @register_tool.  server.py
calls get_registry() to wire up the MCP list_tools / call_tool handlers.
"""

from collections.abc import Callable, Coroutine
from typing import Any

_REGISTRY: dict[str, tuple[str, dict[str, Any], Callable[..., Coroutine[Any, Any, Any]]]] = {}


def register_tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
) -> Callable[
    [Callable[..., Coroutine[Any, Any, Any]]],
    Callable[..., Coroutine[Any, Any, Any]],
]:
    """Decorator factory that registers an async tool handler."""

    def decorator(
        fn: Callable[..., Coroutine[Any, Any, Any]],
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        _REGISTRY[name] = (description, input_schema, fn)
        return fn

    return decorator


def get_registry() -> dict[
    str, tuple[str, dict[str, Any], Callable[..., Coroutine[Any, Any, Any]]]
]:
    """Return a snapshot of the current tool registry."""
    return dict(_REGISTRY)
