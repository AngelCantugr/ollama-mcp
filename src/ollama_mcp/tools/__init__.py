"""Tool registry with auto-discovery.

Each MCP tool module decorates its handler with `@register_tool`. Importing
this package walks every submodule under `ollama_mcp.tools` and imports it,
which triggers the decorators and populates `_REGISTRY` as a side effect.
`server.py` then calls `get_registry()` to wire up MCP `list_tools` /
`call_tool` handlers — it never has to enumerate tool modules manually,
which means parallel PRs adding new tools don't conflict on `server.py`.
"""

import importlib
import pkgutil
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


def _autodiscover() -> None:
    """Import every submodule of `ollama_mcp.tools` so `@register_tool` fires.

    Runs once at first import of the `ollama_mcp.tools` package. Submodules
    are skipped silently if they don't exist (e.g. before they're written),
    but import errors from existing modules propagate so we don't silently
    ship a broken tool.
    """
    for module_info in pkgutil.iter_modules(__path__, prefix=f"{__name__}."):
        if module_info.name == __name__:
            continue
        importlib.import_module(module_info.name)


_autodiscover()
