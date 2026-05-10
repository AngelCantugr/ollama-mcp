"""MCP stdio server entry point.

Stdout is reserved for the MCP wire protocol — all logging goes to stderr.
The tool registry starts empty at Wave 1; handlers are added in later waves.
"""

import asyncio
import json

import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

import ollama_mcp.tools.discovery  # noqa: F401
import ollama_mcp.tools.runner  # noqa: F401
from ollama_mcp import logging as mcp_logging
from ollama_mcp.storage import get_connection, migrate
from ollama_mcp.tools import get_registry

_log = mcp_logging.get_logger("ollama_mcp.server")

app = Server("ollama-mcp")


@app.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
async def list_tools() -> list[types.Tool]:
    registry = get_registry()
    return [
        types.Tool(name=name, description=desc, inputSchema=schema)
        for name, (desc, schema, _) in registry.items()
    ]


@app.call_tool()  # type: ignore[untyped-decorator]
async def call_tool(name: str, arguments: dict[str, object]) -> list[types.TextContent]:
    registry = get_registry()
    if name not in registry:
        # Unknown tool — return error text; MCP layer will wrap it
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
    _, _, handler = registry[name]
    result = await handler(arguments)
    return [types.TextContent(type="text", text=json.dumps(result))]


async def _main() -> None:
    _log.info("ollama-mcp server starting")
    # Run migrations at startup so every tool sees a ready schema and we fail
    # fast before opening the MCP stdio transport if storage is unhealthy.
    conn = get_connection()
    try:
        migrate(conn)
    except Exception:
        _log.error("database migration failed during startup", exc_info=True)
        raise
    finally:
        conn.close()

    async with stdio_server() as (read_stream, write_stream):
        init_opts: InitializationOptions = app.create_initialization_options()
        await app.run(read_stream, write_stream, init_opts)


def main() -> None:
    asyncio.run(_main())
