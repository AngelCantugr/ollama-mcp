"""Smoke test: all modules must import without error."""


def test_all_modules_import() -> None:
    import ollama_mcp  # noqa: F401
    import ollama_mcp.client  # noqa: F401
    import ollama_mcp.envelope  # noqa: F401
    import ollama_mcp.errors  # noqa: F401
    import ollama_mcp.logging  # noqa: F401
    import ollama_mcp.paths  # noqa: F401
    import ollama_mcp.server  # noqa: F401
    import ollama_mcp.tools  # noqa: F401
