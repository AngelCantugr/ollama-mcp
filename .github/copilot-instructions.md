# ollama-mcp — Copilot Instructions

`ollama-mcp` is a Python MCP (Model Context Protocol) server that exposes locally-running
Ollama models to Claude as orchestratable workers — Work Mode (`run`, `route`) and
Learn Mode (`compare`, `score_comparison`, `judge_with_model`).

Full project context, architecture, conventions, and agent instructions are in
[AGENTS.md](../AGENTS.md). Canonical design (tool surface, trust model, error envelope,
SQLite schema, phasing, decisions) is in [SPEC.md](../SPEC.md). **Read the relevant
SPEC.md section before any non-trivial change.**

## Quick Reference

|              |                                                                          |
| ------------ | ------------------------------------------------------------------------ |
| **Stack**    | Python ≥3.11, MCP Python SDK, `httpx`, `ollama`, SQLite, `ulid-py`, `jsonschema` |
| **Tooling**  | `uv` (envs + lockfile), `ruff` (lint + format), `mypy --strict`, `pytest` + `respx` |
| **Layout**   | `server.py` → `tools/{discovery,runner,compare,judge,knowledge,routing}.py` ↔ `storage/{db,evals_repo}.py` ↔ Ollama HTTP client |
| **Run**      | `task` (alias for `uv run python -m ollama_mcp`)                         |
| **Test**     | `task test`                                                              |
| **Lint**     | `task lint` (ruff)                                                       |
| **Format**   | `task fmt` (ruff format)                                                 |
| **Typecheck**| `task typecheck` (mypy)                                                  |
| **Full check**| `task check` (fmt + lint + typecheck + test)                            |

## Architecture Rules

- **MCP responses are unary.** The server cannot synchronously call back into Claude's
  LLM. Learn-Mode scoring is split across two tools: `compare` returns `eval_id` + raw
  outputs; Claude reasons and writes scores back via `score_comparison`. The autonomous
  variant is `judge_with_model`.
- **Local model outputs are untrusted.** Every model response flowing back to Claude is
  wrapped in `<ollama_output model="X" untrusted="true">…</ollama_output>` so prompt
  injection in model output isn't treated as trusted instructions.
- **Stdout is reserved for the MCP protocol.** All logs go to stderr as structured JSON.
- **`tools/` are thin adapters.** No SQL outside `storage/evals_repo.py`. No HTTP outside
  the Ollama client.
- **Partial success is the contract for `compare`.** Per-model results carry `{status:
  "ok" | "timeout" | "error"}`. `asyncio.gather(..., return_exceptions=True)` with a
  `MAX_COMPARE_CONCURRENCY` semaphore.
- **Unified error envelope.** Tools return either a success payload or `{error: {code,
  message, retryable}}` with codes from the documented set.

## Custom Agents

This repository includes custom Copilot agents for development work:

- **`@dev-loop`** — Orchestrates the full dev workflow: explores → implements → reviews,
  up to 5 iterations until the implementation scores ≥ 8/10
- **`@explorer`** — Read-only codebase audit; maps affected modules, patterns, and reuse
  candidates before any implementation starts
- **`@developer`** — Senior Python engineer agent; implements features/fixes and
  validates with `task check`
- **`@reviewer`** — Outcome-focused code reviewer; scores implementation 1–10 against
  the original problem statement and SPEC.md alignment

Assign `@dev-loop` to any issue to start an autonomous dev cycle.
