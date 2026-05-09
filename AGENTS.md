# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, GitHub Copilot, Cursor, Codex, Aider) when working with code in this repository. `CLAUDE.md` is a symlink to this file.

## Project

`ollama-mcp` is a Python MCP (Model Context Protocol) server that exposes locally-running Ollama models to Claude as a fleet of orchestratable workers. Two operating modes:

- **Work Mode** — Claude routes prompts to the right local model (`run`, `route`).
- **Learn Mode** — Claude compares models on the same prompt, scores outputs, and persists eval history so routing decisions improve over time (`compare`, `score_comparison`, `judge_with_model`).

The full design — tool surface, trust model, error envelope, SQLite schema, three-phase implementation plan, and explicit decisions on prior open questions — lives in [SPEC.md](SPEC.md). **Read SPEC.md before any non-trivial change.** When the spec and the code disagree, fix one of them deliberately; do not silently drift.

## Commands

Tooling is driven by `Taskfile.yaml` (go-task), **not** `make`. The Python toolchain is managed by [`uv`](https://docs.astral.sh/uv/).

| Task | Command |
| --- | --- |
| Sync the dev environment from the lockfile | `task setup` (`uv sync --all-extras`) |
| Run the MCP server locally (stdio) | `task` (alias for `task default`) |
| Run all tests | `task test` (wraps `uv run pytest`) |
| Run a single test | `task test:one -- -k "test_name"` |
| Re-run last failing tests | `task test:rerun` |
| Lint | `task lint` (ruff check) |
| Auto-fix lint | `task lint:fix` (ruff check --fix) |
| Format | `task fmt` (ruff format) |
| Type-check | `task typecheck` (mypy) |
| Full local check (run before pushing) | `task check` (`fmt` + `lint` + `typecheck` + `test`) |
| Watch logs from a debug run | `task logs` |

No git pre-commit hooks are configured — run `task check` before pushing. CI enforces `lint`, `typecheck`, and `test`, so fixing locally saves a round trip.

## Architecture

Entry flow: `server.py` (MCP `stdio` transport) → tool dispatcher → `tools/*.py` handlers → `storage/evals_repo.py` and the Ollama REST client.

### MCP server loop

`server.py` registers all tool handlers with the MCP Python SDK and runs over stdio. Stdout is **reserved** for the MCP protocol — never log there. All logging goes to **stderr** as structured JSON (see Observability in SPEC.md). Tool handlers are async and must respect the configured `timeout_ms` per call; they never block on unbounded I/O.

### Tool packages (the extension point)

Each MCP tool lives under `tools/` as a thin adapter that validates input, calls into business logic / `storage/`, and returns the unified response shape. Concrete groupings:

- `tools/discovery.py` — `list_models`, `health`
- `tools/runner.py` — `run`, `route`
- `tools/compare.py` — `compare` (fan-out to multiple models)
- `tools/judge.py` — `score_comparison` (Claude-as-judge write-back), `judge_with_model` (LLM-as-judge variant)
- `tools/knowledge.py` — `log_eval`, `delete_eval`, `export_evals`, `get_model_insights`, `classify_prompt`
- `tools/routing.py` — `get_routing_config`, `update_routing_rule`, `suggest_routing_updates`, `reset_routing`

Adding a new MCP tool = new function in the right module + register it in `server.py`. **Do not put SQL or HTTP calls directly in `tools/`** — those belong in `storage/` and a dedicated Ollama client.

### Storage layer

`storage/db.py` opens the SQLite connection and runs ordered migration files from `storage/migrations/` on startup (`001_init.sql`, `002_*.sql`, …). `storage/evals_repo.py` is the only module allowed to issue SQL — keeps `tools/` thin and makes schema changes audit-able. Eval rows carry `task_type`, `tags`, `prompt_hash`, `judge_model`, and `schema_version` so `get_model_insights` can compute `best_at` from `task_type` aggregation (see SPEC.md → Data Models).

### Ollama client

A single async `httpx.AsyncClient` instance against `OLLAMA_HOST` (default `http://localhost:11434`). All requests honour the per-call `timeout_ms`. Network errors are translated into the standard error envelope (`OLLAMA_UNREACHABLE`, `MODEL_NOT_FOUND`, `MODEL_TIMEOUT`).

### Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL |
| `DATA_DIR` | `~/.ollama-mcp` | Root for DB, exports, config — paths must resolve under this |
| `DB_PATH` | `${DATA_DIR}/evals.db` | SQLite path |
| `ROUTING_CONFIG` | `${DATA_DIR}/routing.json` | Routing rules file (validated against `config/routing.schema.json` on load) |
| `OLLAMA_TIMEOUT_MS` | `120000` | Default per-call timeout |
| `MAX_COMPARE_CONCURRENCY` | `1` | Fan-out concurrency in `compare` (default sequential for single-GPU boxes) |
| `LOG_LEVEL` | `info` | `debug` \| `info` \| `warn` \| `error` |

## Non-obvious patterns

- **MCP responses are unary.** The server cannot synchronously call back into Claude's LLM. The Learn-Mode scoring path is therefore split: `compare` returns raw outputs + an `eval_id`, then Claude reasons and calls `score_comparison(eval_id, scores, …)` in a second MCP call. The `judge_with_model` tool is the autonomous LLM-as-judge alternative when Claude isn't in the loop.
- **Local model outputs are untrusted input.** Every model response flowing back to Claude is wrapped in `<ollama_output model="X" untrusted="true">…</ollama_output>` so prompt-injection payloads in model output don't get treated as trusted instructions. Do not strip the envelope.
- **Partial success is the contract for `compare`.** One model timing out or erroring never fails the whole call. Each per-model result carries `{status: "ok" | "timeout" | "error", error?: string}`. Implementation uses `asyncio.gather(..., return_exceptions=True)` with a `MAX_COMPARE_CONCURRENCY` semaphore.
- **Path hardening.** `DB_PATH` and `ROUTING_CONFIG` are resolved to absolute paths and asserted to live under `DATA_DIR`. Symlinks pointing outside `DATA_DIR` are rejected. The DB file is created with `0600` permissions.
- **Unified error envelope.** All tools return either a success payload or `{error: {code, message, retryable}}` with codes `OLLAMA_UNREACHABLE`, `MODEL_NOT_FOUND`, `MODEL_TIMEOUT`, `INVALID_INPUT`, `DB_ERROR`. `route` falls back to the configured `default` model on `MODEL_NOT_FOUND` and surfaces `matched_rule: "fallback"`.
- **Stdout is reserved for MCP.** All logging — including `--debug` raw Ollama dumps — goes to stderr.
- **Self-improving routing is a Claude-driven loop**, not server magic: `get_model_insights` → `suggest_routing_updates` (server proposes diffs) → `update_routing_rule` (Claude applies). The router itself reads `routing.json` statically.

## Conventions

- **Commit messages** use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`). Release notes group by prefix.
- **Python style.** `ruff` enforces lint + format. `mypy` runs in strict mode on `tools/`, `storage/`, and `server.py`. Public functions get type hints; internal helpers can omit them when obvious.
- **Comments WHY, not WHAT.** Comment non-obvious constraints (MCP protocol quirks, untrusted-output reasoning, SQLite isolation behaviour). Don't narrate code.
- **Errors are values.** Catch `httpx.HTTPError`, `sqlite3.DatabaseError`, `TimeoutError` at the tool boundary and translate to the error envelope. Never let exceptions escape into the MCP response stream.
- **Tests.** `pytest` + `pytest-asyncio`. Mock Ollama with `respx` against `httpx` — never hit a real Ollama in CI. SQLite tests use a temp `DATA_DIR` per test (see `tests/conftest.py`).
- **No streaming in v1.** Tool responses are unary. Do not add streaming primitives without a SPEC update.

## Custom agents

This repo ships GitHub-Copilot-compatible custom agents under `.github/agents/`:

- **`@dev-loop`** — Orchestrates: explores → implements → reviews, up to 5 iterations until the reviewer scores ≥ 8/10
- **`@explorer`** — Read-only audit; maps affected modules, patterns, and reuse candidates before any code is written
- **`@developer`** — Senior Python engineer agent; implements features/fixes and validates with `task check`
- **`@reviewer`** — Outcome-focused reviewer; scores 1–10 against the original problem statement (not just acceptance criteria)

Assign `@dev-loop` to any issue to start an autonomous dev cycle.
