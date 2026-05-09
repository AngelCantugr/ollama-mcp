#!/bin/bash
# Quick CI gate for subagentStop.
# Skips formatting check and tests — the developer agent runs `task check`
# internally before marking work done, and pytest may be slow mid-loop.
# `task ci:quick` runs ruff lint + mypy which catches most mistakes early
# without paying for a full test run.
set -uo pipefail

output=$(task ci:quick 2>&1) || {
  jq -n --arg r "Quick CI check failed (task ci:quick). Fix before finishing:

$output" '{"decision":"block","reason":$r}'
  exit 0
}

echo '{"decision":"approve"}'
