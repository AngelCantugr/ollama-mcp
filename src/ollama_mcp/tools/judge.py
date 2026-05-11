"""Learn-mode scoring tool handlers.

``score_comparison`` — Claude-as-judge: persists scores Claude already
reasoned over for a prior ``compare`` eval row.

``judge_with_model`` — LLM-as-judge: runs the comparison internally, then
asks a designated local Ollama model to score the outputs autonomously.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from ollama_mcp import client
from ollama_mcp.errors import ErrorCode, make_error
from ollama_mcp.logging import get_logger, log_tool_call
from ollama_mcp.storage import get_repo
from ollama_mcp.tools import register_tool

_DEFAULT_TIMEOUT_MS = 120_000
_log = get_logger("ollama_mcp.tools.judge")

_VALID_TASK_TYPES = frozenset(
    ["code", "summary", "explanation", "reasoning", "extraction", "translation", "general", "other"]
)


# ---------------------------------------------------------------------------
# score_comparison — Claude-as-judge write-back
# ---------------------------------------------------------------------------


@register_tool(
    name="score_comparison",
    description=(
        "Persist Claude's scoring of a prior compare result. The server has no LLM — "
        "this tool just writes the scores Claude provides into the eval row identified "
        "by eval_id."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "eval_id": {"type": "string"},
            "criteria": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "scores": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["score", "reasoning"],
                },
                "description": "Map of model name to {score, reasoning}.",
            },
            "winner": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["eval_id", "criteria", "scores", "winner"],
    },
)
async def score_comparison(arguments: dict[str, Any]) -> dict[str, Any]:
    """Persist Claude's scores for a prior compare eval row."""
    start = time.monotonic()

    # --- input validation ---
    eval_id = arguments.get("eval_id")
    criteria = arguments.get("criteria")
    scores = arguments.get("scores")
    winner = arguments.get("winner")
    notes = arguments.get("notes")

    if not isinstance(eval_id, str) or not eval_id:
        return _invalid_input(
            start, "tool=score_comparison", "Field 'eval_id' must be a non-empty string."
        )
    if not isinstance(criteria, list) or len(criteria) == 0:
        return _invalid_input(
            start, "tool=score_comparison", "Field 'criteria' must be a non-empty list."
        )
    if not all(isinstance(c, str) for c in criteria):
        return _invalid_input(
            start, "tool=score_comparison", "Field 'criteria' must contain only strings."
        )
    if not isinstance(scores, dict) or len(scores) == 0:
        return _invalid_input(
            start, "tool=score_comparison", "Field 'scores' must be a non-empty object."
        )
    for model_name, entry in scores.items():
        if not isinstance(entry, dict):
            return _invalid_input(
                start,
                "tool=score_comparison",
                f"scores[{model_name!r}] must be an object with 'score' and 'reasoning'.",
            )
        if "score" not in entry or not isinstance(entry["score"], (int, float)):
            return _invalid_input(
                start,
                "tool=score_comparison",
                f"scores[{model_name!r}]['score'] must be a number.",
            )
        if "reasoning" not in entry or not isinstance(entry["reasoning"], str):
            return _invalid_input(
                start,
                "tool=score_comparison",
                f"scores[{model_name!r}]['reasoning'] must be a string.",
            )
    if not isinstance(winner, str) or not winner:
        return _invalid_input(
            start, "tool=score_comparison", "Field 'winner' must be a non-empty string."
        )
    if notes is not None and not isinstance(notes, str):
        return _invalid_input(
            start, "tool=score_comparison", "Field 'notes' must be a string when provided."
        )

    # --- persist scores ---
    try:
        repo = get_repo()
        repo.update_scores(
            eval_id=eval_id,
            winner=winner,
            scores=scores,
            criteria=criteria,
            judge_model=None,
            notes=notes,
        )
    except KeyError:
        return _invalid_input(start, "tool=score_comparison", f"unknown eval_id {eval_id!r}")
    except Exception as exc:
        duration_ms = _duration_ms(start)
        log_tool_call(
            tool="score_comparison",
            duration_ms=duration_ms,
            model=None,
            status="error",
            error_code=ErrorCode.DB_ERROR.value,
            eval_id=eval_id,
        )
        _log.exception("DB error in score_comparison: %s", exc)
        return {
            "error": make_error(ErrorCode.DB_ERROR, str(exc))["error"],
            "duration_ms": duration_ms,
        }

    duration_ms = _duration_ms(start)
    log_tool_call(
        tool="score_comparison",
        duration_ms=duration_ms,
        model=None,
        status="ok",
        error_code=None,
        eval_id=eval_id,
    )
    return {"logged": True, "eval_id": eval_id}


# ---------------------------------------------------------------------------
# judge_with_model — LLM-as-judge autonomous variant
# ---------------------------------------------------------------------------


@register_tool(
    name="judge_with_model",
    description=(
        "Run a prompt through multiple models and ask a designated local model to "
        "score the outputs. Fully autonomous — no Claude reasoning step. Logs an eval "
        "row with judge_model set."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "models": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "description": "Models to compare. At least 2 (otherwise scoring is meaningless).",
            },
            "judge_model": {"type": "string"},
            "criteria": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Criteria the judge model uses (e.g. ['clarity', 'conciseness']).",
            },
            "task_type": {
                "type": "string",
                "enum": [
                    "code",
                    "summary",
                    "explanation",
                    "reasoning",
                    "extraction",
                    "translation",
                    "general",
                    "other",
                ],
            },
            "timeout_ms": {"type": "integer", "minimum": 1},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["prompt", "models", "judge_model", "criteria"],
    },
)
async def judge_with_model(arguments: dict[str, Any]) -> dict[str, Any]:
    """Run compare + LLM-as-judge autonomously and log the scored eval."""
    start = time.monotonic()

    # --- input validation ---
    prompt = arguments.get("prompt")
    models = arguments.get("models")
    judge_model = arguments.get("judge_model")
    criteria = arguments.get("criteria")
    task_type = arguments.get("task_type")
    timeout_arg = arguments.get("timeout_ms")
    tags = arguments.get("tags")

    if not isinstance(prompt, str) or not prompt:
        return _invalid_input(
            start, "tool=judge_with_model", "Field 'prompt' must be a non-empty string."
        )
    if not isinstance(models, list) or len(models) < 2:
        return _invalid_input(
            start, "tool=judge_with_model", "Field 'models' must be a list with at least 2 entries."
        )
    if not all(isinstance(m, str) for m in models):
        return _invalid_input(
            start, "tool=judge_with_model", "Field 'models' must contain only strings."
        )
    if not isinstance(judge_model, str) or not judge_model:
        return _invalid_input(
            start, "tool=judge_with_model", "Field 'judge_model' must be a non-empty string."
        )
    if not isinstance(criteria, list) or len(criteria) == 0:
        return _invalid_input(
            start, "tool=judge_with_model", "Field 'criteria' must be a non-empty list."
        )
    if not all(isinstance(c, str) for c in criteria):
        return _invalid_input(
            start, "tool=judge_with_model", "Field 'criteria' must contain only strings."
        )
    if task_type is not None and task_type not in _VALID_TASK_TYPES:
        return _invalid_input(
            start,
            "tool=judge_with_model",
            f"Field 'task_type' must be one of {sorted(_VALID_TASK_TYPES)!r} when provided.",
        )
    if tags is not None and (
        not isinstance(tags, list) or not all(isinstance(t, str) for t in tags)
    ):
        return _invalid_input(
            start, "tool=judge_with_model", "Field 'tags' must be a list of strings when provided."
        )

    # --- resolve timeout ---
    timeout_ms: int
    if timeout_arg is None:
        timeout_ms = _resolve_timeout_ms()
    elif isinstance(timeout_arg, int) and timeout_arg >= 1:
        timeout_ms = timeout_arg
    else:
        return _invalid_input(
            start,
            "tool=judge_with_model",
            "Field 'timeout_ms' must be an integer >= 1 when provided.",
        )

    # --- comparison phase: call each model sequentially (single-GPU default) ---
    compare_results: list[dict[str, Any]] = []
    for model in models:
        model_start = time.monotonic()
        try:
            result = await client.generate(model=model, prompt=prompt, timeout_ms=timeout_ms)
        except httpx.TimeoutException:
            compare_results.append(
                {
                    "model": model,
                    "response": None,
                    "status": "timeout",
                    "duration_ms": _duration_ms(model_start),
                }
            )
            continue
        except httpx.ConnectError:
            compare_results.append(
                {
                    "model": model,
                    "response": None,
                    "status": "error",
                    "error": "Cannot reach Ollama",
                    "duration_ms": _duration_ms(model_start),
                }
            )
            continue
        except httpx.HTTPError as exc:
            compare_results.append(
                {
                    "model": model,
                    "response": None,
                    "status": "error",
                    "error": str(exc),
                    "duration_ms": _duration_ms(model_start),
                }
            )
            continue

        if "error" in result:
            compare_results.append(
                {
                    "model": model,
                    "response": None,
                    "status": "error",
                    "error": str(result.get("error")),
                    "duration_ms": _duration_ms(model_start),
                }
            )
            continue

        raw_response = result.get("response", "")
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)

        compare_results.append(
            {
                "model": model,
                "response": raw_response,
                "status": "ok",
                "duration_ms": _duration_ms(model_start),
            }
        )

    # At least one model must have succeeded for judging to be meaningful.
    successful = [r for r in compare_results if r["status"] == "ok"]
    if not successful:
        duration_ms = _duration_ms(start)
        log_tool_call(
            tool="judge_with_model",
            duration_ms=duration_ms,
            model=judge_model,
            status="error",
            error_code=ErrorCode.MODEL_TIMEOUT.value,
            eval_id=None,
        )
        return {
            "error": make_error(
                ErrorCode.MODEL_TIMEOUT,
                "All comparison models failed; cannot judge.",
            )["error"],
            "duration_ms": duration_ms,
        }

    # --- persist partial eval row ---
    try:
        repo = get_repo()
        eval_id = repo.insert_partial(
            prompt=prompt,
            models=models,
            task_type=task_type,
            tags=tags,
        )
    except Exception as exc:
        duration_ms = _duration_ms(start)
        log_tool_call(
            tool="judge_with_model",
            duration_ms=duration_ms,
            model=judge_model,
            status="error",
            error_code=ErrorCode.DB_ERROR.value,
            eval_id=None,
        )
        _log.exception("DB error inserting partial eval in judge_with_model: %s", exc)
        return {
            "error": make_error(ErrorCode.DB_ERROR, str(exc))["error"],
            "duration_ms": duration_ms,
        }

    # --- build judge prompt ---
    candidates_text = "\n".join(
        f"---\nModel: {r['model']}\nResponse: {r['response']}\n" for r in successful
    )
    judge_prompt = (
        f"You are an evaluator. Score each candidate response on the given criteria.\n\n"
        f"Original prompt: {prompt}\n"
        f"Criteria: {', '.join(criteria)}\n\n"
        f"Candidates:\n{candidates_text}\n"
        f"Reply with a JSON object exactly in this shape, nothing else:\n"
        f"{{\n"
        f'  "scores": {{\n'
        f'    "<model_name>": {{"score": <number 1-10>, "reasoning": "<string>"}}\n'
        f"  }},\n"
        f'  "winner": "<model_name with highest score>"\n'
        f"}}"
    )

    # --- call judge model ---
    try:
        judge_result = await client.generate(
            model=judge_model, prompt=judge_prompt, timeout_ms=timeout_ms
        )
    except httpx.TimeoutException:
        duration_ms = _duration_ms(start)
        log_tool_call(
            tool="judge_with_model",
            duration_ms=duration_ms,
            model=judge_model,
            status="error",
            error_code=ErrorCode.MODEL_TIMEOUT.value,
            eval_id=eval_id,
        )
        return {
            "error": make_error(ErrorCode.MODEL_TIMEOUT, f"Judge model {judge_model!r} timed out")[
                "error"
            ],
            "duration_ms": duration_ms,
        }
    except httpx.ConnectError:
        duration_ms = _duration_ms(start)
        log_tool_call(
            tool="judge_with_model",
            duration_ms=duration_ms,
            model=judge_model,
            status="error",
            error_code=ErrorCode.OLLAMA_UNREACHABLE.value,
            eval_id=eval_id,
        )
        return {
            "error": make_error(ErrorCode.OLLAMA_UNREACHABLE, "Cannot reach Ollama")["error"],
            "duration_ms": duration_ms,
        }
    except httpx.HTTPError as exc:
        duration_ms = _duration_ms(start)
        log_tool_call(
            tool="judge_with_model",
            duration_ms=duration_ms,
            model=judge_model,
            status="error",
            error_code=ErrorCode.OLLAMA_UNREACHABLE.value,
            eval_id=eval_id,
        )
        return {
            "error": make_error(ErrorCode.OLLAMA_UNREACHABLE, str(exc))["error"],
            "duration_ms": duration_ms,
        }

    if "error" in judge_result:
        duration_ms = _duration_ms(start)
        err = judge_result.get("error", {})
        code = (
            err.get("code", ErrorCode.OLLAMA_UNREACHABLE.value)
            if isinstance(err, dict)
            else ErrorCode.OLLAMA_UNREACHABLE.value
        )
        log_tool_call(
            tool="judge_with_model",
            duration_ms=duration_ms,
            model=judge_model,
            status="error",
            error_code=code if isinstance(code, str) else ErrorCode.OLLAMA_UNREACHABLE.value,
            eval_id=eval_id,
        )
        return {
            "error": dict(err)
            if isinstance(err, dict)
            else make_error(ErrorCode.OLLAMA_UNREACHABLE, str(err))["error"],
            "duration_ms": duration_ms,
        }

    raw_judge_output = judge_result.get("response", "")
    if not isinstance(raw_judge_output, str):
        raw_judge_output = str(raw_judge_output)

    # --- parse judge output defensively (trust model's JSON, not its prose) ---
    parsed = _parse_judge_output(raw_judge_output)
    if parsed is None:
        duration_ms = _duration_ms(start)
        log_tool_call(
            tool="judge_with_model",
            duration_ms=duration_ms,
            model=judge_model,
            status="error",
            error_code=ErrorCode.INVALID_INPUT.value,
            eval_id=eval_id,
        )
        return {
            "error": make_error(ErrorCode.INVALID_INPUT, "judge model output unparseable")["error"],
            "duration_ms": duration_ms,
        }

    judge_scores, winner = parsed

    # --- update eval with scored results ---
    try:
        repo.update_scores(
            eval_id=eval_id,
            winner=winner,
            scores=judge_scores,
            criteria=criteria,
            judge_model=judge_model,
            notes=None,
        )
    except KeyError:
        # Shouldn't happen since we just inserted, but handle defensively.
        duration_ms = _duration_ms(start)
        log_tool_call(
            tool="judge_with_model",
            duration_ms=duration_ms,
            model=judge_model,
            status="error",
            error_code=ErrorCode.DB_ERROR.value,
            eval_id=eval_id,
        )
        return {
            "error": make_error(
                ErrorCode.DB_ERROR, f"eval_id {eval_id!r} disappeared before scoring"
            )["error"],
            "duration_ms": duration_ms,
        }
    except Exception as exc:
        duration_ms = _duration_ms(start)
        log_tool_call(
            tool="judge_with_model",
            duration_ms=duration_ms,
            model=judge_model,
            status="error",
            error_code=ErrorCode.DB_ERROR.value,
            eval_id=eval_id,
        )
        _log.exception("DB error updating scores in judge_with_model: %s", exc)
        return {
            "error": make_error(ErrorCode.DB_ERROR, str(exc))["error"],
            "duration_ms": duration_ms,
        }

    duration_ms = _duration_ms(start)
    log_tool_call(
        tool="judge_with_model",
        duration_ms=duration_ms,
        model=judge_model,
        status="ok",
        error_code=None,
        eval_id=eval_id,
    )
    return {
        "eval_id": eval_id,
        "winner": winner,
        "scores": judge_scores,
        "criteria": criteria,
        "judge_model": judge_model,
        "logged": True,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_judge_output(raw: str) -> tuple[dict[str, Any], str] | None:
    """Extract (scores, winner) from the judge model's raw text output.

    Returns ``None`` if the output cannot be parsed or fails structural
    validation — the caller must then return an INVALID_INPUT error rather
    than forwarding whatever the judge said.
    """
    # The judge model may wrap the JSON in markdown code fences.
    text = raw.strip()
    if text.startswith("```"):
        # Strip opening fence (e.g. ```json or ```) and closing ```
        lines = text.splitlines()
        # Drop first line (``` or ```json) and trailing ```
        inner_lines = lines[1:]
        while inner_lines and inner_lines[-1].strip().startswith("```"):
            inner_lines.pop()
        text = "\n".join(inner_lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON object inside prose output
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return None
        try:
            data = json.loads(text[start_idx : end_idx + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    scores_raw = data.get("scores")
    if not isinstance(scores_raw, dict) or not scores_raw:
        return None

    # Validate each score entry
    scores: dict[str, Any] = {}
    for model_name, entry in scores_raw.items():
        if not isinstance(entry, dict):
            return None
        score_val = entry.get("score")
        reasoning_val = entry.get("reasoning")
        if not isinstance(score_val, (int, float)):
            return None
        if not isinstance(reasoning_val, str):
            return None
        scores[model_name] = {"score": score_val, "reasoning": reasoning_val}

    # Determine winner: use judge's explicit winner if valid, else compute from scores.
    winner_raw = data.get("winner")
    if isinstance(winner_raw, str) and winner_raw in scores:
        winner = winner_raw
    else:
        # Fall back to highest-scored model
        winner = max(scores, key=lambda m: scores[m]["score"])

    return scores, winner


def _resolve_timeout_ms() -> int:
    raw = os.environ.get("OLLAMA_TIMEOUT_MS", str(_DEFAULT_TIMEOUT_MS))
    try:
        ms = int(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT_MS
    return ms if ms > 0 else _DEFAULT_TIMEOUT_MS


def _duration_ms(start: float) -> int:
    return max(1, int(round((time.monotonic() - start) * 1000)))


def _invalid_input(start: float, tool: str, message: str) -> dict[str, Any]:
    duration_ms = _duration_ms(start)
    log_tool_call(
        tool=tool,
        duration_ms=duration_ms,
        model=None,
        status="error",
        error_code=ErrorCode.INVALID_INPUT.value,
        eval_id=None,
    )
    return {
        "error": make_error(ErrorCode.INVALID_INPUT, message)["error"],
        "duration_ms": duration_ms,
    }
