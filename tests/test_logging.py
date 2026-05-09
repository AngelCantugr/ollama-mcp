"""Tests for structured JSON logging."""

import json
import logging

import pytest

from ollama_mcp import logging as mcp_logging


def test_get_logger_returns_logger() -> None:
    logger = mcp_logging.get_logger("test.logger")
    assert isinstance(logger, logging.Logger)


def test_logger_writes_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    logger = mcp_logging.get_logger("test.stderr")
    logger.setLevel(logging.INFO)
    logger.info("hello stderr")
    captured = capsys.readouterr()
    assert captured.out == "", "nothing must go to stdout"
    assert "hello stderr" in captured.err


def test_logger_output_is_json(capsys: pytest.CaptureFixture[str]) -> None:
    logger = mcp_logging.get_logger("test.json_fmt")
    logger.setLevel(logging.INFO)
    logger.info("structured message")
    captured = capsys.readouterr()
    # Each log line should be valid JSON
    for line in captured.err.strip().splitlines():
        parsed = json.loads(line)
        assert "msg" in parsed
        assert "level" in parsed
        assert "ts" in parsed


def test_logger_fields(capsys: pytest.CaptureFixture[str]) -> None:
    logger = mcp_logging.get_logger("test.fields")
    logger.setLevel(logging.DEBUG)
    logger.debug("field test")
    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    parsed = json.loads(line)
    assert parsed["level"] == "debug"
    assert parsed["logger"] == "test.fields"
    assert parsed["msg"] == "field test"


def test_log_level_filtering(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "error")
    # Create a fresh logger so the level is re-evaluated
    logger = mcp_logging.get_logger("test.filter_level")
    # Force level re-evaluation (get_logger honours LOG_LEVEL on creation)
    logger.setLevel(logging.ERROR)
    logger.info("should be suppressed")
    logger.error("should appear")
    captured = capsys.readouterr()
    lines = [line for line in captured.err.strip().splitlines() if line]
    msgs = [json.loads(line)["msg"] for line in lines]
    assert not any("should be suppressed" in m for m in msgs)
    assert any("should appear" in m for m in msgs)


def test_log_tool_call_writes_json(capsys: pytest.CaptureFixture[str]) -> None:
    mcp_logging.log_tool_call(
        tool="run",
        duration_ms=42.5,
        model="llama3",
        status="ok",
        error_code=None,
        eval_id="01HV...",
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    line = captured.err.strip().splitlines()[-1]
    outer = json.loads(line)
    inner = json.loads(outer["msg"])
    assert inner["tool"] == "run"
    assert inner["duration_ms"] == 42.5
    assert inner["model"] == "llama3"
    assert inner["status"] == "ok"
    assert "error_code" not in inner
    assert "eval_id" in inner


def test_log_tool_call_omits_none_fields(capsys: pytest.CaptureFixture[str]) -> None:
    mcp_logging.log_tool_call(
        tool="health",
        duration_ms=1.0,
        model=None,
        status="ok",
        error_code=None,
        eval_id=None,
    )
    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    outer = json.loads(line)
    inner = json.loads(outer["msg"])
    assert "model" not in inner
    assert "error_code" not in inner
    assert "eval_id" not in inner
