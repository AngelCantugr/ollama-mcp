"""Tests for the error envelope module."""

import pytest

from ollama_mcp.errors import (
    ErrorCode,
    ErrorEnvelope,
    make_error,
)


def test_all_error_codes_exist() -> None:
    codes = {ec.value for ec in ErrorCode}
    assert codes == {
        "OLLAMA_UNREACHABLE",
        "MODEL_NOT_FOUND",
        "MODEL_TIMEOUT",
        "INVALID_INPUT",
        "DB_ERROR",
    }


@pytest.mark.parametrize(
    "code,expected_retryable",
    [
        (ErrorCode.OLLAMA_UNREACHABLE, True),
        (ErrorCode.MODEL_NOT_FOUND, False),
        (ErrorCode.MODEL_TIMEOUT, True),
        (ErrorCode.INVALID_INPUT, False),
        (ErrorCode.DB_ERROR, True),
    ],
)
def test_make_error_shape(code: ErrorCode, expected_retryable: bool) -> None:
    env: ErrorEnvelope = make_error(code, "test message")

    assert "error" in env
    detail = env["error"]
    assert detail["code"] == code.value
    assert detail["message"] == "test message"
    assert detail["retryable"] is expected_retryable


def test_make_error_returns_typed_dict() -> None:
    env = make_error(ErrorCode.INVALID_INPUT, "bad input")
    # TypedDict is a plain dict at runtime
    assert isinstance(env, dict)
    assert isinstance(env["error"], dict)


def test_error_code_is_string_enum() -> None:
    # Codes compare equal to their string values (used in JSON serialisation)
    assert ErrorCode.OLLAMA_UNREACHABLE == "OLLAMA_UNREACHABLE"
    # With StrEnum, str() returns the value directly
    assert str(ErrorCode.MODEL_NOT_FOUND) == "MODEL_NOT_FOUND"


def test_make_error_all_codes() -> None:
    """Every ErrorCode must produce a valid envelope without raising."""
    for code in ErrorCode:
        env = make_error(code, f"message for {code.value}")
        assert env["error"]["code"] == code.value
