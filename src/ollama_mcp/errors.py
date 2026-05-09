import enum
from typing import TypedDict


class ErrorCode(enum.StrEnum):
    OLLAMA_UNREACHABLE = "OLLAMA_UNREACHABLE"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    INVALID_INPUT = "INVALID_INPUT"
    DB_ERROR = "DB_ERROR"


RETRYABLE: dict[ErrorCode, bool] = {
    ErrorCode.OLLAMA_UNREACHABLE: True,
    ErrorCode.MODEL_NOT_FOUND: False,
    ErrorCode.MODEL_TIMEOUT: True,
    ErrorCode.INVALID_INPUT: False,
    ErrorCode.DB_ERROR: True,  # transient lock contention is plausible
}


class ErrorDetail(TypedDict):
    code: str
    message: str
    retryable: bool


class ErrorEnvelope(TypedDict):
    error: ErrorDetail


def make_error(code: ErrorCode, message: str) -> ErrorEnvelope:
    return {"error": {"code": code.value, "message": message, "retryable": RETRYABLE[code]}}
