"""Structured errors shared by inference transports."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping


class InferenceServiceError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        status_code: int = 400,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        if not code:
            raise ValueError("error code must not be empty")
        self.code = code
        self.message = message
        self.retryable = bool(retryable)
        self.status_code = int(status_code)
        self.details = MappingProxyType(dict(details or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "details": dict(self.details),
            }
        }
