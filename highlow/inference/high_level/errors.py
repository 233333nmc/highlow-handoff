"""Errors raised by the framework-free high-level inference boundary."""

from __future__ import annotations


class HighLevelError(RuntimeError):
    """Base class for high-level inference failures."""


class UnsupportedVersionError(HighLevelError):
    """A requested prompt or response-schema version is not registered."""


class HighLevelValidationError(HighLevelError, ValueError):
    """A high-level request or structured model result is invalid."""


class BackendProtocolError(HighLevelError):
    """A model backend returned a response that violates its protocol."""


class BackendTransportError(HighLevelError):
    """A model backend could not be reached or returned an HTTP failure."""


__all__ = [
    "BackendProtocolError",
    "BackendTransportError",
    "HighLevelError",
    "HighLevelValidationError",
    "UnsupportedVersionError",
]
