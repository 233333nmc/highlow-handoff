"""Small JSON conversion helper for framework-free service adapters."""

from __future__ import annotations

import base64
import binascii
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def to_jsonable(value: Any) -> Any:
    """Convert service results to JSON-compatible values.

    Core contracts intentionally allow native tensors and arrays for efficient
    in-process use.  Network transports may use this conservative conversion;
    high-throughput deployments can replace it with a binary codec without
    changing service or backend interfaces.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "data": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return to_jsonable(tolist())
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def from_jsonable(value: Any) -> Any:
    """Decode values emitted by :func:`to_jsonable`.

    Only the exact two-key base64 envelope produced for ``bytes`` is treated
    specially.  Other mappings are traversed without imposing a schema, so
    service-specific decoders can safely apply this at opaque payload fields.
    """

    if isinstance(value, Mapping):
        if set(value) == {"encoding", "data"} and value.get("encoding") == "base64":
            encoded = value.get("data")
            if not isinstance(encoded, str):
                raise ValueError("base64 data must be a string")
            try:
                return base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("invalid base64 data") from exc
        return {str(key): from_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [from_jsonable(item) for item in value]
    return value


__all__ = ["from_jsonable", "to_jsonable"]
