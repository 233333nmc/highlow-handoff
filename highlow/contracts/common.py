"""Common request metadata used across process boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Transport-independent correlation and deadline metadata."""

    request_id: str = field(default_factory=lambda: str(uuid4()))
    trace_id: str = ""
    api_version: str = "v1"
    deadline_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not self.api_version:
            raise ValueError("api_version must not be empty")
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive when provided")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))
