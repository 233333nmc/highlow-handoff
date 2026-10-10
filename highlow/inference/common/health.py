"""Transport-neutral health responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class HealthState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class HealthReport:
    service: str
    state: HealthState
    version: str = "v1"
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.service:
            raise ValueError("service name must not be empty")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @property
    def ready(self) -> bool:
        return self.state is HealthState.READY
