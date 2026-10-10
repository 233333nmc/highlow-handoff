"""Generic episode lifecycle values passed through a simulator bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class EpisodeContext:
    task_id: str
    instruction: str
    episode_id: str = field(default_factory=lambda: str(uuid4()))
    seed: int | None = None
    simulator: str = ""
    simulator_version: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if not self.task_id:
            raise ValueError("task_id must not be empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    episode_id: str
    success: bool | None = None
    terminated: bool = False
    truncated: bool = False
    step_count: int = 0
    failure_reason: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_count < 0:
            raise ValueError("step_count must be non-negative")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
