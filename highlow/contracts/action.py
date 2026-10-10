"""Policy-neutral action payloads returned through a simulator bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ActionEnvelope:
    episode_id: str
    step_id: int
    schema_id: str
    data: Any = field(repr=False)
    horizon: int = 1
    coordinate_frame: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        if not self.schema_id:
            raise ValueError("action schema_id must not be empty")
        if self.horizon < 1:
            raise ValueError("action horizon must be at least one")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
