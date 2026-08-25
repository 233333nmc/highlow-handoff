"""Simulator-neutral observation containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Modality:
    """One named sensor or state value with an explicit schema."""

    name: str
    schema_id: str
    data: Any = field(repr=False)
    timestamp_s: float | None = None
    encoding: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("modality name must not be empty")
        if not self.schema_id:
            raise ValueError("modality schema_id must not be empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ObservationEnvelope:
    episode_id: str
    step_id: int
    timestamp_s: float
    modalities: Mapping[str, Modality]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")
        modalities = dict(self.modalities)
        for key, modality in modalities.items():
            if key != modality.name:
                raise ValueError(
                    f"modality key {key!r} does not match modality name {modality.name!r}"
                )
        object.__setattr__(self, "modalities", MappingProxyType(modalities))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def require(self, name: str) -> Modality:
        try:
            return self.modalities[name]
        except KeyError as exc:
            raise KeyError(f"required observation modality {name!r} is missing") from exc
