"""Framework-free contracts for low-level policy inference.

This module deliberately contains only standard-library and ``highlow.contracts``
imports.  Model-framework integration belongs in ``low_level.plugins``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from highlow.contracts import (
    ActionEnvelope,
    EpisodeContext,
    EpisodeResult,
    ObservationEnvelope,
    RequestContext,
)


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _validated_strings(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError(f"{field_name} entries must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} entries must be unique")
    return normalized


class ContextMode(str, Enum):
    """How backend state is scoped across requests."""

    STATELESS = "stateless"
    EPISODE = "episode"


class TemporalMode(str, Enum):
    """Whether predictions consume only the current step or temporal state."""

    SINGLE_STEP = "single_step"
    HISTORY = "history"


class PredictionMode(str, Enum):
    """Native prediction granularity of a policy backend."""

    SINGLE_ACTION = "single_action"
    ACTION_CHUNK = "action_chunk"


def _enum_value(enum_type: type[Enum], value: Any, field_name: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(repr(member.value) for member in enum_type)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


@dataclass(frozen=True, slots=True)
class PolicyManifest:
    """Describes a policy without importing its model framework.

    Schema identifiers describe payload semantics, while modality names describe
    where those payloads must appear in an :class:`ObservationEnvelope`.
    """

    policy_id: str
    policy_version: str
    backend_id: str = ""
    context_mode: ContextMode = ContextMode.EPISODE
    temporal_mode: TemporalMode = TemporalMode.SINGLE_STEP
    prediction_mode: PredictionMode = PredictionMode.SINGLE_ACTION
    supported_api_versions: tuple[str, ...] = ("v1",)
    observation_schema_ids: tuple[str, ...] = ()
    action_schema_ids: tuple[str, ...] = ()
    required_modalities: tuple[str, ...] = ()
    optional_modalities: tuple[str, ...] = ()
    max_concurrent_episodes: int = 1
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("policy_id must not be empty")
        if not isinstance(self.policy_version, str) or not self.policy_version:
            raise ValueError("policy_version must not be empty")
        backend_id = self.backend_id or self.policy_id
        if not isinstance(backend_id, str) or not backend_id:
            raise ValueError("backend_id must not be empty")
        if self.max_concurrent_episodes < 1:
            raise ValueError("max_concurrent_episodes must be at least one")

        api_versions = _validated_strings(
            self.supported_api_versions, "supported_api_versions"
        )
        if not api_versions:
            raise ValueError("supported_api_versions must not be empty")
        observation_schemas = _validated_strings(
            self.observation_schema_ids, "observation_schema_ids"
        )
        action_schemas = _validated_strings(
            self.action_schema_ids, "action_schema_ids"
        )
        required = _validated_strings(self.required_modalities, "required_modalities")
        optional = _validated_strings(self.optional_modalities, "optional_modalities")
        overlap = set(required).intersection(optional)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"modalities cannot be both required and optional: {names}")

        object.__setattr__(self, "backend_id", backend_id)
        object.__setattr__(
            self,
            "context_mode",
            _enum_value(ContextMode, self.context_mode, "context_mode"),
        )
        object.__setattr__(
            self,
            "temporal_mode",
            _enum_value(TemporalMode, self.temporal_mode, "temporal_mode"),
        )
        object.__setattr__(
            self,
            "prediction_mode",
            _enum_value(PredictionMode, self.prediction_mode, "prediction_mode"),
        )
        object.__setattr__(self, "supported_api_versions", api_versions)
        object.__setattr__(self, "observation_schema_ids", observation_schemas)
        object.__setattr__(self, "action_schema_ids", action_schemas)
        object.__setattr__(self, "required_modalities", required)
        object.__setattr__(self, "optional_modalities", optional)
        object.__setattr__(self, "capabilities", _frozen_mapping(self.capabilities))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class LowLevelInferenceRequest:
    """One canonical observation plus request-scoped policy conditioning."""

    observation: ObservationEnvelope
    conditioning: Mapping[str, Any] = field(default_factory=dict)
    context: RequestContext = field(default_factory=RequestContext)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ObservationEnvelope):
            raise TypeError("observation must be an ObservationEnvelope")
        if not isinstance(self.context, RequestContext):
            raise TypeError("context must be a RequestContext")
        object.__setattr__(self, "conditioning", _frozen_mapping(self.conditioning))

    @property
    def episode_id(self) -> str:
        return self.observation.episode_id

    @property
    def step_id(self) -> int:
        return self.observation.step_id

    @property
    def request_context(self) -> RequestContext:
        """Compatibility alias that makes the metadata role explicit."""

        return self.context


@dataclass(frozen=True, slots=True)
class PolicyPrediction:
    """Backend result with a bridge-ready action and diagnostic metadata."""

    action: ActionEnvelope
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action, ActionEnvelope):
            raise TypeError("action must be an ActionEnvelope")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    def to_action_envelope(self) -> ActionEnvelope:
        return self.action


@runtime_checkable
class Backend(Protocol):
    """Lifecycle and inference interface implemented by low-level plugins.

    Episode lifecycle calls must be idempotent for an identical
    :class:`EpisodeContext`.  The service can resume a multi-phase reset after
    a failure, but a call that raises may already have changed backend state.
    Idempotency makes retrying that uncertain phase safe.
    """

    @property
    def manifest(self) -> PolicyManifest:
        ...

    def open(self) -> None:
        ...

    def begin_episode(self, episode: EpisodeContext) -> None:
        ...

    def infer(
        self, request: LowLevelInferenceRequest
    ) -> PolicyPrediction | ActionEnvelope:
        ...

    def end_episode(
        self,
        episode: EpisodeContext,
        result: EpisodeResult | None = None,
    ) -> None:
        ...

    def close(self) -> None:
        ...


class LowLevelError(RuntimeError):
    """Base exception for the generic low-level inference layer."""


class LifecycleError(LowLevelError):
    """Raised when a service or episode operation occurs in the wrong state."""


class PredictionContractError(LowLevelError):
    """Raised when a backend result does not match its request or manifest."""


__all__ = [
    "Backend",
    "ContextMode",
    "LifecycleError",
    "LowLevelError",
    "LowLevelInferenceRequest",
    "PolicyManifest",
    "PolicyPrediction",
    "PredictionMode",
    "PredictionContractError",
    "TemporalMode",
]
