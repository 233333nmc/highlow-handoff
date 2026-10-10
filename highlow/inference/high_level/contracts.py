"""Framework- and simulator-neutral high-level inference contracts.

The public contracts in this module are deliberately small dataclasses.  They
can cross a process boundary without importing a model framework, and they keep
episode correlation separate from planner/verifier semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from highlow.contracts import EpisodeContext, RequestContext
from highlow.inference.common import HealthState


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _tuple_of_strings(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


class HighLevelOperation(str, Enum):
    """The two semantic operations supported by the high-level service."""

    PLANNER = "planner"
    VERIFIER = "verifier"


def coerce_operation(value: HighLevelOperation | str) -> HighLevelOperation:
    if isinstance(value, HighLevelOperation):
        return value
    normalized = str(value).strip().lower()
    aliases = {
        "plan": HighLevelOperation.PLANNER,
        "planner": HighLevelOperation.PLANNER,
        "verify": HighLevelOperation.VERIFIER,
        "verifier": HighLevelOperation.VERIFIER,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(
            f"unsupported high-level operation {value!r}; expected one of {allowed}"
        ) from exc


class VerifierDecision(str, Enum):
    SAME = "same"
    SWITCH = "switch"


class VerifierStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STUCK = "stuck"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class HighLevelFrame:
    """One ordered visual-evidence item supplied by the orchestrator.

    ``data`` is either encoded image bytes, an HTTP(S) URL, or a ``data:`` URL.
    Frame selection and ordering belong to the orchestrator; a backend only
    converts this value to its transport's multimodal representation.
    """

    data: bytes | str = field(repr=False)
    mime_type: str = "image/jpeg"
    frame_id: int | None = None
    timestamp_s: float | None = None
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, (bytes, str)) or not self.data:
            raise ValueError("frame data must be non-empty bytes or a URL string")
        if not self.mime_type.startswith("image/"):
            raise ValueError("frame mime_type must start with 'image/'")
        if self.frame_id is not None and self.frame_id < 0:
            raise ValueError("frame_id must be non-negative when provided")
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Backend-neutral generation controls."""

    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least one")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        object.__setattr__(self, "extra", _frozen_mapping(self.extra))


@dataclass(frozen=True, slots=True)
class HighLevelRequest:
    """Semantic request accepted by :class:`HighLevelService`.

    The service derives the final prompt and response schema from ``operation``
    and the requested versions.  ``structured_context`` intentionally remains
    generic: planner v1 understands keys such as ``planner_mode``,
    ``current_subtask`` and ``completed_subtasks``; verifier v1 requires
    ``current_subtask``.  Future versions may interpret different keys without
    changing this process-boundary contract.
    """

    request_context: RequestContext
    operation: HighLevelOperation | str
    episode_id: str = ""
    episode_context: EpisodeContext | None = None
    task_instruction: str = ""
    structured_context: Mapping[str, Any] = field(default_factory=dict)
    frames: tuple[HighLevelFrame, ...] = ()
    prompt_version: str = ""
    schema_version: str = ""
    generation: GenerationOptions = field(default_factory=GenerationOptions)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operation = coerce_operation(self.operation)
        object.__setattr__(self, "operation", operation)

        episode_id = str(self.episode_id).strip()
        task_instruction = str(self.task_instruction).strip()
        if self.episode_context is not None:
            if episode_id and episode_id != self.episode_context.episode_id:
                raise ValueError(
                    "episode_id does not match episode_context.episode_id"
                )
            episode_id = episode_id or self.episode_context.episode_id
            task_instruction = task_instruction or self.episode_context.instruction.strip()
        if not episode_id:
            raise ValueError("episode_id or episode_context is required")
        if not task_instruction:
            raise ValueError("task_instruction or episode_context.instruction is required")

        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "task_instruction", task_instruction)
        object.__setattr__(
            self, "structured_context", _frozen_mapping(self.structured_context)
        )
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class PlannerOutput:
    subtasks: tuple[str, ...]
    confidence: float | None = None

    def __post_init__(self) -> None:
        subtasks = tuple(str(item).strip() for item in self.subtasks)
        if not subtasks or any(not item for item in subtasks):
            raise ValueError("planner output must contain non-empty subtasks")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("planner confidence must be in [0, 1]")
        object.__setattr__(self, "subtasks", subtasks)


@dataclass(frozen=True, slots=True)
class VerifierOutput:
    decision: VerifierDecision | str
    status: VerifierStatus | str
    next_subtask: str = ""
    evidence: str = ""
    confidence: float | None = None

    def __post_init__(self) -> None:
        decision = VerifierDecision(self.decision)
        status = VerifierStatus(self.status)
        next_subtask = str(self.next_subtask).strip()
        if decision is VerifierDecision.SWITCH:
            if status is not VerifierStatus.COMPLETED:
                raise ValueError("verifier switch requires completed status")
            if not next_subtask:
                raise ValueError("verifier switch requires a next_subtask")
        else:
            if status is VerifierStatus.COMPLETED:
                raise ValueError("completed verifier status requires a switch")
            if next_subtask:
                raise ValueError("same verifier decision requires an empty next_subtask")
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "next_subtask", next_subtask)
        object.__setattr__(self, "evidence", str(self.evidence).strip())
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("verifier confidence must be in [0, 1]")


HighLevelOutput = PlannerOutput | VerifierOutput


@dataclass(frozen=True, slots=True)
class HighLevelResponse:
    request_context: RequestContext
    episode_id: str
    operation: HighLevelOperation | str
    output: HighLevelOutput
    prompt_version: str
    schema_version: str
    raw_text: str = field(default="", repr=False)
    backend: str = ""
    model: str = ""
    finish_reason: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)
    timing_ms: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operation = coerce_operation(self.operation)
        if operation is HighLevelOperation.PLANNER and not isinstance(
            self.output, PlannerOutput
        ):
            raise ValueError("planner response requires PlannerOutput")
        if operation is HighLevelOperation.VERIFIER and not isinstance(
            self.output, VerifierOutput
        ):
            raise ValueError("verifier response requires VerifierOutput")
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if not self.prompt_version or not self.schema_version:
            raise ValueError("response prompt/schema versions must not be empty")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "usage", _frozen_mapping(self.usage))
        object.__setattr__(
            self,
            "timing_ms",
            MappingProxyType({str(k): float(v) for k, v in self.timing_ms.items()}),
        )
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class BackendInferenceRequest:
    """Rendered prompt and schema passed from the service to a model backend."""

    request_context: RequestContext
    operation: HighLevelOperation | str
    prompt: str
    response_schema: Mapping[str, Any]
    schema_name: str
    frames: tuple[HighLevelFrame, ...] = ()
    generation: GenerationOptions = field(default_factory=GenerationOptions)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("backend prompt must not be empty")
        if not self.schema_name.strip():
            raise ValueError("backend schema_name must not be empty")
        object.__setattr__(self, "operation", coerce_operation(self.operation))
        object.__setattr__(self, "response_schema", _frozen_mapping(self.response_schema))
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class BackendInferenceResponse:
    """Raw and parsed structured output returned by a model backend."""

    raw_text: str
    structured_output: Mapping[str, Any]
    backend: str
    model: str
    finish_reason: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)
    timing_ms: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.structured_output, Mapping):
            raise TypeError("structured_output must be a mapping")
        object.__setattr__(
            self, "structured_output", _frozen_mapping(self.structured_output)
        )
        object.__setattr__(self, "usage", _frozen_mapping(self.usage))
        object.__setattr__(
            self,
            "timing_ms",
            MappingProxyType({str(k): float(v) for k, v in self.timing_ms.items()}),
        )
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class HighLevelCapabilities:
    component: str
    backend: str
    model: str
    protocol: str
    operations: tuple[HighLevelOperation | str, ...]
    prompt_versions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    schema_versions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    supports_images: bool = True
    supports_structured_output: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operations = tuple(coerce_operation(item) for item in self.operations)
        prompt_versions = {
            str(key): _tuple_of_strings(value)
            for key, value in self.prompt_versions.items()
        }
        schema_versions = {
            str(key): _tuple_of_strings(value)
            for key, value in self.schema_versions.items()
        }
        object.__setattr__(self, "operations", operations)
        object.__setattr__(
            self, "prompt_versions", MappingProxyType(prompt_versions)
        )
        object.__setattr__(
            self, "schema_versions", MappingProxyType(schema_versions)
        )
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class HealthStatus:
    state: HealthState | str
    component: str
    checked_at_s: float
    detail: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", HealthState(self.state))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def healthy(self) -> bool:
        return self.state is not HealthState.UNAVAILABLE


__all__ = [
    "BackendInferenceRequest",
    "BackendInferenceResponse",
    "GenerationOptions",
    "HealthState",
    "HealthStatus",
    "HighLevelCapabilities",
    "HighLevelFrame",
    "HighLevelOperation",
    "HighLevelOutput",
    "HighLevelRequest",
    "HighLevelResponse",
    "PlannerOutput",
    "VerifierDecision",
    "VerifierOutput",
    "VerifierStatus",
    "coerce_operation",
]
