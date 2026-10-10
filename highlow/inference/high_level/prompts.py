"""Versioned planner/verifier prompts, JSON schemas, and validators."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .contracts import (
    HighLevelOperation,
    HighLevelOutput,
    HighLevelRequest,
    PlannerOutput,
    VerifierDecision,
    VerifierOutput,
    VerifierStatus,
    coerce_operation,
)
from .errors import HighLevelValidationError, UnsupportedVersionError


PLANNER_PROMPT_V1 = "planner.v1"
PLANNER_SCHEMA_V1 = "planner.v1"
VERIFIER_PROMPT_V1 = "verifier.v1"
VERIFIER_SCHEMA_V1 = "verifier.v1"


PLANNER_JSON_SCHEMA_V1: Mapping[str, Any] = MappingProxyType(
    {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "HighLevelPlannerV1",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "subtasks": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["subtasks"],
    }
)


VERIFIER_JSON_SCHEMA_V1: Mapping[str, Any] = MappingProxyType(
    {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "HighLevelVerifierV1",
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decision": {"const": "same"},
                    "status": {
                        "type": "string",
                        "enum": ["running", "failed", "stuck", "uncertain"],
                    },
                    "next_subtask": {"const": ""},
                    "evidence": {"type": "string"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": [
                    "decision",
                    "status",
                    "next_subtask",
                    "evidence",
                ],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decision": {"const": "switch"},
                    "status": {"const": "completed"},
                    "next_subtask": {"type": "string", "minLength": 1},
                    "evidence": {"type": "string"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": [
                    "decision",
                    "status",
                    "next_subtask",
                    "evidence",
                ],
            },
        ],
    }
)


Renderer = Callable[[HighLevelRequest], str]
Validator = Callable[[Mapping[str, Any], HighLevelRequest], HighLevelOutput]
ResponseSchemaResolver = Callable[[HighLevelRequest], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class PromptSchemaSpec:
    operation: HighLevelOperation
    prompt_version: str
    schema_version: str
    schema_name: str
    response_schema: Mapping[str, Any]
    renderer: Renderer
    validator: Validator
    response_schema_resolver: ResponseSchemaResolver | None = None

    def response_schema_for(
        self, request: HighLevelRequest
    ) -> Mapping[str, Any]:
        """Return the response schema specialization for one request."""

        if self.response_schema_resolver is None:
            return self.response_schema
        return self.response_schema_resolver(request)


def _json_context(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise HighLevelValidationError(
            "structured_context must contain JSON-serializable values"
        ) from exc


def _frame_summary(request: HighLevelRequest) -> str:
    descriptors = [
        {
            "frame_id": frame.frame_id,
            "timestamp_s": frame.timestamp_s,
            "source": frame.source,
        }
        for frame in request.frames
    ]
    return json.dumps(descriptors, ensure_ascii=False, separators=(",", ":"))


def _planner_mode(request: HighLevelRequest) -> str:
    value = str(request.structured_context.get("planner_mode", "next")).lower()
    return "all" if "all" in value else "next"


def planner_response_schema_v1(
    request: HighLevelRequest,
) -> Mapping[str, Any]:
    """Specialize planner.v1 cardinality without mutating its base schema."""

    if request.operation is not HighLevelOperation.PLANNER:
        raise HighLevelValidationError(
            "planner.v1 response schema requires operation='planner'"
        )
    if _planner_mode(request) == "all":
        return PLANNER_JSON_SCHEMA_V1

    properties = dict(PLANNER_JSON_SCHEMA_V1["properties"])
    subtasks = dict(properties["subtasks"])
    subtasks["maxItems"] = 1
    properties["subtasks"] = subtasks
    return MappingProxyType(
        {
            **dict(PLANNER_JSON_SCHEMA_V1),
            "properties": MappingProxyType(properties),
        }
    )


def render_planner_v1(request: HighLevelRequest) -> str:
    if request.operation is not HighLevelOperation.PLANNER:
        raise HighLevelValidationError("planner.v1 requires operation='planner'")
    planner_mode = _planner_mode(request)
    cardinality = (
        "Generate the complete short ordered list of remaining subtasks."
        if planner_mode == "all"
        else "Generate exactly one next subtask."
    )
    return (
        "You are the high-level planner for a robot manipulation system.\n"
        f"Prompt version: {PLANNER_PROMPT_V1}\n"
        f"Response schema version: {PLANNER_SCHEMA_V1}\n"
        f"Task instruction: {request.task_instruction}\n"
        f"Planner mode: {planner_mode}\n"
        f"Structured progress context: {_json_context(request.structured_context)}\n"
        f"Available visual evidence: {_frame_summary(request)}\n"
        f"{cardinality}\n"
        "Use object-level, directly executable phases that include the intended "
        "target state. Do not split a phase into grasp/lift/move/release motor "
        "primitives. Preserve dependency order. Return only the JSON object "
        "required by the response schema, without Markdown or explanation."
    )


def render_verifier_v1(request: HighLevelRequest) -> str:
    if request.operation is not HighLevelOperation.VERIFIER:
        raise HighLevelValidationError("verifier.v1 requires operation='verifier'")
    current_subtask = str(
        request.structured_context.get("current_subtask", "")
    ).strip()
    if not current_subtask:
        raise HighLevelValidationError(
            "verifier.v1 requires structured_context['current_subtask']"
        )
    return (
        "You are a completion verifier for one robot manipulation subtask.\n"
        f"Prompt version: {VERIFIER_PROMPT_V1}\n"
        f"Response schema version: {VERIFIER_SCHEMA_V1}\n"
        f"Task instruction: {request.task_instruction}\n"
        f"Current low-level subtask: {current_subtask}\n"
        f"Structured execution context: {_json_context(request.structured_context)}\n"
        f"Ordered visual evidence: {_frame_summary(request)}\n"
        "Judge the visible goal state of the current subtask. Use decision='same' "
        "when it should keep running. Use decision='switch' only when the evidence "
        "shows that the current subtask is completed. Status must be one of "
        "running/completed/failed/stuck/uncertain. A switch requires status='completed' "
        "and a non-empty next_subtask. Decision='same' must not use completed status "
        "and must leave next_subtask empty. Give one short evidence statement. "
        "Return only the JSON object required by the response schema, without "
        "Markdown or explanation."
    )


def _confidence(payload: Mapping[str, Any]) -> float | None:
    value = payload.get("confidence")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HighLevelValidationError("confidence must be a number")
    confidence = float(value)
    if not 0 <= confidence <= 1:
        raise HighLevelValidationError("confidence must be in [0, 1]")
    return confidence


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise HighLevelValidationError(
            f"structured output contains unknown fields: {', '.join(unknown)}"
        )


def validate_planner_v1(
    payload: Mapping[str, Any], request: HighLevelRequest
) -> PlannerOutput:
    _reject_unknown(payload, {"subtasks", "confidence"})
    subtasks = payload.get("subtasks")
    if not isinstance(subtasks, list) or not subtasks:
        raise HighLevelValidationError("planner.v1 requires a non-empty subtasks list")
    if any(not isinstance(item, str) or not item.strip() for item in subtasks):
        raise HighLevelValidationError("every planner subtask must be a non-empty string")
    cleaned = tuple(item.strip() for item in subtasks)
    if _planner_mode(request) == "next" and len(cleaned) != 1:
        raise HighLevelValidationError(
            "planner.v1 in next mode must return exactly one subtask"
        )
    return PlannerOutput(subtasks=cleaned, confidence=_confidence(payload))


def validate_verifier_v1(
    payload: Mapping[str, Any], request: HighLevelRequest
) -> VerifierOutput:
    del request
    _reject_unknown(
        payload,
        {"decision", "status", "next_subtask", "evidence", "confidence"},
    )
    required = ("decision", "status", "next_subtask", "evidence")
    missing = [key for key in required if key not in payload]
    if missing:
        raise HighLevelValidationError(
            f"verifier.v1 is missing required fields: {', '.join(missing)}"
        )
    if not isinstance(payload["next_subtask"], str):
        raise HighLevelValidationError("next_subtask must be a string")
    if not isinstance(payload["evidence"], str):
        raise HighLevelValidationError("evidence must be a string")
    try:
        decision = VerifierDecision(payload["decision"])
        status = VerifierStatus(payload["status"])
    except (TypeError, ValueError) as exc:
        raise HighLevelValidationError(
            "verifier decision or status is outside the v1 enum"
        ) from exc
    next_subtask = payload["next_subtask"].strip()
    if decision is VerifierDecision.SWITCH:
        if status is not VerifierStatus.COMPLETED:
            raise HighLevelValidationError(
                "verifier switch requires status='completed'"
            )
        if not next_subtask:
            raise HighLevelValidationError(
                "verifier switch requires a non-empty next_subtask"
            )
    else:
        if status is VerifierStatus.COMPLETED:
            raise HighLevelValidationError(
                "verifier completed status requires decision='switch'"
            )
        if next_subtask:
            raise HighLevelValidationError(
                "verifier same decision requires an empty next_subtask"
            )
    return VerifierOutput(
        decision=decision,
        status=status,
        next_subtask=next_subtask,
        evidence=payload["evidence"],
        confidence=_confidence(payload),
    )


class PromptSchemaRegistry:
    """Registry that resolves an explicit operation/prompt/schema triple."""

    def __init__(self) -> None:
        self._specs: dict[tuple[HighLevelOperation, str, str], PromptSchemaSpec] = {}
        self._defaults: dict[HighLevelOperation, tuple[str, str]] = {}

    def register(self, spec: PromptSchemaSpec, *, default: bool = False) -> None:
        key = (spec.operation, spec.prompt_version, spec.schema_version)
        if key in self._specs:
            raise ValueError(f"prompt/schema spec is already registered: {key!r}")
        self._specs[key] = spec
        if default or spec.operation not in self._defaults:
            self._defaults[spec.operation] = (
                spec.prompt_version,
                spec.schema_version,
            )

    def resolve(
        self,
        operation: HighLevelOperation | str,
        prompt_version: str = "",
        schema_version: str = "",
    ) -> PromptSchemaSpec:
        op = coerce_operation(operation)
        try:
            default_prompt, default_schema = self._defaults[op]
        except KeyError as exc:
            raise UnsupportedVersionError(
                f"no prompt/schema versions are registered for {op.value}"
            ) from exc
        key = (
            op,
            prompt_version or default_prompt,
            schema_version or default_schema,
        )
        try:
            return self._specs[key]
        except KeyError as exc:
            raise UnsupportedVersionError(
                "unsupported high-level prompt/schema combination: "
                f"operation={op.value!r}, prompt={key[1]!r}, schema={key[2]!r}"
            ) from exc

    def prompt_versions(self) -> Mapping[str, tuple[str, ...]]:
        result: dict[str, set[str]] = {}
        for operation, prompt_version, _ in self._specs:
            result.setdefault(operation.value, set()).add(prompt_version)
        return MappingProxyType(
            {key: tuple(sorted(values)) for key, values in result.items()}
        )

    def schema_versions(self) -> Mapping[str, tuple[str, ...]]:
        result: dict[str, set[str]] = {}
        for operation, _, schema_version in self._specs:
            result.setdefault(operation.value, set()).add(schema_version)
        return MappingProxyType(
            {key: tuple(sorted(values)) for key, values in result.items()}
        )


def default_prompt_schema_registry() -> PromptSchemaRegistry:
    registry = PromptSchemaRegistry()
    registry.register(
        PromptSchemaSpec(
            operation=HighLevelOperation.PLANNER,
            prompt_version=PLANNER_PROMPT_V1,
            schema_version=PLANNER_SCHEMA_V1,
            schema_name="high_level_planner_v1",
            response_schema=PLANNER_JSON_SCHEMA_V1,
            renderer=render_planner_v1,
            validator=validate_planner_v1,
            response_schema_resolver=planner_response_schema_v1,
        ),
        default=True,
    )
    registry.register(
        PromptSchemaSpec(
            operation=HighLevelOperation.VERIFIER,
            prompt_version=VERIFIER_PROMPT_V1,
            schema_version=VERIFIER_SCHEMA_V1,
            schema_name="high_level_verifier_v1",
            response_schema=VERIFIER_JSON_SCHEMA_V1,
            renderer=render_verifier_v1,
            validator=validate_verifier_v1,
        ),
        default=True,
    )
    return registry


__all__ = [
    "PLANNER_JSON_SCHEMA_V1",
    "PLANNER_PROMPT_V1",
    "PLANNER_SCHEMA_V1",
    "PromptSchemaRegistry",
    "PromptSchemaSpec",
    "VERIFIER_JSON_SCHEMA_V1",
    "VERIFIER_PROMPT_V1",
    "VERIFIER_SCHEMA_V1",
    "default_prompt_schema_registry",
    "planner_response_schema_v1",
    "render_planner_v1",
    "render_verifier_v1",
    "validate_planner_v1",
    "validate_verifier_v1",
]
