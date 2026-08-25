"""Pure-ASGI JSON transport for :mod:`highlow.inference.low_level`."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from highlow.contracts import (
    EpisodeContext,
    EpisodeResult,
    Modality,
    ObservationEnvelope,
    RequestContext,
)
from highlow.inference.common import (
    HealthReport,
    HealthState,
    InferenceServiceError,
    JsonAsgiApp,
    Route,
    from_jsonable,
)

from .core import LifecycleError, LowLevelInferenceRequest, PredictionContractError
from .service import LowLevelService


def _object(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a JSON object")
    return value


def _optional_object(
    payload: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    return _object(payload.get(field_name, {}), field_name)


def _request_context(value: Mapping[str, Any]) -> RequestContext:
    kwargs: dict[str, Any] = {
        "trace_id": value.get("trace_id", ""),
        "api_version": value.get("api_version", "v1"),
        "deadline_ms": value.get("deadline_ms"),
        "metadata": _object(value.get("metadata", {}), "context.metadata"),
    }
    if "request_id" in value:
        kwargs["request_id"] = value["request_id"]
    return RequestContext(**kwargs)


def _episode_context(payload: Mapping[str, Any]) -> EpisodeContext:
    kwargs: dict[str, Any] = {
        "task_id": payload["task_id"],
        "instruction": payload.get("instruction", ""),
        "seed": payload.get("seed"),
        "simulator": payload.get("simulator", ""),
        "simulator_version": payload.get("simulator_version", ""),
        "metadata": _object(payload.get("metadata", {}), "metadata"),
    }
    if "episode_id" in payload:
        kwargs["episode_id"] = payload["episode_id"]
    return EpisodeContext(**kwargs)


def _episode_result(value: Mapping[str, Any], episode_id: str) -> EpisodeResult:
    supplied_episode_id = value.get("episode_id", episode_id)
    if supplied_episode_id != episode_id:
        raise ValueError("result.episode_id must match episode_id")
    return EpisodeResult(
        episode_id=episode_id,
        success=value.get("success"),
        terminated=value.get("terminated", False),
        truncated=value.get("truncated", False),
        step_count=value.get("step_count", 0),
        failure_reason=value.get("failure_reason", ""),
        metrics=_object(value.get("metrics", {}), "result.metrics"),
    )


def _observation(value: Mapping[str, Any]) -> ObservationEnvelope:
    modality_values = _object(value["modalities"], "observation.modalities")
    modalities: dict[str, Modality] = {}
    for name, raw_modality in modality_values.items():
        modality = _object(raw_modality, f"observation.modalities.{name}")
        declared_name = modality.get("name", name)
        if declared_name != name:
            raise ValueError(f"modality {name!r} declares a different name")
        modalities[name] = Modality(
            name=name,
            schema_id=modality["schema_id"],
            data=from_jsonable(modality.get("data")),
            timestamp_s=modality.get("timestamp_s"),
            encoding=modality.get("encoding", ""),
            metadata=_object(
                modality.get("metadata", {}),
                f"observation.modalities.{name}.metadata",
            ),
        )
    return ObservationEnvelope(
        episode_id=value["episode_id"],
        step_id=value["step_id"],
        timestamp_s=value["timestamp_s"],
        modalities=modalities,
        metadata=_object(value.get("metadata", {}), "observation.metadata"),
    )


def _translate_errors(handler: Callable[[Mapping[str, Any]], Any]):
    def translated(payload: Mapping[str, Any]) -> Any:
        try:
            return handler(payload)
        except LifecycleError as exc:
            raise InferenceServiceError(
                "INVALID_LIFECYCLE",
                str(exc),
                status_code=409,
            ) from exc
        except PredictionContractError as exc:
            raise InferenceServiceError(
                "PREDICTION_CONTRACT_ERROR",
                str(exc),
                status_code=422,
            ) from exc

    return translated


def create_app(service: LowLevelService) -> JsonAsgiApp:
    """Create a dependency-free ASGI app around one low-level service."""

    if not isinstance(service, LowLevelService):
        raise TypeError("service must be a LowLevelService")

    def health(_: Mapping[str, Any]) -> HealthReport:
        if service.is_closed:
            state = HealthState.UNAVAILABLE
        elif not service.is_open:
            state = HealthState.STARTING
        else:
            state = HealthState.READY
        return HealthReport(
            service="low-level-service",
            state=state,
            details={
                "backend_id": service.manifest.backend_id,
                "backend_open": service.is_open,
                "active_episode_count": len(service.active_contexts),
            },
        )

    def capabilities(_: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "component": "low-level-service",
            "manifest": service.manifest,
            "active_episode_ids": tuple(service.active_contexts),
        }

    def open_episode(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        episode_payload = payload.get("episode", payload)
        episode = _episode_context(_object(episode_payload, "episode"))
        service.open_context(episode)
        return {"episode_id": episode.episode_id, "state": "open"}

    def close_episode(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        episode_id = payload["episode_id"]
        raw_result = payload.get("result")
        if raw_result is None and any(
            field in payload
            for field in (
                "success",
                "terminated",
                "truncated",
                "step_count",
                "failure_reason",
                "metrics",
            )
        ):
            # JsonHttpInferenceClient deliberately sends opaque dataclasses
            # directly.  Accept a top-level EpisodeResult as well as the
            # explicit {episode_id, result: {...}} wire shape.
            raw_result = payload
        result = None
        if raw_result is not None:
            result = _episode_result(_object(raw_result, "result"), episode_id)
        service.close_context(episode_id, result)
        return {"episode_id": episode_id, "state": "closed"}

    def reset_episode(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        episode_id = payload["episode_id"]
        service.reset_context(episode_id)
        return {"episode_id": episode_id, "state": "open", "reset": True}

    def infer(payload: Mapping[str, Any]):
        observation = _observation(_object(payload["observation"], "observation"))
        request = LowLevelInferenceRequest(
            observation=observation,
            conditioning=_optional_object(payload, "conditioning"),
            context=_request_context(_optional_object(payload, "context")),
        )
        return service.infer(request)

    handlers = {
        "health": _translate_errors(health),
        "capabilities": _translate_errors(capabilities),
        "open": _translate_errors(open_episode),
        "close": _translate_errors(close_episode),
        "reset": _translate_errors(reset_episode),
        "infer": _translate_errors(infer),
    }
    routes = [
        Route("GET", "/health", handlers["health"]),
        Route("GET", "/capabilities", handlers["capabilities"]),
        Route("GET", "/v1/health", handlers["health"]),
        Route("GET", "/v1/capabilities", handlers["capabilities"]),
        Route("POST", "/v1/episodes/open", handlers["open"]),
        Route("POST", "/v1/episodes/close", handlers["close"]),
        Route("POST", "/v1/episodes/reset", handlers["reset"]),
        Route("POST", "/v1/infer", handlers["infer"]),
    ]
    return JsonAsgiApp(
        routes,
        on_startup=service.open,
        on_shutdown=service.close,
    )


__all__ = ["create_app"]
