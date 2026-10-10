"""Pure-ASGI HTTP facade for :class:`HighLevelService`."""

from __future__ import annotations

from typing import Any, Mapping

from highlow.contracts import EpisodeContext, RequestContext
from highlow.inference.common import (
    InferenceServiceError,
    JsonAsgiApp,
    Route,
    from_jsonable,
)

from .contracts import GenerationOptions, HighLevelFrame, HighLevelRequest
from .errors import (
    BackendProtocolError,
    BackendTransportError,
    HighLevelError,
    UnsupportedVersionError,
)
from .service import HighLevelService


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a JSON object")
    return value


def _request_context(value: Any) -> RequestContext:
    payload = _mapping(value, "request_context")
    kwargs: dict[str, Any] = {
        "trace_id": str(payload.get("trace_id", "")),
        "api_version": str(payload.get("api_version", "v1")),
        "deadline_ms": payload.get("deadline_ms"),
        "metadata": _mapping(
            payload.get("metadata", {}), "request_context.metadata"
        ),
    }
    if "request_id" in payload:
        kwargs["request_id"] = str(payload["request_id"])
    return RequestContext(**kwargs)


def _episode_context(value: Any) -> EpisodeContext | None:
    if value is None:
        return None
    payload = _mapping(value, "episode_context")
    return EpisodeContext(
        task_id=str(payload.get("task_id", "")),
        instruction=str(payload.get("instruction", "")),
        episode_id=str(payload.get("episode_id", "")),
        seed=payload.get("seed"),
        simulator=str(payload.get("simulator", "")),
        simulator_version=str(payload.get("simulator_version", "")),
        metadata=_mapping(payload.get("metadata", {}), "episode_context.metadata"),
    )


def _frame(value: Any, index: int) -> HighLevelFrame:
    payload = _mapping(value, f"frames[{index}]")
    data = from_jsonable(
        payload.get("url", payload.get("data_url", payload.get("data")))
    )
    valid_url = isinstance(data, str) and data.startswith(
        ("data:", "http://", "https://")
    )
    if not isinstance(data, bytes) and not valid_url:
        raise ValueError(
            f"frames[{index}] must provide encoded bytes or an HTTP(S)/data URL"
        )
    return HighLevelFrame(
        data=data,
        mime_type=str(payload.get("mime_type", "image/jpeg")),
        frame_id=payload.get("frame_id"),
        timestamp_s=payload.get("timestamp_s"),
        source=str(payload.get("source", "")),
        metadata=_mapping(payload.get("metadata", {}), f"frames[{index}].metadata"),
    )


def _generation(value: Any) -> GenerationOptions:
    if value is None:
        return GenerationOptions()
    payload = _mapping(value, "generation")
    return GenerationOptions(
        max_tokens=int(payload.get("max_tokens", 128)),
        temperature=float(payload.get("temperature", 0.0)),
        top_p=float(payload.get("top_p", 1.0)),
        seed=payload.get("seed"),
        extra=_mapping(payload.get("extra", {}), "generation.extra"),
    )


def high_level_request_from_dict(payload: Mapping[str, Any]) -> HighLevelRequest:
    """Decode the versioned JSON request used by the ASGI endpoint."""

    raw_context = payload.get("request_context", payload.get("context"))
    if raw_context is None:
        raise KeyError("request_context is required")
    raw_frames = payload.get("frames", [])
    if not isinstance(raw_frames, list):
        raise TypeError("frames must be a JSON array")
    return HighLevelRequest(
        request_context=_request_context(raw_context),
        operation=payload.get("operation", ""),
        episode_id=str(payload.get("episode_id", "")),
        episode_context=_episode_context(payload.get("episode_context")),
        task_instruction=str(payload.get("task_instruction", "")),
        structured_context=_mapping(
            payload.get("structured_context", {}), "structured_context"
        ),
        frames=tuple(_frame(value, index) for index, value in enumerate(raw_frames)),
        prompt_version=str(payload.get("prompt_version", "")),
        schema_version=str(payload.get("schema_version", "")),
        generation=_generation(payload.get("generation")),
        metadata=_mapping(payload.get("metadata", {}), "metadata"),
    )


def _service_error(exc: HighLevelError) -> InferenceServiceError:
    if isinstance(exc, UnsupportedVersionError):
        return InferenceServiceError(
            "UNSUPPORTED_VERSION", str(exc), status_code=400, retryable=False
        )
    if isinstance(exc, BackendTransportError):
        return InferenceServiceError(
            "BACKEND_UNAVAILABLE", str(exc), status_code=503, retryable=True
        )
    if isinstance(exc, BackendProtocolError):
        return InferenceServiceError(
            "BACKEND_PROTOCOL_ERROR", str(exc), status_code=502, retryable=False
        )
    return InferenceServiceError(
        "HIGH_LEVEL_INFERENCE_ERROR", str(exc), status_code=422, retryable=False
    )


def create_app(service: HighLevelService) -> JsonAsgiApp:
    """Create a dependency-free ASGI app around an injected service."""

    if not callable(getattr(service, "infer", None)):
        raise TypeError("service must provide infer(request)")

    def health(_: Mapping[str, Any]) -> Any:
        return service.health()

    def capabilities(_: Mapping[str, Any]) -> Any:
        return service.capabilities()

    def infer(payload: Mapping[str, Any]) -> Any:
        request = high_level_request_from_dict(payload)
        try:
            return service.infer(request)
        except HighLevelError as exc:
            raise _service_error(exc) from exc

    return JsonAsgiApp(
        [
            Route("GET", "/health", health),
            Route("GET", "/capabilities", capabilities),
            Route("POST", "/v1/high-level/infer", infer),
        ]
    )


__all__ = ["create_app", "high_level_request_from_dict"]
