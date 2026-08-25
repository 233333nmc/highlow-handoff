"""SGLang backend using its OpenAI-compatible chat-completions API."""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, runtime_checkable
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

from .contracts import (
    BackendInferenceRequest,
    BackendInferenceResponse,
    HealthState,
    HealthStatus,
    HighLevelCapabilities,
    HighLevelFrame,
    HighLevelOperation,
)
from .errors import BackendProtocolError, BackendTransportError


@dataclass(frozen=True, slots=True)
class HttpRequest:
    method: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes | None = field(default=None, repr=False)
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("HTTP timeout must be positive")
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    body: bytes = field(repr=False)
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))


@runtime_checkable
class HttpTransport(Protocol):
    def send(self, request: HttpRequest) -> HttpResponse:
        """Send one request and return its status, headers, and raw body."""


class UrllibTransport:
    """Standard-library transport used when no test transport is injected."""

    def send(self, request: HttpRequest) -> HttpResponse:
        raw_request = urllib_request.Request(
            request.url,
            data=request.body,
            headers=dict(request.headers),
            method=request.method,
        )
        try:
            with urllib_request.urlopen(raw_request, timeout=request.timeout_s) as response:
                return HttpResponse(
                    status_code=int(response.status),
                    body=response.read(),
                    headers=dict(response.headers.items()),
                )
        except urllib_error.HTTPError as exc:
            return HttpResponse(
                status_code=int(exc.code),
                body=exc.read(),
                headers=dict(exc.headers.items()) if exc.headers else {},
            )
        except (OSError, urllib_error.URLError) as exc:
            raise BackendTransportError(
                f"SGLang request failed for {request.url}: {exc}"
            ) from exc


TransportCallable = Callable[[HttpRequest], HttpResponse]


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _decode_json_response(response: HttpResponse, *, endpoint: str) -> Mapping[str, Any]:
    if not 200 <= response.status_code < 300:
        excerpt = response.body.decode("utf-8", errors="replace")[:1000]
        raise BackendTransportError(
            f"SGLang endpoint {endpoint} returned HTTP {response.status_code}: {excerpt}"
        )
    try:
        payload = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendProtocolError(
            f"SGLang endpoint {endpoint} did not return valid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise BackendProtocolError(
            f"SGLang endpoint {endpoint} returned a non-object JSON payload"
        )
    return payload


def _structured_object(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise BackendProtocolError(
            "SGLang completion content is not a valid JSON object"
        ) from exc
    if not isinstance(value, Mapping):
        raise BackendProtocolError("SGLang structured output must be a JSON object")
    return value


def _completion_text(body: Mapping[str, Any]) -> tuple[str, str]:
    try:
        choice = body["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise BackendProtocolError(
            "SGLang response is missing choices[0].message.content"
        ) from exc
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping) and item.get("type") in {None, "text"}
        )
    if not isinstance(content, str) or not content.strip():
        raise BackendProtocolError("SGLang completion content must be a non-empty string")
    finish_reason = str(choice.get("finish_reason", ""))
    return content.strip(), finish_reason


class SGLangOpenAIBackend:
    """Schema-constrained multimodal backend for a separately hosted SGLang model."""

    _RESERVED_GENERATION_KEYS = {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "response_format",
        "user",
    }

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 120.0,
        transport: HttpTransport | TransportCallable | None = None,
        chat_path: str = "",
        health_path: str = "/health",
    ) -> None:
        base_url = str(base_url).strip().rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if not str(model).strip():
            raise ValueError("model must not be empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        if base_url.endswith("/v1"):
            api_root = base_url
            server_root = base_url[:-3].rstrip("/")
        else:
            api_root = f"{base_url}/v1"
            server_root = base_url
        self.base_url = base_url
        self.model = str(model).strip()
        self.api_key = str(api_key)
        self.timeout_s = float(timeout_s)
        self.chat_url = (
            f"{server_root}/{chat_path.lstrip('/')}"
            if chat_path
            else f"{api_root}/chat/completions"
        )
        self.health_url = f"{server_root}/{health_path.lstrip('/')}"
        self.models_url = f"{api_root}/models"
        self._transport = transport or UrllibTransport()

    def _send(self, request: HttpRequest) -> HttpResponse:
        sender = getattr(self._transport, "send", None)
        response = sender(request) if callable(sender) else self._transport(request)
        if not isinstance(response, HttpResponse):
            raise BackendProtocolError("injected transport must return HttpResponse")
        return response

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _image_url(frame: HighLevelFrame) -> str:
        if isinstance(frame.data, bytes):
            encoded = base64.b64encode(frame.data).decode("ascii")
            return f"data:{frame.mime_type};base64,{encoded}"
        value = frame.data.strip()
        if value.startswith(("data:", "http://", "https://")):
            return value
        raise BackendProtocolError(
            "string frame data must be an HTTP(S) URL or a data URL"
        )

    def _payload(self, request: BackendInferenceRequest) -> Mapping[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": self._image_url(frame)},
            }
            for frame in request.frames
        ]
        content.append({"type": "text", "text": request.prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": request.generation.temperature,
            "top_p": request.generation.top_p,
            "max_tokens": request.generation.max_tokens,
            "user": request.request_context.request_id,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "strict": True,
                    "schema": _plain_json(request.response_schema),
                },
            },
        }
        if request.generation.seed is not None:
            payload["seed"] = int(request.generation.seed)
        collisions = self._RESERVED_GENERATION_KEYS.intersection(
            request.generation.extra
        )
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"generation.extra cannot override reserved keys: {names}")
        payload.update(_plain_json(request.generation.extra))
        return payload

    def infer(self, request: BackendInferenceRequest) -> BackendInferenceResponse:
        payload = self._payload(request)
        timeout_s = self.timeout_s
        if request.request_context.deadline_ms is not None:
            timeout_s = min(timeout_s, request.request_context.deadline_ms / 1000.0)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        started = time.perf_counter_ns()
        response = self._send(
            HttpRequest(
                method="POST",
                url=self.chat_url,
                headers=self._headers(),
                body=encoded,
                timeout_s=timeout_s,
            )
        )
        finished = time.perf_counter_ns()
        body = _decode_json_response(response, endpoint=self.chat_url)
        raw_text, finish_reason = _completion_text(body)
        structured = _structured_object(raw_text)
        usage = body.get("usage", {})
        if not isinstance(usage, Mapping):
            usage = {}
        timing_ms: dict[str, float] = {
            "transport": (finished - started) / 1_000_000.0,
        }
        server_timing = body.get("timing", {})
        if isinstance(server_timing, Mapping):
            for key, value in server_timing.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    timing_ms[f"server_{key}"] = float(value) * (
                        1000.0 if str(key).endswith("_s") else 1.0
                    )
        return BackendInferenceResponse(
            raw_text=raw_text,
            structured_output=structured,
            backend="sglang-openai",
            model=str(body.get("model") or self.model),
            finish_reason=finish_reason,
            usage=dict(usage),
            timing_ms=timing_ms,
            metadata={
                "response_id": str(body.get("id", "")),
                "chat_endpoint": self.chat_url,
            },
        )

    def capabilities(self) -> HighLevelCapabilities:
        return HighLevelCapabilities(
            component="sglang-openai-backend",
            backend="sglang-openai",
            model=self.model,
            protocol="openai-chat-completions",
            operations=(
                HighLevelOperation.PLANNER,
                HighLevelOperation.VERIFIER,
            ),
            supports_images=True,
            supports_structured_output=True,
            metadata={
                "chat_endpoint": self.chat_url,
                "health_endpoint": self.health_url,
                "models_endpoint": self.models_url,
                "structured_output_mode": "json_schema",
            },
        )

    def health(self) -> HealthStatus:
        checked_at_s = time.time()
        try:
            response = self._send(
                HttpRequest(
                    method="GET",
                    url=self.health_url,
                    headers={"Accept": "application/json"},
                    timeout_s=min(self.timeout_s, 10.0),
                )
            )
        except Exception as exc:
            return HealthStatus(
                state=HealthState.UNAVAILABLE,
                component="sglang-openai-backend",
                checked_at_s=checked_at_s,
                detail=str(exc),
                metadata={"endpoint": self.health_url, "model": self.model},
            )
        ready = 200 <= response.status_code < 300
        detail = response.body.decode("utf-8", errors="replace")[:500]
        return HealthStatus(
            state=HealthState.READY if ready else HealthState.UNAVAILABLE,
            component="sglang-openai-backend",
            checked_at_s=checked_at_s,
            detail=detail,
            metadata={
                "endpoint": self.health_url,
                "http_status": response.status_code,
                "model": self.model,
            },
        )


__all__ = [
    "HttpRequest",
    "HttpResponse",
    "HttpTransport",
    "SGLangOpenAIBackend",
    "TransportCallable",
    "UrllibTransport",
]
