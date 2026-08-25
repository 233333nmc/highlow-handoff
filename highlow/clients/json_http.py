"""Schema-agnostic JSON-over-HTTP inference client using the stdlib."""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen

from highlow.inference.common.serialization import from_jsonable, to_jsonable


RequestEncoder = Callable[[Any], Any]
ResponseDecoder = Callable[[Any], Any]
HttpTransport = Callable[
    [str, str, bytes | None, Mapping[str, str], float | None],
    Any,
]

def _stdlib_transport(
    method: str,
    url: str,
    body: bytes | None,
    headers: Mapping[str, str],
    timeout_s: float | None,
) -> bytes:
    request = Request(url, data=body, headers=dict(headers), method=method)
    if timeout_s is None:
        with urlopen(request) as response:  # noqa: S310 - URL is caller configuration
            return response.read()
    with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        return response.read()


class JsonHttpInferenceClient:
    """Call an arbitrary high/low JSON service without knowing its schema.

    Codecs are injected at the contract boundary.  The default encoder handles
    dataclasses, enums, mappings, bytes and array-like ``tolist`` values; the
    default decoder restores transport envelopes such as base64-encoded bytes
    while otherwise preserving ordinary parsed JSON values.
    """

    def __init__(
        self,
        base_url: str,
        *,
        infer_endpoint: str = "/infer",
        open_context_endpoint: str | None = None,
        reset_context_endpoint: str | None = None,
        close_context_endpoint: str | None = None,
        capabilities_endpoint: str | None = None,
        health_endpoint: str | None = None,
        request_encoder: RequestEncoder = to_jsonable,
        response_decoder: ResponseDecoder = from_jsonable,
        transport: HttpTransport = _stdlib_transport,
        timeout_s: float | None = 30.0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not infer_endpoint:
            raise ValueError("infer_endpoint must not be empty")
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive when provided")
        for name, value in (
            ("request_encoder", request_encoder),
            ("response_decoder", response_decoder),
            ("transport", transport),
        ):
            if not callable(value):
                raise TypeError(f"{name} must be callable")
        self.base_url = base_url.rstrip("/")
        self.infer_endpoint = infer_endpoint
        self.open_context_endpoint = open_context_endpoint
        self.reset_context_endpoint = reset_context_endpoint
        self.close_context_endpoint = close_context_endpoint
        self.capabilities_endpoint = capabilities_endpoint
        self.health_endpoint = health_endpoint
        self.request_encoder = request_encoder
        self.response_decoder = response_decoder
        self.transport = transport
        self.timeout_s = timeout_s
        merged_headers = {"Accept": "application/json", "Content-Type": "application/json"}
        merged_headers.update(dict(headers or {}))
        self.headers = MappingProxyType(merged_headers)

    def infer(self, request: Any) -> Any:
        return self._request("POST", self.infer_endpoint, request)

    def open_context(self, context: Any) -> Any:
        return self._optional_context(self.open_context_endpoint, context)

    def reset_context(self, context: Any) -> Any:
        return self._optional_context(self.reset_context_endpoint, context)

    def close_context(self, context: Any) -> Any:
        return self._optional_context(self.close_context_endpoint, context)

    def capabilities(self) -> Any:
        if self.capabilities_endpoint is None:
            return {}
        return self._request("GET", self.capabilities_endpoint, None)

    def health(self) -> Any:
        if self.health_endpoint is None:
            return {"status": "unknown", "transport": "json_http"}
        return self._request("GET", self.health_endpoint, None)

    def _optional_context(self, endpoint: str | None, context: Any) -> Any:
        if endpoint is None:
            return None
        return self._request("POST", endpoint, context)

    def _request(self, method: str, endpoint: str, value: Any) -> Any:
        body = None
        if value is not None:
            encoded = self.request_encoder(value)
            body = encoded if isinstance(encoded, bytes) else json.dumps(
                encoded,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        result = self.transport(
            method,
            self._url(endpoint),
            body,
            self.headers,
            self.timeout_s,
        )
        if isinstance(result, bytes):
            payload = json.loads(result) if result else None
        elif isinstance(result, str):
            payload = json.loads(result) if result else None
        else:
            # Injected transports may return an already-parsed JSON value.
            payload = result
        return self.response_decoder(payload)

    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"


__all__ = [
    "HttpTransport",
    "JsonHttpInferenceClient",
    "RequestEncoder",
    "ResponseDecoder",
]
