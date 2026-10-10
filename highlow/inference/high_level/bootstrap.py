"""Deployment composition helpers for the SGLang-backed high-level API."""

from __future__ import annotations

import os
from typing import Mapping

from highlow.inference.common import JsonAsgiApp

from .api import create_app
from .service import HighLevelService
from .sglang import HttpTransport, SGLangOpenAIBackend, TransportCallable


def create_sglang_app(
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    timeout_s: float = 120.0,
    chat_path: str = "",
    health_path: str = "/health",
    transport: HttpTransport | TransportCallable | None = None,
) -> JsonAsgiApp:
    """Compose the public high-level service around an external SGLang server."""

    backend = SGLangOpenAIBackend(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_s=timeout_s,
        chat_path=chat_path,
        health_path=health_path,
        transport=transport,
    )
    return create_app(HighLevelService(backend))


def create_sglang_app_from_env(
    environ: Mapping[str, str] | None = None,
) -> JsonAsgiApp:
    """ASGI factory suitable for ``uvicorn ... --factory``.

    Required variables are ``HIGHLOW_SGLANG_BASE_URL`` and
    ``HIGHLOW_SGLANG_MODEL``.  API key, timeout, chat path, and health path are
    optional and remain deployment concerns rather than service contracts.
    """

    values = os.environ if environ is None else environ
    try:
        base_url = values["HIGHLOW_SGLANG_BASE_URL"]
        model = values["HIGHLOW_SGLANG_MODEL"]
    except KeyError as exc:
        raise RuntimeError(f"missing required environment variable: {exc.args[0]}") from exc
    try:
        timeout_s = float(values.get("HIGHLOW_SGLANG_TIMEOUT_S", "120"))
    except ValueError as exc:
        raise RuntimeError("HIGHLOW_SGLANG_TIMEOUT_S must be numeric") from exc
    return create_sglang_app(
        base_url=base_url,
        model=model,
        api_key=values.get("HIGHLOW_SGLANG_API_KEY", ""),
        timeout_s=timeout_s,
        chat_path=values.get("HIGHLOW_SGLANG_CHAT_PATH", ""),
        health_path=values.get("HIGHLOW_SGLANG_HEALTH_PATH", "/health"),
    )


__all__ = ["create_sglang_app", "create_sglang_app_from_env"]
