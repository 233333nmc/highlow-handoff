"""Generic deployment composition helpers for low-level policy servers."""

from __future__ import annotations

import importlib
import json
import os
from typing import Any, Mapping

from highlow.inference.common import JsonAsgiApp

from .api import create_app
from .registry import BackendFactory
from .service import LowLevelService


def load_backend_factory(path: str) -> BackendFactory:
    """Load an explicit ``module:callable`` backend factory."""

    module_name, separator, attribute_name = str(path).partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("backend factory must use 'module:callable' syntax")
    module = importlib.import_module(module_name)
    try:
        factory = getattr(module, attribute_name)
    except AttributeError as exc:
        raise ValueError(f"backend factory {path!r} does not exist") from exc
    if not callable(factory):
        raise TypeError(f"backend factory {path!r} is not callable")
    return factory


def create_policy_app(
    factory: BackendFactory,
    *,
    backend_config: Mapping[str, Any] | None = None,
) -> JsonAsgiApp:
    """Create a lifecycle-managed API for any low-level backend factory."""

    if not callable(factory):
        raise TypeError("factory must be callable")
    backend = factory(**dict(backend_config or {}))
    return create_app(LowLevelService(backend))


def create_policy_app_from_env(
    environ: Mapping[str, str] | None = None,
) -> JsonAsgiApp:
    """ASGI factory suitable for ``uvicorn ... --factory``.

    ``HIGHLOW_LOW_LEVEL_FACTORY`` names a plugin composition callable.  The
    optional ``HIGHLOW_LOW_LEVEL_CONFIG`` is a JSON object expanded as keyword
    arguments.  This avoids a central switch statement as new policies are
    added.
    """

    values = os.environ if environ is None else environ
    try:
        factory_path = values["HIGHLOW_LOW_LEVEL_FACTORY"]
    except KeyError as exc:
        raise RuntimeError(
            "missing required environment variable: HIGHLOW_LOW_LEVEL_FACTORY"
        ) from exc
    raw_config = values.get("HIGHLOW_LOW_LEVEL_CONFIG", "{}")
    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise RuntimeError("HIGHLOW_LOW_LEVEL_CONFIG must be valid JSON") from exc
    if not isinstance(config, Mapping):
        raise RuntimeError("HIGHLOW_LOW_LEVEL_CONFIG must be a JSON object")
    return create_policy_app(
        load_backend_factory(factory_path),
        backend_config=config,
    )


__all__ = [
    "create_policy_app",
    "create_policy_app_from_env",
    "load_backend_factory",
]
