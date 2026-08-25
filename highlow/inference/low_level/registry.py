"""Thread-safe registry of low-level backend factories."""

from __future__ import annotations

from threading import RLock
from typing import Any, Callable

from .core import Backend


BackendFactory = Callable[..., Backend]


class BackendRegistrationError(ValueError):
    """Raised when a backend factory registration is invalid or duplicated."""


class UnknownBackendError(LookupError):
    """Raised when a requested backend has not been registered."""


def _backend_key(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise BackendRegistrationError("backend name must not be empty")
    return name.strip().casefold()


class BackendRegistry:
    """Maps stable backend identifiers to cheap, framework-lazy factories."""

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}
        self._lock = RLock()

    def register(
        self,
        name: str,
        factory: BackendFactory,
        *,
        replace: bool = False,
    ) -> BackendFactory:
        key = _backend_key(name)
        if not callable(factory):
            raise BackendRegistrationError("backend factory must be callable")
        with self._lock:
            if key in self._factories and not replace:
                raise BackendRegistrationError(f"backend {key!r} is already registered")
            self._factories[key] = factory
        return factory

    def unregister(self, name: str) -> BackendFactory:
        key = _backend_key(name)
        with self._lock:
            try:
                return self._factories.pop(key)
            except KeyError as exc:
                raise UnknownBackendError(f"unknown backend {key!r}") from exc

    def get_factory(self, name: str) -> BackendFactory:
        key = _backend_key(name)
        with self._lock:
            try:
                return self._factories[key]
            except KeyError as exc:
                raise UnknownBackendError(f"unknown backend {key!r}") from exc

    def create(self, name: str, /, **kwargs: Any) -> Backend:
        backend = self.get_factory(name)(**kwargs)
        if not isinstance(backend, Backend):
            raise BackendRegistrationError(
                f"factory for {_backend_key(name)!r} did not return a Backend"
            )
        return backend

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._factories))

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            key = _backend_key(name)
        except BackendRegistrationError:
            return False
        with self._lock:
            return key in self._factories


default_registry = BackendRegistry()


__all__ = [
    "BackendFactory",
    "BackendRegistrationError",
    "BackendRegistry",
    "UnknownBackendError",
    "default_registry",
]
