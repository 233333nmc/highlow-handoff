"""Outbound model-backend port used by the high-level service."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .contracts import (
    BackendInferenceRequest,
    BackendInferenceResponse,
    HealthStatus,
    HighLevelCapabilities,
)


@runtime_checkable
class HighLevelBackend(Protocol):
    """A model-serving backend with no planner state of its own."""

    def infer(self, request: BackendInferenceRequest) -> BackendInferenceResponse:
        """Generate one schema-constrained response."""

    def capabilities(self) -> HighLevelCapabilities:
        """Describe static serving capabilities without running inference."""

    def health(self) -> HealthStatus:
        """Return a current, non-throwing health snapshot."""


__all__ = ["HighLevelBackend"]
