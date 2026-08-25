"""Stateless planner/verifier service built around a pluggable backend."""

from __future__ import annotations

import time
from typing import Mapping

from .backend import HighLevelBackend
from .contracts import (
    BackendInferenceRequest,
    HealthState,
    HealthStatus,
    HighLevelCapabilities,
    HighLevelOperation,
    HighLevelRequest,
    HighLevelResponse,
)
from .prompts import PromptSchemaRegistry, default_prompt_schema_registry


class HighLevelService:
    """Render, invoke, and validate one high-level operation.

    The service is intentionally episode-stateless.  Scheduling, prefetch,
    stale-response rejection, and committing a subtask to a low-level policy
    remain orchestrator responsibilities.
    """

    def __init__(
        self,
        backend: HighLevelBackend,
        *,
        registry: PromptSchemaRegistry | None = None,
    ) -> None:
        self._backend = backend
        self._registry = registry or default_prompt_schema_registry()

    def infer(self, request: HighLevelRequest) -> HighLevelResponse:
        spec = self._registry.resolve(
            request.operation,
            request.prompt_version,
            request.schema_version,
        )
        prompt = spec.renderer(request)
        backend_request = BackendInferenceRequest(
            request_context=request.request_context,
            operation=request.operation,
            prompt=prompt,
            response_schema=spec.response_schema,
            schema_name=spec.schema_name,
            frames=request.frames,
            generation=request.generation,
            metadata={
                **dict(request.metadata),
                "episode_id": request.episode_id,
                "prompt_version": spec.prompt_version,
                "schema_version": spec.schema_version,
            },
        )
        backend_response = self._backend.infer(backend_request)
        output = spec.validator(backend_response.structured_output, request)
        return HighLevelResponse(
            request_context=request.request_context,
            episode_id=request.episode_id,
            operation=request.operation,
            output=output,
            prompt_version=spec.prompt_version,
            schema_version=spec.schema_version,
            raw_text=backend_response.raw_text,
            backend=backend_response.backend,
            model=backend_response.model,
            finish_reason=backend_response.finish_reason,
            usage=backend_response.usage,
            timing_ms=backend_response.timing_ms,
            metadata=backend_response.metadata,
        )

    def capabilities(self) -> HighLevelCapabilities:
        backend = self._backend.capabilities()
        registered_operations = tuple(
            operation
            for operation in (
                HighLevelOperation.PLANNER,
                HighLevelOperation.VERIFIER,
            )
            if operation in backend.operations
        )
        return HighLevelCapabilities(
            component="high-level-service",
            backend=backend.backend,
            model=backend.model,
            protocol=backend.protocol,
            operations=registered_operations,
            prompt_versions=self._registry.prompt_versions(),
            schema_versions=self._registry.schema_versions(),
            supports_images=backend.supports_images,
            supports_structured_output=backend.supports_structured_output,
            metadata={
                **dict(backend.metadata),
                "backend_component": backend.component,
            },
        )

    def health(self) -> HealthStatus:
        try:
            backend = self._backend.health()
        except Exception as exc:  # A health endpoint must remain non-throwing.
            return HealthStatus(
                state=HealthState.UNAVAILABLE,
                component="high-level-service",
                checked_at_s=time.time(),
                detail=f"backend health check failed: {exc}",
            )
        return HealthStatus(
            state=backend.state,
            component="high-level-service",
            checked_at_s=time.time(),
            detail=backend.detail,
            metadata={
                **dict(backend.metadata),
                "backend_component": backend.component,
            },
        )


__all__ = ["HighLevelService"]
