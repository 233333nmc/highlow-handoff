"""Lifecycle integration for the existing HyVLA policy wrapper."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Mapping

from highlow.contracts import ActionEnvelope, EpisodeContext, EpisodeResult

from ...core import (
    LifecycleError,
    LowLevelInferenceRequest,
    PolicyManifest,
    PolicyPrediction,
    PredictionContractError,
)
from ...registry import BackendRegistry, default_registry
from .input_adapter import HyVLAInputAdapter
from .manifest import HYVLA_MANIFEST, STATE_ACTION_SCHEMA_ID


PolicyFactory = Callable[[Mapping[str, Any]], Any]


class HyVLABackend:
    """Lifecycle adapter around ``HyVLAPolicyWrapper`` with injectable factory."""

    def __init__(
        self,
        policy_config: Mapping[str, Any] | None = None,
        *,
        policy_factory: PolicyFactory,
        input_adapter: HyVLAInputAdapter | None = None,
        manifest: PolicyManifest = HYVLA_MANIFEST,
        coordinate_frame: str = "robot.base",
    ) -> None:
        self._policy_config = MappingProxyType(dict(policy_config or {}))
        if not callable(policy_factory):
            raise TypeError("policy_factory must be callable")
        self._policy_factory = policy_factory
        self._input_adapter = input_adapter or HyVLAInputAdapter()
        self._manifest = manifest
        self._coordinate_frame = coordinate_frame
        self._policy: Any | None = None
        self._episode: EpisodeContext | None = None
        self._last_instruction: str | None = None
        self._closed = False

    @property
    def manifest(self) -> PolicyManifest:
        return self._manifest

    @property
    def is_open(self) -> bool:
        return self._policy is not None and not self._closed

    def open(self) -> None:
        if self._closed:
            raise LifecycleError("HyVLA backend is closed")
        if self._policy is not None:
            return
        policy = self._policy_factory(dict(self._policy_config))
        if not callable(getattr(policy, "get_action", None)):
            raise TypeError("HyVLA policy must provide get_action(batch)")
        if not callable(getattr(policy, "reset", None)):
            raise TypeError("HyVLA policy must provide reset()")
        self._policy = policy

    def begin_episode(self, episode: EpisodeContext) -> None:
        if self._policy is None or self._closed:
            raise LifecycleError("open() must be called before begin_episode()")
        if self._episode is not None:
            if self._episode == episode:
                # Required by the generic lifecycle contract: a caller may
                # retry after the original response was lost.
                return
            raise LifecycleError(f"episode {self._episode.episode_id!r} is already active")
        self._policy.reset()
        self._episode = episode
        self._last_instruction = None

    def infer(self, request: LowLevelInferenceRequest) -> PolicyPrediction:
        policy = self._policy
        episode = self._episode
        if policy is None or self._closed:
            raise LifecycleError("HyVLA backend is not open")
        if episode is None:
            raise LifecycleError("begin_episode() must be called before infer()")
        if request.episode_id != episode.episode_id:
            raise PredictionContractError("request does not belong to the active episode")

        instruction_value = request.conditioning.get("instruction", episode.instruction)
        if not isinstance(instruction_value, str):
            raise PredictionContractError("conditioning.instruction must be a string")
        instruction_changed = (
            self._last_instruction is not None
            and instruction_value != self._last_instruction
        )
        stale_action_count = 0
        if instruction_changed:
            invalidate = getattr(policy, "invalidate_action_cache", None)
            if callable(invalidate):
                invalidated = invalidate()
                if isinstance(invalidated, int):
                    stale_action_count = invalidated

        batch = self._input_adapter.build_batch(request.observation, instruction_value)
        action_data = policy.get_action(batch)
        self._last_instruction = instruction_value

        diagnostics: dict[str, Any] = {
            "backend_id": self.manifest.backend_id,
            "policy_id": self.manifest.policy_id,
            "policy_version": self.manifest.policy_version,
            "request_id": request.context.request_id,
            "trace_id": request.context.trace_id,
            "instruction_changed": instruction_changed,
            "stale_action_count": stale_action_count,
        }
        policy_metrics = {
            "used_action_cache": "last_action_used_cache",
            "model_forward_ms": "last_model_forward_ms",
            "action_call_ms": "last_action_call_ms",
        }
        for output_name, attribute_name in policy_metrics.items():
            value = getattr(policy, attribute_name, None)
            if value is not None:
                diagnostics[output_name] = value

        action = ActionEnvelope(
            episode_id=request.episode_id,
            step_id=request.step_id,
            schema_id=STATE_ACTION_SCHEMA_ID,
            data=action_data,
            horizon=1,
            coordinate_frame=self._coordinate_frame,
            metadata=diagnostics,
        )
        return PolicyPrediction(action=action, metadata=diagnostics)

    def end_episode(
        self,
        episode: EpisodeContext,
        result: EpisodeResult | None = None,
    ) -> None:
        if self._episode is None:
            # Cleanup is idempotent so a lost close response can be retried.
            return
        if episode.episode_id != self._episode.episode_id:
            raise PredictionContractError("episode does not match the active HyVLA context")
        try:
            if self._policy is not None:
                self._policy.reset()
        finally:
            self._episode = None
            self._last_instruction = None

    def close(self) -> None:
        if self._closed:
            return
        policy = self._policy
        self._episode = None
        self._last_instruction = None
        self._policy = None
        self._closed = True
        close_policy = getattr(policy, "close", None)
        if callable(close_policy):
            close_policy()


def create_hyvla_backend(**kwargs: Any) -> HyVLABackend:
    return HyVLABackend(**kwargs)


def register_hyvla(
    registry: BackendRegistry = default_registry,
    *,
    replace: bool = False,
) -> None:
    registry.register(HYVLA_MANIFEST.backend_id, create_hyvla_backend, replace=replace)


__all__ = [
    "HyVLABackend",
    "PolicyFactory",
    "create_hyvla_backend",
    "register_hyvla",
]
