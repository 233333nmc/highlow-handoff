"""Composable policy hook for an official RoboTwin-owned control loop."""

from __future__ import annotations

from threading import RLock
from typing import Any, Mapping

from highlow.bridge import DualSystemBridge, SimulatorAdapter
from highlow.contracts import EpisodeContext, EpisodeResult, RequestContext

class RoboTwinPolicyHook:
    """Convert and infer without ever calling a RoboTwin environment method."""

    def __init__(
        self,
        adapter: SimulatorAdapter[Mapping[str, Any], Any],
        bridge: DualSystemBridge,
    ) -> None:
        self.adapter = adapter
        self.bridge = bridge
        self._context: EpisodeContext | None = None
        self._ready = False
        self._lock = RLock()

    @property
    def context(self) -> EpisodeContext | None:
        return self._context

    @property
    def ready(self) -> bool:
        return self._ready

    def start_episode(
        self,
        task_id: str,
        instruction: str,
        *,
        episode_id: str | None = None,
        seed: int | None = None,
        simulator_version: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> EpisodeContext:
        with self._lock:
            context = self._context
            if context is None:
                context = self.adapter.create_episode_context(
                    task_id,
                    instruction,
                    episode_id=episode_id,
                    seed=seed,
                    simulator_version=simulator_version,
                    metadata=metadata,
                )
                # Retain auto-generated ids before remote opening starts.  A
                # transient failure can then be retried without orphaning a
                # stateful low-level context under a new id.
                self._context = context
            elif not self._matches_start_request(
                context,
                task_id,
                instruction,
                episode_id=episode_id,
                seed=seed,
                simulator_version=simulator_version,
                metadata=metadata,
            ):
                raise RuntimeError("a different RoboTwin policy episode is active")
            self.bridge.open_context(context)
            self._ready = True
            return context

    def act(
        self,
        native_observation: Mapping[str, Any],
        step_id: int,
        timestamp_s: float,
        *,
        request_context: RequestContext | None = None,
        observation_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        with self._lock:
            if self._context is None or not self._ready:
                raise RuntimeError("start_episode() must be called before act()")
            observation = self.adapter.encode_observation(
                self._context,
                native_observation,
                step_id=step_id,
                timestamp_s=timestamp_s,
                metadata=observation_metadata,
            )
            action = self.bridge.infer(
                observation,
                request_context=request_context,
            )
            return self.adapter.decode_action(action)

    def reset(self) -> None:
        """Reset policy context only; the caller remains responsible for env reset."""

        with self._lock:
            if self._context is None:
                return
            if not self._ready:
                raise RuntimeError("the RoboTwin policy episode has not finished opening")
            self.bridge.reset_context(self._context)

    def finish(self, result: EpisodeResult | None = None) -> None:
        """Finish policy context only; this does not close/reset the environment."""

        with self._lock:
            if self._context is None:
                return
            if result is not None and result.episode_id != self._context.episode_id:
                raise ValueError("episode result does not match the active episode")
            close_value: EpisodeContext | EpisodeResult = result or self._context
            self.bridge.close_context(close_value)
            self._context = None
            self._ready = False

    @staticmethod
    def _matches_start_request(
        context: EpisodeContext,
        task_id: str,
        instruction: str,
        *,
        episode_id: str | None,
        seed: int | None,
        simulator_version: str,
        metadata: Mapping[str, Any] | None,
    ) -> bool:
        return (
            context.task_id == task_id
            and context.instruction == instruction
            and (episode_id is None or context.episode_id == episode_id)
            and context.seed == seed
            and context.simulator_version == simulator_version
            and dict(context.metadata) == dict(metadata or {})
        )

    finish_episode = finish


__all__ = ["RoboTwinPolicyHook"]
