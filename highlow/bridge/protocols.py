"""Lightweight ports used by the simulator-facing bridge.

The protocols in this module intentionally describe behaviour rather than a
particular model implementation or transport.  Importing them must stay safe
in simulator processes that do not have the model runtime installed.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, TypeVar, runtime_checkable

from highlow.contracts import ActionEnvelope, EpisodeContext, ObservationEnvelope


NativeObservation = TypeVar("NativeObservation")
NativeAction = TypeVar("NativeAction")


@runtime_checkable
class SimulatorAdapter(Protocol[NativeObservation, NativeAction]):
    """Convert simulator-native values to and from canonical contracts.

    An adapter is deliberately *not* a simulator driver.  In particular it
    has no ``reset``, ``step`` or ``take_action`` method: the official
    simulator loop remains the sole owner of those operations.
    """

    def create_episode_context(
        self,
        task_id: str,
        instruction: str,
        *,
        episode_id: str | None = None,
        seed: int | None = None,
        simulator_version: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> EpisodeContext:
        """Build canonical metadata for one simulator-owned episode."""

    def encode_observation(
        self,
        context: EpisodeContext,
        observation: NativeObservation,
        *,
        step_id: int,
        timestamp_s: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> ObservationEnvelope:
        """Convert one already-produced simulator observation."""

    def decode_action(self, action: ActionEnvelope) -> NativeAction:
        """Convert one canonical action for the simulator loop to execute."""


@runtime_checkable
class InferenceClient(Protocol):
    """Minimal common port for a high- or low-level inference service."""

    def infer(self, request: Any) -> Any:
        """Run one synchronous inference request."""


class HighLevelClient(InferenceClient, Protocol):
    """Marker protocol for a high-level inference client."""


class LowLevelClient(InferenceClient, Protocol):
    """Marker protocol for a low-level inference client."""


__all__ = [
    "HighLevelClient",
    "InferenceClient",
    "LowLevelClient",
    "SimulatorAdapter",
]
