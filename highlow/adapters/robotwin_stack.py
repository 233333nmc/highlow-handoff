"""Ready-made wiring for RoboTwin observations and the bundled services."""

from __future__ import annotations

from typing import Any

from highlow.bridge import (
    BridgeTurn,
    DualSystemBridge,
    low_level_request,
    periodic_high_level_trigger,
    planner_then_verifier_request,
)

from .robotwin import encode_robotwin_rgb_frame


def robotwin_high_level_request(turn: BridgeTurn) -> Any:
    """Plan first, then verify, with on-demand encoding of all RoboTwin RGB views."""

    return planner_then_verifier_request(
        turn,
        frame_encoder=encode_robotwin_rgb_frame,
    )


def create_robotwin_bridge(
    high_client: Any,
    low_client: Any,
    *,
    verifier_interval_steps: int | None = None,
) -> DualSystemBridge:
    """Compose a bridge without taking ownership of the RoboTwin loop.

    With no interval the high level plans once.  Supplying an interval also
    schedules verifier calls at that many simulator steps; callers with
    event/oracle boundaries can still construct :class:`DualSystemBridge`
    directly and inject their own trigger.
    """

    trigger = (
        None
        if verifier_interval_steps is None
        else periodic_high_level_trigger(verifier_interval_steps)
    )
    return DualSystemBridge(
        high_client,
        low_client,
        high_request_factory=robotwin_high_level_request,
        low_request_factory=low_level_request,
        high_level_trigger=trigger,
    )


__all__ = ["create_robotwin_bridge", "robotwin_high_level_request"]
