"""Simulator-safe bridge interfaces and coordinator."""

from .dual_system import BridgeTurn, DualSystemBridge
from .protocols import HighLevelClient, InferenceClient, LowLevelClient, SimulatorAdapter
from .service_factories import (
    FrameEncoder,
    FramePayload,
    low_level_request,
    periodic_high_level_trigger,
    planner_request,
    planner_then_verifier_request,
    verifier_request,
)

__all__ = [
    "BridgeTurn",
    "DualSystemBridge",
    "FrameEncoder",
    "FramePayload",
    "HighLevelClient",
    "InferenceClient",
    "LowLevelClient",
    "SimulatorAdapter",
    "low_level_request",
    "periodic_high_level_trigger",
    "planner_request",
    "planner_then_verifier_request",
    "verifier_request",
]
