"""Lazy request factories for the bundled high- and low-level services."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from highlow.contracts import Modality

from .dual_system import BridgeTurn


FramePayload = bytes | str | tuple[bytes | str, str]
FrameEncoder = Callable[[str, Modality], FramePayload | None]


def _high_level_frames(
    turn: BridgeTurn,
    frame_encoder: FrameEncoder | None = None,
) -> tuple[Any, ...]:
    """Build frames, optionally encoding simulator-native image payloads."""

    from highlow.inference.high_level import HighLevelFrame

    frames = []
    for name, modality in turn.observation.modalities.items():
        if not modality.schema_id.startswith("image."):
            continue
        data = modality.data
        encoding = modality.encoding.lower()
        if isinstance(data, bytes):
            mime_type = {
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "png": "image/png",
                "webp": "image/webp",
            }.get(encoding)
            if mime_type is None:
                continue
        elif isinstance(data, str) and (
            data.startswith("data:")
            or data.startswith("http://")
            or data.startswith("https://")
        ):
            mime_type = str(modality.metadata.get("mime_type", "image/jpeg"))
        elif frame_encoder is not None:
            encoded = frame_encoder(name, modality)
            if encoded is None:
                continue
            if isinstance(encoded, tuple):
                if len(encoded) != 2:
                    raise ValueError("frame encoder tuple must be (data, mime_type)")
                data, mime_type = encoded
            else:
                data = encoded
                mime_type = str(modality.metadata.get("mime_type", "image/jpeg"))
            if not isinstance(data, (bytes, str)):
                raise TypeError("frame encoder must return bytes, URL string, or None")
        else:
            continue
        frames.append(
            HighLevelFrame(
                data=data,
                mime_type=mime_type,
                frame_id=turn.observation.step_id,
                timestamp_s=modality.timestamp_s,
                source=name,
            )
        )
    return tuple(frames)


def planner_request(
    turn: BridgeTurn,
    *,
    frame_encoder: FrameEncoder | None = None,
) -> Any:
    """Build a bundled high-level planner request for one bridge turn."""

    from highlow.inference.high_level import HighLevelRequest

    return HighLevelRequest(
        request_context=turn.request_context,
        operation="planner",
        episode_context=turn.episode_context,
        task_instruction=turn.episode_context.instruction,
        structured_context={
            "planner_mode": "next",
            "current_subtask": turn.instruction,
            "instruction_revision": turn.instruction_revision,
            "high_level_call_count": turn.high_level_call_count,
            "step_id": turn.observation.step_id,
        },
        frames=_high_level_frames(turn, frame_encoder),
    )


def verifier_request(
    turn: BridgeTurn,
    *,
    frame_encoder: FrameEncoder | None = None,
) -> Any:
    """Build a bundled high-level verifier request for one bridge turn."""

    from highlow.inference.high_level import HighLevelRequest

    return HighLevelRequest(
        request_context=turn.request_context,
        operation="verifier",
        episode_context=turn.episode_context,
        task_instruction=turn.episode_context.instruction,
        structured_context={
            "current_subtask": turn.instruction,
            "instruction_revision": turn.instruction_revision,
            "high_level_call_count": turn.high_level_call_count,
            "step_id": turn.observation.step_id,
        },
        frames=_high_level_frames(turn, frame_encoder),
    )


def planner_then_verifier_request(
    turn: BridgeTurn,
    *,
    frame_encoder: FrameEncoder | None = None,
) -> Any:
    """Plan on the first high-level call, then verify later boundaries."""

    if turn.high_level_call_count == 0:
        return planner_request(turn, frame_encoder=frame_encoder)
    return verifier_request(turn, frame_encoder=frame_encoder)


def periodic_high_level_trigger(every_n_steps: int) -> Callable[[BridgeTurn], bool]:
    """Create a simple initial-plus-periodic high-level schedule."""

    if every_n_steps < 1:
        raise ValueError("every_n_steps must be positive")

    def trigger(turn: BridgeTurn) -> bool:
        return turn.high_level_call_count == 0 or (
            turn.observation.step_id % every_n_steps == 0
        )

    return trigger


def low_level_request(turn: BridgeTurn, high_response: Any) -> Any:
    """Build the bundled stateful, per-step low-level request."""

    from highlow.inference.low_level import LowLevelInferenceRequest

    conditioning = {
        "instruction": turn.instruction,
        "instruction_revision": turn.instruction_revision,
        "high_level_call_count": turn.high_level_call_count,
    }
    if high_response is not None:
        operation = (
            high_response.get("operation")
            if isinstance(high_response, Mapping)
            else getattr(high_response, "operation", None)
        )
        if operation is not None:
            conditioning["high_level_operation"] = getattr(
                operation, "value", str(operation)
            )
    return LowLevelInferenceRequest(
        observation=turn.observation,
        conditioning=conditioning,
        context=turn.request_context,
    )


__all__ = [
    "FrameEncoder",
    "FramePayload",
    "low_level_request",
    "periodic_high_level_trigger",
    "planner_request",
    "planner_then_verifier_request",
    "verifier_request",
]
