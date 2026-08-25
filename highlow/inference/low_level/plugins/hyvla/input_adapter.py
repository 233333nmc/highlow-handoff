"""Canonical observation to HyVLA batch conversion."""

from __future__ import annotations

from typing import Any

from highlow.contracts import ObservationEnvelope

from ...core import PredictionContractError
from .manifest import (
    END_EFFECTOR_STATE_MODALITY,
    HEAD_RGB_MODALITY,
    LEFT_RGB_MODALITY,
    RGB_SCHEMA_ID,
    RIGHT_RGB_MODALITY,
    STATE_ACTION_SCHEMA_ID,
)


class HyVLAInputAdapter:
    """Build the current ``HyVLAPolicyWrapper.get_action`` batch contract."""

    _CAMERA_BATCH_KEYS = {
        HEAD_RGB_MODALITY: "top_head",
        LEFT_RGB_MODALITY: "hand_left",
        RIGHT_RGB_MODALITY: "hand_right",
    }

    def __init__(self, *, max_state_dim: int = 32) -> None:
        if max_state_dim < 16:
            raise ValueError("max_state_dim must be at least 16")
        self.max_state_dim = max_state_dim

    def build_batch(
        self,
        observation: ObservationEnvelope,
        instruction: str,
    ) -> dict[str, Any]:
        # Lazy by design: plugin discovery and backend construction do not need
        # numerical or model frameworks installed.
        import numpy as np

        if not isinstance(observation, ObservationEnvelope):
            raise TypeError("observation must be an ObservationEnvelope")
        if not isinstance(instruction, str):
            raise TypeError("HyVLA instruction must be a string")

        batch: dict[str, Any] = {}
        expected_shape: tuple[int, int, int] | None = None
        for modality_name, policy_camera_name in self._CAMERA_BATCH_KEYS.items():
            modality = observation.require(modality_name)
            if modality.schema_id != RGB_SCHEMA_ID:
                raise PredictionContractError(
                    f"modality {modality_name!r} has schema {modality.schema_id!r}; "
                    f"expected {RGB_SCHEMA_ID!r}"
                )
            native_array = isinstance(modality.data, np.ndarray)
            image = np.asarray(modality.data)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise PredictionContractError(
                    f"modality {modality_name!r} must have HWC shape (H, W, 3)"
                )
            if native_array and image.dtype != np.uint8:
                raise PredictionContractError(
                    f"modality {modality_name!r} must contain uint8 pixels"
                )
            if not native_array:
                if not np.issubdtype(image.dtype, np.number):
                    raise PredictionContractError(
                        f"JSON modality {modality_name!r} must contain numbers"
                    )
                if (
                    not np.all(np.isfinite(image))
                    or np.any(image < 0)
                    or np.any(image > 255)
                    or np.any(image != np.floor(image))
                ):
                    raise PredictionContractError(
                        f"JSON modality {modality_name!r} must contain integers in [0, 255]"
                    )
                image = image.astype(np.uint8)
            shape = tuple(int(value) for value in image.shape)
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise PredictionContractError("all HyVLA RGB modalities must share a shape")

            batch[f"observation.images.{policy_camera_name}"] = (
                image.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
            )
            batch[f"raw_images.{policy_camera_name}"] = image

        state_modality = observation.require(END_EFFECTOR_STATE_MODALITY)
        if state_modality.schema_id != STATE_ACTION_SCHEMA_ID:
            raise PredictionContractError(
                f"modality {END_EFFECTOR_STATE_MODALITY!r} has schema "
                f"{state_modality.schema_id!r}; expected {STATE_ACTION_SCHEMA_ID!r}"
            )
        state = np.asarray(state_modality.data, dtype=np.float32)
        if state.ndim == 1:
            state = state[np.newaxis, ...]
        if state.ndim != 2 or state.shape != (1, 16):
            raise PredictionContractError(
                "robot.end_effector_state must have shape (16,) or (1, 16)"
            )
        padded_state = np.zeros((1, self.max_state_dim), dtype=np.float32)
        padded_state[:, :16] = state
        batch["observation.state"] = padded_state
        batch["task"] = [instruction]
        return batch


__all__ = ["HyVLAInputAdapter"]
