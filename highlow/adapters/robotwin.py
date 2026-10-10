"""Pure conversion adapter for observations and actions used by RoboTwin."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import Any

from highlow.contracts import ActionEnvelope, EpisodeContext, Modality, ObservationEnvelope


HEAD_RGB = "camera.head.rgb"
LEFT_RGB = "camera.left.rgb"
RIGHT_RGB = "camera.right.rgb"
END_EFFECTOR_STATE = "robot.end_effector_state"

RGB_SCHEMA_ID = "image.rgb.uint8.hwc@1"
EE_ACTION_SCHEMA_ID = "robot.dual_arm.ee_pose_gripper.wxyz@1"
EE_STATE_SCHEMA_ID = EE_ACTION_SCHEMA_ID


def encode_robotwin_rgb_frame(name: str, modality: Modality) -> tuple[bytes, str] | None:
    """Encode a raw canonical RoboTwin RGB modality as JPEG on demand.

    Imports stay inside the call so simulator/bridge discovery remains light.
    The bridge invokes this only on scheduled high-level ticks; low-level
    inference continues to receive the original array without a copy.
    """

    if name not in {HEAD_RGB, LEFT_RGB, RIGHT_RGB}:
        return None
    if modality.schema_id != RGB_SCHEMA_ID:
        raise ValueError(
            f"RoboTwin RGB modality {name!r} has unsupported schema "
            f"{modality.schema_id!r}"
        )

    import numpy as np
    from PIL import Image

    image = np.asarray(modality.data)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"RoboTwin RGB modality {name!r} must have HWC shape")
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number):
            raise ValueError(f"RoboTwin RGB modality {name!r} must be numeric")
        if (
            not np.all(np.isfinite(image))
            or np.any(image < 0)
            or np.any(image > 255)
            or np.any(image != np.floor(image))
        ):
            raise ValueError(
                f"RoboTwin RGB modality {name!r} must contain integers in [0, 255]"
            )
        image = image.astype(np.uint8)

    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue(), "image/jpeg"


def _required_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"RoboTwin observation field {path!r} must be a mapping")
    return value


def _required(value: Mapping[str, Any], key: str, path: str) -> Any:
    try:
        return value[key]
    except KeyError as exc:
        full_path = f"{path}.{key}" if path else key
        raise KeyError(f"RoboTwin observation is missing {full_path!r}") from exc


def _pose_values(value: Any, path: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        # numpy arrays are not registered as Sequence, but expose a safe
        # iterable/length duck type.  Avoid importing numpy in this module.
        if not hasattr(value, "__iter__") or not hasattr(value, "__len__"):
            raise TypeError(f"RoboTwin observation field {path!r} must be a sequence")
    values = tuple(value)
    if len(values) != 7:
        raise ValueError(f"RoboTwin observation field {path!r} must contain 7 values")
    return values


class RoboTwinAdapter:
    """Translate RoboTwin dictionaries without owning its environment loop."""

    simulator_name = "robotwin"
    action_schema_id = EE_ACTION_SCHEMA_ID

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
        kwargs: dict[str, Any] = {
            "task_id": task_id,
            "instruction": instruction,
            "seed": seed,
            "simulator": self.simulator_name,
            "simulator_version": simulator_version,
            "metadata": metadata or {},
        }
        if episode_id is not None:
            kwargs["episode_id"] = episode_id
        return EpisodeContext(**kwargs)

    def encode_observation(
        self,
        context: EpisodeContext,
        observation: Mapping[str, Any],
        *,
        step_id: int,
        timestamp_s: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> ObservationEnvelope:
        root = _required_mapping(observation, "")
        sensors = _required_mapping(_required(root, "observation", ""), "observation")
        head = _required_mapping(
            _required(sensors, "head_camera", "observation"),
            "observation.head_camera",
        )
        left = _required_mapping(
            _required(sensors, "left_camera", "observation"),
            "observation.left_camera",
        )
        right = _required_mapping(
            _required(sensors, "right_camera", "observation"),
            "observation.right_camera",
        )
        endpose = _required_mapping(_required(root, "endpose", ""), "endpose")

        left_pose = _pose_values(
            _required(endpose, "left_endpose", "endpose"),
            "endpose.left_endpose",
        )
        right_pose = _pose_values(
            _required(endpose, "right_endpose", "endpose"),
            "endpose.right_endpose",
        )
        state = (
            *left_pose,
            _required(endpose, "left_gripper", "endpose"),
            *right_pose,
            _required(endpose, "right_gripper", "endpose"),
        )

        image_metadata = {"layout": "HWC", "color_space": "RGB"}
        modalities = {
            HEAD_RGB: Modality(
                name=HEAD_RGB,
                schema_id=RGB_SCHEMA_ID,
                data=_required(head, "rgb", "observation.head_camera"),
                timestamp_s=timestamp_s,
                encoding="raw",
                metadata=image_metadata,
            ),
            LEFT_RGB: Modality(
                name=LEFT_RGB,
                schema_id=RGB_SCHEMA_ID,
                data=_required(left, "rgb", "observation.left_camera"),
                timestamp_s=timestamp_s,
                encoding="raw",
                metadata=image_metadata,
            ),
            RIGHT_RGB: Modality(
                name=RIGHT_RGB,
                schema_id=RGB_SCHEMA_ID,
                data=_required(right, "rgb", "observation.right_camera"),
                timestamp_s=timestamp_s,
                encoding="raw",
                metadata=image_metadata,
            ),
            END_EFFECTOR_STATE: Modality(
                name=END_EFFECTOR_STATE,
                schema_id=EE_STATE_SCHEMA_ID,
                data=state,
                timestamp_s=timestamp_s,
                encoding="native",
                metadata={
                    "coordinate_frame": "robotwin",
                    "quaternion_order": "wxyz",
                    "arm_order": ("left", "right"),
                },
            ),
        }
        return ObservationEnvelope(
            episode_id=context.episode_id,
            step_id=step_id,
            timestamp_s=timestamp_s,
            modalities=modalities,
            metadata=metadata or {},
        )

    def decode_action(self, action: ActionEnvelope) -> Any:
        if action.schema_id != self.action_schema_id:
            raise ValueError(
                f"unsupported RoboTwin action schema {action.schema_id!r}; "
                f"expected {self.action_schema_id!r}"
            )
        if action.horizon != 1:
            raise ValueError("RoboTwinAdapter expects one action per simulator tick")
        data = action.data
        try:
            size = len(data)
        except TypeError as exc:
            raise TypeError("RoboTwin action data must be a 16-value sequence") from exc
        if size != 16:
            raise ValueError("RoboTwin action data must contain exactly 16 values")
        return data

    # Readable conversion aliases.  They perform no lifecycle operation.
    episode_context = create_episode_context
    observation_envelope = encode_observation
    action_from_envelope = decode_action


__all__ = [
    "EE_ACTION_SCHEMA_ID",
    "EE_STATE_SCHEMA_ID",
    "END_EFFECTOR_STATE",
    "HEAD_RGB",
    "LEFT_RGB",
    "RGB_SCHEMA_ID",
    "RIGHT_RGB",
    "RoboTwinAdapter",
    "encode_robotwin_rgb_frame",
]
