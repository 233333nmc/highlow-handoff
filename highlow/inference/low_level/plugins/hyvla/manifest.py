"""Framework-free HyVLA schemas and capability declaration."""

from ...core import ContextMode, PolicyManifest, PredictionMode, TemporalMode


HEAD_RGB_MODALITY = "camera.head.rgb"
LEFT_RGB_MODALITY = "camera.left.rgb"
RIGHT_RGB_MODALITY = "camera.right.rgb"
END_EFFECTOR_STATE_MODALITY = "robot.end_effector_state"

RGB_SCHEMA_ID = "image.rgb.uint8.hwc@1"
STATE_ACTION_SCHEMA_ID = "robot.dual_arm.ee_pose_gripper.wxyz@1"


HYVLA_MANIFEST = PolicyManifest(
    policy_id="hyvla",
    policy_version="1",
    backend_id="hyvla",
    context_mode=ContextMode.EPISODE,
    temporal_mode=TemporalMode.HISTORY,
    # The current wrapper caches its model-produced chunk internally and this
    # backend returns one executable action per request.
    prediction_mode=PredictionMode.SINGLE_ACTION,
    observation_schema_ids=(RGB_SCHEMA_ID, STATE_ACTION_SCHEMA_ID),
    action_schema_ids=(STATE_ACTION_SCHEMA_ID,),
    required_modalities=(
        HEAD_RGB_MODALITY,
        LEFT_RGB_MODALITY,
        RIGHT_RGB_MODALITY,
        END_EFFECTOR_STATE_MODALITY,
    ),
    capabilities={
        "instruction_conditioning": True,
        "instruction_updates": True,
        "action_cache": True,
    },
    metadata={"native_action_layout": "dual-arm xyz+quat_wxyz+gripper"},
)


__all__ = [
    "END_EFFECTOR_STATE_MODALITY",
    "HEAD_RGB_MODALITY",
    "HYVLA_MANIFEST",
    "LEFT_RGB_MODALITY",
    "RGB_SCHEMA_ID",
    "RIGHT_RGB_MODALITY",
    "STATE_ACTION_SCHEMA_ID",
]
