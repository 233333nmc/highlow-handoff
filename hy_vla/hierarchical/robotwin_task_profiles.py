"""Task profiles and simulator-only first-subtask oracles for RoboTwin.

The oracles in this module are benchmark instrumentation. They read the
simulator's internal state only to calibrate a fixed handoff time; they are not
used as a deployment completion detector or passed to the high-level model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


FirstSubtaskOracle = Callable[[Any], bool]
PhaseOracle = Callable[[Any, int], bool]


@dataclass(frozen=True, slots=True)
class RoboTwinTaskProfile:
    """Static high-level phases plus read-only simulator oracles."""

    task_name: str
    task_instruction: str
    phase_names: tuple[str, ...]
    phase_prompts: tuple[str, ...]
    first_subtask_complete: FirstSubtaskOracle
    phase_complete: PhaseOracle | None = None

    def is_phase_complete(self, env: Any, phase_index: int) -> bool:
        """Return whether a specific high-level phase is complete."""

        index = int(phase_index)
        if self.phase_complete is not None:
            return bool(self.phase_complete(env, index))
        return index == 0 and bool(self.first_subtask_complete(env))


def _position(actor_or_pose: Any) -> np.ndarray:
    pose = actor_or_pose.get_pose() if hasattr(actor_or_pose, "get_pose") else actor_or_pose
    value = pose.p if hasattr(pose, "p") else pose
    return np.asarray(value, dtype=float)


def _quaternion(actor: Any) -> np.ndarray:
    pose = actor.get_pose()
    return np.asarray(pose.q, dtype=float)


def _functional_point_xy(actor: Any, point_id: int) -> np.ndarray:
    point = _functional_point(actor, point_id)
    return point[:2]


def _functional_point(actor: Any, point_id: int) -> np.ndarray:
    try:
        point = actor.get_functional_point(point_id, "pose")
    except TypeError:
        point = actor.get_functional_point(point_id)
    value = point.p if hasattr(point, "p") else point
    return np.asarray(value, dtype=float)


def _left_open(env: Any) -> bool:
    return bool(env.is_left_gripper_open())


def _both_open(env: Any) -> bool:
    return bool(env.is_left_gripper_open()) and bool(env.is_right_gripper_open())


def _burger_first_subtask_complete(env: Any) -> bool:
    try:
        return (
            float(
                np.linalg.norm(
                    _functional_point_xy(env.tray, 0)
                    - _functional_point_xy(env.hamburg, 0)
                )
            )
            < 0.08
            and _left_open(env)
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


def _cans_first_subtask_complete(env: Any) -> bool:
    try:
        return (
            float(
                np.linalg.norm(
                    _position(env.object1)[:2]
                    - _functional_point_xy(env.plasticbox, 1)
                )
            )
            < 0.04
            and _left_open(env)
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


def _bottles_first_subtask_complete(env: Any) -> bool:
    try:
        target_xy = np.asarray([-0.45, 0.0], dtype=float)
        eps_xy = np.asarray([0.221, 0.325], dtype=float)
        for bottle in env.bottles:
            position = _position(bottle)
            if (
                np.all(np.abs(position[:2] - target_xy) < eps_xy)
                and 0.2 < float(position[2]) < 0.7
            ):
                return True
        return False
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


def _blocks_first_subtask_complete(env: Any) -> bool:
    try:
        target = np.asarray(env.block1_target_pose, dtype=float)
        position = _position(env.block1)
        return (
            float(np.linalg.norm(position[:2] - target[:2])) < 0.04
            and abs(float(position[2]) - float(target[2])) < 0.04
            and _both_open(env)
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


def _block_stacked_on(upper: Any, lower: Any) -> bool:
    upper_position = _position(upper)
    lower_position = _position(lower)
    target = np.asarray(
        [lower_position[0], lower_position[1], lower_position[2] + 0.05],
        dtype=float,
    )
    eps = np.asarray([0.025, 0.025, 0.012], dtype=float)
    return bool(np.all(np.abs(upper_position - target) < eps))


def _blocks_phase_complete(env: Any, phase_index: int) -> bool:
    try:
        if phase_index == 0:
            return _blocks_first_subtask_complete(env)
        if phase_index == 1:
            return (
                _blocks_first_subtask_complete(env)
                and _block_stacked_on(env.block2, env.block1)
                and _both_open(env)
            )
        if phase_index == 2:
            return (
                _blocks_first_subtask_complete(env)
                and _block_stacked_on(env.block2, env.block1)
                and _block_stacked_on(env.block3, env.block2)
                and _both_open(env)
            )
        return False
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


def _bowls_first_subtask_complete(env: Any) -> bool:
    try:
        target = np.asarray(env.bowl1_target_pose, dtype=float)
        position = _position(env.bowl1)
        return (
            float(np.linalg.norm(position[:2] - target[:2])) < 0.04
            and abs(float(position[2]) - float(target[2])) < 0.04
            and _both_open(env)
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


def _shoes_first_subtask_complete(env: Any) -> bool:
    try:
        position = _position(env.left_shoe)
        quaternion = _quaternion(env.left_shoe)
        if quaternion[0] < 0:
            quaternion = -quaternion
        target_xy = np.asarray([0.0, -0.17], dtype=float)
        target_quaternion = np.asarray([0.5, 0.5, -0.5, -0.5], dtype=float)
        target_z = float(_position(env.shoe_box)[2]) + 0.01
        return (
            np.all(np.abs(position[:2] - target_xy) < 0.05)
            and np.all(np.abs(quaternion - target_quaternion) < 0.07)
            and abs(float(position[2]) - target_z) < 0.03
            and _left_open(env)
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


def _hanging_mug_first_subtask_complete(env: Any) -> bool:
    try:
        mug_position = _position(env.mug)
        middle_position = np.asarray(env.middle_pos[:3], dtype=float)
        return (
            float(np.linalg.norm(mug_position[:2] - middle_position[:2])) < 0.06
            and abs(float(mug_position[2]) - float(middle_position[2])) < 0.08
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


def _put_object_cabinet_first_subtask_complete(env: Any) -> bool:
    try:
        qpos = np.asarray(env.cabinet.get_qpos(), dtype=float)
        return qpos.size > 0 and abs(float(qpos[0])) > 0.08
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


def _handover_block_first_subtask_complete(env: Any) -> bool:
    try:
        box_position = _position(env.box)
        middle_position = np.asarray(env.block_middle_pose[:3], dtype=float)
        return (
            float(np.linalg.norm(box_position[:2] - middle_position[:2])) < 0.08
            and abs(float(box_position[2]) - float(middle_position[2])) < 0.08
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


def _handover_mic_first_subtask_complete(env: Any) -> bool:
    try:
        microphone_position = _position(env.microphone)
        middle_position = np.asarray(env.handover_middle_pose[:3], dtype=float)
        return (
            float(np.linalg.norm(microphone_position[:2] - middle_position[:2])) < 0.08
            and abs(float(microphone_position[2]) - float(middle_position[2])) < 0.08
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


_PROFILES: dict[str, RoboTwinTaskProfile] = {
    "place_burger_fries": RoboTwinTaskProfile(
        task_name="place_burger_fries",
        task_instruction=(
            "Use dual arm to pick the hamburg and french fries and put them on the tray."
        ),
        phase_names=("handle hamburger", "handle french fries"),
        phase_prompts=(
            "Place the hamburg on the tray.",
            "Place the french fries on the tray.",
        ),
        first_subtask_complete=_burger_first_subtask_complete,
    ),
    "place_cans_plasticbox": RoboTwinTaskProfile(
        task_name="place_cans_plasticbox",
        task_instruction="Place both cans into the plastic box using the two arms.",
        phase_names=("place left can", "place right can"),
        phase_prompts=(
            "Place the left can into the plastic box.",
            "Place the right can into the plastic box.",
        ),
        first_subtask_complete=_cans_first_subtask_complete,
    ),
    "put_bottles_dustbin": RoboTwinTaskProfile(
        task_name="put_bottles_dustbin",
        task_instruction="Put all three bottles into the dustbin.",
        phase_names=("place first bottle", "place remaining bottles"),
        phase_prompts=(
            "Put one bottle into the dustbin.",
            "Put the remaining bottles into the dustbin.",
        ),
        first_subtask_complete=_bottles_first_subtask_complete,
    ),
    "stack_blocks_three": RoboTwinTaskProfile(
        task_name="stack_blocks_three",
        task_instruction="Stack the red, green, and blue blocks in order.",
        phase_names=("place red block", "stack green block", "stack blue block"),
        phase_prompts=(
            "Place the red block at the center of the table.",
            "Stack the green block on the red block.",
            "Stack the blue block on the green block.",
        ),
        first_subtask_complete=_blocks_first_subtask_complete,
        phase_complete=_blocks_phase_complete,
    ),
    "stack_bowls_three": RoboTwinTaskProfile(
        task_name="stack_bowls_three",
        task_instruction="Stack the three bowls at the center of the table.",
        phase_names=("place first bowl", "stack remaining bowls"),
        phase_prompts=(
            "Place the first bowl at the center of the table.",
            "Stack the remaining two bowls on the first bowl.",
        ),
        first_subtask_complete=_bowls_first_subtask_complete,
    ),
    "place_dual_shoes": RoboTwinTaskProfile(
        task_name="place_dual_shoes",
        task_instruction="Place both shoes into the shoe box with their toes pointing left.",
        phase_names=("place left shoe", "place right shoe"),
        phase_prompts=(
            "Place the left shoe into the left side of the shoe box with the toe pointing left.",
            "Place the right shoe into the right side of the shoe box with the toe pointing left.",
        ),
        first_subtask_complete=_shoes_first_subtask_complete,
    ),
    "hanging_mug": RoboTwinTaskProfile(
        task_name="hanging_mug",
        task_instruction="Hang the mug on the rack.",
        phase_names=("prepare mug for handover", "hang mug on rack"),
        phase_prompts=(
            "Place the mug at the middle handover position.",
            "Transfer the mug to the right arm and hang it on the rack.",
        ),
        first_subtask_complete=_hanging_mug_first_subtask_complete,
    ),
    "put_object_cabinet": RoboTwinTaskProfile(
        task_name="put_object_cabinet",
        task_instruction="Open the cabinet and put the object inside.",
        phase_names=("open cabinet", "place object in cabinet"),
        phase_prompts=(
            "Open the cabinet drawer.",
            "Pick up the object and place it inside the open cabinet.",
        ),
        first_subtask_complete=_put_object_cabinet_first_subtask_complete,
    ),
    "handover_block": RoboTwinTaskProfile(
        task_name="handover_block",
        task_instruction="Hand over the red block and place it on the blue target block.",
        phase_names=("move block to handover point", "place block on target"),
        phase_prompts=(
            "Move the red block to the center handover point.",
            "Transfer the red block to the other arm and place it on the blue target block.",
        ),
        first_subtask_complete=_handover_block_first_subtask_complete,
    ),
    "handover_mic": RoboTwinTaskProfile(
        task_name="handover_mic",
        task_instruction="Hand over the microphone from one arm to the other.",
        phase_names=("move microphone to handover point", "complete microphone handover"),
        phase_prompts=(
            "Move the microphone to the center handover point.",
            "Transfer the microphone to the other arm and lift it away from the handover point.",
        ),
        first_subtask_complete=_handover_mic_first_subtask_complete,
    ),
}


def get_robotwin_task_profile(task_name: str) -> RoboTwinTaskProfile:
    """Return the task profile used by the hierarchical benchmark."""

    normalized = str(task_name).strip()
    try:
        return _PROFILES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROFILES))
        raise ValueError(
            f"Unsupported hierarchical RoboTwin task {normalized!r}. "
            f"Supported tasks: {supported}."
        ) from exc


def supported_robotwin_task_names() -> tuple[str, ...]:
    """Return task names with a validated first-subtask profile."""

    return tuple(sorted(_PROFILES))
