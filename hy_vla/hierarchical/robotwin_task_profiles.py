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


@dataclass(frozen=True, slots=True)
class RoboTwinTaskProfile:
    """Static high-level phases plus a read-only first-subtask oracle."""

    task_name: str
    task_instruction: str
    phase_names: tuple[str, ...]
    phase_prompts: tuple[str, ...]
    first_subtask_complete: FirstSubtaskOracle


def _position(actor_or_pose: Any) -> np.ndarray:
    pose = actor_or_pose.get_pose() if hasattr(actor_or_pose, "get_pose") else actor_or_pose
    value = pose.p if hasattr(pose, "p") else pose
    return np.asarray(value, dtype=float)


def _quaternion(actor: Any) -> np.ndarray:
    pose = actor.get_pose()
    return np.asarray(pose.q, dtype=float)


def _functional_point_xy(actor: Any, point_id: int) -> np.ndarray:
    point = actor.get_functional_point(point_id)
    value = point.p if hasattr(point, "p") else point
    return np.asarray(value, dtype=float)[:2]


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
        phase_names=("place red block", "stack green and blue blocks"),
        phase_prompts=(
            "Place the red block at the center of the table.",
            "Stack the green block on the red block, then stack the blue block on the green block.",
        ),
        first_subtask_complete=_blocks_first_subtask_complete,
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
