"""Tests for task-configurable high-level handoff profiles."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from hy_vla.hierarchical import (
    BurgerFriesHighLevelController,
    get_robotwin_task_profile,
    supported_robotwin_task_names,
)


class _Pose:
    def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0)) -> None:
        self.p = np.asarray(p, dtype=float)
        self.q = np.asarray(q, dtype=float)


class _Actor:
    def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0), points=None) -> None:
        self.pose = _Pose(p, q)
        self.points = points or {}

    def get_pose(self):
        return self.pose

    def get_functional_point(self, point_id):
        return self.points[point_id]


class RobotwinTaskProfileTest(unittest.TestCase):
    def test_five_new_tasks_are_supported(self) -> None:
        supported = set(supported_robotwin_task_names())
        self.assertTrue(
            {
                "place_cans_plasticbox",
                "put_bottles_dustbin",
                "stack_blocks_three",
                "stack_bowls_three",
                "place_dual_shoes",
            }.issubset(supported)
        )

    def test_first_subtask_oracles(self) -> None:
        cans = SimpleNamespace(
            plasticbox=_Actor([0, 0, 0], points={1: np.array([0.1, -0.1, 0.7])}),
            object1=_Actor([0.11, -0.11, 0.7]),
            is_left_gripper_open=lambda: True,
        )
        self.assertTrue(
            get_robotwin_task_profile("place_cans_plasticbox").first_subtask_complete(cans)
        )

        dustbin = SimpleNamespace(
            bottles=[
                _Actor([-0.45, 0.0, 0.35]),
                _Actor([0.3, 0.2, 0.74]),
                _Actor([-0.1, 0.1, 0.74]),
            ]
        )
        self.assertTrue(
            get_robotwin_task_profile("put_bottles_dustbin").first_subtask_complete(
                dustbin
            )
        )

        blocks = SimpleNamespace(
            block1_target_pose=[0.0, -0.13, 0.75, 0, 1, 0, 0],
            block1=_Actor([0.0, -0.13, 0.75]),
            is_left_gripper_open=lambda: True,
            is_right_gripper_open=lambda: True,
        )
        self.assertTrue(
            get_robotwin_task_profile("stack_blocks_three").first_subtask_complete(
                blocks
            )
        )

        bowls = SimpleNamespace(
            bowl1_target_pose=np.array([0.0, -0.1, 0.76]),
            bowl1=_Actor([0.0, -0.1, 0.76]),
            is_left_gripper_open=lambda: True,
            is_right_gripper_open=lambda: True,
        )
        self.assertTrue(
            get_robotwin_task_profile("stack_bowls_three").first_subtask_complete(
                bowls
            )
        )

        shoes = SimpleNamespace(
            shoe_box=_Actor([0.0, -0.13, 0.74]),
            left_shoe=_Actor(
                [0.0, -0.17, 0.75],
                q=[0.5, 0.5, -0.5, -0.5],
            ),
            is_left_gripper_open=lambda: True,
        )
        self.assertTrue(
            get_robotwin_task_profile("place_dual_shoes").first_subtask_complete(shoes)
        )

    def test_controller_accepts_task_profile_phases(self) -> None:
        controller = BurgerFriesHighLevelController(
            task_name="stack_blocks_three",
            task_instruction="Stack blocks.",
            phase_names=("place red", "finish stack"),
            phase_prompts=("Place red.", "Finish stack."),
            enable_forced_boundaries=False,
        )
        initial = controller.step(timestamp_s=0.0, frame_id=0)
        self.assertIsNotNone(initial)
        assert initial is not None
        self.assertEqual("Place red.", initial.next_subtask)

        handoff = controller.trigger_handoff(
            timestamp_s=1.0,
            frame_id=25,
            trigger="simulator_oracle_stack_blocks_three_first_subtask_completed",
        )
        self.assertIsNotNone(handoff)
        assert handoff is not None
        self.assertEqual("Finish stack.", handoff.next_subtask)


if __name__ == "__main__":
    unittest.main()
