"""Regression test for the forced-boundary high/low controller."""

from __future__ import annotations

import unittest

from hy_vla.hierarchical import BurgerFriesHighLevelController, FrameRecord


class ForcedBoundaryControllerTest(unittest.TestCase):
    def test_initial_plan_then_one_unverified_five_second_handoff(self) -> None:
        fps = 25.0
        controller = BurgerFriesHighLevelController(
            forced_boundary_s=5.0,
            max_forced_boundaries=1,
            history_fps=1.0,
            source_fps=fps,
            max_memory_frames=256,
        )

        decisions = []
        for frame_id in range(260):
            timestamp_s = frame_id / fps
            controller.observe(
                FrameRecord(
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    camera_images={"head_camera": object()}
                    if frame_id % int(fps) == 0
                    else {},
                )
            )
            decision = controller.step(
                timestamp_s=timestamp_s,
                frame_id=frame_id,
            )
            if decision is not None:
                decisions.append(decision)
            controller.record_low_level_step(
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                state=[timestamp_s, 0.0, 0.0, 0.0],
                action=[0.01, 0.0, 0.0, 0.0],
                low_level_call_ms=4.0,
                model_forward_ms=20.0 if frame_id % 7 == 0 else 0.0,
                used_action_cache=(frame_id % 7 != 0),
            )

        self.assertEqual(2, len(decisions))

        initial, boundary = decisions
        self.assertEqual("initial", initial.trigger)
        self.assertEqual("Place the hamburg on the tray.", initial.next_subtask)
        self.assertIsNone(initial.execution_report)

        self.assertEqual("forced_boundary", boundary.trigger)
        self.assertEqual("Place the french fries on the tray.", boundary.next_subtask)
        self.assertIsNotNone(boundary.execution_report)
        report = boundary.execution_report
        assert report is not None
        self.assertEqual("unknown", report.status)
        self.assertEqual(125, report.low_level_steps)
        self.assertEqual(18, report.model_forward_calls)
        self.assertEqual([0, 25, 50, 75, 100, 125], report.sampled_frame_ids)
        self.assertAlmostEqual(5.0, report.end_timestamp_s - report.start_timestamp_s)

    def test_external_completion_handoff_disables_timer(self) -> None:
        fps = 25.0
        controller = BurgerFriesHighLevelController(
            forced_boundary_s=0.1,
            max_forced_boundaries=1,
            enable_forced_boundaries=False,
            history_fps=1.0,
            source_fps=fps,
        )

        initial = None
        for frame_id in range(51):
            timestamp_s = frame_id / fps
            controller.observe(
                FrameRecord(
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    camera_images={"head_camera": object()}
                    if frame_id % int(fps) == 0
                    else {},
                )
            )
            decision = controller.step(
                timestamp_s=timestamp_s,
                frame_id=frame_id,
            )
            if decision is not None:
                initial = decision
            controller.record_low_level_step(
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                state=[timestamp_s, 0.0, 0.0, 0.0],
                action=[0.01, 0.0, 0.0, 0.0],
                low_level_call_ms=4.0,
                model_forward_ms=20.0 if frame_id % 7 == 0 else 0.0,
                used_action_cache=(frame_id % 7 != 0),
            )

        self.assertIsNotNone(initial)
        handoff = controller.trigger_handoff(
            timestamp_s=2.04,
            frame_id=51,
            trigger="simulator_oracle_hamburger_placed",
            report_status="completed",
        )

        self.assertIsNotNone(handoff)
        assert handoff is not None
        self.assertEqual("simulator_oracle_hamburger_placed", handoff.trigger)
        self.assertEqual("Place the french fries on the tray.", handoff.next_subtask)
        self.assertIsNotNone(handoff.execution_report)
        report = handoff.execution_report
        assert report is not None
        self.assertEqual("completed", report.status)
        self.assertEqual(
            "simulator_oracle_hamburger_placed",
            report.trigger,
        )
        self.assertEqual(51, report.low_level_steps)


if __name__ == "__main__":
    unittest.main()
