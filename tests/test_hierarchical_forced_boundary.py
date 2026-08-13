"""Regression test for the forced-boundary high/low controller."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time
import unittest

import numpy as np

from hy_vla.hierarchical import (
    BurgerFriesHighLevelController,
    FrameRecord,
    JsonTextHighLevelBackend,
)
from robotwin_eval import deploy_policy


class _Decision:
    def __init__(
        self,
        *,
        trigger: str,
        next_subtask: str,
        should_refresh_low_level: bool = True,
    ) -> None:
        self.trigger = trigger
        self.next_subtask = next_subtask
        self.phase_index = 0
        self.context_frames: list[int] = []
        self.confidence = 1.0
        self.should_refresh_low_level = should_refresh_low_level
        self.reason = trigger
        self.prompt = ""
        self.raw_prediction = ""
        self.prediction_parse_ok = True
        self.prediction_parse_error = None
        self.execution_report = None
        self.timing_ms: dict[str, float] = {}
        self.planner_action = "switch" if should_refresh_low_level else "continue"
        self.subtask_status = "completed" if should_refresh_low_level else "unfinished"
        self.planner_mode = "test"
        self.planner_candidate_subtask = ""
        self.planner_candidate_source = ""
        self.request_kind = "test"
        self.output_schema = "test"
        self.high_level_status = self.subtask_status
        self.verifier_next_subtask = ""
        self.evidence = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "trigger": self.trigger,
            "next_subtask": self.next_subtask,
            "should_refresh_low_level": self.should_refresh_low_level,
            "timing_ms": dict(self.timing_ms),
            "planner_action": self.planner_action,
            "subtask_status": self.subtask_status,
            "high_level_status": self.high_level_status,
            "verifier_next_subtask": self.verifier_next_subtask,
            "evidence": self.evidence,
        }


class _AsyncController:
    source_fps = 25.0
    task_name = "place_burger_fries"

    def __init__(self) -> None:
        self.observed: list[FrameRecord] = []
        self.begin_calls = 0
        self.finished_prediction: object | None = None

    def should_capture_visual(self, timestamp_s: float) -> bool:
        return True

    def observe(self, frame: FrameRecord) -> None:
        self.observed.append(frame)

    def step(self, *, timestamp_s: float, frame_id: int):
        return _Decision(
            trigger="initial",
            next_subtask="Place the hamburg on the tray.",
        )

    def begin_step(self, *, timestamp_s: float, frame_id: int):
        self.begin_calls += 1
        if self.begin_calls > 1:
            return None
        return SimpleNamespace(
            timestamp_s=timestamp_s,
            request_started_ns=time.perf_counter_ns(),
            initial=False,
        )

    def predict_request(self, request: object) -> object:
        return object()

    def finish_step(
        self,
        request: object,
        prediction: object,
        *,
        commit_timestamp_s: float | None = None,
        commit_frame_id: int | None = None,
    ):
        del commit_timestamp_s, commit_frame_id
        self.finished_prediction = prediction
        return _Decision(
            trigger="forced_boundary",
            next_subtask="Place the french fries on the tray.",
        )


class _AsyncModel:
    def __init__(self, controller: _AsyncController) -> None:
        self.high_level_controller = controller
        self.high_level_current_instruction = None
        self.high_level_decisions: list[dict[str, object]] = []
        self.pending_instruction_handoff = None
        self.high_level_instruction_mode = "subtask"
        self.high_level_log_path = None
        self.hierarchical_async_enabled = True
        self.hierarchical_async_executor = ThreadPoolExecutor(max_workers=1)
        self.hierarchical_async_future = None
        self.hierarchical_async_request = None
        self.hierarchical_async_request_id = 0
        self.hierarchical_async_drop_stale_s = 0.0
        self.hierarchical_handoff_fired = False
        self.cache_invalidations = 0
        self.action_cache = [object(), object()]

    def invalidate_action_cache(self) -> int:
        self.cache_invalidations += 1
        stale_actions = len(self.action_cache)
        self.action_cache.clear()
        return stale_actions


class _TaskEnv:
    def get_instruction(self) -> str:
        return "Put the burger and french fries on the tray."


class ForcedBoundaryControllerTest(unittest.TestCase):
    def test_publish_continue_decision_does_not_clear_low_level_cache(self) -> None:
        controller = _AsyncController()
        model = _AsyncModel(controller)
        model.high_level_current_instruction = "Place the hamburg on the tray."
        decision = _Decision(
            trigger="forced_boundary",
            next_subtask="Place the french fries on the tray.",
            should_refresh_low_level=False,
        )

        instruction = deploy_policy._publish_high_level_decision(
            _TaskEnv(),
            model,
            original_instruction="Put the burger and french fries on the tray.",
            decision=decision,
            frame_id=25,
            timestamp_s=1.0,
        )

        self.assertEqual("Place the hamburg on the tray.", instruction)
        self.assertEqual(0, model.cache_invalidations)
        self.assertFalse(model.hierarchical_handoff_fired)
        self.assertIsNotNone(model.pending_instruction_handoff)
        self.assertEqual(1, len(model.high_level_decisions))
        self.assertEqual(
            0.0,
            model.high_level_decisions[0]["timing_ms"]["stale_action_cache_discarded"],
        )
        timing = model.high_level_decisions[0]["timing_ms"]
        self.assertEqual(2.0, timing["action_cache_len_before_high_level_commit"])
        self.assertEqual(2.0, timing["action_cache_len_after_high_level_commit"])
        self.assertEqual(0.0, timing["action_cache_cleared"])
        self.assertEqual(1.0, timing["same_decision_kept_action_cache"])
        model.hierarchical_async_executor.shutdown(wait=True)

    def test_async_worker_returns_current_instruction_until_prediction_ready(self) -> None:
        controller = _AsyncController()
        model = _AsyncModel(controller)
        observation = {
            "observation": {
                "head_camera": {
                    "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                },
            },
        }

        first_instruction = deploy_policy._resolve_instruction(
            _TaskEnv(),
            model,
            observation,
            frame_id=0,
            timestamp_s=0.0,
        )

        self.assertEqual("Place the hamburg on the tray.", first_instruction)
        self.assertEqual(1, controller.begin_calls)
        self.assertIsNotNone(model.hierarchical_async_future)
        self.assertEqual(1, model.cache_invalidations)

        assert model.hierarchical_async_future is not None
        model.hierarchical_async_future.result(timeout=1.0)
        second_instruction = deploy_policy._resolve_instruction(
            _TaskEnv(),
            model,
            observation,
            frame_id=1,
            timestamp_s=0.04,
        )

        self.assertEqual("Place the french fries on the tray.", second_instruction)
        self.assertEqual(2, model.cache_invalidations)
        self.assertIsNone(model.hierarchical_async_future)
        self.assertIsNone(model.hierarchical_async_request)
        self.assertIsNotNone(controller.finished_prediction)
        self.assertEqual(2, len(model.high_level_decisions))
        async_decision = model.high_level_decisions[-1]
        self.assertEqual("forced_boundary", async_decision["trigger"])
        timing = async_decision["timing_ms"]
        assert isinstance(timing, dict)
        self.assertIn("request_to_instruction_ready", timing)
        self.assertIn("async_response_age", timing)
        self.assertIn("action_cache_len_before_high_level_commit", timing)
        self.assertIn("action_cache_len_after_high_level_commit", timing)
        model.hierarchical_async_executor.shutdown(wait=True)

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
                instruction="Place the hamburg on the tray.",
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
        self.assertEqual("Place the hamburg on the tray.", report.low_level_instruction)
        self.assertEqual(
            {"Place the hamburg on the tray.": 18},
            report.model_forward_instruction_counts,
        )
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

    def test_minimal_prompt_schema_commits_next_subtask(self) -> None:
        controller = BurgerFriesHighLevelController(
            forced_boundary_s=1.0,
            max_forced_boundaries=1,
            history_fps=1.0,
            source_fps=25.0,
            prompt_mode="minimal",
            backend=JsonTextHighLevelBackend(
                lambda **_kwargs: '{"next_subtask":"Place the french fries on the tray."}'
            ),
        )
        for frame_id in range(26):
            timestamp_s = frame_id / 25.0
            controller.observe(
                FrameRecord(
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    camera_images={"head_camera": object()}
                    if frame_id % 25 == 0
                    else {},
                )
            )
            initial = controller.step(timestamp_s=timestamp_s, frame_id=frame_id)
            if initial is not None:
                break

        self.assertIsNotNone(initial)
        assert initial is not None
        self.assertEqual("Place the french fries on the tray.", initial.next_subtask)
        self.assertIn('"next_subtask"', initial.raw_prediction)
        self.assertNotIn("confidence", initial.prompt)

    def test_minimal_status_can_continue_without_refreshing_low_level(self) -> None:
        def generate_fn(**kwargs: object) -> str:
            if kwargs["execution_report"] is None:
                return (
                    '{"action":"switch",'
                    '"next_subtask":"Place the hamburg on the tray.",'
                    '"subtask_status":"unknown"}'
                )
            return (
                '{"action":"continue",'
                '"next_subtask":"Place the hamburg on the tray.",'
                '"subtask_status":"unfinished"}'
            )

        controller = BurgerFriesHighLevelController(
            forced_boundary_s=1.0,
            max_forced_boundaries=2,
            history_fps=1.0,
            source_fps=25.0,
            prompt_mode="minimal_status",
            backend=JsonTextHighLevelBackend(generate_fn),
        )
        decisions = []
        for frame_id in range(27):
            timestamp_s = frame_id / 25.0
            controller.observe(
                FrameRecord(
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    camera_images={"head_camera": object()}
                    if frame_id % 25 == 0
                    else {},
                )
            )
            decision = controller.step(timestamp_s=timestamp_s, frame_id=frame_id)
            if decision is not None:
                decisions.append(decision)
            controller.record_low_level_step(
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                state=[timestamp_s, 0.0, 0.0, 0.0],
                action=[0.01, 0.0, 0.0, 0.0],
                low_level_call_ms=1.0,
                model_forward_ms=2.0,
                used_action_cache=False,
            )

        self.assertEqual(2, len(decisions))
        initial, boundary = decisions
        self.assertTrue(initial.should_refresh_low_level)
        self.assertEqual("switch", initial.planner_action)
        self.assertFalse(boundary.should_refresh_low_level)
        self.assertEqual("continue", boundary.planner_action)
        self.assertEqual("unfinished", boundary.subtask_status)
        self.assertEqual("Place the hamburg on the tray.", boundary.next_subtask)
        self.assertEqual(0, boundary.phase_index)
        self.assertIn(
            "Active subtask: Place the hamburg on the tray.",
            boundary.memory_text,
        )
        self.assertIn("action", boundary.prompt)
        self.assertNotIn("confidence", boundary.prompt)
        self.assertNotIn("Current time", boundary.prompt)
        self.assertNotIn("steps=", boundary.prompt)
        self.assertNotIn("duration=", boundary.prompt)

    def test_completion_status_uses_profile_planner_candidate(self) -> None:
        backend_calls = 0

        def generate_fn(**kwargs: object) -> str:
            nonlocal backend_calls
            backend_calls += 1
            self.assertIsNotNone(kwargs["execution_report"])
            return '{"completed":true}'

        controller = BurgerFriesHighLevelController(
            forced_boundary_s=1.0,
            max_forced_boundaries=1,
            history_fps=1.0,
            source_fps=25.0,
            prompt_mode="completion_status",
            planner_mode="profile_all",
            backend=JsonTextHighLevelBackend(generate_fn),
        )
        decisions = []
        for frame_id in range(27):
            timestamp_s = frame_id / 25.0
            controller.observe(
                FrameRecord(
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    camera_images={"head_camera": object()}
                    if frame_id % 25 == 0
                    else {},
                )
            )
            decision = controller.step(timestamp_s=timestamp_s, frame_id=frame_id)
            if decision is not None:
                decisions.append(decision)
            controller.record_low_level_step(
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                state=[timestamp_s, 0.0, 0.0, 0.0],
                action=[0.01, 0.0, 0.0, 0.0],
                low_level_call_ms=1.0,
                model_forward_ms=2.0,
                used_action_cache=False,
            )

        self.assertEqual(2, len(decisions))
        initial, boundary = decisions
        self.assertEqual("Place the hamburg on the tray.", initial.next_subtask)
        self.assertEqual(1.0, initial.timing_ms["backend_predict_skipped"])
        self.assertEqual(1, backend_calls)

        self.assertTrue(boundary.should_refresh_low_level)
        self.assertEqual("switch", boundary.planner_action)
        self.assertEqual("completed", boundary.subtask_status)
        self.assertEqual("completed", boundary.high_level_status)
        self.assertEqual("", boundary.evidence)
        self.assertEqual("profile_all", boundary.planner_mode)
        self.assertEqual("profile_all", boundary.planner_candidate_source)
        self.assertEqual(
            "Place the french fries on the tray.",
            boundary.planner_candidate_subtask,
        )
        self.assertEqual(
            "Place the french fries on the tray.",
            boundary.next_subtask,
        )
        self.assertNotIn("Suggested next subtask", boundary.prompt)
        self.assertIn("completed=true", boundary.prompt)
        self.assertNotIn("status=", boundary.prompt)

    def test_vlm_plan_all_generates_subtasks_with_repeated_next_prompts(self) -> None:
        calls: list[str | None] = []

        def generate_fn(**kwargs: object) -> str:
            calls.append(kwargs.get("output_schema"))
            if len(calls) == 1:
                return '{"next_subtask":"Move the burger onto the tray."}'
            if len(calls) == 2:
                return '{"next_subtask":"Move the fries onto the tray."}'
            return '{"completed":true}'

        controller = BurgerFriesHighLevelController(
            forced_boundary_s=1.0,
            max_forced_boundaries=1,
            history_fps=1.0,
            source_fps=25.0,
            prompt_mode="completion_status",
            planner_mode="vlm_plan_all",
            backend=JsonTextHighLevelBackend(generate_fn),
        )
        decisions = []
        for frame_id in range(27):
            timestamp_s = frame_id / 25.0
            controller.observe(
                FrameRecord(
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    camera_images={"head_camera": object()}
                    if frame_id % 25 == 0
                    else {},
                )
            )
            decision = controller.step(timestamp_s=timestamp_s, frame_id=frame_id)
            if decision is not None:
                decisions.append(decision)
            controller.record_low_level_step(
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                state=[timestamp_s, 0.0, 0.0, 0.0],
                action=[0.01, 0.0, 0.0, 0.0],
                low_level_call_ms=1.0,
                model_forward_ms=2.0,
                used_action_cache=False,
            )

        self.assertEqual(["minimal", "minimal", "completion_bool"], calls)
        self.assertEqual(2, len(decisions))
        initial, boundary = decisions
        self.assertEqual("Move the burger onto the tray.", initial.next_subtask)
        self.assertEqual("planner_all", initial.request_kind)
        self.assertEqual("minimal", initial.output_schema)
        self.assertIn("backend_plan_all_step_1_predict", initial.timing_ms)
        self.assertEqual("Move the fries onto the tray.", boundary.next_subtask)
        self.assertEqual("vlm_plan_all_cache", boundary.planner_candidate_source)
        self.assertEqual("completion_bool", boundary.output_schema)

    def test_vlm_plan_next_calls_model_after_completed_verifier(self) -> None:
        calls: list[str | None] = []

        def generate_fn(**kwargs: object) -> str:
            calls.append(kwargs.get("output_schema"))
            if len(calls) == 1:
                return '{"next_subtask":"Move the burger onto the tray."}'
            if kwargs.get("output_schema") == "completion_bool":
                return '{"completed":true}'
            return '{"next_subtask":"Move the fries onto the tray."}'

        controller = BurgerFriesHighLevelController(
            forced_boundary_s=1.0,
            max_forced_boundaries=1,
            history_fps=1.0,
            source_fps=25.0,
            prompt_mode="completion_status",
            planner_mode="vlm_plan_next",
            backend=JsonTextHighLevelBackend(generate_fn),
        )
        decisions = []
        for frame_id in range(27):
            timestamp_s = frame_id / 25.0
            controller.observe(
                FrameRecord(
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    camera_images={"head_camera": object()}
                    if frame_id % 25 == 0
                    else {},
                )
            )
            decision = controller.step(timestamp_s=timestamp_s, frame_id=frame_id)
            if decision is not None:
                decisions.append(decision)
            controller.record_low_level_step(
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                state=[timestamp_s, 0.0, 0.0, 0.0],
                action=[0.01, 0.0, 0.0, 0.0],
                low_level_call_ms=1.0,
                model_forward_ms=2.0,
                used_action_cache=False,
            )

        self.assertEqual(["next_subtask", "completion_bool", "next_subtask"], calls)
        self.assertEqual(2, len(decisions))
        initial, boundary = decisions
        self.assertEqual("Move the burger onto the tray.", initial.next_subtask)
        self.assertEqual("planner_next", initial.request_kind)
        self.assertEqual("Move the fries onto the tray.", boundary.next_subtask)
        self.assertEqual("switch", boundary.planner_action)
        self.assertEqual("completed", boundary.subtask_status)
        self.assertIn("backend_next_planner_predict", boundary.timing_ms)

    def test_vlm_plan_next_uses_prefetched_subtask_when_ready(self) -> None:
        calls: list[tuple[str | None, bool, str]] = []

        def generate_fn(**kwargs: object) -> str:
            prompt = str(kwargs.get("prompt") or "")
            calls.append(
                (
                    kwargs.get("output_schema"),
                    bool(kwargs.get("allow_text_only")),
                    prompt,
                )
            )
            if len(calls) == 1:
                return '{"next_subtask":"Move the burger onto the tray."}'
            if "Generate the next object-level robot subtask" in prompt:
                return '{"next_subtask":"Move the fries onto the tray."}'
            return '{"completed":true}'

        controller = BurgerFriesHighLevelController(
            forced_boundary_s=1.0,
            max_forced_boundaries=1,
            history_fps=1.0,
            source_fps=25.0,
            prompt_mode="completion_status",
            planner_mode="vlm_plan_next",
            backend=JsonTextHighLevelBackend(generate_fn),
        )
        controller.observe(
            FrameRecord(
                frame_id=0,
                timestamp_s=0.0,
                camera_images={"head_camera": object()},
            )
        )
        initial = controller.step(timestamp_s=0.0, frame_id=0)
        self.assertIsNotNone(initial)
        self.assertEqual("Move the burger onto the tray.", initial.next_subtask)

        prefetch_request = controller.begin_prefetch_next_request(
            timestamp_s=0.0,
            frame_id=0,
        )
        self.assertIsNotNone(prefetch_request)
        assert prefetch_request is not None
        prefetch_prediction = controller.predict_request(prefetch_request)
        self.assertTrue(
            controller.finish_prefetch_next_request(
                prefetch_request,
                prefetch_prediction,
                ready_timestamp_s=0.2,
            )
        )

        boundary = None
        for frame_id in range(1, 27):
            timestamp_s = frame_id / 25.0
            controller.observe(
                FrameRecord(
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    camera_images={"head_camera": object()}
                    if frame_id % 25 == 0
                    else {},
                )
            )
            decision = controller.step(timestamp_s=timestamp_s, frame_id=frame_id)
            if decision is not None:
                boundary = decision
            controller.record_low_level_step(
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                state=[timestamp_s, 0.0, 0.0, 0.0],
                action=[0.01, 0.0, 0.0, 0.0],
                low_level_call_ms=1.0,
                model_forward_ms=2.0,
                used_action_cache=False,
            )

        self.assertEqual(
            ["next_subtask", "next_subtask", "completion_bool"],
            [call[0] for call in calls],
        )
        self.assertTrue(calls[1][1])
        self.assertIsNotNone(boundary)
        assert boundary is not None
        self.assertEqual("Move the fries onto the tray.", boundary.next_subtask)
        self.assertEqual("completed", boundary.subtask_status)
        self.assertEqual(1.0, boundary.timing_ms["backend_next_planner_prefetch_used"])
        self.assertNotIn("backend_next_planner_predict", boundary.timing_ms)

    def test_unseen_visual_context_sends_only_frames_not_seen_by_high_level(self) -> None:
        seen_recent_frame_ids: list[list[int]] = []

        def generate_fn(**kwargs: object) -> str:
            recent_frames = kwargs["recent_frames"]
            assert isinstance(recent_frames, list)
            seen_recent_frame_ids.append(
                [int(frame.frame_id) for frame in recent_frames]
            )
            if len(seen_recent_frame_ids) == 1:
                return '{"next_subtask":"Place the hamburg on the tray."}'
            return '{"next_subtask":"Place the french fries on the tray."}'

        controller = BurgerFriesHighLevelController(
            forced_boundary_s=2.0,
            max_forced_boundaries=1,
            history_fps=1.0,
            source_fps=25.0,
            prompt_mode="minimal",
            visual_context_mode="unseen",
            backend=JsonTextHighLevelBackend(generate_fn),
        )
        decisions = []
        for frame_id in range(52):
            timestamp_s = frame_id / 25.0
            controller.observe(
                FrameRecord(
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    camera_images={"head_camera": object()}
                    if frame_id % 25 == 0
                    else {},
                )
            )
            decision = controller.step(timestamp_s=timestamp_s, frame_id=frame_id)
            if decision is not None:
                decisions.append(decision)
            controller.record_low_level_step(
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                state=[timestamp_s, 0.0, 0.0, 0.0],
                action=[0.01, 0.0, 0.0, 0.0],
                low_level_call_ms=1.0,
                model_forward_ms=2.0,
                used_action_cache=False,
            )

        self.assertEqual([[0], [25, 50]], seen_recent_frame_ids)
        self.assertEqual(2, len(decisions))
        self.assertEqual([25, 50], decisions[1].context_frames)
        self.assertEqual("unseen_memory", decisions[1].visual_context_mode)

    def test_next_planner_unseen_context_keeps_frames_until_verifier(self) -> None:
        seen_recent_frame_ids: list[list[int]] = []

        def generate_fn(**kwargs: object) -> str:
            recent_frames = kwargs["recent_frames"]
            assert isinstance(recent_frames, list)
            seen_recent_frame_ids.append(
                [int(frame.frame_id) for frame in recent_frames]
            )
            if len(seen_recent_frame_ids) == 1:
                return '{"next_subtask":"Place the hamburg on the tray."}'
            if len(seen_recent_frame_ids) == 2:
                return '{"completed":true}'
            return '{"next_subtask":"Place the french fries on the tray."}'

        controller = BurgerFriesHighLevelController(
            forced_boundary_s=2.17,
            max_forced_boundaries=1,
            history_fps=0.5,
            source_fps=25.0,
            prompt_mode="completion_status",
            planner_mode="vlm_plan_next",
            visual_context_mode="unseen_memory",
            backend=JsonTextHighLevelBackend(generate_fn),
        )
        decisions = []
        for frame_id in range(56):
            timestamp_s = frame_id / 25.0
            controller.observe(
                FrameRecord(
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    camera_images={"head_camera": object()}
                    if frame_id in {0, 50, 55}
                    else {},
                )
            )
            decision = controller.step(timestamp_s=timestamp_s, frame_id=frame_id)
            if decision is not None:
                decisions.append(decision)
            controller.record_low_level_step(
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                state=[timestamp_s, 0.0, 0.0, 0.0],
                action=[0.01, 0.0, 0.0, 0.0],
                low_level_call_ms=1.0,
                model_forward_ms=2.0,
                used_action_cache=False,
            )

        self.assertEqual([[], [0, 50, 55], []], seen_recent_frame_ids)
        self.assertEqual(2, len(decisions))
        self.assertEqual([0, 50, 55], decisions[1].context_frames)
        self.assertEqual("verifier", decisions[1].request_kind)
        self.assertEqual("vlm_plan_next", decisions[1].planner_mode)
        self.assertEqual("unseen_memory", decisions[1].visual_context_mode)


if __name__ == "__main__":
    unittest.main()
