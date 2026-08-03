# coding=utf-8
# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboTwin entry point for Hy-VLA evaluation.

This is the single file imported by RoboTwin's ``script/eval_policy.py``
via ``importlib.import_module(<policy_name>)``. It exposes exactly four
public symbols (the contract RoboTwin expects):

* ``encode_obs(observation, instruction)`` -- dict packing helper used
  by ``eval``; exposed so user scripts can replay it offline.
* ``get_model(usr_args)`` -- factory returning a ``HyVLAPolicyWrapper``.
* ``eval(TASK_ENV, model, observation)`` -- per-step closed-loop hook.
* ``reset_model(model)`` -- per-episode hook.

Drop this directory under ``robotwin/policy/hy_vla/`` (or any other
RoboTwin policy slot) and point ``--config`` at the bundled
``deploy_policy.yml``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from hy_vla.hierarchical import (
    BurgerFriesHighLevelController,
    FrameRecord,
    Qwen25VlmBackend,
    Qwen36VllmBackend,
    SharedHunyuanVlmBackend,
    get_robotwin_task_profile,
)

from .policy_wrapper import HyVLAPolicyWrapper, build_policy


# ---------------------------------------------------------------------------
# Observation packing
# ---------------------------------------------------------------------------
def _to_chw_float(img: np.ndarray) -> np.ndarray:
    """``(H, W, 3) uint8`` -> ``(1, 3, H, W) float32 in [0, 1]``."""
    return (img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0)


def _pad_state(state: np.ndarray, max_state_dim: int = 32) -> np.ndarray:
    """Pad a 16-d EE state to ``max_state_dim`` with zeros."""
    if state.shape[-1] == max_state_dim:
        return state
    shape = list(state.shape)
    cur = shape[-1]
    shape[-1] = max_state_dim
    out = np.zeros(shape, dtype=state.dtype)
    out[..., :cur] = state
    return out


def encode_obs(observation: dict[str, Any], instruction: str) -> dict[str, Any]:
    """Pack a RoboTwin ``observation`` dict + language instruction into the
    batch format expected by ``HyVLAPolicyWrapper.get_action``.

    The dual-arm 16-d EE state uses RoboTwin's native quaternion layout
    (wxyz). The wxyz->xyzw flip happens inside the wrapper.
    """
    eepose_16d = np.array([
        *observation["endpose"]["left_endpose"],   # 7-d (xyz + quat_wxyz)
        observation["endpose"]["left_gripper"],    # 1-d
        *observation["endpose"]["right_endpose"],  # 7-d
        observation["endpose"]["right_gripper"],   # 1-d
    ], dtype=np.float32)

    head = observation["observation"]["head_camera"]["rgb"]
    left = observation["observation"]["left_camera"]["rgb"]
    right = observation["observation"]["right_camera"]["rgb"]

    return {
        # Visual inputs (single-frame; the MEM path replaces these later
        # with a (1, K, C, H, W) stack built from raw_images.* below).
        "observation.images.top_head": _to_chw_float(head),
        "observation.images.hand_left": _to_chw_float(left),
        "observation.images.hand_right": _to_chw_float(right),
        # Padded EE state, ``(1, max_state_dim)``.
        "observation.state": _pad_state(eepose_16d[np.newaxis, :], max_state_dim=32),
    # Language instruction (HyVLA.prepare_language expects a list).
        "task": [instruction],
        # Raw uint8 frames -- the MEM video-encoder path needs the
        # un-normalized HWC layout to grow its per-camera history buffer.
        "raw_images.top_head": head,
        "raw_images.hand_left": left,
        "raw_images.hand_right": right,
    }


# ---------------------------------------------------------------------------
# RoboTwin hooks
# ---------------------------------------------------------------------------
def get_model(usr_args: dict[str, Any]) -> HyVLAPolicyWrapper:
    """Factory called once per evaluation run by RoboTwin."""
    model = build_policy(usr_args)
    _maybe_attach_hierarchical_controller(model, usr_args)
    return model


def eval(TASK_ENV, model: HyVLAPolicyWrapper, observation: dict[str, Any]) -> None:  # noqa: A001
    """Per-step closed-loop hook.

    RoboTwin calls this in a tight loop; we pack the observation, query
    the wrapper for one 16-d action, and forward it back via
    ``TASK_ENV.take_action(..., action_type='ee')``.
    """
    controller = getattr(model, "high_level_controller", None)
    if controller is not None:
        # RoboTwin increments this once per completed rollout. Keep it with
        # each decision so steady-state latency analysis can drop warm-up
        # episodes without inferring episode boundaries from duplicate logs.
        model.hierarchical_episode_index = int(getattr(TASK_ENV, "test_num", 0))
    frame_id = int(
        getattr(
            TASK_ENV,
            "take_action_cnt",
            len(controller.buffer) if controller is not None else 0,
        )
    )
    source_fps = controller.source_fps if controller is not None else 25.0
    timestamp_s = frame_id / max(source_fps, 1e-6)

    instruction = _resolve_instruction(
        TASK_ENV,
        model,
        observation,
        frame_id=frame_id,
        timestamp_s=timestamp_s,
    )
    batch = encode_obs(observation, instruction)
    state = batch["observation.state"][0, :16].copy()
    action_started = time.perf_counter()
    action = model.get_action(batch)
    action_finished = time.perf_counter()
    pending_handoff = getattr(model, "pending_instruction_handoff", None)
    if pending_handoff is not None:
        ready_ns = int(pending_handoff["instruction_ready_ns"])
        first_action_ms = (time.perf_counter_ns() - ready_ns) / 1_000_000.0
        pending_handoff["decision"].timing_ms["instruction_ready_to_first_action"] = (
            first_action_ms
        )
        pending_handoff["decision"].timing_ms["instruction_ready_first_action_forward"] = (
            (action_finished - action_started) * 1000.0
        )
        _append_high_level_log(
            model,
            int(pending_handoff["frame_id"]),
            float(pending_handoff["timestamp_s"]),
            pending_handoff["decision"],
        )
        model.pending_instruction_handoff = None
        if pending_handoff["decision"].trigger != "initial":
            model.hierarchical_first_action_after_handoff_logged = True
    if controller is not None:
        controller.record_low_level_step(
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            state=state,
            action=action,
            low_level_call_ms=(action_finished - action_started) * 1000.0,
            model_forward_ms=float(getattr(model, "last_model_forward_ms", 0.0)),
            used_action_cache=bool(getattr(model, "last_action_used_cache", False)),
        )
    TASK_ENV.take_action(action, action_type="ee")
    _maybe_trigger_oracle_handoff(
        TASK_ENV,
        model,
        frame_id=frame_id,
        timestamp_s=timestamp_s,
    )


def reset_model(model: HyVLAPolicyWrapper) -> str:
    """Per-episode reset hook."""
    message = model.reset()
    controller = getattr(model, "high_level_controller", None)
    if controller is not None:
        controller.reset()
        model.high_level_current_instruction = None
        model.high_level_decisions = []
        model.pending_instruction_handoff = None
        model.hierarchical_oracle_streak = 0
        model.hierarchical_oracle_handoff_fired = False
        model.hierarchical_handoff_fired = False
        model.hierarchical_first_action_after_handoff_logged = False
    return message


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _maybe_attach_hierarchical_controller(
    model: HyVLAPolicyWrapper,
    usr_args: dict[str, Any],
) -> None:
    if not _as_bool(usr_args.get("hierarchical_enabled", False)):
        return

    task_name = str(usr_args.get("task_name", "place_burger_fries"))
    profile = get_robotwin_task_profile(task_name)

    backend_name = str(usr_args.get("hierarchical_backend", "scripted")).strip().lower()
    backend = None
    if backend_name == "shared_hunyuan_vlm":
        backend = SharedHunyuanVlmBackend(
            vlm=model.policy.model.dual_tower.vlm,
            checkpoint_path=str(usr_args.get("ckpt_path")),
            camera_name=str(
                usr_args.get("hierarchical_vlm_camera", "head_camera")
            ),
            max_images=int(usr_args.get("hierarchical_vlm_max_images", 8)),
            max_new_tokens=int(
                usr_args.get("hierarchical_vlm_max_new_tokens", 160)
            ),
        )
    elif backend_name == "qwen25_vl":
        qwen_model_path = usr_args.get("hierarchical_qwen_model_path")
        if not qwen_model_path:
            raise ValueError(
                "hierarchical_qwen_model_path is required when "
                "hierarchical_backend=qwen25_vl"
            )
        backend = Qwen25VlmBackend(
            model_path=str(qwen_model_path),
            device=str(usr_args.get("hierarchical_qwen_device", "auto")),
            camera_name=str(
                usr_args.get("hierarchical_vlm_camera", "head_camera")
            ),
            max_images=int(usr_args.get("hierarchical_vlm_max_images", 6)),
            max_new_tokens=int(
                usr_args.get("hierarchical_vlm_max_new_tokens", 128)
            ),
            attention_implementation=str(
                usr_args.get(
                    "hierarchical_qwen_attention_implementation",
                    "flash_attention_2",
                )
            ),
            min_pixels=int(
                usr_args.get(
                    "hierarchical_qwen_min_pixels",
                    256 * 28 * 28,
                )
            ),
            max_pixels=int(
                usr_args.get(
                    "hierarchical_qwen_max_pixels",
                    768 * 28 * 28,
                )
            ),
        )
    elif backend_name == "qwen36_vllm_mtp":
        qwen_server_url = usr_args.get("hierarchical_qwen36_server_url")
        if not qwen_server_url:
            raise ValueError(
                "hierarchical_qwen36_server_url is required when "
                "hierarchical_backend=qwen36_vllm_mtp"
            )
        backend = Qwen36VllmBackend(
            server_url=str(qwen_server_url),
            model_name=str(
                usr_args.get(
                    "hierarchical_qwen36_model_name",
                    "qwen3.6-27b-mtp",
                )
            ),
            camera_name=str(
                usr_args.get("hierarchical_vlm_camera", "head_camera")
            ),
            max_images=int(usr_args.get("hierarchical_vlm_max_images", 6)),
            max_new_tokens=int(
                usr_args.get("hierarchical_vlm_max_new_tokens", 64)
            ),
            request_timeout_s=float(
                usr_args.get("hierarchical_qwen36_request_timeout_s", 120.0)
            ),
            jpeg_quality=int(
                usr_args.get("hierarchical_qwen36_jpeg_quality", 90)
            ),
            image_selection=str(
                usr_args.get("hierarchical_qwen36_image_selection", "tail")
            ),
            image_max_side=int(
                usr_args.get("hierarchical_qwen36_image_max_side", 0)
            ),
        )
    elif backend_name != "scripted":
        raise ValueError(
            "hierarchical_backend must be 'scripted', 'shared_hunyuan_vlm', "
            "'qwen25_vl', or 'qwen36_vllm_mtp', "
            f"got {backend_name!r}"
        )

    handoff_trigger = str(
        usr_args.get("hierarchical_handoff_trigger", "forced_boundary")
    ).strip().lower()
    # Preserve the old burger-only name for existing benchmark scripts.
    if handoff_trigger == "burger_oracle":
        handoff_trigger = "task_oracle"
    if handoff_trigger not in {"forced_boundary", "task_oracle"}:
        raise ValueError(
            "hierarchical_handoff_trigger must be 'forced_boundary', "
            f"'task_oracle', or legacy 'burger_oracle', got {handoff_trigger!r}"
        )

    controller = BurgerFriesHighLevelController(
        task_instruction=str(
            usr_args.get(
                "hierarchical_task_instruction",
                profile.task_instruction,
            )
        ),
        task_name=profile.task_name,
        phase_names=profile.phase_names,
        phase_prompts=profile.phase_prompts,
        forced_boundary_s=float(
            usr_args.get(
                "hierarchical_forced_boundary_s",
                usr_args.get("hierarchical_phase_boundary_s", 5.0),
            )
        ),
        max_forced_boundaries=int(
            usr_args.get("hierarchical_max_forced_boundaries", 1)
        ),
        enable_forced_boundaries=(handoff_trigger == "forced_boundary"),
        recent_frames_length=int(usr_args.get("hierarchical_recent_frames_length", 8)),
        history_fps=float(usr_args.get("hierarchical_history_fps", 1.0)),
        source_fps=float(usr_args.get("hierarchical_source_fps", 25.0)),
        keyframe_merge_distance=int(usr_args.get("hierarchical_keyframe_merge_distance", 5)),
        memory_length=int(usr_args.get("hierarchical_memory_length", 8)),
        max_memory_frames=int(usr_args.get("hierarchical_max_memory_frames", 256)),
        backend=backend,
    )
    model.high_level_controller = controller
    model.high_level_instruction_mode = str(
        usr_args.get("hierarchical_instruction_mode", "subtask")
    )
    model.high_level_current_instruction = None
    model.high_level_decisions = []
    model.hierarchical_task_profile = profile
    model.hierarchical_handoff_trigger = handoff_trigger
    model.hierarchical_oracle_stability_steps = max(
        1,
        int(usr_args.get("hierarchical_oracle_stability_steps", 3)),
    )
    model.hierarchical_oracle_streak = 0
    model.hierarchical_oracle_handoff_fired = False
    model.hierarchical_handoff_fired = False
    model.hierarchical_first_action_after_handoff_logged = False
    # Calibration can stop immediately after an oracle timestamp is captured.
    # It does not mark the episode as successful and is not used for the
    # full forced-boundary latency evaluation.
    model.hierarchical_stop_after_oracle_handoff = _as_bool(
        usr_args.get("hierarchical_stop_after_oracle_handoff", False)
    )
    model.hierarchical_calibration_max_sim_time_s = max(
        0.0,
        float(usr_args.get("hierarchical_calibration_max_sim_time_s", 0.0)),
    )
    model.hierarchical_stop_after_measurement_handoff = _as_bool(
        usr_args.get("hierarchical_stop_after_measurement_handoff", False)
    )
    model.hierarchical_episode_index = None
    log_path = usr_args.get("hierarchical_log_path")
    model.high_level_log_path = Path(log_path) if log_path else None


def _maybe_trigger_oracle_handoff(
    TASK_ENV,
    model: HyVLAPolicyWrapper,
    *,
    frame_id: int,
    timestamp_s: float,
) -> None:
    """Trigger one high-level handoff after a profile-specific simulator event."""

    controller = getattr(model, "high_level_controller", None)
    if controller is None:
        return
    if getattr(model, "hierarchical_handoff_trigger", "") != "task_oracle":
        return
    if bool(getattr(model, "hierarchical_oracle_handoff_fired", False)):
        return
    if bool(getattr(TASK_ENV, "eval_success", False)):
        return

    profile = getattr(model, "hierarchical_task_profile", None)
    if profile is None:
        return
    if not profile.first_subtask_complete(TASK_ENV):
        model.hierarchical_oracle_streak = 0
        return

    model.hierarchical_oracle_streak = (
        int(getattr(model, "hierarchical_oracle_streak", 0)) + 1
    )
    if model.hierarchical_oracle_streak < int(
        getattr(model, "hierarchical_oracle_stability_steps", 3)
    ):
        return

    # `take_action` has already advanced the environment by one control tick.
    # The latest captured observation is frame_id, so the report uses that
    # frame as visual evidence and the post-action tick as the event timestamp.
    event_frame_id = int(getattr(TASK_ENV, "take_action_cnt", frame_id + 1))
    event_timestamp_s = (
        event_frame_id / max(float(controller.source_fps), 1e-6)
    )
    handoff_started = time.perf_counter_ns()
    decision = controller.trigger_handoff(
        timestamp_s=event_timestamp_s,
        frame_id=event_frame_id,
        trigger=f"simulator_oracle_{profile.task_name}_first_subtask_completed",
        report_status="completed",
    )
    handoff_finished = time.perf_counter_ns()
    if decision is None:
        return

    decision.timing_ms["handoff_request_to_decision"] = (
        handoff_finished - handoff_started
    ) / 1_000_000.0
    decision.timing_ms["oracle_stability_steps"] = float(
        getattr(model, "hierarchical_oracle_stability_steps", 3)
    )
    instruction_update_started = time.perf_counter_ns()
    model.high_level_current_instruction = _format_low_level_instruction(
        TASK_ENV.get_instruction(),
        decision,
        mode=getattr(model, "high_level_instruction_mode", "subtask"),
    )
    stale_action_count = model.invalidate_action_cache()
    instruction_update_finished = time.perf_counter_ns()
    decision.timing_ms["instruction_update_total"] = (
        instruction_update_finished - instruction_update_started
    ) / 1_000_000.0
    decision.timing_ms["handoff_to_instruction_ready"] = (
        instruction_update_finished - handoff_started
    ) / 1_000_000.0
    decision.timing_ms["stale_action_cache_discarded"] = float(stale_action_count)
    model.pending_instruction_handoff = {
        "decision": decision,
        "frame_id": event_frame_id,
        "timestamp_s": event_timestamp_s,
        "instruction_ready_ns": instruction_update_finished,
    }
    model.high_level_decisions.append(
        {
            "frame_id": event_frame_id,
            "timestamp_s": event_timestamp_s,
            **decision.to_dict(),
        }
    )
    _append_high_level_log(model, event_frame_id, event_timestamp_s, decision)
    model.hierarchical_oracle_handoff_fired = True
    model.hierarchical_handoff_fired = True


def _resolve_instruction(
    TASK_ENV,
    model: HyVLAPolicyWrapper,
    observation: dict[str, Any],
    *,
    frame_id: int,
    timestamp_s: float,
) -> str:
    original_instruction = TASK_ENV.get_instruction()
    controller = getattr(model, "high_level_controller", None)
    if controller is None:
        return original_instruction

    camera_images: dict[str, Any] = {}
    if controller.should_capture_visual(timestamp_s):
        head_camera = observation["observation"]["head_camera"]["rgb"]
        # The slow controller starts with one camera at 1 FPS. This avoids
        # retaining 25 FPS multi-view frames while preserving real pixels
        # for a future high-level VLM backend.
        camera_images["head_camera"] = np.ascontiguousarray(head_camera).copy()
    frame = FrameRecord(
        frame_id=frame_id,
        timestamp_s=timestamp_s,
        caption=f"RoboTwin {controller.task_name} observation",
        camera_images=camera_images,
        metadata={
            "task_instruction": original_instruction,
            "visual_capture": bool(camera_images),
            "has_head_camera": "head_camera" in observation.get("observation", {}),
            "has_left_camera": "left_camera" in observation.get("observation", {}),
            "has_right_camera": "right_camera" in observation.get("observation", {}),
        },
    )
    controller.observe(frame)
    # The forced-boundary measurement starts after per-step observation
    # bookkeeping and ends once the refreshed instruction is published. It
    # deliberately excludes startup/model-loading/warm-up and the following
    # low-level action forward.
    boundary_started = time.perf_counter_ns()
    decision = controller.step(timestamp_s=timestamp_s, frame_id=frame_id)
    if decision is not None:
        handoff_started = time.perf_counter_ns()
        next_instruction = _format_low_level_instruction(
            original_instruction,
            decision,
            mode=getattr(model, "high_level_instruction_mode", "subtask"),
        )
        stale_action_count = model.invalidate_action_cache()
        handoff_finished = time.perf_counter_ns()
        decision.timing_ms["instruction_update_total"] = (
            handoff_finished - handoff_started
        ) / 1_000_000.0
        # The controller decision has already been computed at this point.
        # This field measures only publication of the text command and cache
        # invalidation; `total` / `backend_online_total` measure planning.
        decision.timing_ms["decision_to_instruction_ready"] = (
            handoff_finished - handoff_started
        ) / 1_000_000.0
        if decision.trigger == "forced_boundary":
            decision.timing_ms["boundary_to_instruction_ready"] = (
                handoff_finished - boundary_started
            ) / 1_000_000.0
        decision.timing_ms["stale_action_cache_discarded"] = float(stale_action_count)
        model.high_level_current_instruction = next_instruction
        model.pending_instruction_handoff = {
            "decision": decision,
            "frame_id": frame_id,
            "timestamp_s": timestamp_s,
            "instruction_ready_ns": handoff_finished,
        }
        model.high_level_decisions.append(
            {
                "frame_id": frame_id,
                "timestamp_s": timestamp_s,
                **decision.to_dict(),
            }
        )
        _append_high_level_log(model, frame_id, timestamp_s, decision)
        model.hierarchical_handoff_fired = True

    return model.high_level_current_instruction or original_instruction


def _format_low_level_instruction(
    original_instruction: str,
    decision,
    *,
    mode: str,
) -> str:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "original":
        return original_instruction
    if normalized_mode == "augmented":
        return (
            f"{original_instruction}\n"
            f"Current subtask: {decision.next_subtask}\n"
            f"Memory: phase={decision.phase_index}; keyframes={decision.memory_keyframes}"
        )
    if normalized_mode == "memory_only":
        return decision.memory_text
    return decision.next_subtask


def _append_high_level_log(
    model: HyVLAPolicyWrapper,
    frame_id: int,
    timestamp_s: float,
    decision,
) -> None:
    log_path = getattr(model, "high_level_log_path", None)
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "frame_id": frame_id,
        "timestamp_s": timestamp_s,
        "episode_index": getattr(model, "hierarchical_episode_index", None),
        **decision.to_dict(),
    }
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = ["encode_obs", "get_model", "eval", "reset_model"]
