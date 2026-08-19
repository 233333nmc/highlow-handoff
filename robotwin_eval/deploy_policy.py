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

from concurrent.futures import Future, ThreadPoolExecutor
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from hy_vla.hierarchical import (
    BurgerFriesHighLevelController,
    FrameRecord,
    Gemma4VllmBackend,
    HighLevelRequest,
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


def _hold_action_from_observation(observation: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            *observation["endpose"]["left_endpose"],
            observation["endpose"]["left_gripper"],
            *observation["endpose"]["right_endpose"],
            observation["endpose"]["right_gripper"],
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# RoboTwin hooks
# ---------------------------------------------------------------------------
def get_model(usr_args: dict[str, Any]) -> HyVLAPolicyWrapper:
    """Factory called once per evaluation run by RoboTwin."""
    model = build_policy(usr_args)
    override = usr_args.get("low_level_instruction_override")
    model.low_level_instruction_override = None if override is None else str(override)
    trace_path = usr_args.get("low_level_instruction_trace_path")
    model.low_level_instruction_trace_path = Path(trace_path) if trace_path else None
    model.low_level_instruction_trace_every_step = _as_bool(
        usr_args.get("low_level_instruction_trace_every_step", False)
    )
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
    pause_remaining = int(
        getattr(model, "hierarchical_handoff_pause_remaining_steps", 0)
    )
    if pause_remaining > 0:
        action = _hold_action_from_observation(observation)
        model.hierarchical_handoff_pause_remaining_steps = pause_remaining - 1
        model.last_action_used_cache = False
        model.last_model_forward_ms = 0.0
        model.last_action_call_ms = 0.0
        model.last_action_chunk_wxyz = None
        _append_low_level_instruction_trace(
            model,
            TASK_ENV,
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            instruction=instruction,
            action=action,
            pause_step=True,
            pause_remaining_steps=pause_remaining,
        )
        TASK_ENV.take_action(action, action_type="ee")
        _maybe_trigger_oracle_handoff(
            TASK_ENV,
            model,
            frame_id=frame_id,
            timestamp_s=timestamp_s,
        )
        return

    action_started = time.perf_counter()
    action_started_ns = time.perf_counter_ns()
    action = model.get_action(batch)
    action_finished = time.perf_counter()
    action_finished_ns = time.perf_counter_ns()
    pending_handoff = getattr(model, "pending_instruction_handoff", None)
    if pending_handoff is not None:
        ready_ns = int(pending_handoff["instruction_ready_ns"])
        ready_to_start_ms = (action_started_ns - ready_ns) / 1_000_000.0
        action_call_ms = (action_finished_ns - action_started_ns) / 1_000_000.0
        first_action_ms = (action_finished_ns - ready_ns) / 1_000_000.0
        pending_handoff["decision"].timing_ms[
            "instruction_ready_to_low_level_start"
        ] = ready_to_start_ms
        pending_handoff["decision"].timing_ms["instruction_ready_to_first_action"] = (
            first_action_ms
        )
        pending_handoff["decision"].timing_ms["instruction_ready_first_action_forward"] = (
            action_call_ms
        )
        pending_handoff["decision"].timing_ms["low_level_resume_action_call"] = (
            action_call_ms
        )
        pending_handoff["decision"].timing_ms["low_level_resume_model_forward"] = (
            float(getattr(model, "last_model_forward_ms", 0.0))
        )
        pending_handoff["decision"].timing_ms["low_level_resume_used_action_cache"] = (
            1.0 if bool(getattr(model, "last_action_used_cache", False)) else 0.0
        )
        boundary_to_ready = pending_handoff["decision"].timing_ms.get(
            "boundary_to_instruction_ready"
        )
        if boundary_to_ready is not None:
            pending_handoff["decision"].timing_ms["boundary_to_first_action"] = (
                float(boundary_to_ready) + first_action_ms
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
            instruction=instruction,
        )
    _append_low_level_instruction_trace(
        model,
        TASK_ENV,
        frame_id=frame_id,
        timestamp_s=timestamp_s,
        instruction=instruction,
        action=action,
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
        future = getattr(model, "hierarchical_async_future", None)
        if future is not None and not future.done():
            future.cancel()
        model.hierarchical_async_future = None
        model.hierarchical_async_request = None
        model.hierarchical_async_request_id = int(
            getattr(model, "hierarchical_async_request_id", 0)
        ) + 1
        prefetch_future = getattr(model, "hierarchical_prefetch_future", None)
        if prefetch_future is not None and not prefetch_future.done():
            prefetch_future.cancel()
        model.hierarchical_prefetch_future = None
        model.hierarchical_prefetch_request = None
        model.hierarchical_oracle_streak = 0
        model.hierarchical_oracle_phase_index = -1
        model.hierarchical_oracle_handoff_fired = False
        model.hierarchical_handoff_fired = False
        model.hierarchical_decision_handoff_logged = False
        model.hierarchical_first_action_after_handoff_logged = False
        model.hierarchical_handoff_pause_remaining_steps = 0
    return message


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_phase_prompts_json(
    value: Any,
    *,
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    if value is None or str(value).strip() == "":
        return fallback
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        if "|||" in text:
            return tuple(text.split("|||"))
        raise
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("hierarchical_phase_prompts_json must be a non-empty JSON list")
    if not all(isinstance(item, str) for item in parsed):
        raise ValueError("hierarchical_phase_prompts_json entries must be strings")
    return tuple(parsed)


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
            output_schema=str(
                usr_args.get("hierarchical_qwen36_output_schema", "legacy")
            ),
            mtp_speculative_tokens=int(
                usr_args.get("hierarchical_qwen36_mtp_speculative_tokens", 1)
            ),
        )
    elif backend_name == "qwen35_vllm_mtp":
        qwen_server_url = usr_args.get("hierarchical_qwen35_server_url")
        if not qwen_server_url:
            raise ValueError(
                "hierarchical_qwen35_server_url is required when "
                "hierarchical_backend=qwen35_vllm_mtp"
            )
        backend = Qwen36VllmBackend(
            server_url=str(qwen_server_url),
            model_name=str(
                usr_args.get(
                    "hierarchical_qwen35_model_name",
                    "qwen3.5-9b-mtp",
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
                usr_args.get("hierarchical_qwen35_request_timeout_s", 120.0)
            ),
            jpeg_quality=int(
                usr_args.get("hierarchical_qwen35_jpeg_quality", 90)
            ),
            image_selection=str(
                usr_args.get("hierarchical_qwen35_image_selection", "tail")
            ),
            image_max_side=int(
                usr_args.get("hierarchical_qwen35_image_max_side", 0)
            ),
            output_schema=str(
                usr_args.get("hierarchical_qwen35_output_schema", "legacy")
            ),
            backend_label="Qwen3.5 vLLM",
            mtp_speculative_tokens=int(
                usr_args.get("hierarchical_qwen35_mtp_speculative_tokens", 2)
            ),
        )
    elif backend_name == "gemma4_vllm":
        backend = Gemma4VllmBackend(
            server_url=str(
                usr_args.get(
                    "hierarchical_gemma4_server_url",
                    "http://127.0.0.1:8002",
                )
            ),
            model_name=str(
                usr_args.get("hierarchical_gemma4_model_name", "gemma-4-12B-it")
            ),
            camera_name=str(
                usr_args.get("hierarchical_vlm_camera", "head_camera")
            ),
            max_images=int(usr_args.get("hierarchical_vlm_max_images", 1)),
            max_new_tokens=int(
                usr_args.get("hierarchical_vlm_max_new_tokens", 48)
            ),
            request_timeout_s=float(
                usr_args.get("hierarchical_gemma4_request_timeout_s", 120.0)
            ),
            jpeg_quality=int(
                usr_args.get("hierarchical_gemma4_jpeg_quality", 90)
            ),
            image_selection=str(
                usr_args.get("hierarchical_gemma4_image_selection", "latest")
            ),
            image_max_side=int(
                usr_args.get("hierarchical_gemma4_image_max_side", 0)
            ),
            output_schema=str(
                usr_args.get("hierarchical_gemma4_output_schema", "minimal_status")
            ),
        )
    elif backend_name != "scripted":
        raise ValueError(
            "hierarchical_backend must be 'scripted', 'shared_hunyuan_vlm', "
            "'qwen25_vl', 'qwen35_vllm_mtp', 'qwen36_vllm_mtp', "
            "or 'gemma4_vllm', "
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

    phase_prompts = _parse_phase_prompts_json(
        usr_args.get("hierarchical_phase_prompts_json"),
        fallback=profile.phase_prompts,
    )
    phase_names = (
        profile.phase_names
        if len(phase_prompts) == len(profile.phase_names)
        else tuple(f"phase {idx}" for idx in range(len(phase_prompts)))
    )
    controller = BurgerFriesHighLevelController(
        task_instruction=str(
            usr_args.get(
                "hierarchical_task_instruction",
                profile.task_instruction,
            )
        ),
        task_name=profile.task_name,
        phase_names=phase_names,
        phase_prompts=phase_prompts,
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
        memory_length=int(usr_args.get("hierarchical_memory_length", 8)),
        max_memory_frames=int(usr_args.get("hierarchical_max_memory_frames", 256)),
        prompt_mode=str(usr_args.get("hierarchical_prompt_mode", "legacy")),
        planner_mode=str(usr_args.get("hierarchical_planner_mode", "inline_vlm")),
        visual_context_mode=str(
            usr_args.get("hierarchical_visual_context_mode", "recent")
        ),
        include_phase_profile_hints=_as_bool(
            usr_args.get("hierarchical_include_phase_profile_hints", True)
        ),
        backend=backend,
    )
    model.high_level_controller = controller
    model.hierarchical_planner_mode = str(
        usr_args.get("hierarchical_planner_mode", "inline_vlm")
    )
    model.high_level_instruction_mode = str(
        usr_args.get("hierarchical_instruction_mode", "subtask")
    )
    model.high_level_current_instruction = None
    model.high_level_decisions = []
    model.hierarchical_task_profile = profile
    model.hierarchical_handoff_trigger = handoff_trigger
    model.hierarchical_async_enabled = _as_bool(
        usr_args.get("hierarchical_async_enabled", False)
    )
    model.hierarchical_async_executor = (
        ThreadPoolExecutor(max_workers=1)
        if bool(model.hierarchical_async_enabled)
        else None
    )
    model.hierarchical_async_future = None
    model.hierarchical_async_request = None
    model.hierarchical_async_request_id = 0
    prefetch_planner_enabled = (
        str(model.hierarchical_planner_mode).strip().lower() == "vlm_plan_next"
    )
    model.hierarchical_prefetch_executor = (
        ThreadPoolExecutor(max_workers=1)
        if prefetch_planner_enabled
        else None
    )
    model.hierarchical_prefetch_future = None
    model.hierarchical_prefetch_request = None
    model.hierarchical_prefetch_request_id = 0
    model.hierarchical_async_drop_stale_s = max(
        0.0,
        float(usr_args.get("hierarchical_async_drop_stale_s", 0.0)),
    )
    model.hierarchical_force_visual_on_decision = _as_bool(
        usr_args.get("hierarchical_force_visual_on_decision", False)
    )
    model.hierarchical_oracle_stability_steps = max(
        1,
        int(usr_args.get("hierarchical_oracle_stability_steps", 3)),
    )
    model.hierarchical_oracle_streak = 0
    model.hierarchical_oracle_phase_index = -1
    model.hierarchical_oracle_handoff_fired = False
    model.hierarchical_handoff_fired = False
    model.hierarchical_decision_handoff_logged = False
    model.hierarchical_first_action_after_handoff_logged = False
    model.hierarchical_handoff_pause_steps = max(
        0,
        int(usr_args.get("hierarchical_handoff_pause_steps", 0)),
    )
    model.hierarchical_handoff_pause_remaining_steps = 0
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
    model.hierarchical_stop_after_decision_handoff = _as_bool(
        usr_args.get("hierarchical_stop_after_decision_handoff", False)
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
    """Trigger high-level handoffs after profile-specific simulator events."""

    controller = getattr(model, "high_level_controller", None)
    if controller is None:
        return
    if getattr(model, "hierarchical_handoff_trigger", "") != "task_oracle":
        return
    if bool(getattr(TASK_ENV, "eval_success", False)):
        return

    profile = getattr(model, "hierarchical_task_profile", None)
    if profile is None:
        return

    active_phase_index = int(getattr(controller, "_active_phase_index", 0))
    phase_count = len(getattr(controller, "phase_prompts", ()) or ())
    if phase_count <= 0 or active_phase_index >= phase_count - 1:
        return
    if int(getattr(model, "hierarchical_oracle_phase_index", -1)) != active_phase_index:
        model.hierarchical_oracle_phase_index = active_phase_index
        model.hierarchical_oracle_streak = 0

    phase_complete_fn = getattr(profile, "is_phase_complete", None)
    try:
        phase_complete = (
            bool(phase_complete_fn(TASK_ENV, active_phase_index))
            if callable(phase_complete_fn)
            else (
                active_phase_index == 0
                and bool(profile.first_subtask_complete(TASK_ENV))
            )
        )
    except Exception:  # noqa: BLE001 - instrumentation must not fail eval.
        phase_complete = False
    if not phase_complete:
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
        trigger=(
            f"simulator_oracle_{profile.task_name}_phase_"
            f"{active_phase_index}_completed"
        ),
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
    decision.timing_ms["oracle_completed_phase_index"] = float(active_phase_index)
    decision.timing_ms["oracle_next_phase_index"] = float(decision.phase_index)
    instruction_update_started = time.perf_counter_ns()
    action_cache_before = _action_cache_len(model)
    model.high_level_current_instruction = _format_low_level_instruction(
        TASK_ENV.get_instruction(),
        decision,
        mode=getattr(model, "high_level_instruction_mode", "subtask"),
    )
    stale_action_count = model.invalidate_action_cache()
    action_cache_after = _action_cache_len(model)
    instruction_update_finished = time.perf_counter_ns()
    decision.timing_ms["instruction_update_total"] = (
        instruction_update_finished - instruction_update_started
    ) / 1_000_000.0
    decision.timing_ms["handoff_to_instruction_ready"] = (
        instruction_update_finished - handoff_started
    ) / 1_000_000.0
    decision.timing_ms["action_cache_len_before_high_level_commit"] = float(
        action_cache_before
    )
    decision.timing_ms["action_cache_len_after_high_level_commit"] = float(
        action_cache_after
    )
    decision.timing_ms["stale_action_cache_discarded"] = float(stale_action_count)
    decision.timing_ms["action_cache_cleared"] = (
        1.0 if stale_action_count > 0 else 0.0
    )
    decision.timing_ms["same_decision_kept_action_cache"] = 0.0
    model.pending_instruction_handoff = {
        "decision": decision,
        "frame_id": event_frame_id,
        "timestamp_s": event_timestamp_s,
        "instruction_ready_ns": instruction_update_finished,
    }
    _arm_handoff_pause(model, decision)
    model.high_level_decisions.append(
        {
            "frame_id": event_frame_id,
            "timestamp_s": event_timestamp_s,
            **decision.to_dict(),
        }
    )
    _append_high_level_log(model, event_frame_id, event_timestamp_s, decision)
    model.hierarchical_oracle_streak = 0
    model.hierarchical_oracle_phase_index = int(decision.phase_index)
    model.hierarchical_oracle_handoff_fired = int(decision.phase_index) >= (
        phase_count - 1
    )
    model.hierarchical_handoff_fired = True
    if int(decision.phase_index) < phase_count - 1:
        _maybe_start_prefetch_next_planner(
            model,
            timestamp_s=event_timestamp_s,
            frame_id=event_frame_id,
        )


def _predict_high_level_request(controller, request: HighLevelRequest):
    return controller.predict_request(request)


def _poll_prefetch_next_planner(
    model: HyVLAPolicyWrapper,
    *,
    timestamp_s: float,
) -> None:
    future: Future | None = getattr(model, "hierarchical_prefetch_future", None)
    request: HighLevelRequest | None = getattr(
        model,
        "hierarchical_prefetch_request",
        None,
    )
    if future is None or request is None or not future.done():
        return

    model.hierarchical_prefetch_future = None
    model.hierarchical_prefetch_request = None
    if future.cancelled():
        return

    controller = getattr(model, "high_level_controller", None)
    if controller is None:
        return
    try:
        prediction = future.result()
    except Exception as exc:  # noqa: BLE001 - keep eval alive and expose error.
        model.hierarchical_prefetch_error = str(exc)
        return
    finish_prefetch = getattr(controller, "finish_prefetch_next_request", None)
    if finish_prefetch is not None:
        finish_prefetch(
            request,
            prediction,
            ready_timestamp_s=timestamp_s,
        )


def _maybe_start_prefetch_next_planner(
    model: HyVLAPolicyWrapper,
    *,
    timestamp_s: float,
    frame_id: int,
) -> None:
    controller = getattr(model, "high_level_controller", None)
    executor: ThreadPoolExecutor | None = getattr(
        model,
        "hierarchical_prefetch_executor",
        None,
    )
    if controller is None or executor is None:
        return

    _poll_prefetch_next_planner(model, timestamp_s=timestamp_s)
    future: Future | None = getattr(model, "hierarchical_prefetch_future", None)
    if future is not None and not future.done():
        return

    begin_prefetch = getattr(controller, "begin_prefetch_next_request", None)
    if begin_prefetch is None:
        return
    request = begin_prefetch(timestamp_s=timestamp_s, frame_id=frame_id)
    if request is None:
        return

    model.hierarchical_prefetch_request_id = int(
        getattr(model, "hierarchical_prefetch_request_id", 0)
    ) + 1
    model.hierarchical_prefetch_request = request
    model.hierarchical_prefetch_future = executor.submit(
        _predict_high_level_request,
        controller,
        request,
    )


def _action_cache_len(model: HyVLAPolicyWrapper) -> int:
    cache = getattr(model, "action_cache", None)
    if cache is None:
        return -1
    try:
        return int(len(cache))
    except TypeError:
        return -1


def _arm_handoff_pause(model: HyVLAPolicyWrapper, decision) -> None:
    if getattr(decision, "trigger", "") == "initial":
        return
    steps = int(getattr(model, "hierarchical_handoff_pause_steps", 0))
    if steps <= 0:
        return
    model.hierarchical_handoff_pause_remaining_steps = steps
    decision.timing_ms["handoff_pause_steps"] = float(steps)


def _publish_high_level_decision(
    TASK_ENV,
    model: HyVLAPolicyWrapper,
    *,
    original_instruction: str,
    decision,
    frame_id: int,
    timestamp_s: float,
    instruction_update_started_ns: int | None = None,
    request_started_ns: int | None = None,
) -> str:
    update_started = instruction_update_started_ns or time.perf_counter_ns()
    should_refresh_low_level = bool(
        getattr(decision, "should_refresh_low_level", True)
    )
    action_cache_before = _action_cache_len(model)
    if should_refresh_low_level:
        next_instruction = _format_low_level_instruction(
            original_instruction,
            decision,
            mode=getattr(model, "high_level_instruction_mode", "subtask"),
        )
        stale_action_count = model.invalidate_action_cache()
    else:
        next_instruction = _current_high_level_instruction(
            model,
            original_instruction,
        )
        stale_action_count = 0
    action_cache_after = _action_cache_len(model)
    update_finished = time.perf_counter_ns()
    decision.timing_ms["instruction_update_total"] = (
        update_finished - update_started
    ) / 1_000_000.0
    decision.timing_ms["decision_to_instruction_ready"] = (
        update_finished - update_started
    ) / 1_000_000.0
    if request_started_ns is not None:
        request_to_ready_ms = (
            update_finished - int(request_started_ns)
        ) / 1_000_000.0
        decision.timing_ms["request_to_decision_ready"] = request_to_ready_ms
        decision.timing_ms["request_to_instruction_ready"] = request_to_ready_ms
        if decision.trigger == "forced_boundary":
            decision.timing_ms["boundary_to_decision_ready"] = request_to_ready_ms
            decision.timing_ms["boundary_to_instruction_ready"] = request_to_ready_ms
    if decision.trigger == "forced_boundary":
        profile = getattr(model, "hierarchical_task_profile", None)
        if profile is not None:
            try:
                # ``finish_step`` advances the controller phase when the VLM
                # says ``switch``. Evaluate the phase that was active at the
                # boundary, not the newly published phase.
                oracle_phase_index = int(decision.phase_index)
                if str(decision.planner_action) == "switch":
                    oracle_phase_index = max(0, oracle_phase_index - 1)
                phase_complete_fn = getattr(profile, "is_phase_complete", None)
                if callable(phase_complete_fn):
                    oracle_completed = bool(
                        phase_complete_fn(TASK_ENV, oracle_phase_index)
                    )
                else:
                    oracle_completed = bool(
                        oracle_phase_index == 0
                        and profile.first_subtask_complete(TASK_ENV)
                    )
                decision.timing_ms["measurement_oracle_phase_index"] = float(
                    oracle_phase_index
                )
                decision.timing_ms["measurement_oracle_completed"] = (
                    1.0 if oracle_completed else 0.0
                )
            except Exception as exc:  # noqa: BLE001 - log availability, keep eval alive.
                decision.timing_ms["measurement_oracle_completed_available"] = 0.0
                decision.timing_ms["measurement_oracle_error"] = float(
                    len(str(exc))
                )
    decision.timing_ms["action_cache_len_before_high_level_commit"] = float(
        action_cache_before
    )
    decision.timing_ms["action_cache_len_after_high_level_commit"] = float(
        action_cache_after
    )
    decision.timing_ms["stale_action_cache_discarded"] = float(stale_action_count)
    decision.timing_ms["action_cache_cleared"] = (
        1.0 if should_refresh_low_level and stale_action_count > 0 else 0.0
    )
    decision.timing_ms["same_decision_kept_action_cache"] = (
        1.0 if not should_refresh_low_level else 0.0
    )
    model.high_level_current_instruction = next_instruction
    if decision.trigger != "initial" or should_refresh_low_level:
        model.pending_instruction_handoff = {
            "decision": decision,
            "frame_id": frame_id,
            "timestamp_s": timestamp_s,
            "instruction_ready_ns": update_finished,
        }
        if should_refresh_low_level:
            _arm_handoff_pause(model, decision)
    model.high_level_decisions.append(
        {
            "frame_id": frame_id,
            "timestamp_s": timestamp_s,
            **decision.to_dict(),
        }
    )
    _append_high_level_log(model, frame_id, timestamp_s, decision)
    if decision.trigger != "initial":
        model.hierarchical_decision_handoff_logged = True
    if should_refresh_low_level:
        model.hierarchical_handoff_fired = True
        _maybe_start_prefetch_next_planner(
            model,
            timestamp_s=timestamp_s,
            frame_id=frame_id,
        )
    return next_instruction


def _poll_async_high_level_request(
    TASK_ENV,
    model: HyVLAPolicyWrapper,
    *,
    original_instruction: str,
    frame_id: int,
    timestamp_s: float,
) -> None:
    future: Future | None = getattr(model, "hierarchical_async_future", None)
    request: HighLevelRequest | None = getattr(
        model,
        "hierarchical_async_request",
        None,
    )
    if future is None or request is None or not future.done():
        return

    model.hierarchical_async_future = None
    model.hierarchical_async_request = None
    if future.cancelled():
        return

    prediction = future.result()
    max_age_s = float(getattr(model, "hierarchical_async_drop_stale_s", 0.0))
    response_age_s = max(0.0, float(timestamp_s) - float(request.timestamp_s))
    if max_age_s > 0 and response_age_s > max_age_s:
        return

    controller = getattr(model, "high_level_controller", None)
    if controller is None:
        return
    finalize_prediction = getattr(
        controller,
        "finalize_prediction_for_request",
        None,
    )
    if finalize_prediction is not None:
        prediction = finalize_prediction(request, prediction)
    decision = controller.finish_step(
        request,
        prediction,
        commit_timestamp_s=timestamp_s,
        commit_frame_id=frame_id,
    )
    decision.timing_ms["async_response_age"] = response_age_s * 1000.0
    decision.timing_ms["async_request_to_decision"] = (
        time.perf_counter_ns() - int(request.request_started_ns)
    ) / 1_000_000.0
    _publish_high_level_decision(
        TASK_ENV,
        model,
        original_instruction=original_instruction,
        decision=decision,
        frame_id=frame_id,
        timestamp_s=timestamp_s,
        request_started_ns=request.request_started_ns,
    )


def _maybe_start_async_high_level_request(
    model: HyVLAPolicyWrapper,
    *,
    timestamp_s: float,
    frame_id: int,
) -> None:
    controller = getattr(model, "high_level_controller", None)
    executor: ThreadPoolExecutor | None = getattr(
        model,
        "hierarchical_async_executor",
        None,
    )
    if controller is None or executor is None:
        return
    future: Future | None = getattr(model, "hierarchical_async_future", None)
    if future is not None and not future.done():
        return

    request = controller.begin_step(timestamp_s=timestamp_s, frame_id=frame_id)
    if request is None or request.initial:
        return
    model.hierarchical_async_request_id = int(
        getattr(model, "hierarchical_async_request_id", 0)
    ) + 1
    model.hierarchical_async_request = request
    model.hierarchical_async_future = executor.submit(
        _predict_high_level_request,
        controller,
        request,
    )


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
        return _apply_low_level_instruction_override(model, original_instruction)
    if hasattr(controller, "set_episode_instruction"):
        controller.set_episode_instruction(original_instruction)

    should_update_now = bool(
        getattr(controller, "should_update", lambda _timestamp_s: True)(
            timestamp_s
        )
    )
    visual_context_mode = str(getattr(controller, "visual_context_mode", "recent"))
    force_decision_visual = bool(
        getattr(model, "hierarchical_force_visual_on_decision", False)
        or visual_context_mode in {"unseen", "unseen_memory"}
    ) and bool(should_update_now)
    camera_images: dict[str, Any] = {}
    if controller.should_capture_visual(timestamp_s) or force_decision_visual:
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
            "visual_capture_reason": (
                "decision" if force_decision_visual else "periodic"
            )
            if camera_images
            else "none",
            "has_head_camera": "head_camera" in observation.get("observation", {}),
            "has_left_camera": "left_camera" in observation.get("observation", {}),
            "has_right_camera": "right_camera" in observation.get("observation", {}),
        },
    )
    controller.observe(frame)
    _poll_prefetch_next_planner(model, timestamp_s=timestamp_s)
    if bool(getattr(model, "hierarchical_async_enabled", False)):
        if model.high_level_current_instruction is None:
            decision = controller.step(timestamp_s=timestamp_s, frame_id=frame_id)
            if decision is not None:
                _publish_high_level_decision(
                    TASK_ENV,
                    model,
                    original_instruction=original_instruction,
                    decision=decision,
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                )
        _poll_async_high_level_request(
            TASK_ENV,
            model,
            original_instruction=original_instruction,
            frame_id=frame_id,
            timestamp_s=timestamp_s,
        )
        _maybe_start_async_high_level_request(
            model,
            timestamp_s=timestamp_s,
            frame_id=frame_id,
        )
        return _apply_low_level_instruction_override(
            model,
            _current_high_level_instruction(model, original_instruction),
        )

    # The forced-boundary measurement starts after per-step observation
    # bookkeeping and ends once the refreshed instruction is published. It
    # deliberately excludes startup/model-loading/warm-up and the following
    # low-level action forward.
    boundary_started = time.perf_counter_ns()
    decision = controller.step(timestamp_s=timestamp_s, frame_id=frame_id)
    if decision is not None:
        _publish_high_level_decision(
            TASK_ENV,
            model,
            original_instruction=original_instruction,
            decision=decision,
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            request_started_ns=boundary_started,
        )

    return _apply_low_level_instruction_override(
        model,
        _current_high_level_instruction(model, original_instruction),
    )


def _current_high_level_instruction(
    model: HyVLAPolicyWrapper,
    original_instruction: str,
) -> str:
    current_instruction = getattr(model, "high_level_current_instruction", None)
    if current_instruction is None:
        return original_instruction
    return str(current_instruction)


def _apply_low_level_instruction_override(
    model: HyVLAPolicyWrapper,
    instruction: str,
) -> str:
    override = getattr(model, "low_level_instruction_override", None)
    if override is None:
        return instruction
    return str(override)


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
            f"High-level phase: {decision.phase_index}"
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


def _append_low_level_instruction_trace(
    model: HyVLAPolicyWrapper,
    TASK_ENV,
    *,
    frame_id: int,
    timestamp_s: float,
    instruction: str,
    action: np.ndarray,
    pause_step: bool = False,
    pause_remaining_steps: int = 0,
) -> None:
    trace_path = getattr(model, "low_level_instruction_trace_path", None)
    if trace_path is None:
        return
    used_cache = bool(getattr(model, "last_action_used_cache", False))
    if used_cache and not bool(
        getattr(model, "low_level_instruction_trace_every_step", False)
    ):
        return
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "frame_id": int(frame_id),
        "timestamp_s": float(timestamp_s),
        "episode_index": int(getattr(TASK_ENV, "test_num", 0)),
        "original_instruction": TASK_ENV.get_instruction(),
        "low_level_instruction": instruction,
        "low_level_instruction_override": getattr(
            model,
            "low_level_instruction_override",
            None,
        ),
        "used_action_cache": used_cache,
        "model_forward_ms": float(getattr(model, "last_model_forward_ms", 0.0)),
        "action_call_ms": float(getattr(model, "last_action_call_ms", 0.0)),
        "action_cache_len_after_step": _action_cache_len(model),
        "handoff_pause_step": bool(pause_step),
        "handoff_pause_remaining_steps": int(pause_remaining_steps),
        "executed_action_wxyz": _json_array(action),
        "high_level_current_instruction": getattr(
            model,
            "high_level_current_instruction",
            None,
        ),
    }
    action_chunk = getattr(model, "last_action_chunk_wxyz", None)
    if action_chunk is not None:
        row["model_forward_action_chunk_wxyz"] = _json_array(action_chunk)
    with trace_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _json_array(value) -> list:
    arr = np.asarray(value, dtype=np.float32)
    return np.round(arr, 6).tolist()


__all__ = ["encode_obs", "get_model", "eval", "reset_model"]
