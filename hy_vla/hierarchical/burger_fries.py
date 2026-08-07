"""Burger / fries hierarchical controller scaffold.

This module is intentionally lightweight:
- it does not depend on a specific high-level VLM yet;
- it provides the state machine and frame buffer that we can connect to
  high-level prompting or a future text-generation backend.

The immediate goal is to profile a two-stage robot task:

1. place the hamburg on the tray
2. place the french fries on the tray

The controller keeps a short rolling visual history, periodically emits a
structured high-level state record, and emits subtask handoff decisions.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence


@dataclass(slots=True)
class FrameRecord:
    """One observed frame in the high-level memory stream."""

    frame_id: int
    timestamp_s: float
    caption: str = ""
    event_tags: tuple[str, ...] = ()
    camera_images: dict[str, Any] = field(default_factory=dict, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def short_label(self) -> str:
        tag_str = ",".join(self.event_tags) if self.event_tags else "-"
        return f"f{self.frame_id}@{self.timestamp_s:.2f}s[{tag_str}]"


@dataclass(slots=True)
class ExecutionReport:
    """Low-level feedback handed to the high-level controller at a boundary.

    The initial forced-boundary baseline deliberately leaves ``status`` as
    ``unknown``. A future verifier can replace it with completed/partial/
    failed without changing the high-level handoff contract.
    """

    trigger: str
    subtask: str
    phase_index: int
    start_timestamp_s: float
    end_timestamp_s: float
    start_frame_id: int | None
    end_frame_id: int
    low_level_steps: int
    model_forward_calls: int
    mean_low_level_call_ms: float
    mean_model_forward_ms: float
    mean_action_l2: float
    state_delta_l2: float | None
    sampled_frame_ids: list[int]
    status: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger,
            "subtask": self.subtask,
            "phase_index": self.phase_index,
            "start_timestamp_s": self.start_timestamp_s,
            "end_timestamp_s": self.end_timestamp_s,
            "duration_s": max(0.0, self.end_timestamp_s - self.start_timestamp_s),
            "start_frame_id": self.start_frame_id,
            "end_frame_id": self.end_frame_id,
            "low_level_steps": self.low_level_steps,
            "model_forward_calls": self.model_forward_calls,
            "mean_low_level_call_ms": self.mean_low_level_call_ms,
            "mean_model_forward_ms": self.mean_model_forward_ms,
            "mean_action_l2": self.mean_action_l2,
            "state_delta_l2": self.state_delta_l2,
            "sampled_frame_ids": list(self.sampled_frame_ids),
            "status": self.status,
        }


@dataclass(slots=True)
class HighLevelDecision:
    """A structured high-level output for the low-level policy."""

    memory_text: str
    next_subtask: str
    phase_index: int
    confidence: float
    should_refresh_low_level: bool
    context_frames: list[int] = field(default_factory=list)
    reason: str = ""
    prompt: str = ""
    raw_prediction: str = ""
    prediction_parse_ok: bool = True
    prediction_parse_error: str | None = None
    trigger: str = "initial"
    execution_report: ExecutionReport | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    planner_action: str = "switch"
    subtask_status: str = "unknown"
    planner_mode: str = "inline_vlm"
    planner_candidate_subtask: str = ""
    planner_candidate_source: str = ""
    request_kind: str = "legacy"
    output_schema: str = ""
    high_level_status: str = "unknown"
    verifier_next_subtask: str = ""
    evidence: str = ""
    visual_context_mode: str = "recent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_text": self.memory_text,
            "next_subtask": self.next_subtask,
            "phase_index": self.phase_index,
            "context_frames": list(self.context_frames),
            "confidence": float(self.confidence),
            "should_refresh_low_level": bool(self.should_refresh_low_level),
            "reason": self.reason,
            "prompt": self.prompt,
            "raw_prediction": self.raw_prediction,
            "prediction_parse_ok": bool(self.prediction_parse_ok),
            "prediction_parse_error": self.prediction_parse_error,
            "trigger": self.trigger,
            "execution_report": (
                self.execution_report.to_dict() if self.execution_report is not None else None
            ),
            "timing_ms": dict(self.timing_ms),
            "planner_action": self.planner_action,
            "subtask_status": self.subtask_status,
            "planner_mode": self.planner_mode,
            "planner_candidate_subtask": self.planner_candidate_subtask,
            "planner_candidate_source": self.planner_candidate_source,
            "request_kind": self.request_kind,
            "output_schema": self.output_schema,
            "high_level_status": self.high_level_status,
            "verifier_next_subtask": self.verifier_next_subtask,
            "evidence": self.evidence,
            "visual_context_mode": self.visual_context_mode,
        }


@dataclass(slots=True)
class HighLevelPrediction:
    """Structured output expected from a high-level memory policy.

    The active high-level planner/verifier path uses text fields only:
    next/current subtask, optional confidence, decision/status, and evidence.
    """

    current_subtask: str
    confidence: float = 1.0
    raw_text: str = ""
    parse_ok: bool = True
    parse_error: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    action: str = "switch"
    subtask_status: str = "unknown"
    subtasks: list[str] = field(default_factory=list)
    high_level_status: str = "unknown"
    verifier_next_subtask: str = ""
    evidence: str = ""


@dataclass(slots=True)
class HighLevelRequest:
    """Immutable request snapshot for synchronous or async high-level planning."""

    prompt: str
    timestamp_s: float
    frame_id: int
    phase_index: int
    phase_prompt: str
    trigger: str
    initial: bool
    duration_s: float
    sampled_history: list[FrameRecord]
    recent_frames: list[FrameRecord]
    memory_frames: list[FrameRecord]
    context_frames: list[int]
    execution_report: ExecutionReport | None
    prompt_build_ms: float = 0.0
    request_started_ns: int = 0
    planner_mode: str = "inline_vlm"
    planner_candidate_subtask: str = ""
    planner_candidate_source: str = ""
    request_kind: str = "legacy"
    output_schema: str = ""
    allow_text_only: bool = False
    visual_context_mode: str = "recent"


@dataclass(slots=True)
class PlannerPrefetchRecord:
    """A text-only next-subtask prediction prepared before a switch is needed."""

    request: HighLevelRequest
    prediction: HighLevelPrediction
    ready_timestamp_s: float


class HighLevelBackend(Protocol):
    """Backend contract for a real or scripted high-level VLM."""

    def predict(
        self,
        *,
        prompt: str,
        memory_frames: Sequence[FrameRecord],
        recent_frames: Sequence[FrameRecord],
        timestamp_s: float,
        phase_index: int,
        phase_prompt: str,
        execution_report: ExecutionReport | None,
        output_schema: str | None = None,
        allow_text_only: bool = False,
    ) -> HighLevelPrediction:
        ...


def _extract_json_object(text: str) -> str:
    """Extract the first JSON object from raw model text."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    raise ValueError("no JSON object found")


def parse_high_level_prediction_text(
    text: str,
    *,
    fallback_subtask: str,
    default_confidence: float = 0.5,
) -> HighLevelPrediction:
    """Parse a high-level JSON prediction.

    Legacy MemER-style schema:

    {
      "current_subtask": "Place the hamburg on the tray.",
      "confidence": 0.8
    }

    The simplified planner schema is also accepted:

    {
      "next_subtask": "Place the hamburg on the tray."
    }

    The status-aware planner schema is:

    {
      "action": "continue" | "switch",
      "next_subtask": "Place the hamburg on the tray.",
      "subtask_status": "unfinished" | "completed" | "unknown"
    }

    The completion-verifier schema is:

    {
      "decision": "same" | "switch",
      "status": "running" | "completed" | "failed" | "stuck" | "uncertain",
      "next_subtask": "",
      "evidence": "short visual evidence"
    }

    The all-at-once planner schema is:

    {
      "subtasks": ["Pick up the burger.", "Place the fries on the tray."],
      "confidence": 0.8
    }
    """

    raw_text = text
    try:
        payload = json.loads(_extract_json_object(text))
        def _clean_str(value: Any) -> str:
            if value is None:
                return ""
            return str(value).strip()

        raw_subtasks = payload.get("subtasks") or payload.get("plan") or []
        if isinstance(raw_subtasks, str):
            subtasks = [raw_subtasks.strip()] if raw_subtasks.strip() else []
        else:
            subtasks = [
                str(item).strip()
                for item in raw_subtasks
                if str(item).strip()
            ]
        verifier_next_subtask = _clean_str(payload.get("next_subtask"))
        current_subtask = (
            verifier_next_subtask
            or _clean_str(payload.get("current_subtask"))
            or _clean_str(payload.get("first_subtask"))
            or (subtasks[0] if subtasks else "")
            or fallback_subtask
        )
        raw_status = _clean_str(
            payload.get("status") or payload.get("subtask_status") or "unknown"
        ).lower()
        if raw_status in {"running", "unfinished", "in_progress"}:
            high_level_status = "running"
            subtask_status = "unfinished"
        elif raw_status == "completed":
            high_level_status = "completed"
            subtask_status = "completed"
        elif raw_status in {"failed", "stuck", "uncertain"}:
            high_level_status = raw_status
            subtask_status = "unknown"
        else:
            high_level_status = "unknown"
            subtask_status = "unknown"
        raw_action = payload.get("decision")
        if raw_action is None:
            raw_action = payload.get("action")
        if raw_action is None:
            if subtask_status == "completed":
                action = "switch"
            elif subtask_status in {"unfinished", "running", "in_progress"}:
                action = "continue"
            elif payload.get("next_subtask") or payload.get("current_subtask"):
                action = "switch"
            else:
                action = "continue"
        else:
            action = str(raw_action).strip().lower()
            if action == "same":
                action = "continue"
        if action not in {"continue", "switch"}:
            action = "continue" if subtask_status != "completed" else "switch"
        confidence = float(payload.get("confidence", default_confidence))
        evidence = (
            _clean_str(payload.get("evidence"))
            or _clean_str(payload.get("reason"))
            or _clean_str(payload.get("rationale"))
        )
        return HighLevelPrediction(
            current_subtask=current_subtask,
            confidence=max(0.0, min(1.0, confidence)),
            raw_text=raw_text,
            parse_ok=True,
            action=action,
            subtask_status=subtask_status,
            subtasks=subtasks,
            high_level_status=high_level_status,
            verifier_next_subtask=verifier_next_subtask,
            evidence=evidence,
        )
    except Exception as exc:  # noqa: BLE001 - keep raw VLM parse failure visible.
        return HighLevelPrediction(
            current_subtask=fallback_subtask,
            confidence=0.0,
            raw_text=raw_text,
            parse_ok=False,
            parse_error=str(exc),
        )


class JsonTextHighLevelBackend:
    """Adapter for a future real high-level VLM.

    The generator receives the text prompt plus memory/recent frame records
    and returns raw text. This class only owns the robust JSON parsing
    contract, so Qwen/Hunyuan/OpenAI/local backends can be plugged in later
    without changing the controller or RoboTwin adapter.
    """

    def __init__(
        self,
        generate_fn: Callable[..., str],
        *,
        default_confidence: float = 0.5,
    ) -> None:
        self.generate_fn = generate_fn
        self.default_confidence = float(default_confidence)

    def predict(
        self,
        *,
        prompt: str,
        memory_frames: Sequence[FrameRecord],
        recent_frames: Sequence[FrameRecord],
        timestamp_s: float,
        phase_index: int,
        phase_prompt: str,
        execution_report: ExecutionReport | None,
        output_schema: str | None = None,
        allow_text_only: bool = False,
    ) -> HighLevelPrediction:
        raw = self.generate_fn(
            prompt=prompt,
            memory_frames=memory_frames,
            recent_frames=recent_frames,
            timestamp_s=timestamp_s,
            phase_index=phase_index,
            phase_prompt=phase_prompt,
            execution_report=execution_report,
            output_schema=output_schema,
            allow_text_only=allow_text_only,
        )
        return parse_high_level_prediction_text(
            raw,
            fallback_subtask=phase_prompt,
            default_confidence=self.default_confidence,
        )


class FixedLatencyBackend:
    """Wrap another backend and sleep to simulate high-level VLM latency."""

    def __init__(self, backend: HighLevelBackend, *, latency_s: float) -> None:
        self.backend = backend
        self.latency_s = max(0.0, float(latency_s))

    def predict(
        self,
        *,
        prompt: str,
        memory_frames: Sequence[FrameRecord],
        recent_frames: Sequence[FrameRecord],
        timestamp_s: float,
        phase_index: int,
        phase_prompt: str,
        execution_report: ExecutionReport | None,
        output_schema: str | None = None,
        allow_text_only: bool = False,
    ) -> HighLevelPrediction:
        if self.latency_s > 0:
            time.sleep(self.latency_s)
        return self.backend.predict(
            prompt=prompt,
            memory_frames=memory_frames,
            recent_frames=recent_frames,
            timestamp_s=timestamp_s,
            phase_index=phase_index,
            phase_prompt=phase_prompt,
            execution_report=execution_report,
            output_schema=output_schema,
            allow_text_only=allow_text_only,
        )


class ScriptedBurgerFriesBackend:
    """Deterministic backend used until a real high-level VLM is attached."""

    def predict(
        self,
        *,
        prompt: str,
        memory_frames: Sequence[FrameRecord],
        recent_frames: Sequence[FrameRecord],
        timestamp_s: float,
        phase_index: int,
        phase_prompt: str,
        execution_report: ExecutionReport | None,
        output_schema: str | None = None,
        allow_text_only: bool = False,
    ) -> HighLevelPrediction:
        del output_schema, allow_text_only, memory_frames, recent_frames

        return HighLevelPrediction(
            current_subtask=phase_prompt,
            confidence=0.9,
            raw_text=(
                "{"
                f'"current_subtask": "{phase_prompt}", '
                '"confidence": 0.9'
                "}"
            ),
        )


class FrameBuffer:
    """Rolling frame buffer with simple FPS-aware subsampling."""

    def __init__(self, maxlen: int | None = None) -> None:
        self._frames: deque[FrameRecord] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def append(self, frame: FrameRecord) -> None:
        self._frames.append(frame)

    def extend(self, frames: list[FrameRecord]) -> None:
        for frame in frames:
            self.append(frame)

    def all(self) -> list[FrameRecord]:
        return list(self._frames)

    def recent(self, max_frames: int) -> list[FrameRecord]:
        if max_frames <= 0:
            return []
        return list(self._frames)[-max_frames:]

    def sample_recent_by_fps(
        self,
        *,
        target_fps: float,
        duration_s: float | None = None,
        source_fps: float | None = None,
    ) -> list[FrameRecord]:
        """Return a time-subsampled slice of the recent history.

        If `source_fps` is known and > `target_fps`, we stride by
        `round(source_fps / target_fps)`. Otherwise we fall back to a
        simple timestamp binning strategy.
        """

        frames = list(self._frames)
        if not frames:
            return []

        if duration_s is not None and duration_s > 0:
            cutoff = frames[-1].timestamp_s - duration_s
            frames = [frame for frame in frames if frame.timestamp_s >= cutoff]
        if len(frames) <= 1 or target_fps <= 0:
            return frames

        if source_fps is not None and source_fps > 0:
            stride = max(1, int(round(source_fps / target_fps)))
            return frames[::stride]

        start = frames[0].timestamp_s
        end = frames[-1].timestamp_s
        step = 1.0 / target_fps
        targets: list[float] = []
        t = start
        while t <= end + 1e-9:
            targets.append(t)
            t += step

        selected: list[FrameRecord] = []
        used_ids: set[int] = set()
        for target in targets:
            closest = min(
                (frame for frame in frames if frame.frame_id not in used_ids),
                key=lambda frame: abs(frame.timestamp_s - target),
                default=None,
            )
            if closest is not None:
                selected.append(closest)
                used_ids.add(closest.frame_id)
        return selected


class BurgerFriesHighLevelController:
    """Configurable high-level controller for sequential RoboTwin tasks.

    It supports two boundary sources with one shared handoff contract:

    - forced boundaries for timing controls;
    - externally confirmed boundaries from a simulator oracle or future
      visual completion detector.

    Both paths produce an ``ExecutionReport`` before publishing the next
    low-level subtask.
    """

    def __init__(
        self,
        *,
        task_name: str = "place_burger_fries",
        task_instruction: str | None = None,
        phase_names: Sequence[str] | None = None,
        phase_prompts: Sequence[str] | None = None,
        forced_boundary_s: float = 5.0,
        max_forced_boundaries: int = 1,
        enable_forced_boundaries: bool = True,
        recent_frames_length: int = 8,
        history_fps: float = 1.0,
        source_fps: float = 25.0,
        memory_length: int = 8,
        max_memory_frames: int = 256,
        prompt_mode: str = "legacy",
        planner_mode: str = "inline_vlm",
        visual_context_mode: str = "recent",
        backend: HighLevelBackend | None = None,
    ) -> None:
        resolved_phase_names = tuple(phase_names or (
            "handle hamburger",
            "handle french fries",
        ))
        resolved_phase_prompts = tuple(phase_prompts or (
            "Place the hamburg on the tray.",
            "Place the french fries on the tray.",
        ))
        if not resolved_phase_names or not resolved_phase_prompts:
            raise ValueError("phase_names and phase_prompts must be non-empty")
        if len(resolved_phase_names) != len(resolved_phase_prompts):
            raise ValueError("phase_names and phase_prompts must have the same length")

        self.task_name = str(task_name)
        self.task_instruction = task_instruction or (
            "Use dual arm to pick the hamburg and frenchfries and put them onto the tray."
        )
        self.forced_boundary_s = max(0.1, float(forced_boundary_s))
        self.max_forced_boundaries = max(0, int(max_forced_boundaries))
        self.enable_forced_boundaries = bool(enable_forced_boundaries)
        self.recent_frames_length = int(recent_frames_length)
        self.history_fps = float(history_fps)
        self.source_fps = float(source_fps)
        del memory_length
        self.backend = backend or ScriptedBurgerFriesBackend()
        self.prompt_mode = str(prompt_mode).strip().lower()
        if self.prompt_mode not in {
            "legacy",
            "minimal",
            "minimal_status",
            "completion_status",
        }:
            raise ValueError(
                "prompt_mode must be 'legacy', 'minimal', 'minimal_status', "
                "or 'completion_status', "
                f"got {prompt_mode!r}"
            )
        self.planner_mode = str(planner_mode).strip().lower()
        if self.planner_mode not in {
            "inline_vlm",
            "vlm_plan_all",
            "vlm_plan_next",
            "profile_all",
            "lazy_profile",
        }:
            raise ValueError(
                "planner_mode must be 'inline_vlm', 'vlm_plan_all', "
                "'vlm_plan_next', 'profile_all', or 'lazy_profile', "
                f"got {planner_mode!r}"
            )
        self.visual_context_mode = str(visual_context_mode).strip().lower()
        if self.visual_context_mode not in {"recent", "unseen"}:
            raise ValueError(
                "visual_context_mode must be 'recent' or 'unseen', "
                f"got {visual_context_mode!r}"
            )
        self.buffer = FrameBuffer(maxlen=max_memory_frames)
        self.memory_text = ""
        self.phase_names = resolved_phase_names
        self.phase_prompts = resolved_phase_prompts
        self._planner_plan_cache: tuple[str, ...] = (
            resolved_phase_prompts if self.planner_mode == "profile_all" else ()
        )
        self._planner_prefetch: PlannerPrefetchRecord | None = None
        self._has_initial_decision = False
        self._active_phase_index = 0
        self._active_subtask = ""
        self._active_start_timestamp_s: float | None = None
        self._active_start_frame_id: int | None = None
        self._forced_boundary_count = 0
        self._last_visual_capture_s = float("-inf")
        self._last_consumed_visual_frame_id: int | None = None
        self._reset_execution_accumulator()

    def reset(self) -> None:
        self.buffer.clear()
        self.memory_text = ""
        self._planner_plan_cache = (
            self.phase_prompts if self.planner_mode == "profile_all" else ()
        )
        self._planner_prefetch = None
        self._has_initial_decision = False
        self._active_phase_index = 0
        self._active_subtask = ""
        self._active_start_timestamp_s = None
        self._active_start_frame_id = None
        self._forced_boundary_count = 0
        self._last_visual_capture_s = float("-inf")
        self._last_consumed_visual_frame_id = None
        self._reset_execution_accumulator()

    def observe(self, frame: FrameRecord) -> None:
        self.buffer.append(frame)
        if frame.camera_images:
            self._last_visual_capture_s = frame.timestamp_s

    def _profile_subtask_for_phase(self, phase_index: int) -> str:
        index = min(max(int(phase_index), 0), len(self.phase_prompts) - 1)
        return self.phase_prompts[index]

    def _planner_candidate_for_phase(self, phase_index: int) -> tuple[str, str]:
        """Return the next-subtask candidate owned by the planner side.

        `inline_vlm` keeps the legacy behavior where the VLM prediction itself
        supplies the subtask text. The profile modes split planning from visual
        completion verification: completion_status decides switch/continue,
        while this candidate supplies the subtask text on switch.
        """

        phase_prompt = self._profile_subtask_for_phase(phase_index)
        if self.planner_mode == "vlm_plan_all":
            if self._planner_plan_cache:
                index = min(max(int(phase_index), 0), len(self._planner_plan_cache) - 1)
                return self._planner_plan_cache[index], "vlm_plan_all_cache"
            return "", "vlm_plan_all_pending"
        if self.planner_mode == "vlm_plan_next":
            return "", "vlm_plan_next_pending"
        if self.planner_mode == "profile_all":
            if not self._planner_plan_cache:
                self._planner_plan_cache = self.phase_prompts
            index = min(max(int(phase_index), 0), len(self._planner_plan_cache) - 1)
            return self._planner_plan_cache[index], "profile_all"
        if self.planner_mode == "lazy_profile":
            return phase_prompt, "lazy_profile"
        return phase_prompt, "inline_vlm_fallback"

    def _synthetic_planner_prediction(
        self,
        request: HighLevelRequest,
    ) -> HighLevelPrediction:
        """Publish the planner candidate without calling a verifier.

        Initial decisions have no current subtask to verify. In the split
        planner/verifier modes, the first low-level instruction therefore comes
        directly from the planner candidate.
        """

        return HighLevelPrediction(
            current_subtask=request.planner_candidate_subtask or request.phase_prompt,
            confidence=1.0,
            raw_text=(
                '{"source":"planner_candidate","subtask_status":"unknown"}'
            ),
            parse_ok=True,
            action="switch",
            subtask_status="unknown",
        )

    def should_update(self, timestamp_s: float) -> bool:
        if not self._has_initial_decision:
            return True
        if not self.enable_forced_boundaries:
            return False
        if self._forced_boundary_count >= self.max_forced_boundaries:
            return False
        if self._active_start_timestamp_s is None:
            return True
        return (timestamp_s - self._active_start_timestamp_s) >= (
            self.forced_boundary_s - 1e-6
        )

    def should_capture_visual(self, timestamp_s: float) -> bool:
        """Keep only low-frequency evidence frames for the slow controller."""

        if self.history_fps <= 0:
            return False
        if self._last_visual_capture_s == float("-inf"):
            return True
        return (timestamp_s - self._last_visual_capture_s) >= (
            (1.0 / self.history_fps) - 1e-6
        )

    def record_low_level_step(
        self,
        *,
        frame_id: int,
        timestamp_s: float,
        state: Sequence[float],
        action: Sequence[float],
        low_level_call_ms: float,
        model_forward_ms: float,
        used_action_cache: bool,
    ) -> None:
        """Accumulate low-level feedback for the active subtask.

        This intentionally reports execution statistics rather than claiming
        task success. Completion verification is a later module.
        """

        if not self._has_initial_decision:
            return

        state_vec = self._to_float_vector(state)
        action_vec = self._to_float_vector(action)
        if self._exec_start_state is None:
            self._exec_start_state = state_vec
            self._exec_start_frame_id = int(frame_id)
        self._exec_end_state = state_vec
        self._exec_low_level_steps += 1
        self._exec_total_low_level_call_ms += max(0.0, float(low_level_call_ms))
        self._exec_total_model_forward_ms += max(0.0, float(model_forward_ms))
        self._exec_action_l2_sum += self._l2_norm(action_vec)
        if not used_action_cache:
            self._exec_model_forward_calls += 1

    def _reset_execution_accumulator(self) -> None:
        self._exec_start_state: list[float] | None = None
        self._exec_end_state: list[float] | None = None
        self._exec_start_frame_id: int | None = None
        self._exec_low_level_steps = 0
        self._exec_model_forward_calls = 0
        self._exec_total_low_level_call_ms = 0.0
        self._exec_total_model_forward_ms = 0.0
        self._exec_action_l2_sum = 0.0

    @staticmethod
    def _to_float_vector(value: Sequence[float]) -> list[float]:
        if hasattr(value, "reshape"):
            value = value.reshape(-1)  # type: ignore[assignment, union-attr]
        return [float(item) for item in value]

    @staticmethod
    def _l2_norm(values: Sequence[float]) -> float:
        return sum(float(value) * float(value) for value in values) ** 0.5

    @classmethod
    def _l2_delta(
        cls,
        first: Sequence[float] | None,
        last: Sequence[float] | None,
    ) -> float | None:
        if first is None or last is None or len(first) != len(last):
            return None
        return cls._l2_norm([float(b) - float(a) for a, b in zip(first, last, strict=True)])

    def _sample_execution_history(self, duration_s: float) -> list[FrameRecord]:
        sampled = self.buffer.sample_recent_by_fps(
            target_fps=self.history_fps,
            duration_s=max(duration_s, 1.0),
            source_fps=self.source_fps,
        )
        visual_samples = [frame for frame in sampled if frame.camera_images]
        return visual_samples or sampled

    def _unseen_visual_context(self) -> list[FrameRecord]:
        """Return visual frames not included in a prior high-level request."""

        last_seen = self._last_consumed_visual_frame_id
        frames = [
            frame
            for frame in self.buffer.all()
            if frame.camera_images
            and (last_seen is None or int(frame.frame_id) > int(last_seen))
        ]
        if frames:
            return frames
        # Avoid an empty VLM request when the scheduler fires before a new
        # periodic capture. Reusing the latest visual frame is explicit in the
        # logged context, and the consumption cursor is not advanced unless a
        # frame is actually present.
        visual_frames = [frame for frame in self.buffer.all() if frame.camera_images]
        return visual_frames[-1:] if visual_frames else []

    def _mark_visual_context_consumed(self, frames: Sequence[FrameRecord]) -> None:
        visual_frame_ids = [
            int(frame.frame_id)
            for frame in frames
            if frame.camera_images
        ]
        if visual_frame_ids:
            self._last_consumed_visual_frame_id = max(visual_frame_ids)

    def _build_execution_report(
        self,
        *,
        timestamp_s: float,
        frame_id: int,
        sampled_history: list[FrameRecord],
        trigger: str,
        status: str,
    ) -> ExecutionReport:
        start_timestamp_s = (
            self._active_start_timestamp_s
            if self._active_start_timestamp_s is not None
            else timestamp_s
        )
        steps = self._exec_low_level_steps
        return ExecutionReport(
            trigger=trigger,
            subtask=self._active_subtask,
            phase_index=self._active_phase_index,
            start_timestamp_s=start_timestamp_s,
            end_timestamp_s=timestamp_s,
            start_frame_id=self._exec_start_frame_id or self._active_start_frame_id,
            end_frame_id=int(frame_id),
            low_level_steps=steps,
            model_forward_calls=self._exec_model_forward_calls,
            mean_low_level_call_ms=(
                self._exec_total_low_level_call_ms / steps if steps else 0.0
            ),
            mean_model_forward_ms=(
                self._exec_total_model_forward_ms / self._exec_model_forward_calls
                if self._exec_model_forward_calls
                else 0.0
            ),
            mean_action_l2=(self._exec_action_l2_sum / steps if steps else 0.0),
            state_delta_l2=self._l2_delta(
                self._exec_start_state,
                self._exec_end_state,
            ),
            sampled_frame_ids=[frame.frame_id for frame in sampled_history],
            status=status,
        )

    def build_memory_text(
        self,
        *,
        phase_index: int,
        recent: list[FrameRecord],
        memory_frames: list[FrameRecord],
        prediction: HighLevelPrediction,
        execution_report: ExecutionReport | None,
    ) -> str:
        phase_name = self.phase_names[min(phase_index, len(self.phase_names) - 1)]
        phase_prompt = prediction.current_subtask or self.phase_prompts[min(phase_index, len(self.phase_prompts) - 1)]
        frame_ids = ", ".join(frame.short_label() for frame in recent[-self.recent_frames_length :]) or "none"
        del memory_frames
        if execution_report is None:
            report_text = "initial planning; no prior low-level execution report"
        else:
            report_text = (
                f"status={execution_report.status}; "
                f"subtask={execution_report.subtask}; "
                f"duration={execution_report.end_timestamp_s - execution_report.start_timestamp_s:.2f}s; "
                f"steps={execution_report.low_level_steps}; "
                f"state_delta_l2={execution_report.state_delta_l2}"
            )
        return (
            f"Task: {self.task_name}\n"
            f"High-level phase: {phase_name}\n"
            f"Active subtask: {phase_prompt}\n"
            f"Execution report: {report_text}\n"
            f"Visual context frames: {frame_ids}"
        )

    def build_prompt(
        self,
        *,
        timestamp_s: float,
        phase_index: int,
        recent: list[FrameRecord],
        memory_frames: list[FrameRecord],
        execution_report: ExecutionReport | None,
    ) -> str:
        del memory_frames
        if execution_report is None and self.planner_mode == "vlm_plan_all":
            return self._build_vlm_plan_next_prompt(
                phase_index=phase_index,
                execution_report=None,
            )
        if execution_report is None and self.planner_mode == "vlm_plan_next":
            return self._build_vlm_plan_next_prompt(
                phase_index=phase_index,
                execution_report=None,
            )
        if self.prompt_mode == "completion_status" and execution_report is None:
            initial_candidate, initial_source = self._planner_candidate_for_phase(
                phase_index
            )
            return (
                "Synthetic initial high-level planner decision.\n"
                "No completion verification is run before the first low-level "
                "subtask.\n"
                f"Task instruction: {self.task_instruction}\n"
                f"Planner mode: {self.planner_mode}\n"
                f"First subtask candidate: {initial_candidate}\n"
                f"Planner source: {initial_source}"
            )
        if self.prompt_mode == "completion_status":
            return self._build_completion_status_prompt(
                timestamp_s=timestamp_s,
                recent=recent,
                execution_report=execution_report,
            )
        if self.prompt_mode in {"minimal", "minimal_status"}:
            return self._build_minimal_prompt(
                timestamp_s=timestamp_s,
                phase_index=phase_index,
                recent=recent,
                execution_report=execution_report,
            )
        recent_desc = "\n".join(f"- {frame.short_label()}" for frame in recent[-self.recent_frames_length :]) or "- none"
        next_subtask = self.phase_prompts[min(phase_index, len(self.phase_prompts) - 1)]
        if execution_report is None:
            report_desc = "Initial planning. No prior execution report."
        else:
            report_desc = json.dumps(execution_report.to_dict(), ensure_ascii=False)
        return (
            "You are the high-level planner for a robot manipulation task.\n"
            f"Task instruction: {self.task_instruction}\n"
            f"Current time: {timestamp_s:.2f}s\n"
            f"Current phase index: {phase_index}\n"
            f"Current suggested next subtask: {next_subtask}\n"
            "Recent frames:\n"
            f"{recent_desc}\n"
            "Previous low-level execution report:\n"
            f"{report_desc}\n"
            "Return JSON with current_subtask and confidence."
        )

    def _build_minimal_prompt(
        self,
        *,
        timestamp_s: float,
        phase_index: int,
        recent: list[FrameRecord],
        execution_report: ExecutionReport | None,
    ) -> str:
        current_subtask = self._active_subtask or "none yet"
        suggested_next = self.phase_prompts[min(phase_index, len(self.phase_prompts) - 1)]
        visual_desc = (
            f"{len(recent[-self.recent_frames_length :])} recent head-camera frame(s)"
            if recent
            else "no recent camera frame"
        )
        if execution_report is None:
            report_desc = "initial planning"
        else:
            report = execution_report.to_dict()
            report_desc = (
                f"trigger={report['trigger']}; "
                f"status={report['status']}; "
                f"executed_subtask={report['subtask']}"
            )
        status_contract = (
            "Return JSON with action, next_subtask, and subtask_status. "
            "Use action='continue' when the current subtask should keep running; "
            "use action='switch' when next_subtask should be published."
            if self.prompt_mode == "minimal_status"
            else (
                "Choose the next low-level subtask instruction. "
                "If the current subtask appears unfinished, return the current "
                "subtask or a short continuation instruction instead of advancing."
            )
        )
        return (
            "You are the high-level planner for a RoboTwin manipulation task.\n"
            f"Task instruction: {self.task_instruction}\n"
            f"Current low-level subtask: {current_subtask}\n"
            f"Suggested next subtask from the task profile: {suggested_next}\n"
            f"Visual evidence: {visual_desc}\n"
            f"Low-level status: {report_desc}\n"
            f"{status_contract}"
        )

    def _build_vlm_plan_all_prompt(self) -> str:
        return (
            "You are the high-level planner for a RoboTwin robot manipulation "
            "task.\n"
            f"Task instruction: {self.task_instruction}\n"
            f"{self._phase_granularity_contract()}"
            "Decompose the task into a short ordered list of low-level subtask "
            "instructions for Hy-VLA. Each subtask should be directly executable "
            "by the low-level policy. Do not include explanations."
        )

    def _phase_granularity_contract(self) -> str:
        examples = "\n".join(
            f"{index}. {prompt}"
            for index, prompt in enumerate(self.phase_prompts)
        )
        return (
            "Subtask granularity rule:\n"
            "- Generate object-level pick-and-place phases, not motor primitives.\n"
            "- Each subtask must include the intended final object state and "
            "target region.\n"
            "- Do not split one object-level phase into grasp-only, lift-only, "
            "move-only, or release-only instructions.\n"
            "- Do not output instructions such as 'pick up ...', 'grasp ...', "
            "or 'move ...' unless they also include the final target placement.\n"
            "Task-profile phase examples:\n"
            f"{examples}\n"
        )

    def _build_vlm_plan_next_prompt(
        self,
        *,
        phase_index: int,
        execution_report: ExecutionReport | None,
        previous_subtask: str | None = None,
    ) -> str:
        if previous_subtask:
            progress_desc = (
                "This is upfront planning. The previous subtask was generated "
                "by the planner but has not necessarily been executed yet."
            )
            current_subtask = previous_subtask
        elif execution_report is None:
            progress_desc = "No subtask has been executed yet."
            current_subtask = "none yet"
        else:
            report = execution_report.to_dict()
            progress_desc = (
                f"The verifier accepted this completed subtask: "
                f"{report['subtask']}"
            )
            current_subtask = self._active_subtask or report["subtask"]
        return (
            "You are the high-level planner for a RoboTwin robot manipulation "
            "task.\n"
            f"Task instruction: {self.task_instruction}\n"
            f"{self._phase_granularity_contract()}"
            f"Previously active subtask: {current_subtask}\n"
            f"Progress summary: {progress_desc}\n"
            f"Next subtask index: {phase_index}\n"
            f"Requested phase-level target: {self._profile_subtask_for_phase(phase_index)}\n"
            "Generate exactly one low-level subtask instruction that should be "
            "sent to Hy-VLA next. Match the requested target granularity and "
            "do not explain."
        )

    def _build_vlm_prefetch_next_prompt(self, *, phase_index: int) -> str:
        current_subtask = self._active_subtask or self._profile_subtask_for_phase(
            self._active_phase_index
        )
        return (
            "You are the high-level planner for a RoboTwin robot manipulation "
            "task.\n"
            f"Task instruction: {self.task_instruction}\n"
            f"{self._phase_granularity_contract()}"
            f"Current published low-level subtask: {current_subtask}\n"
            f"Current subtask index: {self._active_phase_index}\n"
            f"Next subtask index to prepare: {phase_index}\n"
            f"Requested phase-level target: {self._profile_subtask_for_phase(phase_index)}\n"
            "The current subtask has just been sent to the low-level policy and "
            "may still be executing. Prepare exactly one likely next low-level "
            "subtask instruction for Hy-VLA. Match the requested target "
            "granularity. Do not verify completion and do not explain."
        )

    def _build_vlm_plan_all_next_prompt(
        self,
        *,
        phase_index: int,
        planned_subtasks: Sequence[str],
    ) -> str:
        generated = "\n".join(
            f"{index + 1}. {subtask}"
            for index, subtask in enumerate(planned_subtasks)
        ) or "none"
        return (
            "You are the high-level planner for a RoboTwin robot manipulation "
            "task.\n"
            f"Task instruction: {self.task_instruction}\n"
            f"{self._phase_granularity_contract()}"
            "We are generating the complete high-level plan before execution.\n"
            "Generated subtasks so far:\n"
            f"{generated}\n"
            f"Next subtask index to generate: {phase_index}\n"
            f"Requested phase-level target: {self._profile_subtask_for_phase(phase_index)}\n"
            "Assume the generated subtasks above will be executed in order. "
            "Do not repeat any generated subtask. Generate exactly one different "
            "low-level subtask instruction that should come next. Match the "
            "requested target granularity and do not explain."
        )

    def _build_completion_status_prompt(
        self,
        *,
        timestamp_s: float,
        recent: list[FrameRecord],
        execution_report: ExecutionReport | None,
    ) -> str:
        current_subtask = self._active_subtask or self._profile_subtask_for_phase(0)
        if not recent:
            visual_desc = "no current camera frame"
        elif len(recent) == 1:
            visual_desc = "one current head-camera frame"
        elif self.visual_context_mode == "unseen":
            frame_ids = ", ".join(str(frame.frame_id) for frame in recent)
            visual_desc = (
                f"{len(recent)} accumulated head-camera frames not previously "
                f"sent to high-level: [{frame_ids}]"
            )
        else:
            visual_desc = f"{len(recent)} recent head-camera frames"
        del timestamp_s, execution_report
        return (
            "You are a completion verifier for a RoboTwin robot manipulation "
            "subtask.\n"
            f"Task instruction: {self.task_instruction}\n"
            f"Current low-level subtask: {current_subtask}\n"
            f"Visual evidence: {visual_desc}\n"
            "Decide whether the current low-level subtask is already completed "
            "in the provided visual evidence.\n"
            "Use decision='same' when the robot should keep executing the current "
            "subtask. Use decision='switch' only when the provided image(s) show "
            "the current subtask is completed.\n"
            "Use status='running' for normal unfinished execution, "
            "status='completed' for completed execution, status='failed' for a "
            "clear failure, status='stuck' for no-progress/stalled execution, "
            "and status='uncertain' when the image is insufficient.\n"
            "Keep next_subtask as an empty string unless you have a direct "
            "suggestion. The separate planner may own next-subtask generation.\n"
            "Write evidence as one short visual reason."
        )

    def begin_step(
        self,
        *,
        timestamp_s: float,
        frame_id: int,
        force: bool = False,
        trigger: str | None = None,
        report_status: str = "unknown",
    ) -> HighLevelRequest | None:
        """Build a high-level request snapshot without running the backend."""

        request_started_ns = time.perf_counter_ns()
        build_started = time.perf_counter()
        if not force and not self.should_update(timestamp_s):
            return None

        initial = not self._has_initial_decision
        if not initial and self._forced_boundary_count >= self.max_forced_boundaries:
            return None
        decision_trigger = "initial" if initial else (trigger or "forced_boundary")
        phase_index = 0 if initial else min(
            self._active_phase_index + 1,
            len(self.phase_prompts) - 1,
        )
        if initial and self.planner_mode == "vlm_plan_all":
            request_kind = "planner_all"
            output_schema = "minimal"
            allow_text_only = True
        elif initial and self.planner_mode == "vlm_plan_next":
            request_kind = "planner_next"
            output_schema = "minimal"
            allow_text_only = True
        elif (
            not initial
            and (
                self.prompt_mode == "completion_status"
                or self.planner_mode in {
                    "vlm_plan_all",
                    "vlm_plan_next",
                    "profile_all",
                    "lazy_profile",
                }
            )
        ):
            request_kind = "verifier"
            output_schema = "completion_status"
            allow_text_only = False
        else:
            request_kind = "legacy"
            output_schema = self.prompt_mode
            allow_text_only = False
        planner_candidate, planner_candidate_source = self._planner_candidate_for_phase(
            phase_index
        )
        duration_s = (
            self.forced_boundary_s
            if self._active_start_timestamp_s is None
            else max(timestamp_s - self._active_start_timestamp_s, 0.0)
        )
        sampled_history = self._sample_execution_history(duration_s)
        execution_report = (
            None
            if initial
            else self._build_execution_report(
                timestamp_s=timestamp_s,
                frame_id=frame_id,
                sampled_history=sampled_history,
                trigger=decision_trigger,
                status=report_status,
            )
        )
        if self.visual_context_mode == "unseen":
            recent = self._unseen_visual_context()
        else:
            recent = sampled_history[-self.recent_frames_length :]
        context_ids = [frame.frame_id for frame in recent]
        memory_frames: list[FrameRecord] = []
        prompt = self.build_prompt(
            timestamp_s=timestamp_s,
            phase_index=phase_index,
            recent=recent,
            memory_frames=memory_frames,
            execution_report=execution_report,
        )
        build_finished = time.perf_counter()
        request = HighLevelRequest(
            prompt=prompt,
            timestamp_s=timestamp_s,
            frame_id=frame_id,
            phase_index=phase_index,
            phase_prompt=self.phase_prompts[min(phase_index, len(self.phase_prompts) - 1)],
            trigger=decision_trigger,
            initial=initial,
            duration_s=duration_s,
            sampled_history=sampled_history,
            recent_frames=recent,
            memory_frames=memory_frames,
            context_frames=context_ids,
            execution_report=execution_report,
            prompt_build_ms=(build_finished - build_started) * 1000.0,
            request_started_ns=request_started_ns,
            planner_mode=self.planner_mode,
            planner_candidate_subtask=planner_candidate,
            planner_candidate_source=planner_candidate_source,
            request_kind=request_kind,
            output_schema=output_schema,
            allow_text_only=allow_text_only,
            visual_context_mode=self.visual_context_mode,
        )
        if self.visual_context_mode == "unseen":
            self._mark_visual_context_consumed(recent)
        return request

    def predict_request(self, request: HighLevelRequest) -> HighLevelPrediction:
        """Run the configured backend for a prepared high-level request."""

        return self.backend.predict(
            prompt=request.prompt,
            memory_frames=request.memory_frames,
            recent_frames=request.recent_frames,
            timestamp_s=request.timestamp_s,
            phase_index=request.phase_index,
            phase_prompt=request.phase_prompt,
            execution_report=request.execution_report,
            output_schema=request.output_schema or None,
            allow_text_only=request.allow_text_only,
        )

    def begin_prefetch_next_request(
        self,
        *,
        timestamp_s: float,
        frame_id: int,
    ) -> HighLevelRequest | None:
        """Build a text-only request to prepare the next subtask early.

        This is only for `vlm_plan_next`: after the current subtask has been
        published, the high-level planner can prepare one candidate for the
        next phase while the low-level policy continues executing.
        """

        if self.planner_mode != "vlm_plan_next":
            return None
        if not self._has_initial_decision:
            return None
        next_phase_index = int(self._active_phase_index) + 1
        if next_phase_index >= len(self.phase_prompts):
            return None
        if (
            self._planner_prefetch is not None
            and self._planner_prefetch.request.phase_index == next_phase_index
        ):
            return None

        request_started_ns = time.perf_counter_ns()
        build_started = time.perf_counter()
        prompt = self._build_vlm_prefetch_next_prompt(
            phase_index=next_phase_index,
        )
        build_finished = time.perf_counter()
        return HighLevelRequest(
            prompt=prompt,
            timestamp_s=timestamp_s,
            frame_id=frame_id,
            phase_index=next_phase_index,
            phase_prompt=self.phase_prompts[next_phase_index],
            trigger="planner_prefetch_next",
            initial=False,
            duration_s=0.0,
            sampled_history=[],
            recent_frames=[],
            memory_frames=[],
            context_frames=[],
            execution_report=None,
            prompt_build_ms=(build_finished - build_started) * 1000.0,
            request_started_ns=request_started_ns,
            planner_mode=self.planner_mode,
            planner_candidate_subtask="",
            planner_candidate_source="vlm_plan_next_prefetch_pending",
            request_kind="planner_prefetch_next",
            output_schema="minimal",
            allow_text_only=True,
        )

    def finish_prefetch_next_request(
        self,
        request: HighLevelRequest,
        prediction: HighLevelPrediction,
        *,
        ready_timestamp_s: float,
    ) -> bool:
        """Store a prepared next-subtask prediction if it is still useful."""

        if self.planner_mode != "vlm_plan_next":
            return False
        if request.request_kind != "planner_prefetch_next":
            return False
        if request.phase_index <= self._active_phase_index:
            return False
        if not prediction.parse_ok or not prediction.current_subtask:
            return False
        prediction.timing_ms["prefetch_prompt_build"] = float(
            request.prompt_build_ms
        )
        prediction.timing_ms["prefetch_request_to_ready"] = (
            time.perf_counter_ns() - int(request.request_started_ns)
        ) / 1_000_000.0
        self._planner_prefetch = PlannerPrefetchRecord(
            request=request,
            prediction=prediction,
            ready_timestamp_s=float(ready_timestamp_s),
        )
        return True

    def finish_step(
        self,
        request: HighLevelRequest,
        prediction: HighLevelPrediction,
        *,
        commit_timestamp_s: float | None = None,
        commit_frame_id: int | None = None,
    ) -> HighLevelDecision:
        """Commit a high-level prediction to controller state."""

        update_started = time.perf_counter()
        state_timestamp_s = (
            float(request.timestamp_s)
            if commit_timestamp_s is None
            else float(commit_timestamp_s)
        )
        state_frame_id = (
            int(request.frame_id)
            if commit_frame_id is None
            else int(commit_frame_id)
        )
        memory_updated = time.perf_counter()
        if request.request_kind == "planner_all":
            planned = tuple(prediction.subtasks or [prediction.current_subtask])
            self._planner_plan_cache = tuple(item for item in planned if item)
        planner_candidate = request.planner_candidate_subtask
        if not planner_candidate and request.initial:
            planner_candidate = prediction.current_subtask or (
                prediction.subtasks[0] if prediction.subtasks else ""
            )
        if (
            not planner_candidate
            and self.planner_mode in {"inline_vlm", "profile_all", "lazy_profile"}
        ):
            planner_candidate = request.phase_prompt
        if self.planner_mode == "inline_vlm":
            planner_action = prediction.action if prediction.parse_ok else "switch"
            if request.initial:
                planner_action = "switch"
            should_refresh_low_level = planner_action != "continue"
            next_subtask = (
                prediction.current_subtask
                if should_refresh_low_level
                else (self._active_subtask or prediction.current_subtask)
            ) or planner_candidate
        else:
            if request.initial:
                planner_action = "switch"
            else:
                planner_action = prediction.action if prediction.parse_ok else "continue"
            should_refresh_low_level = planner_action != "continue"
            if should_refresh_low_level:
                next_subtask = (
                    planner_candidate
                    or prediction.current_subtask
                    or request.phase_prompt
                )
            else:
                next_subtask = (
                    self._active_subtask
                    or prediction.current_subtask
                    or request.phase_prompt
                )
        decision_phase_index = (
            request.phase_index
            if should_refresh_low_level
            else self._active_phase_index
        )
        memory_text = self.build_memory_text(
            phase_index=decision_phase_index,
            recent=request.recent_frames,
            memory_frames=[],
            prediction=prediction,
            execution_report=request.execution_report,
        )
        confidence = prediction.confidence if prediction.parse_ok else 0.0
        if request.initial:
            reason = "initial high-level plan"
        elif planner_action == "continue":
            reason = (
                f"high-level requested continue after {request.duration_s:.2f}s; "
                f"status={prediction.high_level_status or prediction.subtask_status}"
            )
        elif request.trigger == "forced_boundary":
            if self.planner_mode == "inline_vlm":
                reason = (
                    f"forced {request.duration_s:.2f}s subtask boundary; "
                    "success remains unverified"
                )
            else:
                reason = (
                    f"forced {request.duration_s:.2f}s subtask check; "
                    f"verifier_status={prediction.high_level_status}; "
                    f"planner_source={request.planner_candidate_source}"
                )
        else:
            report_status = (
                request.execution_report.status
                if request.execution_report is not None
                else "unknown"
            )
            reason = (
                f"{request.trigger} boundary after {request.duration_s:.2f}s; "
                f"status={report_status}"
            )
        finished = time.perf_counter()

        self._has_initial_decision = True
        self._active_phase_index = int(decision_phase_index)
        self._active_subtask = next_subtask
        self._active_start_timestamp_s = state_timestamp_s
        self._active_start_frame_id = state_frame_id
        if not request.initial:
            self._forced_boundary_count += 1
        self._reset_execution_accumulator()
        self.memory_text = memory_text
        controller_timing_ms = {
            "history_and_memory_update": (memory_updated - update_started) * 1000.0,
            "prompt_build": float(request.prompt_build_ms),
            "postprocess": (finished - memory_updated) * 1000.0,
            "total": (finished - update_started) * 1000.0,
        }
        for key, value in prediction.timing_ms.items():
            controller_timing_ms[f"backend_{key}"] = float(value)
        return HighLevelDecision(
            memory_text=memory_text,
            next_subtask=next_subtask,
            phase_index=int(decision_phase_index),
            context_frames=request.context_frames,
            confidence=confidence,
            should_refresh_low_level=should_refresh_low_level,
            reason=reason,
            prompt=request.prompt,
            raw_prediction=prediction.raw_text,
            prediction_parse_ok=prediction.parse_ok,
            prediction_parse_error=prediction.parse_error,
            trigger=request.trigger,
            execution_report=request.execution_report,
            timing_ms=controller_timing_ms,
            planner_action=planner_action,
            subtask_status=prediction.subtask_status,
            planner_mode=request.planner_mode,
            planner_candidate_subtask=request.planner_candidate_subtask,
            planner_candidate_source=request.planner_candidate_source,
            request_kind=request.request_kind,
            output_schema=request.output_schema,
            high_level_status=prediction.high_level_status,
            verifier_next_subtask=prediction.verifier_next_subtask,
            evidence=prediction.evidence,
            visual_context_mode=request.visual_context_mode,
        )

    def finalize_prediction_for_request(
        self,
        request: HighLevelRequest,
        prediction: HighLevelPrediction,
    ) -> HighLevelPrediction:
        if (
            request.initial
            and request.request_kind == "planner_all"
            and self.planner_mode == "vlm_plan_all"
        ):
            return self._predict_remaining_plan_all(request, prediction)
        if (
            request.request_kind == "verifier"
            and self.planner_mode == "vlm_plan_next"
            and prediction.parse_ok
            and prediction.action == "switch"
        ):
            prefetched = self._consume_prefetched_next_subtask(
                request,
                prediction,
            )
            if prefetched is not None:
                return prefetched
            return self._predict_lazy_next_subtask(request, prediction)
        return prediction

    def _predict_remaining_plan_all(
        self,
        request: HighLevelRequest,
        first_prediction: HighLevelPrediction,
    ) -> HighLevelPrediction:
        """Generate the full plan through repeated one-step next planners."""

        planned_subtasks = [
            first_prediction.current_subtask or request.phase_prompt
        ]
        raw_parts = [
            "plan_all_step_0_prompt=",
            request.prompt,
            "\nplan_all_step_0_prediction=",
            first_prediction.raw_text,
        ]
        combined_timing = {
            f"plan_all_step_0_{key}": float(value)
            for key, value in first_prediction.timing_ms.items()
        }
        parse_ok = bool(first_prediction.parse_ok)
        parse_error = first_prediction.parse_error
        confidence_values = [float(first_prediction.confidence)]

        for phase_index in range(request.phase_index + 1, len(self.phase_prompts)):
            planner_started = time.perf_counter()
            planner_prompt = self._build_vlm_plan_all_next_prompt(
                phase_index=phase_index,
                planned_subtasks=planned_subtasks,
            )
            planner_prediction = self.backend.predict(
                prompt=planner_prompt,
                memory_frames=[],
                recent_frames=[],
                timestamp_s=request.timestamp_s,
                phase_index=phase_index,
                phase_prompt=self.phase_prompts[phase_index],
                execution_report=None,
                output_schema="minimal",
                allow_text_only=True,
            )
            planner_finished = time.perf_counter()
            planned_subtasks.append(
                planner_prediction.current_subtask
                or self._profile_subtask_for_phase(phase_index)
            )
            raw_parts.extend(
                [
                    f"\nplan_all_step_{phase_index}_prompt=",
                    planner_prompt,
                    f"\nplan_all_step_{phase_index}_prediction=",
                    planner_prediction.raw_text,
                ]
            )
            combined_timing.update(
                {
                    f"plan_all_step_{phase_index}_{key}": float(value)
                    for key, value in planner_prediction.timing_ms.items()
                }
            )
            combined_timing[f"plan_all_step_{phase_index}_predict"] = (
                planner_finished - planner_started
            ) * 1000.0
            confidence_values.append(float(planner_prediction.confidence))
            if not planner_prediction.parse_ok:
                parse_ok = False
                parse_error = planner_prediction.parse_error or parse_error

        combined_timing["plan_all_generated_subtasks"] = float(len(planned_subtasks))
        return HighLevelPrediction(
            current_subtask=planned_subtasks[0],
            confidence=min(confidence_values) if confidence_values else 0.0,
            raw_text="".join(raw_parts),
            parse_ok=parse_ok,
            parse_error=parse_error,
            timing_ms=combined_timing,
            action="switch",
            subtask_status=first_prediction.subtask_status,
            subtasks=planned_subtasks,
            high_level_status=first_prediction.high_level_status,
            verifier_next_subtask=first_prediction.verifier_next_subtask,
            evidence=first_prediction.evidence,
        )

    def _consume_prefetched_next_subtask(
        self,
        request: HighLevelRequest,
        verifier_prediction: HighLevelPrediction,
    ) -> HighLevelPrediction | None:
        record = self._planner_prefetch
        if record is None:
            return None
        if record.request.phase_index != request.phase_index:
            return None

        self._planner_prefetch = None
        planner_prediction = record.prediction
        combined_timing = {
            f"verifier_{key}": float(value)
            for key, value in verifier_prediction.timing_ms.items()
        }
        combined_timing.update(
            {
                f"next_planner_prefetch_{key}": float(value)
                for key, value in planner_prediction.timing_ms.items()
            }
        )
        combined_timing["next_planner_prefetch_used"] = 1.0
        combined_timing["next_planner_prefetch_ready_age"] = max(
            0.0,
            float(request.timestamp_s) - float(record.ready_timestamp_s),
        ) * 1000.0
        return HighLevelPrediction(
            current_subtask=planner_prediction.current_subtask,
            confidence=(
                planner_prediction.confidence
                if planner_prediction.parse_ok
                else verifier_prediction.confidence
            ),
            raw_text=(
                "verifier_prediction="
                f"{verifier_prediction.raw_text}\n"
                "prefetched_next_planner_prompt="
                f"{record.request.prompt}\n"
                "prefetched_next_planner_prediction="
                f"{planner_prediction.raw_text}"
            ),
            parse_ok=verifier_prediction.parse_ok and planner_prediction.parse_ok,
            parse_error=(
                planner_prediction.parse_error
                if not planner_prediction.parse_ok
                else verifier_prediction.parse_error
            ),
            timing_ms=combined_timing,
            action=verifier_prediction.action,
            subtask_status=verifier_prediction.subtask_status,
            subtasks=planner_prediction.subtasks,
            high_level_status=verifier_prediction.high_level_status,
            verifier_next_subtask=verifier_prediction.verifier_next_subtask,
            evidence=verifier_prediction.evidence,
        )

    def _predict_lazy_next_subtask(
        self,
        request: HighLevelRequest,
        verifier_prediction: HighLevelPrediction,
    ) -> HighLevelPrediction:
        """Run Gemma/Qwen planner for the next subtask after verification."""

        planner_started = time.perf_counter()
        planner_prompt = self._build_vlm_plan_next_prompt(
            phase_index=request.phase_index,
            execution_report=request.execution_report,
        )
        planner_prediction = self.backend.predict(
            prompt=planner_prompt,
            memory_frames=[],
            recent_frames=[],
            timestamp_s=request.timestamp_s,
            phase_index=request.phase_index,
            phase_prompt=request.phase_prompt,
            execution_report=request.execution_report,
            output_schema="minimal",
            allow_text_only=True,
        )
        planner_finished = time.perf_counter()
        combined_timing = {
            f"verifier_{key}": float(value)
            for key, value in verifier_prediction.timing_ms.items()
        }
        combined_timing.update(
            {
                f"next_planner_{key}": float(value)
                for key, value in planner_prediction.timing_ms.items()
            }
        )
        combined_timing["next_planner_predict"] = (
            planner_finished - planner_started
        ) * 1000.0
        combined_timing["next_planner_prefetch_used"] = 0.0
        return HighLevelPrediction(
            current_subtask=planner_prediction.current_subtask,
            confidence=(
                planner_prediction.confidence
                if planner_prediction.parse_ok
                else verifier_prediction.confidence
            ),
            raw_text=(
                "verifier_prediction="
                f"{verifier_prediction.raw_text}\n"
                "next_planner_prompt="
                f"{planner_prompt}\n"
                "next_planner_prediction="
                f"{planner_prediction.raw_text}"
            ),
            parse_ok=verifier_prediction.parse_ok and planner_prediction.parse_ok,
            parse_error=(
                planner_prediction.parse_error
                if not planner_prediction.parse_ok
                else verifier_prediction.parse_error
            ),
            timing_ms=combined_timing,
            action=verifier_prediction.action,
            subtask_status=verifier_prediction.subtask_status,
            subtasks=planner_prediction.subtasks,
            high_level_status=verifier_prediction.high_level_status,
            verifier_next_subtask=verifier_prediction.verifier_next_subtask,
            evidence=verifier_prediction.evidence,
        )

    def step(
        self,
        *,
        timestamp_s: float,
        frame_id: int,
        force: bool = False,
        trigger: str | None = None,
        report_status: str = "unknown",
    ) -> HighLevelDecision | None:
        request = self.begin_step(
            timestamp_s=timestamp_s,
            frame_id=frame_id,
            force=force,
            trigger=trigger,
            report_status=report_status,
        )
        if request is None:
            return None

        synthetic_initial = request.initial and self.planner_mode in {
            "profile_all",
            "lazy_profile",
        }
        backend_started = time.perf_counter()
        if synthetic_initial:
            prediction = self._synthetic_planner_prediction(request)
        else:
            prediction = self.predict_request(request)
            prediction = self.finalize_prediction_for_request(request, prediction)
        backend_finished = time.perf_counter()
        decision = self.finish_step(request, prediction)
        if synthetic_initial:
            decision.timing_ms["backend_predict_skipped"] = 1.0
            decision.timing_ms["synthetic_initial_planner"] = 1.0
        decision.timing_ms["backend_predict"] = (
            backend_finished - backend_started
        ) * 1000.0
        decision.timing_ms["sync_request_to_decision"] = (
            time.perf_counter_ns() - request.request_started_ns
        ) / 1_000_000.0
        return decision

    def trigger_handoff(
        self,
        *,
        timestamp_s: float,
        frame_id: int,
        trigger: str,
        report_status: str = "completed",
    ) -> HighLevelDecision | None:
        """Run one externally confirmed subtask handoff.

        The caller owns completion detection. This controller only consumes
        the event, summarizes low-level execution, and emits the next subtask.
        """

        if not self._has_initial_decision:
            raise RuntimeError("cannot trigger a handoff before the initial plan")
        return self.step(
            timestamp_s=timestamp_s,
            frame_id=frame_id,
            force=True,
            trigger=trigger,
            report_status=report_status,
        )
