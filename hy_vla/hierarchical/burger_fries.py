"""Burger / fries hierarchical controller scaffold.

This module is intentionally lightweight:
- it does not depend on a specific high-level VLM yet;
- it provides the state machine, frame buffer and keyframe filter that
  we can later connect to MemER-style high-level prompting or a future
  text-generation backend.

The immediate goal is to profile a two-stage robot task:

1. place the hamburg on the tray
2. place the french fries on the tray

The controller keeps a short rolling visual history, periodically emits a
structured high-level memory record, and optionally selects keyframes
using a MemER-style merge distance.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from statistics import median, median_low
from typing import Any, Callable, Protocol, Sequence


@dataclass(slots=True)
class FrameRecord:
    """One observed frame in the high-level memory stream."""

    frame_id: int
    timestamp_s: float
    caption: str = ""
    event_tags: tuple[str, ...] = ()
    is_keyframe: bool = False
    camera_images: dict[str, Any] = field(default_factory=dict, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def short_label(self) -> str:
        tag_str = ",".join(self.event_tags) if self.event_tags else "-"
        flag = "K" if self.is_keyframe else "-"
        return f"f{self.frame_id}@{self.timestamp_s:.2f}s[{flag}|{tag_str}]"


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
    selected_keyframes: list[int]
    confidence: float
    should_refresh_low_level: bool
    memory_keyframes: list[int] = field(default_factory=list)
    context_frames: list[int] = field(default_factory=list)
    reason: str = ""
    prompt: str = ""
    raw_prediction: str = ""
    prediction_parse_ok: bool = True
    prediction_parse_error: str | None = None
    trigger: str = "initial"
    execution_report: ExecutionReport | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_text": self.memory_text,
            "next_subtask": self.next_subtask,
            "phase_index": self.phase_index,
            "selected_keyframes": list(self.selected_keyframes),
            "memory_keyframes": list(self.memory_keyframes),
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
        }


@dataclass(slots=True)
class HighLevelPrediction:
    """Structured output expected from a high-level memory policy.

    `keyframe_positions` are 1-indexed positions in the current recent
    context, matching the MemER deployment contract. They are mapped to
    absolute frame ids before entering persistent memory.
    """

    current_subtask: str
    keyframe_positions: list[int] = field(default_factory=list)
    confidence: float = 1.0
    raw_text: str = ""
    parse_ok: bool = True
    parse_error: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)


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
    """Parse a MemER-style high-level JSON prediction.

    Expected schema:

    {
      "current_subtask": "Place the hamburg on the tray.",
      "keyframe_positions": [1, 4],
      "confidence": 0.8
    }
    """

    raw_text = text
    try:
        payload = json.loads(_extract_json_object(text))
        current_subtask = str(payload.get("current_subtask") or fallback_subtask)
        raw_positions = payload.get("keyframe_positions", [])
        if raw_positions is None:
            keyframe_positions: list[int] = []
        elif isinstance(raw_positions, int):
            keyframe_positions = [raw_positions]
        else:
            keyframe_positions = [int(x) for x in raw_positions]
        confidence = float(payload.get("confidence", default_confidence))
        return HighLevelPrediction(
            current_subtask=current_subtask,
            keyframe_positions=keyframe_positions,
            confidence=max(0.0, min(1.0, confidence)),
            raw_text=raw_text,
            parse_ok=True,
        )
    except Exception as exc:  # noqa: BLE001 - keep raw VLM parse failure visible.
        return HighLevelPrediction(
            current_subtask=fallback_subtask,
            keyframe_positions=[],
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
    ) -> HighLevelPrediction:
        raw = self.generate_fn(
            prompt=prompt,
            memory_frames=memory_frames,
            recent_frames=recent_frames,
            timestamp_s=timestamp_s,
            phase_index=phase_index,
            phase_prompt=phase_prompt,
            execution_report=execution_report,
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
        )


@dataclass(slots=True)
class CandidateCluster:
    start_frame_id: int
    end_frame_id: int
    representative_frame_id: int
    vote_count: int


class EpisodicKeyframeMemory:
    """Persistent keyframe memory with 1D clustering over frame ids."""

    def __init__(self, *, merge_distance_frames: int = 5, memory_length: int = 8) -> None:
        self.merge_distance_frames = max(0, int(merge_distance_frames))
        self.memory_length = max(0, int(memory_length))
        self._candidate_ids: list[int] = []
        self._frames_by_id: dict[int, FrameRecord] = {}

    def reset(self) -> None:
        self._candidate_ids.clear()
        self._frames_by_id.clear()

    def remember_frames(self, frames: Sequence[FrameRecord]) -> None:
        for frame in frames:
            self._frames_by_id[int(frame.frame_id)] = frame

    def add_candidates(
        self,
        frame_ids: Sequence[int],
        *,
        frames_by_id: dict[int, FrameRecord] | None = None,
    ) -> list[int]:
        accepted = [int(frame_id) for frame_id in frame_ids if int(frame_id) >= 0]
        if not accepted:
            return []
        self._candidate_ids.extend(accepted)
        self._candidate_ids.sort()
        if frames_by_id:
            self._frames_by_id.update(frames_by_id)
        return accepted

    def all_candidates(self) -> list[int]:
        return list(self._candidate_ids)

    def vote_counts(self) -> dict[int, int]:
        return dict(sorted(Counter(self._candidate_ids).items()))

    def clusters(self) -> list[CandidateCluster]:
        if not self._candidate_ids:
            return []
        ordered = sorted(self._candidate_ids)
        clusters: list[list[int]] = [[ordered[0]]]
        for frame_id in ordered[1:]:
            if frame_id - clusters[-1][-1] <= self.merge_distance_frames:
                clusters[-1].append(frame_id)
            else:
                clusters.append([frame_id])
        return [
            CandidateCluster(
                start_frame_id=cluster[0],
                end_frame_id=cluster[-1],
                representative_frame_id=int(median_low(cluster)),
                vote_count=len(cluster),
            )
            for cluster in clusters
        ]

    def selected_frame_ids(self) -> list[int]:
        if self.memory_length == 0:
            return []
        ids = [cluster.representative_frame_id for cluster in self.clusters()]
        return ids[-self.memory_length :]

    def visible_frame_ids(self, context_frame_ids: Sequence[int]) -> list[int]:
        context_set = {int(frame_id) for frame_id in context_frame_ids}
        return [frame_id for frame_id in self.selected_frame_ids() if frame_id not in context_set]

    def visible_frames(self, context_frame_ids: Sequence[int]) -> list[FrameRecord]:
        out: list[FrameRecord] = []
        for frame_id in self.visible_frame_ids(context_frame_ids):
            frame = self._frames_by_id.get(frame_id)
            if frame is not None:
                out.append(frame)
        return out


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
    ) -> HighLevelPrediction:
        if not recent_frames:
            keyframe_positions: list[int] = []
        elif not memory_frames:
            keyframe_positions = [len(recent_frames)]
        else:
            keyframe_positions = []

        return HighLevelPrediction(
            current_subtask=phase_prompt,
            keyframe_positions=keyframe_positions,
            confidence=0.9 if keyframe_positions or memory_frames else 0.7,
            raw_text=(
                "{"
                f'"current_subtask": "{phase_prompt}", '
                f'"keyframe_positions": {keyframe_positions}'
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


class KeyframeFilter:
    """MemER-style 1D merge filter for candidate keyframes."""

    def __init__(self, merge_distance_frames: int = 5) -> None:
        self.merge_distance_frames = max(0, int(merge_distance_frames))

    def cluster(self, candidates: list[FrameRecord]) -> list[list[FrameRecord]]:
        ordered = sorted(candidates, key=lambda frame: frame.frame_id)
        if not ordered:
            return []

        clusters: list[list[FrameRecord]] = [[ordered[0]]]
        for frame in ordered[1:]:
            if frame.frame_id - clusters[-1][-1].frame_id <= self.merge_distance_frames:
                clusters[-1].append(frame)
            else:
                clusters.append([frame])
        return clusters

    def select_representatives(self, candidates: list[FrameRecord]) -> list[FrameRecord]:
        reps: list[FrameRecord] = []
        for cluster in self.cluster(candidates):
            if len(cluster) == 1:
                reps.append(cluster[0])
                continue
            idx = int(median(range(len(cluster))))
            reps.append(cluster[idx])
        return reps


class BurgerFriesHighLevelController:
    """Configurable high-level controller for sequential RoboTwin tasks.

    It supports two boundary sources with one shared handoff contract:

    - forced boundaries for timing controls;
    - externally confirmed boundaries from a simulator oracle or future
      visual completion detector.

    Both paths produce an ``ExecutionReport`` and then update memory before
    publishing the next low-level subtask.
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
        keyframe_merge_distance: int = 5,
        memory_length: int = 8,
        max_memory_frames: int = 256,
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
        self.keyframe_filter = KeyframeFilter(keyframe_merge_distance)
        self.memory = EpisodicKeyframeMemory(
            merge_distance_frames=keyframe_merge_distance,
            memory_length=memory_length,
        )
        self.backend = backend or ScriptedBurgerFriesBackend()
        self.buffer = FrameBuffer(maxlen=max_memory_frames)
        self.memory_text = ""
        self.selected_keyframes: list[int] = []
        self.phase_names = resolved_phase_names
        self.phase_prompts = resolved_phase_prompts
        self._has_initial_decision = False
        self._active_phase_index = 0
        self._active_subtask = ""
        self._active_start_timestamp_s: float | None = None
        self._active_start_frame_id: int | None = None
        self._forced_boundary_count = 0
        self._last_visual_capture_s = float("-inf")
        self._reset_execution_accumulator()

    def reset(self) -> None:
        self.buffer.clear()
        self.memory.reset()
        self.memory_text = ""
        self.selected_keyframes = []
        self._has_initial_decision = False
        self._active_phase_index = 0
        self._active_subtask = ""
        self._active_start_timestamp_s = None
        self._active_start_frame_id = None
        self._forced_boundary_count = 0
        self._last_visual_capture_s = float("-inf")
        self._reset_execution_accumulator()

    def observe(self, frame: FrameRecord) -> None:
        self.buffer.append(frame)
        if frame.camera_images:
            self._last_visual_capture_s = frame.timestamp_s

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

    def _candidate_keyframes(self, recent: list[FrameRecord], phase_index: int) -> list[FrameRecord]:
        if not recent:
            return []

        tagged = [frame for frame in recent if frame.is_keyframe]
        if tagged:
            return tagged

        # Heuristic fallback: keep the tail of the current recent window.
        tail = recent[-min(3, len(recent)) :]
        if phase_index > self._active_phase_index and recent:
            tail = [recent[-1]]
        return tail

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
        selected_keyframes: list[FrameRecord],
        prediction: HighLevelPrediction,
        execution_report: ExecutionReport | None,
    ) -> str:
        phase_name = self.phase_names[min(phase_index, len(self.phase_names) - 1)]
        phase_prompt = prediction.current_subtask or self.phase_prompts[min(phase_index, len(self.phase_prompts) - 1)]
        frame_ids = ", ".join(frame.short_label() for frame in recent[-self.recent_frames_length :]) or "none"
        memory_ids = ", ".join(frame.short_label() for frame in memory_frames) or "none"
        keyframe_ids = ", ".join(frame.short_label() for frame in selected_keyframes) or "none"
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
            f"Memory: focus on {phase_prompt}\n"
            f"Execution report: {report_text}\n"
            f"Persistent keyframes: {memory_ids}\n"
            f"Recent frames: {frame_ids}\n"
            f"Keyframes: {keyframe_ids}"
        )

    def build_prompt(
        self,
        *,
        timestamp_s: float,
        phase_index: int,
        recent: list[FrameRecord],
        memory_frames: list[FrameRecord],
        selected_keyframes: list[FrameRecord],
        execution_report: ExecutionReport | None,
    ) -> str:
        memory_desc = "\n".join(f"- {frame.short_label()}" for frame in memory_frames) or "- none"
        recent_desc = "\n".join(f"- {frame.short_label()}" for frame in recent[-self.recent_frames_length :]) or "- none"
        keyframe_desc = "\n".join(f"- {frame.short_label()}" for frame in selected_keyframes) or "- none"
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
            "Persistent memory keyframes:\n"
            f"{memory_desc}\n"
            "Recent frames:\n"
            f"{recent_desc}\n"
            "Candidate keyframes:\n"
            f"{keyframe_desc}\n"
            "Previous low-level execution report:\n"
            f"{report_desc}\n"
            "Return JSON with current_subtask, keyframe_positions, and confidence. "
            "keyframe_positions are 1-indexed positions in the recent-frame list."
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
        update_started = time.perf_counter()
        self.memory.remember_frames(sampled_history)
        memory_updated = time.perf_counter()
        recent = sampled_history[-self.recent_frames_length :]
        context_ids = [frame.frame_id for frame in recent]
        memory_frames = self.memory.visible_frames(context_ids)
        candidates = self._candidate_keyframes(recent, phase_index)
        selected = self.keyframe_filter.select_representatives(candidates)
        prompt = self.build_prompt(
            timestamp_s=timestamp_s,
            phase_index=phase_index,
            recent=recent,
            memory_frames=memory_frames,
            selected_keyframes=selected,
            execution_report=execution_report,
        )
        prompt_built = time.perf_counter()
        prediction = self.backend.predict(
            prompt=prompt,
            memory_frames=memory_frames,
            recent_frames=recent,
            timestamp_s=timestamp_s,
            phase_index=phase_index,
            phase_prompt=self.phase_prompts[min(phase_index, len(self.phase_prompts) - 1)],
            execution_report=execution_report,
        )
        backend_returned = time.perf_counter()
        if phase_index != self._active_phase_index and recent and not prediction.keyframe_positions:
            prediction.keyframe_positions = [len(recent)]
        mapped_ids = self._map_relative_keyframes(
            prediction.keyframe_positions,
            recent,
        )
        frames_by_id = {frame.frame_id: frame for frame in recent}
        selected_ids = self.memory.add_candidates(mapped_ids, frames_by_id=frames_by_id)
        memory_frames_after = self.memory.visible_frames(context_ids)
        selected_frames_after = [frames_by_id[frame_id] for frame_id in selected_ids if frame_id in frames_by_id]
        memory_text = self.build_memory_text(
            phase_index=phase_index,
            recent=recent,
            memory_frames=memory_frames_after,
            selected_keyframes=selected_frames_after or selected,
            prediction=prediction,
            execution_report=execution_report,
        )
        next_subtask = prediction.current_subtask or self.phase_prompts[min(phase_index, len(self.phase_prompts) - 1)]
        confidence = prediction.confidence if prediction.parse_ok else 0.0
        if initial:
            reason = "initial high-level plan"
        elif decision_trigger == "forced_boundary":
            reason = f"forced {duration_s:.2f}s subtask boundary; success remains unverified"
        else:
            reason = (
                f"{decision_trigger} boundary after {duration_s:.2f}s; "
                f"status={report_status}"
            )
        finished = time.perf_counter()

        self._has_initial_decision = True
        self._active_phase_index = phase_index
        self._active_subtask = next_subtask
        self._active_start_timestamp_s = timestamp_s
        self._active_start_frame_id = int(frame_id)
        if not initial:
            self._forced_boundary_count += 1
        self._reset_execution_accumulator()
        self.memory_text = memory_text
        self.selected_keyframes = selected_ids
        controller_timing_ms = {
            "history_and_memory_update": (memory_updated - update_started) * 1000.0,
            "prompt_build": (prompt_built - memory_updated) * 1000.0,
            "backend_predict": (backend_returned - prompt_built) * 1000.0,
            "postprocess": (finished - backend_returned) * 1000.0,
            "total": (finished - update_started) * 1000.0,
        }
        for key, value in prediction.timing_ms.items():
            controller_timing_ms[f"backend_{key}"] = float(value)
        return HighLevelDecision(
            memory_text=memory_text,
            next_subtask=next_subtask,
            phase_index=phase_index,
            selected_keyframes=selected_ids,
            memory_keyframes=[frame.frame_id for frame in memory_frames_after],
            context_frames=context_ids,
            confidence=confidence,
            should_refresh_low_level=True,
            reason=reason,
            prompt=prompt,
            raw_prediction=prediction.raw_text,
            prediction_parse_ok=prediction.parse_ok,
            prediction_parse_error=prediction.parse_error,
            trigger=decision_trigger,
            execution_report=execution_report,
            timing_ms=controller_timing_ms,
        )

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
        the event, summarizes low-level execution, updates memory, and emits
        the next subtask.
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

    @staticmethod
    def _map_relative_keyframes(
        keyframe_positions: Sequence[int],
        recent: Sequence[FrameRecord],
    ) -> list[int]:
        mapped: list[int] = []
        for position in keyframe_positions:
            idx = int(position) - 1
            if 0 <= idx < len(recent):
                mapped.append(int(recent[idx].frame_id))
        return mapped
