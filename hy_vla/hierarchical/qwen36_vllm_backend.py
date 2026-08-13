"""Qwen3.6 high-level VLM client backed by a local vLLM server.

Qwen3.6 uses a newer architecture than the Transformers version bundled with
the Hy-VLA evaluation environment. Keeping it in a separate vLLM service
avoids changing the low-level runtime and enables Qwen's native MTP decoding.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import requests
from PIL import Image

from .burger_fries import (
    ExecutionReport,
    FrameRecord,
    HighLevelPrediction,
    parse_high_level_prediction_text,
)


class Qwen36VllmBackend:
    """Request high-level decisions from Qwen3.6 served by vLLM with MTP."""

    def __init__(
        self,
        *,
        server_url: str,
        model_name: str = "qwen3.6-27b-mtp",
        camera_name: str = "head_camera",
        max_images: int = 6,
        max_new_tokens: int = 64,
        request_timeout_s: float = 120.0,
        jpeg_quality: int = 90,
        image_selection: str = "tail",
        image_max_side: int = 0,
        output_schema: str = "legacy",
        warmup_calls: int = 3,
        chat_template_kwargs: dict[str, Any] | None = None,
        include_chat_template_kwargs: bool = True,
        backend_label: str = "Qwen3.6 vLLM",
        mtp_speculative_tokens: int = 1,
    ) -> None:
        self.server_url = self._normalize_server_url(server_url)
        self.model_name = str(model_name)
        self.camera_name = str(camera_name)
        self.max_images = max(1, int(max_images))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.request_timeout_s = max(1.0, float(request_timeout_s))
        self.jpeg_quality = min(100, max(1, int(jpeg_quality)))
        self.image_selection = str(image_selection).strip().lower()
        if self.image_selection not in {
            "tail",
            "all",
            "recent_tail",
            "uniform",
            "latest",
            "first_latest",
            "diverse",
        }:
            raise ValueError(
                "image_selection must be one of 'tail', 'all', "
                "'recent_tail', 'uniform', 'latest', 'first_latest', "
                "or 'diverse'; "
                f"got {image_selection!r}"
            )
        self.image_max_side = max(0, int(image_max_side))
        self.output_schema = str(output_schema).strip().lower()
        if self.output_schema not in {
            "legacy",
            "minimal",
            "next_subtask",
            "minimal_status",
            "completion_status",
            "completion_bool",
            "subtask_plan",
        }:
            raise ValueError(
                "output_schema must be 'legacy', 'minimal', 'next_subtask', "
                "'minimal_status', "
                "'completion_status', 'completion_bool', or 'subtask_plan', "
                f"got {output_schema!r}"
            )
        if include_chat_template_kwargs:
            self.chat_template_kwargs = (
                {"enable_thinking": False}
                if chat_template_kwargs is None
                else dict(chat_template_kwargs)
            )
        else:
            self.chat_template_kwargs = {}
        self.backend_label = str(backend_label)
        self.mtp_speculative_tokens = max(0, int(mtp_speculative_tokens))
        self.session = requests.Session()
        self.startup_warmup_ms = self._warmup(max(0, int(warmup_calls)))

    @staticmethod
    def _normalize_server_url(server_url: str) -> str:
        base = str(server_url).strip().rstrip("/")
        if not base:
            raise ValueError("hierarchical_qwen36_server_url must not be empty.")
        if base.endswith("/v1/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    @staticmethod
    def _elapsed_ms(start_ns: int, end_ns: int) -> float:
        return (end_ns - start_ns) / 1_000_000.0

    @staticmethod
    def _as_rgb_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[-1] not in {3, 4}:
                raise ValueError(
                    "Qwen3.6 high-level images must be HWC RGB/RGBA arrays, "
                    f"got shape {image.shape!r}."
                )
            array = image[..., :3]
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            return Image.fromarray(array, mode="RGB")
        raise TypeError(
            "Qwen3.6 high-level images must be PIL.Image.Image or numpy.ndarray, "
            f"got {type(image).__name__}."
        )

    def _encode_image_data_url(self, image: Any) -> tuple[str, int, int, int]:
        pil_image = self._as_rgb_image(image)
        if self.image_max_side > 0 and max(pil_image.size) > self.image_max_side:
            pil_image = pil_image.copy()
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            pil_image.thumbnail(
                (self.image_max_side, self.image_max_side),
                resampling,
            )
        output = io.BytesIO()
        pil_image.save(
            output,
            format="JPEG",
            quality=self.jpeg_quality,
            optimize=False,
        )
        jpeg_bytes = output.getvalue()
        encoded = base64.b64encode(jpeg_bytes).decode("ascii")
        return (
            f"data:image/jpeg;base64,{encoded}",
            len(jpeg_bytes),
            int(pil_image.width),
            int(pil_image.height),
        )

    def _deduplicate_frames(
        self,
        frames: Sequence[FrameRecord],
    ) -> list[FrameRecord]:
        seen_ids: set[int] = set()
        unique: list[FrameRecord] = []
        for frame in frames:
            if frame.frame_id in seen_ids:
                continue
            if frame.camera_images.get(self.camera_name) is None:
                continue
            seen_ids.add(frame.frame_id)
            unique.append(frame)
        return unique

    def _frame_signature(self, frame: FrameRecord) -> np.ndarray:
        image = frame.camera_images.get(self.camera_name)
        if image is None:
            return np.zeros((32 * 32,), dtype=np.float32)
        pil_image = self._as_rgb_image(image).convert("L").resize((32, 32))
        return np.asarray(pil_image, dtype=np.float32).reshape(-1) / 255.0

    @staticmethod
    def _select_uniform_records(
        candidates: Sequence[FrameRecord],
        *,
        max_count: int,
    ) -> list[FrameRecord]:
        if len(candidates) <= max_count:
            return list(candidates)
        if max_count <= 1:
            return [candidates[-1]]
        indices = [
            round(i * (len(candidates) - 1) / (max_count - 1))
            for i in range(max_count)
        ]
        return [candidates[index] for index in indices]

    def _select_diverse_records(
        self,
        candidates: Sequence[FrameRecord],
    ) -> list[FrameRecord]:
        if len(candidates) <= self.max_images:
            return list(candidates)
        if self.max_images <= 1:
            return [candidates[-1]]

        signatures = [self._frame_signature(frame) for frame in candidates]
        selected_indices: set[int] = {len(candidates) - 1}
        while len(selected_indices) < self.max_images:
            best_index = None
            best_score = float("-inf")
            for index, signature in enumerate(signatures):
                if index in selected_indices:
                    continue
                score = min(
                    float(np.mean(np.abs(signature - signatures[selected])))
                    for selected in selected_indices
                )
                # Prefer later frames when two candidates are visually similar.
                score += index * 1e-6
                if score > best_score:
                    best_index = index
                    best_score = score
            if best_index is None:
                break
            selected_indices.add(best_index)
        return [candidates[index] for index in sorted(selected_indices)]

    def _select_frame_records(
        self,
        *,
        memory_frames: Sequence[FrameRecord],
        recent_frames: Sequence[FrameRecord],
    ) -> list[FrameRecord]:
        all_frames = self._deduplicate_frames([*memory_frames, *recent_frames])
        recent = self._deduplicate_frames(recent_frames)
        if not all_frames:
            return []

        if self.image_selection == "latest":
            return [recent[-1] if recent else all_frames[-1]]

        if self.image_selection == "first_latest":
            candidates = recent or all_frames
            if self.max_images <= 1 or len(candidates) <= 1:
                return [candidates[-1]]
            if self.max_images == 2:
                return [candidates[0], candidates[-1]]
            middle = self._select_uniform_records(
                candidates[1:-1],
                max_count=self.max_images - 2,
            )
            return [candidates[0], *middle, candidates[-1]]

        if self.image_selection == "recent_tail":
            candidates = recent or all_frames
            return candidates[-self.max_images :]

        if self.image_selection == "uniform":
            candidates = recent or all_frames
            return self._select_uniform_records(
                candidates,
                max_count=self.max_images,
            )

        if self.image_selection == "diverse":
            candidates = recent or all_frames
            return self._select_diverse_records(candidates)

        if self.image_selection == "all":
            return all_frames[: self.max_images]

        # `tail` preserves the original backend behavior.
        return all_frames[-self.max_images :]

    def _select_images(
        self,
        *,
        memory_frames: Sequence[FrameRecord],
        recent_frames: Sequence[FrameRecord],
    ) -> tuple[list[Any], list[int], list[int | None]]:
        recent_position_by_id = {
            int(frame.frame_id): index
            for index, frame in enumerate(recent_frames, start=1)
        }
        selected_frames = self._select_frame_records(
            memory_frames=memory_frames,
            recent_frames=recent_frames,
        )
        images: list[Any] = []
        frame_ids: list[int] = []
        recent_positions: list[int | None] = []
        for frame in selected_frames:
            image = frame.camera_images.get(self.camera_name)
            if image is None:
                continue
            images.append(image)
            frame_ids.append(int(frame.frame_id))
            recent_positions.append(recent_position_by_id.get(int(frame.frame_id)))
        return images, frame_ids, recent_positions

    def _response_schema(
        self,
        recent_positions: Sequence[int | None],
        *,
        output_schema: str | None = None,
    ) -> str:
        schema = self.output_schema if output_schema is None else str(output_schema)
        if schema == "subtask_plan":
            return (
                "\nReturn exactly one JSON object and no Markdown:\n"
                '{"subtasks": [string], "confidence": number}\n'
                "The subtasks must be an ordered list of low-level robot "
                "instructions. Do not include explanations."
            )
        if schema == "completion_status":
            return (
                "\nReturn exactly one JSON object and no Markdown:\n"
                '{"decision": "same" | "switch", '
                '"status": "running" | "completed" | "failed" | "stuck" | "uncertain", '
                '"next_subtask": string, '
                '"evidence": string}\n'
                "Use decision='same' when the current low-level subtask should "
                "keep running. Use decision='switch' only when the provided "
                "image(s) show the current low-level subtask is completed. Keep "
                "next_subtask as an empty string unless you have a direct "
                "suggestion. Evidence must be one short visual reason."
            )
        if schema == "completion_bool":
            return (
                "\nReturn exactly one JSON object and no Markdown:\n"
                '{"completed": true | false}'
            )
        if schema == "minimal_status":
            return (
                "\nReturn exactly one JSON object and no Markdown:\n"
                '{"action": "continue" | "switch", '
                '"next_subtask": string, '
                '"subtask_status": "unfinished" | "completed" | "unknown"}\n'
                "Use action='continue' when the current subtask appears "
                "unfinished and the low-level policy should keep executing "
                "the current instruction. Use action='switch' only when the "
                "system should publish next_subtask to the low-level policy."
            )
        if schema == "next_subtask":
            return (
                "\nReturn exactly one JSON object and no Markdown:\n"
                '{"next_subtask": string}'
            )
        if schema == "minimal":
            return (
                "\nReturn exactly one JSON object and no Markdown:\n"
                '{"next_subtask": string}\n'
                "The value must be the next low-level robot instruction. "
                "Use only the task instruction, visual evidence, and execution "
                "summary. If the current subtask appears unfinished, return "
                "the current subtask or a short continuation instruction."
            )
        evidence_mapping = ", ".join(
            f"image {index} -> recent position {position}"
            if position is not None
            else f"image {index} -> persistent-memory frame"
            for index, position in enumerate(recent_positions, start=1)
        ) or "none"
        return (
            "\nReturn exactly one JSON object and no Markdown:\n"
            '{"current_subtask": string, "confidence": number}\n'
            "Use only visual evidence and the execution report. If the report "
            "says completed, move to the next unfinished subtask. The compressed "
            f"image mapping is: {evidence_mapping}."
        )

    def _build_payload(
        self,
        *,
        images: Sequence[Any],
        prompt: str,
        recent_positions: Sequence[int | None],
        output_schema: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, float], str, str]:
        content: list[dict[str, Any]] = []
        encoded_bytes = 0
        total_pixels = 0
        widths: list[int] = []
        heights: list[int] = []
        for image in images:
            data_url, byte_count, width, height = self._encode_image_data_url(image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )
            encoded_bytes += byte_count
            total_pixels += width * height
            widths.append(width)
            heights.append(height)
        response_schema = self._response_schema(
            recent_positions,
            output_schema=output_schema,
        )
        final_prompt = prompt + response_schema
        content.append(
            {
                "type": "text",
                "text": final_prompt,
            }
        )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_new_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        image_count = len(widths)
        return payload, {
            "image_count": float(image_count),
            "image_encoded_bytes": float(encoded_bytes),
            "image_total_pixels": float(total_pixels),
            "image_avg_width": (
                sum(widths) / image_count if image_count else 0.0
            ),
            "image_avg_height": (
                sum(heights) / image_count if image_count else 0.0
            ),
        }, response_schema, final_prompt

    def _request(
        self,
        *,
        images: Sequence[Any],
        prompt: str,
        recent_positions: Sequence[int | None],
        output_schema: str | None = None,
    ) -> tuple[str, dict[str, float], str, str]:
        encode_started = time.perf_counter_ns()
        payload, payload_metrics, response_schema, final_prompt = self._build_payload(
            images=images,
            prompt=prompt,
            recent_positions=recent_positions,
            output_schema=output_schema,
        )
        encode_finished = time.perf_counter_ns()

        request_started = time.perf_counter_ns()
        try:
            response = self.session.post(
                self.server_url,
                json=payload,
                timeout=self.request_timeout_s,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"{self.backend_label} request failed for {self.server_url}: {exc}. "
                "Make sure the matching vLLM service is running and fully ready "
                "before starting the RoboTwin eval script."
            ) from exc
        request_finished = time.perf_counter_ns()

        try:
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (KeyError, TypeError, ValueError, requests.HTTPError) as exc:
            excerpt = response.text[:1000]
            raise RuntimeError(
                f"{self.backend_label} returned an invalid chat-completions response: "
                f"{excerpt}"
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                f"{self.backend_label} response did not contain non-empty message content."
            )
        usage = body.get("usage")
        usage_metrics = {
            "usage_available": 1.0 if isinstance(usage, dict) else 0.0,
            "usage_prompt_tokens": float(
                usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
            ),
            "usage_completion_tokens": float(
                usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
            ),
            "usage_total_tokens": float(
                usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
            ),
        }
        return content.strip(), {
            **payload_metrics,
            **usage_metrics,
            "request_encode": self._elapsed_ms(encode_started, encode_finished),
            "vllm_request": self._elapsed_ms(request_started, request_finished),
        }, response_schema, final_prompt

    def _warmup(self, warmup_calls: int) -> float:
        if warmup_calls <= 0:
            return 0.0

        images = [
            Image.new("RGB", (448, 448), color=(0, 0, 0))
            for _ in range(self.max_images)
        ]
        if self.output_schema == "subtask_plan":
            prompt = (
                "Warm up the high-level robot planner. Return "
                '{"subtasks":["warmup"],"confidence":0.0}.'
            )
        elif self.output_schema == "completion_status":
            prompt = (
                "Warm up the high-level robot completion verifier. Return "
                '{"decision":"same","status":"uncertain",'
                '"next_subtask":"","evidence":"warmup"}.'
            )
        elif self.output_schema == "completion_bool":
            prompt = (
                "Warm up the high-level robot completion verifier. Return "
                '{"completed":false}.'
            )
        elif self.output_schema == "minimal_status":
            prompt = (
                "Warm up the high-level robot planner. Return "
                '{"action":"switch","next_subtask":"warmup",'
                '"subtask_status":"unknown"}.'
            )
        elif self.output_schema == "minimal":
            prompt = (
                "Warm up the high-level robot planner. Return "
                '{"next_subtask":"warmup"}.'
            )
        elif self.output_schema == "next_subtask":
            prompt = (
                "Warm up the high-level robot planner. Return "
                '{"next_subtask":"warmup"}.'
            )
        else:
            prompt = (
                "Warm up the high-level robot planner. Return "
                '{"current_subtask":"warmup","confidence":0.0}.'
            )
        started = time.perf_counter_ns()
        for _ in range(warmup_calls):
            self._request(
                images=images,
                prompt=prompt,
                recent_positions=[None] * len(images),
            )
        finished = time.perf_counter_ns()
        return self._elapsed_ms(started, finished)

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
        del timestamp_s, phase_index, execution_report
        online_started = time.perf_counter_ns()
        select_started = time.perf_counter_ns()
        images, image_frame_ids, recent_positions = self._select_images(
            memory_frames=memory_frames,
            recent_frames=recent_frames,
        )
        select_finished = time.perf_counter_ns()
        if not images and not allow_text_only:
            return HighLevelPrediction(
                current_subtask=phase_prompt,
                confidence=0.0,
                raw_text="",
                parse_ok=False,
                parse_error=(
                    f"no {self.camera_name!r} visual evidence available for "
                    f"the {self.backend_label} backend"
                ),
                timing_ms={
                    "select_images": self._elapsed_ms(
                        select_started, select_finished
                    ),
                    "online_total": self._elapsed_ms(
                        online_started, time.perf_counter_ns()
                    ),
                },
            )

        raw_text, request_timing, response_schema, final_prompt = self._request(
            images=images,
            prompt=prompt,
            recent_positions=recent_positions,
            output_schema=output_schema,
        )
        parse_started = time.perf_counter_ns()
        prediction = parse_high_level_prediction_text(
            raw_text,
            fallback_subtask=phase_prompt,
            default_confidence=0.0,
            output_schema=output_schema,
        )
        parse_finished = time.perf_counter_ns()
        online_finished = time.perf_counter_ns()
        evidence_prefix = (
            f"evidence_frame_ids={image_frame_ids}"
            if image_frame_ids
            else "evidence_frame_ids=[]; text_only=true"
        )
        prediction.raw_text = f"{evidence_prefix}\n{prediction.raw_text}"
        prediction.response_schema = response_schema
        prediction.final_prompt = final_prompt
        prediction.evidence_frame_ids = list(image_frame_ids)
        prediction.timing_ms.update(
            {
                "select_images": self._elapsed_ms(select_started, select_finished),
                **request_timing,
                "json_parse": self._elapsed_ms(parse_started, parse_finished),
                "online_total": self._elapsed_ms(online_started, online_finished),
                "startup_warmup": float(self.startup_warmup_ms),
                # The vLLM server is launched with one native MTP layer.
                "mtp_enabled": 1.0 if self.mtp_speculative_tokens > 0 else 0.0,
                "mtp_speculative_tokens": float(self.mtp_speculative_tokens),
            }
        )
        return prediction


__all__ = ["Qwen36VllmBackend"]
