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
        warmup_calls: int = 3,
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
        }:
            raise ValueError(
                "image_selection must be one of 'tail', 'all', "
                "'recent_tail', 'uniform', or 'latest'; "
                f"got {image_selection!r}"
            )
        self.image_max_side = max(0, int(image_max_side))
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

        if self.image_selection == "recent_tail":
            candidates = recent or all_frames
            return candidates[-self.max_images :]

        if self.image_selection == "uniform":
            candidates = recent or all_frames
            if len(candidates) <= self.max_images:
                return candidates
            if self.max_images == 1:
                return [candidates[-1]]
            indices = [
                round(i * (len(candidates) - 1) / (self.max_images - 1))
                for i in range(self.max_images)
            ]
            return [candidates[index] for index in indices]

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

    @staticmethod
    def _response_schema(recent_positions: Sequence[int | None]) -> str:
        evidence_mapping = ", ".join(
            f"image {index} -> recent position {position}"
            if position is not None
            else f"image {index} -> persistent-memory frame"
            for index, position in enumerate(recent_positions, start=1)
        ) or "none"
        return (
            "\nReturn exactly one JSON object and no Markdown:\n"
            '{"current_subtask": string, "keyframe_positions": [int], '
            '"confidence": number}\n'
            "The listed keyframe positions are 1-indexed positions in the full recent "
            "frame list, not positions in the compressed image list. The compressed "
            f"image mapping is: {evidence_mapping}. Only select a recent-frame "
            "position when recording a keyframe. Use only visual evidence and the "
            "execution report. If the report says completed, move to the next "
            "unfinished subtask."
        )

    def _build_payload(
        self,
        *,
        images: Sequence[Any],
        prompt: str,
        recent_positions: Sequence[int | None],
    ) -> tuple[dict[str, Any], dict[str, float]]:
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
        content.append(
            {
                "type": "text",
                "text": prompt + self._response_schema(recent_positions),
            }
        )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_new_tokens,
            "response_format": {"type": "json_object"},
            # Qwen3.6 defaults to thinking mode. A robot handoff needs the
            # short structured answer, not a latent reasoning transcript.
            "chat_template_kwargs": {"enable_thinking": False},
        }
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
        }

    def _request(
        self,
        *,
        images: Sequence[Any],
        prompt: str,
        recent_positions: Sequence[int | None],
    ) -> tuple[str, dict[str, float]]:
        encode_started = time.perf_counter_ns()
        payload, payload_metrics = self._build_payload(
            images=images,
            prompt=prompt,
            recent_positions=recent_positions,
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
                f"Qwen3.6 vLLM request failed for {self.server_url}: {exc}"
            ) from exc
        request_finished = time.perf_counter_ns()

        try:
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (KeyError, TypeError, ValueError, requests.HTTPError) as exc:
            excerpt = response.text[:1000]
            raise RuntimeError(
                "Qwen3.6 vLLM returned an invalid chat-completions response: "
                f"{excerpt}"
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "Qwen3.6 vLLM response did not contain non-empty message content."
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
        }

    def _warmup(self, warmup_calls: int) -> float:
        if warmup_calls <= 0:
            return 0.0

        images = [
            Image.new("RGB", (448, 448), color=(0, 0, 0))
            for _ in range(self.max_images)
        ]
        prompt = (
            "Warm up the high-level robot planner. Return "
            '{"current_subtask":"warmup","keyframe_positions":[],"confidence":0.0}.'
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
    ) -> HighLevelPrediction:
        del timestamp_s, phase_index, execution_report
        online_started = time.perf_counter_ns()
        select_started = time.perf_counter_ns()
        images, image_frame_ids, recent_positions = self._select_images(
            memory_frames=memory_frames,
            recent_frames=recent_frames,
        )
        select_finished = time.perf_counter_ns()
        if not images:
            return HighLevelPrediction(
                current_subtask=phase_prompt,
                confidence=0.0,
                raw_text="",
                parse_ok=False,
                parse_error=(
                    f"no {self.camera_name!r} visual evidence available for "
                    "the Qwen3.6 vLLM backend"
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

        raw_text, request_timing = self._request(
            images=images,
            prompt=prompt,
            recent_positions=recent_positions,
        )
        parse_started = time.perf_counter_ns()
        prediction = parse_high_level_prediction_text(
            raw_text,
            fallback_subtask=phase_prompt,
            default_confidence=0.0,
        )
        parse_finished = time.perf_counter_ns()
        online_finished = time.perf_counter_ns()
        prediction.raw_text = (
            f"evidence_frame_ids={image_frame_ids}\n{prediction.raw_text}"
        )
        prediction.timing_ms.update(
            {
                "select_images": self._elapsed_ms(select_started, select_finished),
                **request_timing,
                "json_parse": self._elapsed_ms(parse_started, parse_finished),
                "online_total": self._elapsed_ms(online_started, online_finished),
                "startup_warmup": float(self.startup_warmup_ms),
                # The vLLM server is launched with one native MTP layer.
                "mtp_enabled": 1.0,
                "mtp_speculative_tokens": 1.0,
            }
        )
        return prediction


__all__ = ["Qwen36VllmBackend"]
