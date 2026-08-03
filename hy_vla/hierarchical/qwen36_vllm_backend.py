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
        warmup_calls: int = 3,
    ) -> None:
        self.server_url = self._normalize_server_url(server_url)
        self.model_name = str(model_name)
        self.camera_name = str(camera_name)
        self.max_images = max(1, int(max_images))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.request_timeout_s = max(1.0, float(request_timeout_s))
        self.jpeg_quality = min(100, max(1, int(jpeg_quality)))
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

    def _encode_image_data_url(self, image: Any) -> str:
        output = io.BytesIO()
        self._as_rgb_image(image).save(
            output,
            format="JPEG",
            quality=self.jpeg_quality,
            optimize=False,
        )
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _select_images(
        self,
        *,
        memory_frames: Sequence[FrameRecord],
        recent_frames: Sequence[FrameRecord],
    ) -> tuple[list[Any], list[int]]:
        seen_ids: set[int] = set()
        images: list[Any] = []
        frame_ids: list[int] = []
        for frame in [*memory_frames, *recent_frames]:
            if frame.frame_id in seen_ids:
                continue
            image = frame.camera_images.get(self.camera_name)
            if image is None:
                continue
            seen_ids.add(frame.frame_id)
            images.append(image)
            frame_ids.append(int(frame.frame_id))
        return images[-self.max_images :], frame_ids[-self.max_images :]

    @staticmethod
    def _response_schema() -> str:
        return (
            "\nReturn exactly one JSON object and no Markdown:\n"
            '{"current_subtask": string, "keyframe_positions": [int], '
            '"confidence": number}\n'
            "The listed keyframe positions are 1-indexed positions in the recent "
            "frame list. Use only visual evidence and the execution report. If the "
            "report says completed, move to the next unfinished subtask."
        )

    def _build_payload(self, *, images: Sequence[Any], prompt: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": self._encode_image_data_url(image)},
            }
            for image in images
        ]
        content.append({"type": "text", "text": prompt + self._response_schema()})
        return {
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

    def _request(
        self,
        *,
        images: Sequence[Any],
        prompt: str,
    ) -> tuple[str, dict[str, float]]:
        encode_started = time.perf_counter_ns()
        payload = self._build_payload(images=images, prompt=prompt)
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
        return content.strip(), {
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
            self._request(images=images, prompt=prompt)
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
        images, image_frame_ids = self._select_images(
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

        raw_text, request_timing = self._request(images=images, prompt=prompt)
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
