"""Independent Qwen2.5-VL backend for high-level robot handoff decisions.

The low-level Hy-VLA action model and this instruction-following VLM have
different jobs:

* Hy-VLA keeps its short visual/action history and produces fast actions.
* Qwen2.5-VL is called only at a high-level handoff. It reads a small set of
  visual evidence frames plus the completed-subtask report, then emits the
  next language subtask as JSON.

This first implementation is deliberately synchronous so that we can measure
real planning latency. The controller/backend interface is unchanged when the
future asynchronous worker is introduced.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from .burger_fries import (
    ExecutionReport,
    FrameRecord,
    HighLevelPrediction,
    parse_high_level_prediction_text,
)


class Qwen25VlmBackend:
    """Use a standalone Qwen2.5-VL Instruct model as the high-level planner."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        device: str = "auto",
        camera_name: str = "head_camera",
        max_images: int = 6,
        max_new_tokens: int = 128,
        attention_implementation: str = "flash_attention_2",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 768 * 28 * 28,
        warmup_calls: int = 3,
    ) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(
                "Qwen25VlmBackend needs a local Qwen2.5-VL checkpoint. "
                f"Run scripts/download_qwen25_vl_persistent.sh first; got {model_path}"
            )

        self.device = self._resolve_device(device)
        self.camera_name = camera_name
        self.max_images = max(1, int(max_images))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=max(28 * 28, int(min_pixels)),
            max_pixels=max(28 * 28, int(max_pixels)),
            trust_remote_code=False,
        )

        model_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            # Pin the whole high-level model to its own logical device. Do
            # not use device_map="auto": it could occupy the low-level GPU.
            "device_map": {"": self.device},
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        if attention_implementation:
            model_kwargs["attn_implementation"] = attention_implementation

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs,
        )
        self.model.eval()
        self.startup_warmup_ms = self._warmup(max(0, int(warmup_calls)))

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Choose a high-level GPU without changing visibility variables."""

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen2.5-VL high-level planning requires an assigned CUDA GPU."
            )

        requested = str(requested).strip().lower()
        visible_count = torch.cuda.device_count()
        if requested in {"", "auto"}:
            if visible_count < 2:
                raise RuntimeError(
                    "The hierarchical Qwen backend needs two platform-assigned GPUs: "
                    "logical cuda:0 for low-level Hy-VLA and logical cuda:1 for Qwen."
                )
            return "cuda:1"

        if not requested.startswith("cuda:"):
            raise ValueError(
                "hierarchical_qwen_device must be 'auto' or a logical CUDA device "
                f"such as 'cuda:1', got {requested!r}"
            )
        try:
            device_index = int(requested.split(":", maxsplit=1)[1])
        except ValueError as exc:
            raise ValueError(
                "hierarchical_qwen_device must be 'auto' or 'cuda:<logical_index>'"
            ) from exc
        if not 0 <= device_index < visible_count:
            raise RuntimeError(
                f"{requested} is unavailable: this lease exposes {visible_count} GPU(s)."
            )
        return requested

    def _sync(self) -> None:
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(self.device))

    @staticmethod
    def _elapsed_ms(start_ns: int, end_ns: int) -> float:
        return (end_ns - start_ns) / 1_000_000.0

    def _prepare_batch(
        self,
        *,
        images: Sequence[Any],
        prompt: str,
    ) -> tuple[dict[str, Any], int, dict[str, float]]:
        timings: dict[str, float] = {}

        template_start = time.perf_counter_ns()
        messages = self._build_messages(images, prompt)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        template_end = time.perf_counter_ns()
        timings["chat_template"] = self._elapsed_ms(template_start, template_end)

        processor_start = time.perf_counter_ns()
        batch = self.processor(
            text=[text],
            images=list(images),
            padding=True,
            return_tensors="pt",
        )
        processor_end = time.perf_counter_ns()
        timings["processor"] = self._elapsed_ms(processor_start, processor_end)

        self._sync()
        h2d_start = time.perf_counter_ns()
        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        self._sync()
        h2d_end = time.perf_counter_ns()
        timings["h2d_copy"] = self._elapsed_ms(h2d_start, h2d_end)
        return batch, int(batch["input_ids"].shape[1]), timings

    def _generate_and_decode(
        self,
        *,
        batch: dict[str, Any],
        input_length: int,
        max_new_tokens: int,
    ) -> tuple[str, dict[str, float]]:
        timings: dict[str, float] = {}
        self._sync()
        generate_start = time.perf_counter_ns()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        self._sync()
        generate_end = time.perf_counter_ns()
        timings["model_generate"] = self._elapsed_ms(generate_start, generate_end)

        decode_start = time.perf_counter_ns()
        raw_text = self.processor.batch_decode(
            output_ids[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        decode_end = time.perf_counter_ns()
        timings["decode"] = self._elapsed_ms(decode_start, decode_end)
        return raw_text, timings

    def _warmup(self, warmup_calls: int) -> float:
        if warmup_calls <= 0:
            return 0.0

        prompt = (
            "Warm up the high-level robot planner. Return JSON with "
            'current_subtask="warmup", confidence=0.0.'
        )
        # Match the largest online visual context so the warmup covers vision
        # preprocessing and attention shapes used by a real handoff.
        images = [
            Image.new("RGB", (448, 448), color=(0, 0, 0))
            for _ in range(self.max_images)
        ]
        started = time.perf_counter_ns()
        for _ in range(warmup_calls):
            batch, input_length, _ = self._prepare_batch(
                images=images,
                prompt=prompt,
            )
            self._generate_and_decode(
                batch=batch,
                input_length=input_length,
                max_new_tokens=min(self.max_new_tokens, 32),
            )
        self._sync()
        finished = time.perf_counter_ns()
        return self._elapsed_ms(started, finished)

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
    def _build_messages(images: Sequence[Any], prompt: str) -> list[dict[str, Any]]:
        response_schema = (
            "\nYou must return exactly one JSON object and no Markdown:\n"
            '{"current_subtask": string, "confidence": number}\n'
            "Use only visual evidence and the execution report. If the "
            "report says completed, move to the next unfinished subtask."
        )
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image} for image in images
        ]
        content.append({"type": "text", "text": prompt + response_schema})
        return [{"role": "user", "content": content}]

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
        del timestamp_s, phase_index, execution_report, output_schema, allow_text_only
        online_start = time.perf_counter_ns()
        select_start = time.perf_counter_ns()
        images, image_frame_ids = self._select_images(
            memory_frames=memory_frames,
            recent_frames=recent_frames,
        )
        select_end = time.perf_counter_ns()
        if not images:
            return HighLevelPrediction(
                current_subtask=phase_prompt,
                confidence=0.0,
                raw_text="",
                parse_ok=False,
                parse_error=(
                    f"no {self.camera_name!r} visual evidence available for "
                    "the high-level Qwen2.5-VL backend"
                ),
                timing_ms={
                    "select_images": self._elapsed_ms(select_start, select_end),
                    "online_total": self._elapsed_ms(online_start, time.perf_counter_ns()),
                },
            )

        batch, input_length, prepare_timing = self._prepare_batch(
            images=images,
            prompt=prompt,
        )
        raw_text, generate_timing = self._generate_and_decode(
            batch=batch,
            input_length=input_length,
            max_new_tokens=self.max_new_tokens,
        )
        parse_start = time.perf_counter_ns()
        prediction = parse_high_level_prediction_text(
            raw_text,
            fallback_subtask=phase_prompt,
            default_confidence=0.0,
        )
        parse_end = time.perf_counter_ns()
        prediction.raw_text = (
            f"evidence_frame_ids={image_frame_ids}\n{prediction.raw_text}"
        )
        online_end = time.perf_counter_ns()
        prediction.timing_ms.update(
            {
                "select_images": self._elapsed_ms(select_start, select_end),
                **prepare_timing,
                **generate_timing,
                "json_parse": self._elapsed_ms(parse_start, parse_end),
                "online_total": self._elapsed_ms(online_start, online_end),
                "startup_warmup": float(self.startup_warmup_ms),
            }
        )
        return prediction


__all__ = ["Qwen25VlmBackend"]
