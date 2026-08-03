"""Shared Hy-VLA VLM backend for low-frequency high-level decisions.

This backend reuses the VLM tower already loaded by ``HyVLAPolicyWrapper``.
It does not load a second multi-billion-parameter model. The caller supplies
the shared ``HunYuanVLMoTForConditionalGeneration`` module and the local
Hy-VLA checkpoint directory containing tokenizer/image-processor assets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from .burger_fries import (
    ExecutionReport,
    FrameRecord,
    HighLevelPrediction,
    parse_high_level_prediction_text,
)


class SharedHunyuanVlmBackend:
    """Generate high-level JSON using the VLM tower inside Hy-VLA.

    The forced-boundary controller remains responsible for scheduling and
    memory bookkeeping. This backend only turns selected evidence images plus
    the controller prompt into a structured high-level prediction.
    """

    def __init__(
        self,
        *,
        vlm: Any,
        checkpoint_path: str | Path,
        camera_name: str = "head_camera",
        max_images: int = 8,
        max_new_tokens: int = 160,
    ) -> None:
        from transformers import (
            AutoImageProcessor,
            AutoTokenizer,
            Qwen3VLVideoProcessor,
        )

        from hy_vla.hunyuan_vl_mot import HunYuanVLMoTProcessor

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(
                "SharedHunyuanVlmBackend needs a local self-contained Hy-VLA "
                f"checkpoint, got {checkpoint_path}"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=False,
        )
        image_processor = AutoImageProcessor.from_pretrained(
            checkpoint_path,
            trust_remote_code=False,
        )
        # The VLA checkpoint ships an image processor but no standalone
        # raw-VLM config for AutoVideoProcessor. The high-level path currently
        # uses images only; ProcessorMixin still requires a video processor.
        self.processor = HunYuanVLMoTProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=Qwen3VLVideoProcessor(),
            chat_template=getattr(tokenizer, "chat_template", None),
        )
        self.vlm = vlm
        self.camera_name = camera_name
        self.max_images = max(1, int(max_images))
        self.max_new_tokens = max(1, int(max_new_tokens))

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
            frame_ids.append(frame.frame_id)
        return images[-self.max_images :], frame_ids[-self.max_images :]

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
        images, image_frame_ids = self._select_images(
            memory_frames=memory_frames,
            recent_frames=recent_frames,
        )
        if not images:
            return HighLevelPrediction(
                current_subtask=phase_prompt,
                confidence=0.0,
                raw_text="",
                parse_ok=False,
                parse_error=(
                    f"no {self.camera_name!r} visual evidence available for "
                    "the high-level VLM"
                ),
            )

        response_schema = (
            "\nReturn exactly one JSON object with these fields:\n"
            '{"current_subtask": string, "keyframe_positions": [int], '
            '"confidence": float}\n'
            "Use only evidence supported by the supplied images. "
            "This is a forced-boundary baseline: do not claim that the "
            "previous subtask succeeded merely because its time budget ended."
        )
        content = [
            *({"type": "image", "image": image} for image in images),
            {"type": "text", "text": prompt + response_schema},
        ]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        batch = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
        )

        try:
            device = next(self.vlm.parameters()).device
        except StopIteration as exc:
            raise RuntimeError("shared VLM has no parameters") from exc
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        input_length = int(batch["input_ids"].shape[1])
        with torch.inference_mode():
            output_ids = self.vlm.generate(
                **batch,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        raw_text = self.processor.tokenizer.batch_decode(
            output_ids[:, input_length:],
            skip_special_tokens=True,
        )[0].strip()
        prediction = parse_high_level_prediction_text(
            raw_text,
            fallback_subtask=phase_prompt,
            default_confidence=0.0,
        )
        if prediction.parse_ok:
            prediction.raw_text = (
                f"evidence_frame_ids={image_frame_ids}\n{prediction.raw_text}"
            )
        return prediction


__all__ = ["SharedHunyuanVlmBackend"]
