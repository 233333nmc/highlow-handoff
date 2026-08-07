"""Gemma 4 high-level planner backend."""

from __future__ import annotations

from typing import Any

from .qwen36_vllm_backend import Qwen36VllmBackend


class Gemma4VllmBackend(Qwen36VllmBackend):
    """OpenAI-compatible backend for Gemma 4 12B.

    The payload shape is the same multimodal chat-completions request used by
    the Qwen3.6 backend. Gemma does not use Qwen's `enable_thinking` chat
    template kwarg, so this wrapper disables model-specific template kwargs.
    """

    def __init__(
        self,
        *,
        server_url: str,
        model_name: str = "gemma-4-12B-it",
        camera_name: str = "head_camera",
        max_images: int = 1,
        max_new_tokens: int = 48,
        request_timeout_s: float = 120.0,
        jpeg_quality: int = 90,
        image_selection: str = "latest",
        image_max_side: int = 0,
        output_schema: str = "minimal_status",
        warmup_calls: int = 3,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            server_url=server_url,
            model_name=model_name,
            camera_name=camera_name,
            max_images=max_images,
            max_new_tokens=max_new_tokens,
            request_timeout_s=request_timeout_s,
            jpeg_quality=jpeg_quality,
            image_selection=image_selection,
            image_max_side=image_max_side,
            output_schema=output_schema,
            warmup_calls=warmup_calls,
            chat_template_kwargs=chat_template_kwargs,
            include_chat_template_kwargs=bool(chat_template_kwargs),
            backend_label="Gemma 4 Transformers API",
        )


__all__ = ["Gemma4VllmBackend"]
