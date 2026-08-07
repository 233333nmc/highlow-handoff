#!/usr/bin/env python3
"""Minimal OpenAI-compatible Gemma 4 chat-completions server.

This is a fallback for Gemma 4 checkpoints that are supported by
Transformers but not yet by vLLM's native/fallback execution path.
"""

from __future__ import annotations

import argparse
import base64
import io
import time
import threading
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor


def _decode_data_url(data_url: str) -> Image.Image:
    prefix = "base64,"
    if prefix not in data_url:
        raise ValueError("only base64 data URLs are supported for image_url")
    encoded = data_url.split(prefix, 1)[1]
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, str):
            converted.append(
                {"role": role, "content": [{"type": "text", "text": content}]}
            )
            continue

        converted_content: list[dict[str, Any]] = []
        if not isinstance(content, list):
            content = [{"type": "text", "text": str(content)}]
        for item in content:
            if not isinstance(item, dict):
                converted_content.append({"type": "text", "text": str(item)})
                continue
            item_type = item.get("type")
            if item_type == "text":
                converted_content.append(
                    {"type": "text", "text": str(item.get("text", ""))}
                )
            elif item_type == "image_url":
                image_url = item.get("image_url", {})
                if isinstance(image_url, dict):
                    url = str(image_url.get("url", ""))
                else:
                    url = str(image_url)
                converted_content.append(
                    {"type": "image", "image": _decode_data_url(url)}
                )
        converted.append({"role": role, "content": converted_content})
    return converted


def _build_app(args: argparse.Namespace) -> FastAPI:
    print(f"Loading Gemma4 processor from {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    print(f"Loading Gemma4 model from {args.model_path} on {args.device}", flush=True)
    started = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map={"": args.device},
    )
    model.eval()
    load_s = time.perf_counter() - started
    device = next(model.parameters()).device
    print(f"Gemma4 model loaded in {load_s:.3f}s on {device}", flush=True)

    assistant_model = None
    assistant_path = str(args.assistant_model_path or "").strip()
    if assistant_path:
        print(
            f"Loading Gemma4 assistant from {assistant_path} on {args.assistant_device}",
            flush=True,
        )
        assistant_started = time.perf_counter()
        assistant_model = AutoModelForCausalLM.from_pretrained(
            assistant_path,
            dtype="auto",
            device_map={"": args.assistant_device},
        )
        assistant_model.eval()
        assistant_model.generation_config.num_assistant_tokens = (
            max(1, int(args.num_assistant_tokens))
        )
        assistant_model.generation_config.num_assistant_tokens_schedule = (
            args.num_assistant_tokens_schedule
        )
        assistant_load_s = time.perf_counter() - assistant_started
        assistant_device = next(assistant_model.parameters()).device
        print(
            f"Gemma4 assistant loaded in {assistant_load_s:.3f}s on {assistant_device}",
            flush=True,
        )
        print(
            "Gemma4 MTP enabled with "
            f"{assistant_model.generation_config.num_assistant_tokens} draft tokens "
            f"and schedule={assistant_model.generation_config.num_assistant_tokens_schedule}",
            flush=True,
        )

    lock = threading.Lock()
    app = FastAPI()

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": args.model_name,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="messages must be a list")

        max_new_tokens = int(payload.get("max_tokens", args.max_new_tokens))
        max_new_tokens = max(1, min(max_new_tokens, args.max_new_tokens_cap))
        temperature = float(payload.get("temperature", 0.0))
        top_p = float(payload.get("top_p", 1.0))
        chat_template_kwargs = payload.get("chat_template_kwargs")
        if not isinstance(chat_template_kwargs, dict):
            chat_template_kwargs = {}
        enable_thinking = bool(chat_template_kwargs.get("enable_thinking", False))

        try:
            converted = _convert_messages(messages)
            inputs = processor.apply_chat_template(
                converted,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            ).to(device)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        input_len = int(inputs["input_ids"].shape[-1])
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        request_started = time.perf_counter()
        with lock, torch.inference_mode():
            if assistant_model is not None:
                outputs = model.generate(
                    **inputs,
                    assistant_model=assistant_model,
                    **generate_kwargs,
                )
            else:
                outputs = model.generate(**inputs, **generate_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
        generation_s = time.perf_counter() - request_started

        completion_ids = outputs[0][input_len:]
        content = processor.decode(
            completion_ids,
            skip_special_tokens=True,
        ).strip()
        completion_tokens = int(completion_ids.shape[-1])
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        return {
            "id": f"chatcmpl-gemma4-{time.time_ns()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": args.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "timing": {"generation_s": generation_s},
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="gemma-4-12B-it")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--assistant-model-path", default="")
    parser.add_argument("--assistant-device", default="cuda:0")
    parser.add_argument("--num-assistant-tokens", type=int, default=4)
    parser.add_argument("--num-assistant-tokens-schedule", default="heuristic")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens-cap", type=int, default=256)
    args = parser.parse_args()

    app = _build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
