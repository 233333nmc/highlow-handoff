# coding=utf-8
"""Profile Hy-VLA multi-frame inference latency on dummy inputs.

This script is intentionally independent of RoboTwin. It answers the first
systems question for a high/low memory design: how expensive is one Hy-VLA
forward for different history lengths and prompt styles?
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from hy_vla import HyVLA, HyVLAConfig


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**3


def profile_once(policy: HyVLA, config: HyVLAConfig, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    img = torch.zeros(
        args.batch_size,
        args.history_frames,
        3,
        args.image_size,
        args.image_size,
        device=device,
        dtype=dtype,
    )
    state = torch.zeros((args.batch_size, config.max_state_dim), device=device, dtype=dtype)
    batch = {
        "observation.images.top_head": img,
        "observation.images.hand_left": img,
        "observation.images.hand_right": img,
        "observation.state": state,
        "task": [args.task] * args.batch_size,
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    with torch.no_grad():
        for _ in range(args.warmup):
            policy.forward_evaluate(batch)["pred"]
        _sync(device)

        for _ in range(args.iters):
            start = time.perf_counter()
            out = policy.forward_evaluate(batch)["pred"]
            _sync(device)
            times.append(time.perf_counter() - start)

    pred = out[..., : config.action_feature.shape[0]]
    times_sorted = sorted(times)
    return {
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "history_frames": args.history_frames,
        "image_size": args.image_size,
        "task": args.task,
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_s_mean": sum(times) / len(times),
        "latency_s_min": min(times),
        "latency_s_p50": times_sorted[len(times_sorted) // 2],
        "latency_s_max": max(times),
        "peak_memory_gb": _max_memory_gb(device),
        "pred_shape": list(pred.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--history-frames", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--task", default="place the burger on the plate and put the fries in the basket")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config = HyVLAConfig.from_pretrained(args.checkpoint)
    policy = HyVLA.from_pretrained(args.checkpoint, config=config)
    policy.enable_video_encoder_if_needed()
    policy = policy.to(device=device, dtype=dtype).eval()

    result = profile_once(policy, config, args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
