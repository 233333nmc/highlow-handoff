#!/usr/bin/env python3
"""Exercise the dual-system bridge through two real HTTP connections.

This is a transport smoke probe, not a simulator rollout.  It deliberately
uses the same JSON clients, RoboTwin adapter, and policy hook as the official
four-function shim while supplying one deterministic synthetic observation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping
import uuid

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from highlow.adapters import (  # noqa: E402
    RoboTwinAdapter,
    RoboTwinPolicyHook,
    create_robotwin_bridge,
)
from highlow.clients import JsonHttpInferenceClient  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return parsed


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _ready_health(client: JsonHttpInferenceClient, component: str) -> Mapping[str, Any]:
    health = client.health()
    if not isinstance(health, Mapping):
        raise RuntimeError(f"{component} health response must be a JSON object")
    state = str(health.get("state", "")).strip().lower()
    if state != "ready":
        raise RuntimeError(
            f"{component} is not ready: state={state or '<missing>'!r}, "
            f"response={dict(health)!r}"
        )
    return dict(health)


def _synthetic_observation() -> dict[str, Any]:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    # Give each view a distinct value so accidental camera aliasing is visible
    # to a backend without making this probe depend on image fixtures.
    head = image.copy()
    left = image.copy()
    right = image.copy()
    head[..., 0] = 32
    left[..., 1] = 64
    right[..., 2] = 96
    return {
        "observation": {
            "head_camera": {"rgb": head},
            "left_camera": {"rgb": left},
            "right_camera": {"rgb": right},
        },
        "endpose": {
            "left_endpose": [0.35, 0.20, 0.30, 1.0, 0.0, 0.0, 0.0],
            "left_gripper": 0.0,
            "right_endpose": [0.35, -0.20, 0.30, 1.0, 0.0, 0.0, 0.0],
            "right_gripper": 0.0,
        },
    }


def _clients(
    high_url: str,
    low_url: str,
    timeout_s: float,
) -> tuple[JsonHttpInferenceClient, JsonHttpInferenceClient]:
    high = JsonHttpInferenceClient(
        high_url,
        infer_endpoint="/v1/high-level/infer",
        health_endpoint="/health",
        capabilities_endpoint="/capabilities",
        timeout_s=timeout_s,
    )
    low = JsonHttpInferenceClient(
        low_url,
        infer_endpoint="/v1/infer",
        open_context_endpoint="/v1/episodes/open",
        reset_context_endpoint="/v1/episodes/reset",
        close_context_endpoint="/v1/episodes/close",
        health_endpoint="/health",
        capabilities_endpoint="/capabilities",
        timeout_s=timeout_s,
    )
    return high, low


def run_probe(high_url: str, low_url: str, timeout_s: float) -> dict[str, Any]:
    high, low = _clients(high_url, low_url, timeout_s)
    high_health = _ready_health(high, "high-level service")
    low_health = _ready_health(low, "low-level service")

    bridge = create_robotwin_bridge(
        high,
        low,
        verifier_interval_steps=None,
    )
    hook = RoboTwinPolicyHook(RoboTwinAdapter(), bridge)
    episode_id = f"http-probe-{uuid.uuid4().hex}"
    started = time.perf_counter()
    action: np.ndarray | None = None
    try:
        context = hook.start_episode(
            "dual-system-http-probe",
            "Hold the current dual-arm pose.",
            episode_id=episode_id,
            metadata={"probe": "scripts/probe_dual_system_http.py"},
        )
        if context.episode_id != episode_id:
            raise RuntimeError("bridge returned an unexpected episode id")
        action = np.asarray(
            hook.act(_synthetic_observation(), step_id=0, timestamp_s=0.0),
            dtype=np.float64,
        )
        if action.shape != (16,):
            raise RuntimeError(
                f"low-level action must have shape (16,), received {action.shape!r}"
            )
        if not np.all(np.isfinite(action)):
            raise RuntimeError("low-level action contains non-finite values")
    finally:
        # A partial remote open retains the episode id specifically so this
        # close can clean up any state that was created before a lost response.
        if hook.context is not None:
            hook.finish()

    if action is None:  # Defensive: the successful path above always assigns it.
        raise RuntimeError("probe completed without an action")
    return {
        "schema_version": "highlow.dual-system-http-probe.v1",
        "status": "completed",
        "episode_id": episode_id,
        "high_url": high.base_url,
        "low_url": low.base_url,
        "high_health": high_health,
        "low_health": low_health,
        "action_dimension": int(action.size),
        "action": action.tolist(),
        "elapsed_s": time.perf_counter() - started,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-url", required=True, help="High-level facade URL")
    parser.add_argument("--low-url", required=True, help="Low-level service URL")
    parser.add_argument(
        "--timeout-s",
        type=_positive_float,
        default=120.0,
        help="Per-request HTTP timeout (default: 120)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Independent JSON result path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    started_at = _utc_now()
    try:
        payload = run_probe(args.high_url, args.low_url, args.timeout_s)
    except Exception as exc:
        payload = {
            "schema_version": "highlow.dual-system-http-probe.v1",
            "status": "failed",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "high_url": args.high_url,
            "low_url": args.low_url,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        _write_json(output, payload)
        print(f"HTTP probe failed; result written to {output}", file=sys.stderr)
        return 1

    payload["started_at"] = started_at
    payload["finished_at"] = _utc_now()
    _write_json(output, payload)
    print(f"HTTP probe passed; result written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
