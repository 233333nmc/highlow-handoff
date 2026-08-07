# coding=utf-8
"""Dry-run the Burger/Fries hierarchical controller on dummy frames.

This script does not require a GPU. It exists to validate the control
surface we want before attaching a real high-level VLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hy_vla.hierarchical import BurgerFriesHighLevelController, FrameRecord


def simulate(controller: BurgerFriesHighLevelController, total_seconds: float, fps: float) -> list[dict]:
    decisions: list[dict] = []
    total_frames = int(round(total_seconds * fps))
    refresh_stride = max(1, int(round(fps / max(controller.history_fps, 1e-6))))

    controller.reset()
    for frame_id in range(total_frames):
        timestamp_s = frame_id / fps
        frame = FrameRecord(
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            caption=f"dummy frame {frame_id}",
            metadata={"simulated": True},
        )
        controller.observe(frame)
        decision = controller.step(timestamp_s=timestamp_s, frame_id=frame_id)
        if decision is not None:
            decisions.append(
                {
                    "timestamp_s": timestamp_s,
                    **decision.to_dict(),
                }
            )
        controller.record_low_level_step(
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            state=[timestamp_s, 0.0, 0.0, 0.0],
            action=[0.01, 0.0, 0.0, 0.0],
            low_level_call_ms=4.0,
            model_forward_ms=20.0 if frame_id % 7 == 0 else 0.0,
            used_action_cache=(frame_id % 7 != 0),
        )
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-seconds", type=float, default=32.0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--history-fps", type=float, default=1.0)
    parser.add_argument("--forced-boundary-s", type=float, default=5.0)
    parser.add_argument("--max-forced-boundaries", type=int, default=1)
    parser.add_argument("--max-memory-frames", type=int, default=256)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    controller = BurgerFriesHighLevelController(
        forced_boundary_s=args.forced_boundary_s,
        max_forced_boundaries=args.max_forced_boundaries,
        history_fps=args.history_fps,
        source_fps=args.fps,
        max_memory_frames=args.max_memory_frames,
    )
    decisions = simulate(controller, total_seconds=args.total_seconds, fps=args.fps)
    print(json.dumps(decisions, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(decisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
