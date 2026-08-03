# coding=utf-8
"""Estimate forced-boundary high-level / low-level scheduling overhead.

This is a systems profiling helper, not a policy evaluator. It uses the
same handoff abstraction as ``BurgerFriesHighLevelController`` and answers:

- how much wall-clock time a synchronous high-level call would block;
- how early an asynchronous high-level call must start before a subtask
  boundary;
- whether a chosen forced-boundary duration is plausible for closed-loop
  control.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class ScheduledCall:
    dispatch_s: float
    ready_s: float
    trigger: str
    phase_at_dispatch: int
    phase_at_ready: int


def _phase_at(t: float, forced_boundary_s: float) -> int:
    return 0 if t < forced_boundary_s else 1


def _build_dispatch_times(
    *,
    total_seconds: float,
    forced_boundary_s: float,
    max_forced_boundaries: int,
    subtask_lookahead_s: float,
) -> list[tuple[float, str]]:
    dispatches: list[tuple[float, str]] = [(0.0, "initial_plan")]
    for boundary_index in range(1, max_forced_boundaries + 1):
        boundary_s = forced_boundary_s * boundary_index
        if boundary_s > total_seconds + 1e-9:
            break
        dispatches.append((
            round(max(0.0, boundary_s - subtask_lookahead_s), 6),
            f"forced_boundary_{boundary_index}",
        ))

    merged: dict[float, str] = {}
    for t, reason in sorted(dispatches):
        if t not in merged:
            merged[t] = reason
        elif reason not in merged[t]:
            merged[t] = f"{merged[t]}+{reason}"
    return [(t, merged[t]) for t in sorted(merged)]


def profile_schedule(args: argparse.Namespace) -> dict:
    dispatches = _build_dispatch_times(
        total_seconds=args.total_seconds,
        forced_boundary_s=args.forced_boundary_s,
        max_forced_boundaries=args.max_forced_boundaries,
        subtask_lookahead_s=args.subtask_lookahead_s,
    )
    calls = [
        ScheduledCall(
            dispatch_s=dispatch_s,
            ready_s=dispatch_s + args.high_level_latency_s,
            trigger=trigger,
            phase_at_dispatch=_phase_at(dispatch_s, args.forced_boundary_s),
            phase_at_ready=_phase_at(
                dispatch_s + args.high_level_latency_s,
                args.forced_boundary_s,
            ),
        )
        for dispatch_s, trigger in dispatches
    ]

    boundary_calls = [call for call in calls if call.trigger.startswith("forced_boundary_")]
    boundary_delays = [
        max(
            0.0,
            call.ready_s - args.forced_boundary_s * int(call.trigger.rsplit("_", 1)[1]),
        )
        for call in boundary_calls
    ]
    async_boundary_delay_s = max(boundary_delays, default=0.0)

    # The initial plan happens before robot motion begins. Only boundary
    # calls produce in-episode synchronous pauses.
    synchronous_blocked_s = len(boundary_calls) * args.high_level_latency_s
    synchronous_wall_s = args.total_seconds + synchronous_blocked_s
    low_level_step_budget_ms = 1000.0 / args.low_level_fps
    high_level_latency_steps = args.high_level_latency_s * args.low_level_fps

    return {
        "task": "place_burger_fries",
        "total_env_seconds": args.total_seconds,
        "low_level_fps": args.low_level_fps,
        "low_level_step_budget_ms": low_level_step_budget_ms,
        "forced_boundary_s": args.forced_boundary_s,
        "max_forced_boundaries": args.max_forced_boundaries,
        "subtask_lookahead_s": args.subtask_lookahead_s,
        "high_level_latency_s": args.high_level_latency_s,
        "high_level_latency_low_level_steps": high_level_latency_steps,
        "num_high_level_calls": len(calls),
        "num_in_episode_boundary_calls": len(boundary_calls),
        "initial_plan_latency_s": args.high_level_latency_s,
        "synchronous_blocked_s": synchronous_blocked_s,
        "synchronous_wall_s": synchronous_wall_s,
        "synchronous_blocked_pct_of_env_time": 100.0 * synchronous_blocked_s / args.total_seconds,
        "async_boundary_delay_s": async_boundary_delay_s,
        "minimum_lookahead_to_hide_latency_s": args.high_level_latency_s,
        "calls": [asdict(call) for call in calls],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-seconds", type=float, default=32.0)
    parser.add_argument("--low-level-fps", type=float, default=25.0)
    parser.add_argument("--forced-boundary-s", type=float, default=5.0)
    parser.add_argument("--max-forced-boundaries", type=int, default=1)
    parser.add_argument("--subtask-lookahead-s", type=float, default=0.0)
    parser.add_argument("--high-level-latency-s", type=float, default=2.0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = profile_schedule(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
