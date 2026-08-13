#!/usr/bin/env python3
"""Write the exact text prompts used by the minimal next benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from hy_vla.hierarchical.burger_fries import BurgerFriesHighLevelController
from hy_vla.hierarchical.qwen36_vllm_backend import Qwen36VllmBackend
from hy_vla.hierarchical.robotwin_task_profiles import get_robotwin_task_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    profile = get_robotwin_task_profile("place_burger_fries")
    controller = BurgerFriesHighLevelController(
        task_name=profile.task_name,
        task_instruction=profile.task_instruction,
        phase_names=profile.phase_names,
        phase_prompts=profile.phase_prompts,
        prompt_mode="completion_status",
        planner_mode="vlm_plan_next",
        include_phase_profile_hints=False,
    )
    backend = object.__new__(Qwen36VllmBackend)

    initial = controller._build_vlm_plan_next_prompt(
        phase_index=0,
        execution_report=None,
    ) + backend._response_schema([], output_schema="next_subtask")
    controller._active_subtask = "Place the hamburger on the serving tray."
    prefetch = controller._build_vlm_prefetch_next_prompt(
        phase_index=1,
    ) + backend._response_schema([], output_schema="next_subtask")
    verifier = controller._build_completion_status_prompt(
        timestamp_s=2.4,
        recent=[],
        execution_report=None,
    ) + backend._response_schema([], output_schema="completion_bool")

    old_fields = ('"status"', '"decision"', '"next_subtask"', '"evidence"')
    if verifier.count('"completed"') != 1:
        raise RuntimeError("verifier schema is not unique")
    if any(field in verifier for field in old_fields):
        raise RuntimeError("verifier still contains an old response field")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "=== INITIAL PLANNER FINAL PROMPT ===\n"
        f"{initial}\n\n"
        "=== PREFETCH PLANNER FINAL PROMPT ===\n"
        f"{prefetch}\n\n"
        "=== VERIFIER FINAL PROMPT ===\n"
        f"{verifier}\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
