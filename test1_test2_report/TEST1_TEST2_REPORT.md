# Fixed-Boundary Incremental Planning: Test 1 and Test 2

This directory is a local, self-contained review package. Open `index.html` in a browser or serve this directory with a static HTTP server. The page lets a viewer select any of the 12 tasks and any of the five episodes, then seek through the Test 1 and Test 2 videos independently with the normal video progress bar.

## Scope

Both runs use the Gemma fixed-boundary incremental `next` controller. At the boundary, the high-level completion verifier receives the configured image history and decides whether the active subtask is complete. Only a positive completion decision publishes the prefetched next subtask; a negative decision keeps the current instruction and low-level action cache. The low-level Hy-VLA policy performs the robot actions.

Common protocol:

- 12 tasks, 5 episodes per task (`n=5`)
- `HANDOFF_TRIGGER=forced_boundary`, task-specific boundary calibration, and up to 10 handoffs per episode
- Gemma 4 server at `http://127.0.0.1:8003`, `GEMMA4_OUTPUT_SCHEMA=completion_bool`
- `HL_PLANNER_MODE=vlm_plan_next`, `HL_ASYNC_ENABLED=False`
- Unseen visual memory sampled at 1 frame per second from a 25 FPS source, up to 8 images
- Videos are the actual rollout videos with recorded subtask traces overlaid; the overlay is not a reconstructed timeline

## Test 1 versus Test 2

Test 1 is the original planner prompt. Its planner request starts with:

```text
Generate one object-level robot subtask.
Task: <task instruction>
Subtask index: <index>
...
```

Test 2 uses the same controller and model settings, but changes the planner contract to:

```text
Generate one robot subtask.
Task: <task instruction>
Subtask index: <index>
Manipulate one object only and finish at one target.
Do not mention another object, another action, a list, or 'then'.
The subtask must name the object and its final target.
Do not output a grasp-only, lift-only, move-only, or release-only step.
```

The change is deliberately narrow. It does not add a phase-target field and does not change the completion verifier, boundary timing, low-level model, image sampling, or `next` handoff logic. The purpose is to reduce compound planner outputs such as “pick up the burger and fries and place both on the tray,” making each published subtask easier to attribute to a single object.

The prompt switch is explicit in code: Test 1 runs with `HL_ONE_OBJECT_ONLY=False`, while `scripts/run_gemma_fixed_next_test2.sh` sets `ONE_OBJECT_ONLY=True`. Re-running the two commands therefore reproduces different planner prompts instead of depending on which prompt was last edited in the source file.

The videos should be interpreted together with the traces: a compound instruction or a missed completion switch can make later low-level behavior look like a planning failure, while an instruction-following failure can occur even when the high-level subtask is reasonable.

## Original task instructions

These are the task strings passed to the policy before planner-generated subtasks are produced.

| Task id | Original instruction |
| --- | --- |
| `place_cans_plasticbox` | Use dual arm to pick and place cans into plasticbox |
| `stack_bowls_three` | stack the three bowls on top of each other |
| `stack_bowls_two` | stack the two bowls on top of each other |
| `place_dual_shoes` | use both arms to pick up the two shoes on the table and put them in the shoebox, with the shoe tip pointing to the left |
| `hanging_mug` | Use left arm to pick the mug on the table, rotate the mug and put the mug down in the middle of the table, use the right arm to pick the mug and hang it onto the rack. |
| `put_object_cabinet` | use {a} to open the cabinet's drawer, and use another arm to put the object on the table to the drawer. |
| `handover_block` | use the left arm to grasp the red block on the table, handover it to the right arm and place it on the blue pad |
| `handover_mic` | Use one arm to grasp the microphone on the table and handover it to the other arm |
| `stack_blocks_two` | there are two blocks on the table, the color of the blocks is <red, green>, <move the blocks to the center of the table>, and <stack the geen block on the red block> |
| `blocks_ranking_rgb` | <Place> the red block, green block, and blue block <in the order> of red, green, and blue from left to right, <placing in a row>. |
| `place_burger_fries` | Use dual arm to pick the hamburg and frenchfries and put them onto the tray. |
| `stack_blocks_three` | there are three blocks on the table, the color of the blocks is <red, green and blue>, <move the blocks to the center of the table>, and <stack the blue block on the green block, and the green block on the red block> |

## Reproduction commands

Run these from the Hy-VLA repository directory. The Gemma service on port 8003 must already be running, and the RoboTwin Hy-VLA environment must be available.

### Test 1

This is the original five-episode run. It calibrates task-specific boundaries and writes them to `profile_results/gemma_fixed_next_12tasks_n5_20260818/boundary_calibration.tsv`.

```bash
cd /mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory/Hy-Embodied-0.5-VLA
TEST_NUM=5 \
RUN_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory/profile_results/gemma_fixed_next_12tasks_n5_20260818 \
bash scripts/run_gemma_fixed_next_12tasks_n5.sh
```

### Test 2

This run reuses Test 1's boundary calibration and applies the one-object-only prompt.

```bash
cd /mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory/Hy-Embodied-0.5-VLA
TEST_NUM=5 bash scripts/run_gemma_fixed_next_test2.sh
```

The wrapper keeps the output separate in `profile_results/gemma_fixed_next_test2`. To regenerate one annotated video after a rollout (repeat for the desired task and episode):

```bash
cd /mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory/Hy-Embodied-0.5-VLA
python scripts/overlay_subtasks_on_video.py \
  --video /path/to/source_episode.mp4 \
  --trace /mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory/profile_results/gemma_fixed_next_test2/<task>/low_level_instruction_trace.jsonl \
  --output /mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory/profile_results/gemma_fixed_next_test2/<task>/annotated_videos/episode0_subtask.mp4 \
  --episode-index 0
```

The existing batch output contains the full per-task traces, raw high-level JSONL, low-level instruction trace, and source rollout videos in addition to the copied annotated videos in this review package.

## Viewing and eventual GitHub upload

For local viewing:

```bash
cd /mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory/Hy-Embodied-0.5-VLA/test1_test2_report
python3 -m http.server 8768 --bind 0.0.0.0
```

Then open `http://127.0.0.1:8768/`. Serving the directory is preferable to opening the HTML directly because some browsers restrict video loading from `file://` pages. The native `<video controls>` element provides a draggable timeline.

This package has not been committed or pushed. Once reviewed, the directory can be added to the requested branch. Before uploading, check repository hosting limits and repository policy: the package contains approximately 175 MB of annotated MP4s, with each video kept as a separate file so a viewer can stream or download only the selected episode.
