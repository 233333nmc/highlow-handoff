# Fixed-Boundary Incremental Planning: Test 1 vs Test 2

This branch compares two Gemma high-level planner prompts with the same fixed-boundary high-level/low-level controller. It contains the evaluation commands, original task instructions, and 120 rollout videos annotated with the actual subtask sent to Hy-VLA.

Each prompt version covers 12 tasks with five episodes per task. This README works directly in a private GitHub repository and does not require GitHub Pages.

## Experiment logic

The high-level Gemma planner generates the first object-level subtask from the full RoboTwin task instruction. Hy-VLA executes that subtask until a task-specific fixed boundary. At the boundary, low-level execution pauses while Gemma receives unseen head-camera history sampled at 1 Hz and returns only:

```json
{"completed": true}
```

or:

```json
{"completed": false}
```

When the answer is `true`, the controller publishes the prefetched next subtask and clears actions cached under the old instruction. When the answer is `false`, it retains the current instruction and action cache. Each episode permits up to ten fixed-boundary checks.

Common settings:

- High-level model: Gemma 4 through the server at `http://127.0.0.1:8003`
- Low-level model: Hy-VLA
- Planner mode: incremental `vlm_plan_next`
- Completion output: `completion_bool`
- Visual memory: unseen frames sampled at 1 Hz, up to eight images
- High-level execution: blocking (`HL_ASYNC_ENABLED=False`)
- Evaluation: 12 tasks, five episodes per prompt version

## Test 1 and Test 2

### Test 1: original planner prompt

Test 1 asks Gemma to generate one object-level robot subtask. It does not explicitly prohibit mentioning multiple objects, so some outputs combine actions such as placing both the burger and fries in one instruction.

### Test 2: one-object-only planner prompt

Test 2 adds only these constraints:

```text
Manipulate one object only and finish at one target.
Do not mention another object, another action, a list, or 'then'.
```

The completion verifier, image history, fixed boundaries, switching logic, and Hy-VLA policy are unchanged. The implementation exposes an explicit switch: Test 1 runs with `HL_ONE_OBJECT_ONLY=False`, while the Test 2 wrapper sets `ONE_OBJECT_ONLY=True`.

## Annotated video examples

GitHub may not embed repository MP4 files as inline players in a README. The links below open the actual annotated files for repository collaborators.

| Version | Burger and fries, episode 0 |
| --- | --- |
| Test 1 | [Open annotated MP4](test1_test2_report/videos/test1/place_burger_fries/episode0_subtask.mp4) |
| Test 2 | [Open annotated MP4](test1_test2_report/videos/test2/place_burger_fries/episode0_subtask.mp4) |

## All tasks and videos

Each Test 1 or Test 2 link opens a directory containing episodes 0-4.

| Task | Original RoboTwin instruction | Test 1 | Test 2 |
| --- | --- | --- | --- |
| Place cans in plastic box | `Use dual arm to pick and place cans into plasticbox` | [5 videos](test1_test2_report/videos/test1/place_cans_plasticbox/) | [5 videos](test1_test2_report/videos/test2/place_cans_plasticbox/) |
| Stack three bowls | `stack the three bowls on top of each other` | [5 videos](test1_test2_report/videos/test1/stack_bowls_three/) | [5 videos](test1_test2_report/videos/test2/stack_bowls_three/) |
| Stack two bowls | `stack the two bowls on top of each other` | [5 videos](test1_test2_report/videos/test1/stack_bowls_two/) | [5 videos](test1_test2_report/videos/test2/stack_bowls_two/) |
| Place two shoes | `use both arms to pick up the two shoes on the table and put them in the shoebox, with the shoe tip pointing to the left` | [5 videos](test1_test2_report/videos/test1/place_dual_shoes/) | [5 videos](test1_test2_report/videos/test2/place_dual_shoes/) |
| Hang a mug | `Use left arm to pick the mug on the table, rotate the mug and put the mug down in the middle of the table, use the right arm to pick the mug and hang it onto the rack.` | [5 videos](test1_test2_report/videos/test1/hanging_mug/) | [5 videos](test1_test2_report/videos/test2/hanging_mug/) |
| Put an object in a cabinet | `use {a} to open the cabinet's drawer, and use another arm to put the object on the table to the drawer.` | [5 videos](test1_test2_report/videos/test1/put_object_cabinet/) | [5 videos](test1_test2_report/videos/test2/put_object_cabinet/) |
| Handover a block | `use the left arm to grasp the red block on the table, handover it to the right arm and place it on the blue pad` | [5 videos](test1_test2_report/videos/test1/handover_block/) | [5 videos](test1_test2_report/videos/test2/handover_block/) |
| Handover a microphone | `Use one arm to grasp the microphone on the table and handover it to the other arm` | [5 videos](test1_test2_report/videos/test1/handover_mic/) | [5 videos](test1_test2_report/videos/test2/handover_mic/) |
| Stack two blocks | `there are two blocks on the table, the color of the blocks is <red, green>, <move the blocks to the center of the table>, and <stack the geen block on the red block>` | [5 videos](test1_test2_report/videos/test1/stack_blocks_two/) | [5 videos](test1_test2_report/videos/test2/stack_blocks_two/) |
| Arrange RGB blocks | `<Place> the red block, green block, and blue block <in the order> of red, green, and blue from left to right, <placing in a row>.` | [5 videos](test1_test2_report/videos/test1/blocks_ranking_rgb/) | [5 videos](test1_test2_report/videos/test2/blocks_ranking_rgb/) |
| Place burger and fries | `Use dual arm to pick the hamburg and frenchfries and put them onto the tray.` | [5 videos](test1_test2_report/videos/test1/place_burger_fries/) | [5 videos](test1_test2_report/videos/test2/place_burger_fries/) |
| Stack three blocks | `there are three blocks on the table, the color of the blocks is <red, green and blue>, <move the blocks to the center of the table>, and <stack the blue block on the green block, and the green block on the red block>` | [5 videos](test1_test2_report/videos/test1/stack_blocks_three/) | [5 videos](test1_test2_report/videos/test2/stack_blocks_three/) |

## Reproduction commands

The Gemma service must already be running on port 8003.

```bash
cd /mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory/Hy-Embodied-0.5-VLA

# Test 1: original planner prompt and task-specific boundary calibration
TEST_NUM=5 bash scripts/run_gemma_fixed_next_12tasks_n5.sh

# Test 2: one-object-only prompt, reusing Test 1 boundary calibration
TEST_NUM=5 bash scripts/run_gemma_fixed_next_test2.sh
```

The complete experiment notes are in [the detailed report](test1_test2_report/TEST1_TEST2_REPORT.md). The local interactive comparison page is retained in the experiment directory but requires a normal static HTTP server because GitHub does not execute repository HTML as a hosted website.
