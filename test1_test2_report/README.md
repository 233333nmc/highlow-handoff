# Fixed-Boundary Incremental Planning: Test 1 vs Test 2

This folder is a GitHub-README-compatible review package for the two Gemma fixed-boundary experiments. It does not require GitHub Pages or a separate web server. Open the video links below in a private repository where the viewer has repository access.

The videos are annotated with the subtask instruction recorded during the actual rollout. Each task has five episodes for each prompt version.

## Inline preview

GitHub's Markdown renderer may sanitize or omit HTML video players in a repository README. The fallback links below always point to the actual MP4 files, which a repository collaborator can open or download. For guaranteed inline playback with a seek bar, use the local `index.html` viewer or upload videos as GitHub file attachments; repository-relative MP4 links alone do not guarantee an embedded player. See [GitHub's documentation on embedded HTML](https://docs.github.com/en/repositories/working-with-files/using-files/working-with-non-code-files#viewing-html-elements).

### Place burger and fries, episode 0

**Test 1: original prompt**

<video controls preload="metadata" width="640" src="videos/test1/place_burger_fries/episode0_subtask.mp4"></video>

[Open Test 1 episode 0](videos/test1/place_burger_fries/episode0_subtask.mp4)

**Test 2: one-object-only prompt**

<video controls preload="metadata" width="640" src="videos/test2/place_burger_fries/episode0_subtask.mp4"></video>

[Open Test 2 episode 0](videos/test2/place_burger_fries/episode0_subtask.mp4)

## Experiments

Both tests use the same robot policy and evaluation protocol:

- High-level backend: Gemma 4 at `http://127.0.0.1:8003`
- Low-level policy: Hy-VLA
- Planner: incremental `vlm_plan_next`
- Handoff: fixed, task-specific simulator boundary; up to 10 handoffs per episode
- Visual input: unseen memory sampled at 1 Hz, up to 8 frames
- High-level execution: blocking (`HL_ASYNC_ENABLED=False`)
- Samples: 12 tasks, 5 episodes per task (`n=5`)

### Test 1: original planner prompt

The planner was asked to generate one object-level robot subtask. In practice, some outputs could combine multiple objects or actions, for example putting the burger and fries into one instruction.

### Test 2: one-object-only planner prompt

The planner prompt was changed only to add these constraints:

```text
Manipulate one object only and finish at one target.
Do not mention another object, another action, a list, or 'then'.
```

The completion verifier, fixed boundaries, image sampling, low-level Hy-VLA policy, and `next` switching logic are unchanged.

The switch is explicit in code: Test 1 uses `HL_ONE_OBJECT_ONLY=False`, and the Test 2 wrapper sets `ONE_OBJECT_ONLY=True`. Re-running the two commands below therefore keeps the prompt versions separate.

## Video index

Each link opens one annotated MP4. `T1` is the original prompt and `T2` is the one-object-only prompt.

| Task | Original task instruction | Videos |
| --- | --- | --- |
| Place cans in plastic box | `Use dual arm to pick and place cans into plasticbox` | [T1 E0](videos/test1/place_cans_plasticbox/episode0_subtask.mp4) · [T1 E1](videos/test1/place_cans_plasticbox/episode1_subtask.mp4) · [T1 E2](videos/test1/place_cans_plasticbox/episode2_subtask.mp4) · [T1 E3](videos/test1/place_cans_plasticbox/episode3_subtask.mp4) · [T1 E4](videos/test1/place_cans_plasticbox/episode4_subtask.mp4) · [T2 E0](videos/test2/place_cans_plasticbox/episode0_subtask.mp4) · [T2 E1](videos/test2/place_cans_plasticbox/episode1_subtask.mp4) · [T2 E2](videos/test2/place_cans_plasticbox/episode2_subtask.mp4) · [T2 E3](videos/test2/place_cans_plasticbox/episode3_subtask.mp4) · [T2 E4](videos/test2/place_cans_plasticbox/episode4_subtask.mp4) |
| Stack three bowls | `stack the three bowls on top of each other` | [T1 E0](videos/test1/stack_bowls_three/episode0_subtask.mp4) · [T1 E1](videos/test1/stack_bowls_three/episode1_subtask.mp4) · [T1 E2](videos/test1/stack_bowls_three/episode2_subtask.mp4) · [T1 E3](videos/test1/stack_bowls_three/episode3_subtask.mp4) · [T1 E4](videos/test1/stack_bowls_three/episode4_subtask.mp4) · [T2 E0](videos/test2/stack_bowls_three/episode0_subtask.mp4) · [T2 E1](videos/test2/stack_bowls_three/episode1_subtask.mp4) · [T2 E2](videos/test2/stack_bowls_three/episode2_subtask.mp4) · [T2 E3](videos/test2/stack_bowls_three/episode3_subtask.mp4) · [T2 E4](videos/test2/stack_bowls_three/episode4_subtask.mp4) |
| Stack two bowls | `stack the two bowls on top of each other` | [T1 E0](videos/test1/stack_bowls_two/episode0_subtask.mp4) · [T1 E1](videos/test1/stack_bowls_two/episode1_subtask.mp4) · [T1 E2](videos/test1/stack_bowls_two/episode2_subtask.mp4) · [T1 E3](videos/test1/stack_bowls_two/episode3_subtask.mp4) · [T1 E4](videos/test1/stack_bowls_two/episode4_subtask.mp4) · [T2 E0](videos/test2/stack_bowls_two/episode0_subtask.mp4) · [T2 E1](videos/test2/stack_bowls_two/episode1_subtask.mp4) · [T2 E2](videos/test2/stack_bowls_two/episode2_subtask.mp4) · [T2 E3](videos/test2/stack_bowls_two/episode3_subtask.mp4) · [T2 E4](videos/test2/stack_bowls_two/episode4_subtask.mp4) |
| Place two shoes | `use both arms to pick up the two shoes on the table and put them in the shoebox, with the shoe tip pointing to the left` | [T1 E0](videos/test1/place_dual_shoes/episode0_subtask.mp4) · [T1 E1](videos/test1/place_dual_shoes/episode1_subtask.mp4) · [T1 E2](videos/test1/place_dual_shoes/episode2_subtask.mp4) · [T1 E3](videos/test1/place_dual_shoes/episode3_subtask.mp4) · [T1 E4](videos/test1/place_dual_shoes/episode4_subtask.mp4) · [T2 E0](videos/test2/place_dual_shoes/episode0_subtask.mp4) · [T2 E1](videos/test2/place_dual_shoes/episode1_subtask.mp4) · [T2 E2](videos/test2/place_dual_shoes/episode2_subtask.mp4) · [T2 E3](videos/test2/place_dual_shoes/episode3_subtask.mp4) · [T2 E4](videos/test2/place_dual_shoes/episode4_subtask.mp4) |
| Hang a mug | `Use left arm to pick the mug on the table, rotate the mug and put the mug down in the middle of the table, use the right arm to pick the mug and hang it onto the rack.` | [T1 E0](videos/test1/hanging_mug/episode0_subtask.mp4) · [T1 E1](videos/test1/hanging_mug/episode1_subtask.mp4) · [T1 E2](videos/test1/hanging_mug/episode2_subtask.mp4) · [T1 E3](videos/test1/hanging_mug/episode3_subtask.mp4) · [T1 E4](videos/test1/hanging_mug/episode4_subtask.mp4) · [T2 E0](videos/test2/hanging_mug/episode0_subtask.mp4) · [T2 E1](videos/test2/hanging_mug/episode1_subtask.mp4) · [T2 E2](videos/test2/hanging_mug/episode2_subtask.mp4) · [T2 E3](videos/test2/hanging_mug/episode3_subtask.mp4) · [T2 E4](videos/test2/hanging_mug/episode4_subtask.mp4) |
| Put an object in a cabinet | `use {a} to open the cabinet's drawer, and use another arm to put the object on the table to the drawer.` | [T1 E0](videos/test1/put_object_cabinet/episode0_subtask.mp4) · [T1 E1](videos/test1/put_object_cabinet/episode1_subtask.mp4) · [T1 E2](videos/test1/put_object_cabinet/episode2_subtask.mp4) · [T1 E3](videos/test1/put_object_cabinet/episode3_subtask.mp4) · [T1 E4](videos/test1/put_object_cabinet/episode4_subtask.mp4) · [T2 E0](videos/test2/put_object_cabinet/episode0_subtask.mp4) · [T2 E1](videos/test2/put_object_cabinet/episode1_subtask.mp4) · [T2 E2](videos/test2/put_object_cabinet/episode2_subtask.mp4) · [T2 E3](videos/test2/put_object_cabinet/episode3_subtask.mp4) · [T2 E4](videos/test2/put_object_cabinet/episode4_subtask.mp4) |
| Handover a block | `use the left arm to grasp the red block on the table, handover it to the right arm and place it on the blue pad` | [T1 E0](videos/test1/handover_block/episode0_subtask.mp4) · [T1 E1](videos/test1/handover_block/episode1_subtask.mp4) · [T1 E2](videos/test1/handover_block/episode2_subtask.mp4) · [T1 E3](videos/test1/handover_block/episode3_subtask.mp4) · [T1 E4](videos/test1/handover_block/episode4_subtask.mp4) · [T2 E0](videos/test2/handover_block/episode0_subtask.mp4) · [T2 E1](videos/test2/handover_block/episode1_subtask.mp4) · [T2 E2](videos/test2/handover_block/episode2_subtask.mp4) · [T2 E3](videos/test2/handover_block/episode3_subtask.mp4) · [T2 E4](videos/test2/handover_block/episode4_subtask.mp4) |
| Handover a microphone | `Use one arm to grasp the microphone on the table and handover it to the other arm` | [T1 E0](videos/test1/handover_mic/episode0_subtask.mp4) · [T1 E1](videos/test1/handover_mic/episode1_subtask.mp4) · [T1 E2](videos/test1/handover_mic/episode2_subtask.mp4) · [T1 E3](videos/test1/handover_mic/episode3_subtask.mp4) · [T1 E4](videos/test1/handover_mic/episode4_subtask.mp4) · [T2 E0](videos/test2/handover_mic/episode0_subtask.mp4) · [T2 E1](videos/test2/handover_mic/episode1_subtask.mp4) · [T2 E2](videos/test2/handover_mic/episode2_subtask.mp4) · [T2 E3](videos/test2/handover_mic/episode3_subtask.mp4) · [T2 E4](videos/test2/handover_mic/episode4_subtask.mp4) |
| Stack two blocks | `there are two blocks on the table, the color of the blocks is <red, green>, <move the blocks to the center of the table>, and <stack the geen block on the red block>` | [T1 E0](videos/test1/stack_blocks_two/episode0_subtask.mp4) · [T1 E1](videos/test1/stack_blocks_two/episode1_subtask.mp4) · [T1 E2](videos/test1/stack_blocks_two/episode2_subtask.mp4) · [T1 E3](videos/test1/stack_blocks_two/episode3_subtask.mp4) · [T1 E4](videos/test1/stack_blocks_two/episode4_subtask.mp4) · [T2 E0](videos/test2/stack_blocks_two/episode0_subtask.mp4) · [T2 E1](videos/test2/stack_blocks_two/episode1_subtask.mp4) · [T2 E2](videos/test2/stack_blocks_two/episode2_subtask.mp4) · [T2 E3](videos/test2/stack_blocks_two/episode3_subtask.mp4) · [T2 E4](videos/test2/stack_blocks_two/episode4_subtask.mp4) |
| Arrange RGB blocks | `<Place> the red block, green block, and blue block <in the order> of red, green, and blue from left to right, <placing in a row>.` | [T1 E0](videos/test1/blocks_ranking_rgb/episode0_subtask.mp4) · [T1 E1](videos/test1/blocks_ranking_rgb/episode1_subtask.mp4) · [T1 E2](videos/test1/blocks_ranking_rgb/episode2_subtask.mp4) · [T1 E3](videos/test1/blocks_ranking_rgb/episode3_subtask.mp4) · [T1 E4](videos/test1/blocks_ranking_rgb/episode4_subtask.mp4) · [T2 E0](videos/test2/blocks_ranking_rgb/episode0_subtask.mp4) · [T2 E1](videos/test2/blocks_ranking_rgb/episode1_subtask.mp4) · [T2 E2](videos/test2/blocks_ranking_rgb/episode2_subtask.mp4) · [T2 E3](videos/test2/blocks_ranking_rgb/episode3_subtask.mp4) · [T2 E4](videos/test2/blocks_ranking_rgb/episode4_subtask.mp4) |
| Place burger and fries | `Use dual arm to pick the hamburg and frenchfries and put them onto the tray.` | [T1 E0](videos/test1/place_burger_fries/episode0_subtask.mp4) · [T1 E1](videos/test1/place_burger_fries/episode1_subtask.mp4) · [T1 E2](videos/test1/place_burger_fries/episode2_subtask.mp4) · [T1 E3](videos/test1/place_burger_fries/episode3_subtask.mp4) · [T1 E4](videos/test1/place_burger_fries/episode4_subtask.mp4) · [T2 E0](videos/test2/place_burger_fries/episode0_subtask.mp4) · [T2 E1](videos/test2/place_burger_fries/episode1_subtask.mp4) · [T2 E2](videos/test2/place_burger_fries/episode2_subtask.mp4) · [T2 E3](videos/test2/place_burger_fries/episode3_subtask.mp4) · [T2 E4](videos/test2/place_burger_fries/episode4_subtask.mp4) |
| Stack three blocks | `there are three blocks on the table, the color of the blocks is <red, green and blue>, <move the blocks to the center of the table>, and <stack the blue block on the green block, and the green block on the red block>` | [T1 E0](videos/test1/stack_blocks_three/episode0_subtask.mp4) · [T1 E1](videos/test1/stack_blocks_three/episode1_subtask.mp4) · [T1 E2](videos/test1/stack_blocks_three/episode2_subtask.mp4) · [T1 E3](videos/test1/stack_blocks_three/episode3_subtask.mp4) · [T1 E4](videos/test1/stack_blocks_three/episode4_subtask.mp4) · [T2 E0](videos/test2/stack_blocks_three/episode0_subtask.mp4) · [T2 E1](videos/test2/stack_blocks_three/episode1_subtask.mp4) · [T2 E2](videos/test2/stack_blocks_three/episode2_subtask.mp4) · [T2 E3](videos/test2/stack_blocks_three/episode3_subtask.mp4) · [T2 E4](videos/test2/stack_blocks_three/episode4_subtask.mp4) |

## Reproduction commands

Run from `high_level_memory/Hy-Embodied-0.5-VLA`. The Gemma service must already be available on port 8003.

```bash
# Test 1: original planner prompt; calibrates task-specific boundaries
TEST_NUM=5 bash scripts/run_gemma_fixed_next_12tasks_n5.sh

# Test 2: one-object-only planner prompt; reuses Test 1 calibration
TEST_NUM=5 bash scripts/run_gemma_fixed_next_test2.sh
```

The detailed protocol and the local selectable viewer are in [TEST1_TEST2_REPORT.md](TEST1_TEST2_REPORT.md) and [index.html](index.html).
