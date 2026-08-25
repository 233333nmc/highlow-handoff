pgrep: uptime
# LingBot-VLA 2.0 Subtask-Following Diagnostics

This folder combines the previous **12-task LingBot-VLA 2.0 fixed-boundary evaluation** with **15 focused hanging-mug videos**. The goal is to separate two questions:

1. Does the high-level controller publish the intended subtask and clear the old action cache?
2. Does LingBot-VLA strictly stop at the subtask boundary and follow only the newly published instruction?

There are 75 annotated MP4 files in this package: 60 from the 12-task LingBot set and 15 from three five-episode mug diagnostics.


The MP4 files are also linked below. GitHub may offer them as downloads rather than inline playback, especially for a private repository. The links still preserve the exact video files and can be opened after cloning the branch.

## What was tested

- High-level model: Gemma 4 12B instruction model
- Low-level model: LingBot-VLA 2.0
- High-level mode: vlm_plan_next
- Handoff trigger: task-specific fixed boundary
- High-level execution: blocking (HL_ASYNC_ENABLED=False)
- Visual context: unseen-memory images sampled at 1 Hz
- Completion output: {"completed": true} or {"completed": false}
- Planner constraint: one object only, one target, no compound action or list
- Low-level action smoothing: qpos smoothing with alpha=0.25
- Samples: five episodes for the 11 non-burger tasks
- Burger task: one real rollout is available in this folder; the five viewer slots are copies of that one annotated rollout so that all task cards have the same layout. They are **not five independent samples**.

## Control logic

The controller separates semantic instruction generation from visual completion detection:

1. The planner generates one object-level subtask and sends it to LingBot-VLA.
2. LingBot-VLA executes that instruction and produces action chunks.
3. At the task's fixed boundary, low-level execution is inspected and the high-level verifier receives the accumulated 1 Hz camera evidence.
4. The verifier returns only completed=true or completed=false.
5. completed=true publishes the prefetched next subtask and clears action chunks associated with the old instruction.
6. completed=false keeps the current instruction and its action cache. The low-level policy continues to run.

The final simulator result is a **task-level** result. It does not directly label every object-level subtask. Therefore, a task can succeed even when the verifier failed to switch: LingBot-VLA may continue the learned behavior under the old instruction, action chunks may still be available, and the environment may accept the resulting final state. Conversely, a no-switch episode can visibly repeat the same motion and fail.

## Main observation: hanging mug, episode 1

This is the clearest example of the completion gate affecting behavior:

- Original task: “Pick the ceramic mug from the table, rotate it, place it in the middle, and hang it onto the medium-sized rack.”
- Active instruction remained: Lift the cylindrical drinking mug black color from the table.
- At every logged boundary, Gemma returned completed=false.
- The controller therefore never published the hanging-placement subtask and retained the current instruction/action state.
- The video shows the arm repeatedly attempting the mug pickup instead of progressing to the rack-placement instruction.

**Video:** [hanging_mug episode 1](./videos/hanging_mug/episode1_subtask.mp4)

This episode demonstrates that the high-level completion decision is not cosmetic: when it does not switch, the low-level policy is repeatedly driven by the old subtask.

## Targeted search for a switch-then-success case

The first five-episode mug batch did not contain the desired positive example:

- Episode 0: Gemma switched, but the task failed.
- Episode 1: Gemma never switched; the arm repeatedly attempted the pickup and the task failed.
- Episode 2: Gemma never switched and the task failed.
- Episode 3: Gemma never switched, but the task succeeded. This is a low-level completion without a high-level transition, not evidence that switching helped.
- Episode 4: Gemma switched, but the task failed.

I also started a targeted rerun with a new seed. The run using a 1.74-second calibrated boundary was stopped after two successful no-switch episodes and a third long-tail episode. A second run using the previously validated 2.70-second mug boundary produced one failed no-switch episode before being stopped. Neither partial rerun has yet produced a switch-then-success case.

This is an important negative result: the current data demonstrate both “switch + failure” and “no switch + success,” but not yet “switch + success.” A clean positive example requires an episode whose initial planner output separates the pickup stage from the rack-placement stage, followed by a correct completion decision at the boundary and a successful final simulator result.

### Delayed wrong-instruction diagnostic

To isolate whether a later high-level publication can rescue low-level execution, an opt-in diagnostic mode was added. During the first 5.4 simulated seconds, the low-level policy receives the deliberately wrong instruction "Put one bottle into the dustbin." The high-level controller only records observations during this interval. At 5.4 seconds it is released and generates the real mug subtask, after which the normal fixed-boundary verifier logic resumes.

A one-episode smoke test confirmed the intended transition:

1. Low-level trace before 5.4 seconds: "Put one bottle into the dustbin."
2. At frame 135 (5.4 seconds): high-level planner publishes "Lift the beige mug from the table".
3. Subsequent verifier/switch behavior remains controlled by the normal completion protocol.

## Why many other episodes still succeed without a switch

The opposite pattern also appears in this batch. In several episodes the last recorded verifier action is continue, yet the simulator reports task success. This does not mean the VLM correctly recognized completion. It means the low-level policy and task environment can sometimes finish the remaining behavior without receiving a new high-level instruction. Possible contributors include:

- the low-level policy learned a broad behavior from the original task context;
- the current instruction is sufficient to induce more than one physical motion;
- previously generated action chunks continue to be consumed;
- the simulator checks the final scene state rather than the semantic boundary between subtasks.

The report should therefore be read as a diagnostic of **high-level switching versus low-level completion**, not as a claim that task success proves correct VLM completion detection.

For reference, the following counts use the **last logged boundary of each episode**. “No switch + success” is an observational correlation, not a causal label.

| Task | Episodes | Successful tasks | Last boundary: continue | Continue + successful task |
|---|---:|---:|---:|---:|
| Place cans in plastic box | 5 | 5 | 3 | 3 |
| Stack three bowls | 5 | 5 | 2 | 2 |
| Stack two bowls | 5 | 5 | 2 | 2 |
| Place two shoes | 5 | 5 | 5 | 5 |
| Hang a mug | 5 | 1 | 4 | 1 |
| Put an object in a cabinet | 5 | 5 | 4 | 4 |
| Handover a block | 5 | 5 | 5 | 5 |
| Handover a microphone | 5 | 5 | 5 | 5 |
| Stack two blocks | 5 | 5 | 3 | 3 |
| Arrange RGB blocks | 5 | 4 | 5 | 4 |
| Stack three blocks | 5 | 4 | 2 | 1 |

These counts are intentionally kept separate from subtask-level accuracy: the logs contain repeated boundary checks, while the simulator emits one final task result.

## Original task instructions and videos

Each link points to an annotated MP4 with the generated subtask overlaid on the video timeline.

| Task and original instruction | Episode videos |
|---|---|
| **Place cans in plastic box**<br>“Pick the cylindrical can with smooth surface, set it in the smooth exterior blue plasticbox, then repeat with the red can.” | [ep0](./videos/place_cans_plasticbox/episode0_subtask.mp4) · [ep1](./videos/place_cans_plasticbox/episode1_subtask.mp4) · [ep2](./videos/place_cans_plasticbox/episode2_subtask.mp4) · [ep3](./videos/place_cans_plasticbox/episode3_subtask.mp4) · [ep4](./videos/place_cans_plasticbox/episode4_subtask.mp4) |
| **Stack three bowls**<br>“Start stacking the light blue bowl with darkened rim from the bottom to the top.” | [ep0](./videos/stack_bowls_three/episode0_subtask.mp4) · [ep1](./videos/stack_bowls_three/episode1_subtask.mp4) · [ep2](./videos/stack_bowls_three/episode2_subtask.mp4) · [ep3](./videos/stack_bowls_three/episode3_subtask.mp4) · [ep4](./videos/stack_bowls_three/episode4_subtask.mp4) |
| **Stack two bowls**<br>“Place the small smooth brown-rimmed bowl onto the small smooth brown-rimmed bowl to form a stack.” | [ep0](./videos/stack_bowls_two/episode0_subtask.mp4) · [ep1](./videos/stack_bowls_two/episode1_subtask.mp4) · [ep2](./videos/stack_bowls_two/episode2_subtask.mp4) · [ep3](./videos/stack_bowls_two/episode3_subtask.mp4) · [ep4](./videos/stack_bowls_two/episode4_subtask.mp4) |
| **Place two shoes**<br>“Grab two the white shoe, point tips left, place in the rectangular Nike box for shoes.” | [ep0](./videos/place_dual_shoes/episode0_subtask.mp4) · [ep1](./videos/place_dual_shoes/episode1_subtask.mp4) · [ep2](./videos/place_dual_shoes/episode2_subtask.mp4) · [ep3](./videos/place_dual_shoes/episode3_subtask.mp4) · [ep4](./videos/place_dual_shoes/episode4_subtask.mp4) |
| **Hang a mug**<br>“Pick the ceramic mug from the table, rotate it, place it in the middle, and hang it onto the medium-sized rack.” | [ep0](./videos/hanging_mug/episode0_subtask.mp4) · [ep1](./videos/hanging_mug/episode1_subtask.mp4) · [ep2](./videos/hanging_mug/episode2_subtask.mp4) · [ep3](./videos/hanging_mug/episode3_subtask.mp4) · [ep4](./videos/hanging_mug/episode4_subtask.mp4) |
| **Put an object in a cabinet**<br>“Open the cabinet with smooth finish's drawer and set the hand-sized black-gray mouse inside with the other arm.” | [ep0](./videos/put_object_cabinet/episode0_subtask.mp4) · [ep1](./videos/put_object_cabinet/episode1_subtask.mp4) · [ep2](./videos/put_object_cabinet/episode2_subtask.mp4) · [ep3](./videos/put_object_cabinet/episode3_subtask.mp4) · [ep4](./videos/put_object_cabinet/episode4_subtask.mp4) |
| **Handover a block**<br>“Grab the red block with the left arm, switch to the right arm, and set it on the blue pad.” | [ep0](./videos/handover_block/episode0_subtask.mp4) · [ep1](./videos/handover_block/episode1_subtask.mp4) · [ep2](./videos/handover_block/episode2_subtask.mp4) · [ep3](./videos/handover_block/episode3_subtask.mp4) · [ep4](./videos/handover_block/episode4_subtask.mp4) |
| **Handover a microphone**<br>“Take hold of the long teal microphone with white grooves and hand it over.” | [ep0](./videos/handover_mic/episode0_subtask.mp4) · [ep1](./videos/handover_mic/episode1_subtask.mp4) · [ep2](./videos/handover_mic/episode2_subtask.mp4) · [ep3](./videos/handover_mic/episode3_subtask.mp4) · [ep4](./videos/handover_mic/episode4_subtask.mp4) |
| **Stack two blocks**<br>“Move red block and green block to the table's center, then stack green block on red block.” | [ep0](./videos/stack_blocks_two/episode0_subtask.mp4) · [ep1](./videos/stack_blocks_two/episode1_subtask.mp4) · [ep2](./videos/stack_blocks_two/episode2_subtask.mp4) · [ep3](./videos/stack_blocks_two/episode3_subtask.mp4) · [ep4](./videos/stack_blocks_two/episode4_subtask.mp4) |
| **Arrange RGB blocks**<br>“Place red block on the left, then set green block to its right and blue block next to green block.” | [ep0](./videos/blocks_ranking_rgb/episode0_subtask.mp4) · [ep1](./videos/blocks_ranking_rgb/episode1_subtask.mp4) · [ep2](./videos/blocks_ranking_rgb/episode2_subtask.mp4) · [ep3](./videos/blocks_ranking_rgb/episode3_subtask.mp4) · [ep4](./videos/blocks_ranking_rgb/episode4_subtask.mp4) |
| **Place burger and fries**<br>“Pick the small yellow-marked hamburg wrapper and the Red carton box fries crispy golden, place them on the flat brown serving tray.” | [ep0 real rollout](./videos/place_burger_fries/episode0_subtask.mp4) · [ep1 copy](./videos/place_burger_fries/episode1_subtask.mp4) · [ep2 copy](./videos/place_burger_fries/episode2_subtask.mp4) · [ep3 copy](./videos/place_burger_fries/episode3_subtask.mp4) · [ep4 copy](./videos/place_burger_fries/episode4_subtask.mp4) |
| **Stack three blocks**<br>“Place red block, green block, and blue block at the table's center, then stack blue block on green block and green block on red block.” | [ep0](./videos/stack_blocks_three/episode0_subtask.mp4) · [ep1](./videos/stack_blocks_three/episode1_subtask.mp4) · [ep2](./videos/stack_blocks_three/episode2_subtask.mp4) · [ep3](./videos/stack_blocks_three/episode3_subtask.mp4) · [ep4](./videos/stack_blocks_three/episode4_subtask.mp4) |

## Interpretation

The videos answer two separate questions:

1. **Did Gemma publish the next instruction?** This is controlled by the completion verifier and is visible in the overlaid subtask changes.
2. **Could LingBot-VLA finish the simulator task under the current instruction?** This is reflected by the final simulator result.

These are not the same measurement. The hanging-mug episode 1 is a useful failure case because the first question is clearly negative and the low-level behavior visibly loops. The other successful no-switch episodes are useful counterexamples: they show that a capable low-level policy can sometimes complete the task without a correct high-level transition. A future evaluation should add per-subtask simulator state labels or an intervention that prevents the low-level policy from receiving later-task context if we want to attribute failures more cleanly.

## Reproduction context

The videos were generated from the fixed-boundary Gemma controller with LingBot-VLA 2.0 as the low-level server. The original experiment roots, logs, and per-episode traces remain outside this presentation folder under:

profile_results/lingbot_smooth025_burger_20260821/

profile_results/lingbot_smooth025_other11_n5_20260821/

profile_results/lingbot_smooth025_last3_n5_20260822/

This README is a presentation and interpretation layer; it does not change the raw rollout logs.

## Focused Mug Diagnostics

The 15 mug videos are deliberately separated from the 12-task benchmark. They use the same LingBot-VLA 2.0 low-level server and the same RoboTwin `hanging_mug` task, but change how the subtask is published.

### Run 1: Delayed wrong-instruction diagnostic

For the first 5.4 simulated seconds, the low-level policy receives the deliberately wrong instruction:

```text
Put one bottle into the dustbin.
```

After 5.4 seconds, the normal Gemma fixed-boundary `vlm_plan_next` path is released. The high-level model can inspect images, decide `continue` or `switch`, and publish the next subtask. This run tests whether a later high-level handoff can recover from an already corrupted physical state.

- [Episode 0](./videos/mug_delayed_wrong_release_retry2/episode0_subtask.mp4)
- [Episode 1](./videos/mug_delayed_wrong_release_retry2/episode1_subtask.mp4)
- [Episode 2](./videos/mug_delayed_wrong_release_retry2/episode2_subtask.mp4)
- [Episode 3](./videos/mug_delayed_wrong_release_retry2/episode3_subtask.mp4)
- [Episode 4](./videos/mug_delayed_wrong_release_retry2/episode4_subtask.mp4)

### Run 2: Fixed-time handoff after a wrong prefix

This is a low-level control experiment. Gemma is disabled completely. The schedule is:

```text
0.0 s: Put one bottle into the dustbin.
5.4 s: Place the mug at the middle handover position.
```

At 5.4 seconds the controller clears the old action cache and LingBot-VLA receives a fresh low-level request. The run isolates whether a known correct instruction can recover the robot after an incorrect prefix.

- [Episode 0](./videos/mug_fixed_subtask_control/episode0_subtask.mp4)
- [Episode 1](./videos/mug_fixed_subtask_control/episode1_subtask.mp4)
- [Episode 2](./videos/mug_fixed_subtask_control/episode2_subtask.mp4)
- [Episode 3](./videos/mug_fixed_subtask_control/episode3_subtask.mp4)
- [Episode 4](./videos/mug_fixed_subtask_control/episode4_subtask.mp4)

### Run 3: Clean first-subtask control

The same first-stage instruction is provided from the beginning, with no wrong prefix and no VLM:

```text
Place the mug at the middle handover position.
```

This is the clean baseline for comparing basic low-level execution against recovery after an error.

- [Episode 0](./videos/mug_clean_first_subtask/episode0_subtask.mp4)
- [Episode 1](./videos/mug_clean_first_subtask/episode1_subtask.mp4)
- [Episode 2](./videos/mug_clean_first_subtask/episode2_subtask.mp4)
- [Episode 3](./videos/mug_clean_first_subtask/episode3_subtask.mp4)
- [Episode 4](./videos/mug_clean_first_subtask/episode4_subtask.mp4)

## What The 15 Mug Videos Show

The fixed-time handoff logs show that the software handoff itself works: at 5.4 seconds the old action chunk is discarded and LingBot-VLA performs a new inference under the mug instruction. However, the videos do not show strict subtask termination.

In the clean baseline, the same first-stage instruction remains active for the entire rollout. One episode completes the full hanging task even though no second subtask is ever published; the other episodes partially execute, overshoot, or drop the mug. This means the low-level policy is responsive to the instruction but does not expose a hard `done` or `hold` state at the subtask endpoint.

The wrong-prefix run adds a second difficulty: the new trajectory starts from a physically disturbed state. A correct text instruction and a fresh action rollout therefore do not guarantee recovery. These videos should be interpreted as evidence that **subtask publication and low-level subtask termination are separate capabilities**.

The 12-task LingBot videos provide the broader context. They show many cases where the simulator succeeds even when high-level switching is absent, because the low-level policy can sometimes complete a broad learned behavior under the current instruction. They also show cases where a missing switch produces visible repetition. Final task success alone is therefore not a valid substitute for subtask-level completion labels.

## Suggested Reading Order

1. Watch the five episodes under **Run 3: Clean first-subtask control**.
2. Watch the corresponding five episodes under **Run 2: Fixed-time handoff after a wrong prefix**.
3. Compare them with **Run 1: Delayed wrong-instruction diagnostic**, where Gemma decides whether to switch.
4. Use the 12-task links above to see how the same issue appears across different objects and manipulation skills.
