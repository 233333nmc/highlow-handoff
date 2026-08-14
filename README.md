# RoboTwin High-Level / Low-Level Handoff Report

Updated: 2026-08-09

**Rendered experiment report:** https://233333nmc.github.io/highlow-handoff/

The report includes the measured interaction latency, exact model inputs and
outputs, verifier evidence frames, and representative rollout videos.

This repository records the RoboTwin high-level / low-level handoff baseline, Gemma4-12B experiments, and the current `next` / `all` planner debugging results.

## 1. Work This Week

This week focused on three items:

- Cleaned up the high-level input / output interface and removed fields that do not affect control.
- Added and tested the `next` / `all` planner paths and the `same` / `switch` verifier path.
- Ran Gemma4-12B forced-boundary, verifier, and planner experiments.

## 2. Current Code Logic

### 2.1 Planner Modes

The current controller supports these planner modes:

- `inline_vlm`: the VLM directly returns the next subtask.
- `vlm_plan_all`: the VLM generates the full subtask list at the beginning of the episode.
- `vlm_plan_next`: after the current subtask is published to the low-level policy, the controller immediately starts generating and caching the next candidate subtask in the background. It does not wait until the verifier confirms completion. For example, after subtask 1 is published, subtask 2 is prefetched; after subtask 2 is published, subtask 3 is prefetched. The verifier only decides when to consume the cached candidate. If the candidate is not ready, the controller falls back to synchronous generation.
- `profile_all` / `lazy_profile`: deterministic debugging modes.

### 2.2 Verifier / Output Schema

The current high-level output schemas include:

- `minimal`
- `minimal_status`
- `completion_status`

`completion_status` is used for `same` / `switch` decisions. `minimal_status` is the older minimal control interface.

The main verifier output is:

```json
{
  "decision": "same | switch",
  "status": "running | completed | failed | stuck | uncertain",
  "next_subtask": "",
  "evidence": "short visual reason"
}
```

Control behavior:

- `confidence` is logging-only and does not affect control.
- `switch` clears the old instruction's action cache.
- `same` / `running` / `uncertain` keeps the current action cache.

### 2.3 Visual Context

The default Gemma4 forced-boundary verifier configuration uses
`unseen_memory` visual context:

```text
HL_FORCE_VISUAL_ON_DECISION=True
HL_HISTORY_FPS=1.0
HL_VLM_MAX_IMAGES=8
HL_VISUAL_CONTEXT_MODE=unseen_memory
GEMMA4_IMAGE_SELECTION=tail
```

In this mode, the controller stores one low-frequency head-camera frame every
1 second. At a handoff boundary, the verifier receives all buffered frames
that have not yet been sent to the high-level model plus the current boundary
observation. After the verifier call, the image payloads for those frames are
discarded from the buffer.

The code also supports other visual-context settings:

- `HL_RECENT_FRAMES_LENGTH>1`: use the tail of recent visual frames.
- `HL_VISUAL_CONTEXT_MODE=recent`: use the sampled recent visual frames.
- `HL_VISUAL_CONTEXT_MODE=unseen_memory`: use accumulated unseen visual frames.

The latest code forces the current boundary frame into verifier requests, so a fixed-boundary check does not accidentally reuse the previous periodic frame.

- The current control path no longer depends on keyframes.
- History-based keyframe selection is disabled by default.
- The default `next` / `all` logic is not affected by keyframe code.

### 2.4 `next` / `all` Logic

`vlm_plan_next`:

- After the current subtask is published to the low-level policy, the controller immediately prefetches the next candidate subtask in the background.
- For example, after subtask 1 is published, it prefetches subtask 2. Once subtask 2 is published, it prefetches subtask 3.
- At the boundary, the verifier only decides whether the current subtask is completed.
- If completed and the prefetched candidate is ready, the controller publishes the candidate directly.
- If the candidate is not ready, the controller falls back to synchronous generation.

`vlm_plan_all`:

- The VLM generates the full subtask list at the beginning of the episode.
- At the boundary, the controller only runs the completion verifier.
- If the verifier says the current subtask is completed, the controller consumes the next subtask from the cached list.
- If the verifier says the current subtask is not completed, the controller keeps the current subtask unchanged.

### 2.5 Planner Prompt Granularity

The Gemma4 planner still needs explicit prompt constraints to control subtask granularity. The desired output is object-level phases, not motor primitives such as grasp / lift / move / release.

Later text-only tests corrected an earlier assumption. For `stack_blocks_three`, generic "do not be too fine-grained" constraints were not enough. Gemma4 often followed the literal order of the original instruction and returned:

```text
Move red block to the center
Move green block to the center
Move blue block to the center
Stack blue block on green block
Stack green block on red block
```

The final spatial relation is correct, but the ordered handoff sequence is not. It asks for `blue on green` before `green on red`, so it is not an executable dependency order.

`place_burger_fries` showed a related granularity problem. During one `vlm_plan_all` step-by-step planning debug run, Gemma4 returned:

```text
step 0: pick up the hamburger with the left arm
step 1: place the hamburger on the tray
```

This is not the object-level phase that we want to send to Hy-VLA. The more appropriate granularity is:

```text
step 0: Place the hamburg on the tray.
step 1: Place the french fries on the tray.
```

This example shows that even when the final task semantics are simple, Gemma4 may split one object-level phase into `pick up` / `place` substeps, which does not match the desired low-level instruction granularity.

With task profile hints, five text-only `stack_blocks_three` trials returned the desired output consistently:

```json
{
  "subtasks": [
    "Place the red block at the center.",
    "Stack the green block on the red block.",
    "Stack the blue block on the green block."
  ]
}
```

The current code therefore keeps `HL_INCLUDE_PHASE_PROFILE_HINTS` and `HL_PHASE_PROMPTS_JSON` as debugging / benchmark stabilization hooks. They should not be presented as a final-method dependency. The planner prompt still needs more work.

## 3. Gemma4-12B Results

### 3.1 `place_burger_fries` Completion Verifier

- formal episodes: `n=10`
- `boundary_to_instruction_ready` mean: `632.4 ms`
- median: `631.9 ms`
- min / max: `617.5 / 648.4 ms`
- This measures the single-image completion verifier latency. It does not include the following low-level recovery / action forward.

### 3.2 `place_burger_fries` `next` / `all` Comparison

Sections 3.1 and 3.2 both use `place_burger_fries`, but they are not the same code path:

- 3.1 measures the completion verifier itself. At the boundary, Gemma4 looks at the image and decides whether the current subtask is completed.
- 3.2 compares planner strategies. `next` and `all` differ in when the next subtask candidate is generated. At the boundary, the controller mainly runs the verifier and consumes a planner candidate.
- `vlm_plan_next` generates the next candidate after the current subtask is published, so the boundary may avoid planner waiting time.
- `vlm_plan_all` generates the full list at episode start, so the boundary consumes the cached next item.

Therefore, the latency values in 3.1 and 3.2 should not be treated as repeated measurements of the same operation. They answer different questions.

Intuition:

```text
3.1:
  After the boundary, Gemma4 starts looking at the image and verifies whether
  the current subtask is completed.
  This mainly reflects single-frame completion verifier cost.

3.2:
  The next subtask candidate has already been generated or cached by the planner.
  After the boundary, the system mainly runs the verifier and, if needed,
  consumes the prepared candidate.
  Therefore boundary-to-instruction-ready can be shorter.
```

So the lower 3.2 latency does not mean that the same verifier became faster. It means that after the planner and verifier were separated, part of the planning cost moved before the boundary. 3.2 still requires checking whether the candidate subtask is correct, whether the verifier decision is correct, and whether the low-level policy is actually controlled by the new subtask.

Results:

- `vlm_plan_next_prefetch`: mean `424.3 ms`, median `422.7 ms`, min / max `410.8 / 450.8 ms`
- `vlm_plan_all`: mean `446.4 ms`, median `444.8 ms`, min / max `428.7 / 472.4 ms`
- `vlm_plan_all_repeated_next`: mean `441.6 ms`, median `431.7 ms`, min / max `424.4 / 470.0 ms`

Takeaway:

- `next` is better suited for hiding next-instruction planning latency behind low-level execution.
- `all` is usable, but depends more strongly on the stability of the full generated subtask list.

### 3.3 `stack_blocks_three` Single-Frame Verifier Issue

To test fixed-time triggering followed by one-image high-level verification, `stack_blocks_three` was configured as a two-subtask task:

- subtask 1: `Place the red block at the center.`
- subtask 2: `Stack the green block on the red block, then stack the blue block on the green block.`

This experiment already fixed two implementation issues:

- the forced-boundary verifier now uses the current boundary head-camera frame instead of the previous periodic frame;
- the `completion_status` prompt explicitly asks Gemma4 to return JSON, avoiding parser fallback to `unknown` when the model responds in natural language.

However, single-frame completion detection is still unstable:

| Fixed boundary | Context frame | JSON parse | VLM decision | Simulator oracle | Result |
|---:|---|---:|---|---|---|
| `3.00s` | frame `75` | `2/2` | `continue/running` `2/2` | first phase completed `2/2` | false negative |
| `3.50s` | frame `88` | `2/2` | `switch` `1/2`, `continue` `1/2` | not stable against oracle | unstable |
| `4.00s` | frame `100` | `1/2` | `continue/unknown` `2/2` | not stable against oracle | unstable |

Typical false negative:

```text
Simulator oracle: red block phase is completed.
Gemma4 verifier: same / running
Evidence: the robot is currently interacting with the green block, not the red block.
```

Takeaway:

- The fixed-boundary handoff mechanism works.
- The current boundary frame is correctly sent to the high-level VLM.
- The switch / continue action-cache logic is logged correctly.
- The remaining issue is that Gemma4 cannot reliably determine `stack_blocks_three` subtask completion from a single current image.

Therefore, `stack_blocks_three` should not currently use a single-frame VLM verifier as the stable subtask boundary detector. More promising next steps are multi-frame visual context or a dedicated completion-detection benchmark for verifier prompt / model tuning.

### 3.4 Low-Level Instruction Sensitivity

We also ran instruction probes to check whether the Hy-VLA low-level policy is actually controlled by high-level subtask instructions. The probe disables the high-level controller and only changes the low-level instruction:

- full task instruction;
- correct first-subtask instruction;
- wrong subtask instruction;
- empty instruction.

Each run records the actual `low_level_instruction` received by the low-level policy and saves rollout video / trace when enabled.

Current observations:

| Task | Probe variants | Observation |
|---|---|---|
| `place_burger_fries` | full / correct subtask / wrong subtask / empty | All four variants can complete the full burger + fries task. This suggests that, for this task, the low-level policy may rely strongly on vision and task priors. Even a partial, wrong, or empty instruction may still lead to full-task completion. |
| `stack_blocks_three` | full / correct subtask / wrong subtask / empty | In a single-run probe, full and correct-subtask instructions succeeded, while wrong / empty instructions failed. This task is more useful than burger/fries for observing whether the instruction changes low-level behavior, but it still needs more samples. |
| `put_bottles_dustbin` | full / correct subtask | The full task can run, but the probe did not fully cover wrong / empty variants, so this is not a main conclusion yet. |
| `hanging_mug` | full / correct subtask | The task itself has low success rate, so the current probe is not suitable for judging language sensitivity. |

Interpretation:

- Current evidence suggests that, without additional training, we may not be able to strictly guarantee that Hy-VLA follows the high-level subtask instruction. In tasks such as `place_burger_fries`, the low-level policy may keep completing the full task even when the high-level system does not publish the full task decomposition.
- This is not a final conclusion. Possible explanations include:
  - the Hy-VLA checkpoint may mainly be trained on full-task instructions, so subtask instructions are not hard constraints;
  - subtask-level fine-tuning / LoRA may be needed for stricter obedience to high-level phase instructions;
  - our understanding of Hy-VLA language conditioning may still be incomplete;
  - there may still be a code-path issue, such as the full instruction influencing the low-level policy somewhere, or the action-cache / trace logging not fully covering the true model input.
- We have checked the instruction trace and confirmed the recorded `low_level_instruction`, but that alone is not a final proof. To demonstrate that the high-level controller is effective, final success is not enough. We also need to show the low-level instruction trace, whether the action cache was cleared, and whether the first real low-level model forward after a switch used the new instruction.
- `stack_blocks_three` is a better candidate for the handoff demo because wrong / empty instructions are more likely to fail, making it easier to observe whether the high-level instruction actually changes low-level behavior.

## 4. Action Cache

The controller records stale action-cache behavior whenever the high-level state is committed.

Current logic:

- `decision=switch` clears the old instruction's action cache.
- `decision=same` / `running` / `uncertain` keeps the current action cache.
- Each high-level commit records these fields in `hierarchical_decisions.jsonl`:
  - `action_cache_len_before_high_level_commit`
  - `action_cache_len_after_high_level_commit`
  - `stale_action_cache_discarded`
  - `action_cache_cleared`
  - `same_decision_kept_action_cache`

The latest `next/all` and `stack_blocks_three` fixed-boundary experiments still need a consolidated action-cache statistics table. Old multi-task results are no longer included in this README.

## 5. Reproduction

The commands below write caches, logs, model files, and evaluation results under the user's personal directory. They do not use system temporary directories. Most experiment files live under `high_level_memory`. The Gemma4 transformers API server uses `rt/g4_transformers` under the personal directory by default; this can be changed with `GEMMA4_TRANSFORMERS_RUNTIME_ROOT`.

### 5.1 Directory Layout

Recommended layout:

```text
<WORK_ROOT>/
├── Hy-Embodied-0.5-VLA/       # this repository
├── RoboTwin/                   # RoboTwin simulator
├── checkpoints/                # low-level / high-level checkpoints
├── envs/                       # unified Python 3.10 evaluation environment
├── profile_results/            # handoff logs and timing results
├── model_cache/                # persistent Hugging Face cache
└── runtime_cache/              # evaluation runtime cache
```

In the commands below, `<WORK_ROOT>` is:

```text
/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
```

If running on another machine, update `WORK_ROOT` accordingly.

The Gemma4 transformers server also has a default runtime cache:

```text
/mnt/cpfs/epic-user/niumuchun-20260612/rt/g4_transformers/
```

This path is controlled by `GEMMA4_TRANSFORMERS_RUNTIME_ROOT` in `scripts/serve_gemma4_12b_transformers_api.sh`.

### 5.2 Get the Code

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT"

git clone -b highlow-handoff-baseline \
  https://github.com/233333nmc/highlow-handoff.git \
  Hy-Embodied-0.5-VLA

git clone https://github.com/robotwin-Platform/RoboTwin.git RoboTwin
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"
```

If the repository already exists:

```bash
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"
git checkout highlow-handoff-baseline
git pull
```

### 5.3 Create the RoboTwin + Hy-VLA Evaluation Environment

The evaluation uses a unified Python 3.10 environment so the RoboTwin simulator, Hy-VLA low-level policy, and high-level handoff controller run in the same process.

Prerequisites:

- Linux;
- available CUDA GPU;
- `/usr/bin/python3.10`;
- `uv`;
- Gemma4 MTP reproduction requires two GPUs visible to the current lease.

If `uv` is not installed, install it for the current environment first. Then run:

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"

bash scripts/install_robotwin_hyvla_eval_env.sh
```

This creates:

```text
$WORK_ROOT/envs/robotwin_hyvla_eval/
```

The script installs CUDA PyTorch, Hy-VLA inference dependencies, RoboTwin simulator dependencies, and cuRobo. It also points `TMPDIR`, Hugging Face cache, uv cache, and matplotlib cache to the personal directory.

### 5.4 Download RoboTwin Simulation Assets

These experiments run RoboTwin inference/evaluation. They do not require RoboTwin training demonstrations. They do require simulator assets such as backgrounds, objects, and embodiment files.

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"

bash scripts/download_robotwin_assets_persistent.sh
```

Assets are saved to:

```text
$WORK_ROOT/RoboTwin/assets/
```

### 5.5 Download the Low-Level Checkpoint

The Hy-VLA low-level checkpoint is:

```text
tencent/Hy-Embodied-0.5-VLA-RoboTwin
```

Use a persistent Hugging Face cache and save the model under the shared checkpoint directory:

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
export HF_HOME="$WORK_ROOT/model_cache/huggingface"
export HF_HUB_CACHE="$WORK_ROOT/model_cache/huggingface/hub"
export TMPDIR="$WORK_ROOT/.cache/tmp"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TMPDIR"

hf download tencent/Hy-Embodied-0.5-VLA-RoboTwin \
  --local-dir "$WORK_ROOT/checkpoints/Hy-Embodied-0.5-VLA-RoboTwin"
```

Expected final directory:

```text
$WORK_ROOT/checkpoints/Hy-Embodied-0.5-VLA-RoboTwin/
```

### 5.6 Download Gemma4-12B

The main Gemma4 model is:

```text
google/gemma-4-12B-it
```

The model terms may require accepting the license on Hugging Face and setting `HF_TOKEN` in the current shell.

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"

# If Hugging Face authentication is required:
# export HF_TOKEN=...

bash scripts/download_gemma4_12b.sh
```

The main model is saved to:

```text
$WORK_ROOT/checkpoints/google/gemma-4-12B-it/
```

Native MTP also needs a compatible assistant checkpoint. This checkpoint is not generated by `download_gemma4_12b.sh`; prepare it separately and pass it to the server as:

```text
$WORK_ROOT/checkpoints/google/gemma-4-12B-it-assistant/
```

If `GEMMA4_ASSISTANT_MODEL_PATH` is not provided, the server can still run normal Gemma4 inference, but it will not reproduce the native MTP setup used here.

### 5.7 Start the Gemma4 API Server

The high-level controller calls Gemma4 through a local OpenAI-compatible HTTP API. Keep this server running in one terminal, and run RoboTwin evaluation from another terminal.

`GEMMA4_TRANSFORMERS_PATH` must point to a Transformers version that supports `gemma4_unified`. If the compatible version is already in the script's default cache, it can be omitted; otherwise set it explicitly.

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"

GEMMA4_PORT=8003 \
GEMMA4_MODEL_PATH="$WORK_ROOT/checkpoints/google/gemma-4-12B-it" \
GEMMA4_ASSISTANT_MODEL_PATH="$WORK_ROOT/checkpoints/google/gemma-4-12B-it-assistant" \
GEMMA4_DEVICE=cuda:1 \
GEMMA4_ASSISTANT_DEVICE=cuda:0 \
GEMMA4_NUM_ASSISTANT_TOKENS=4 \
GEMMA4_NUM_ASSISTANT_TOKENS_SCHEDULE=heuristic \
bash scripts/serve_gemma4_12b_transformers_api.sh
```

`cuda:0` and `cuda:1` are relative to the GPUs visible in the current lease. Do not manually override `CUDA_VISIBLE_DEVICES`. If only one GPU is visible, use normal inference without the assistant checkpoint, or request two visible GPUs.

To override the default runtime cache:

```bash
GEMMA4_TRANSFORMERS_RUNTIME_ROOT="$WORK_ROOT/runtime_cache/gemma4_transformers"
```

### 5.8 Run `vlm_plan_next`

`vlm_plan_next` publishes the current instruction, immediately prefetches the next candidate subtask, and later lets the boundary verifier decide whether to consume that candidate.

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"

LOG_DIR="$WORK_ROOT/profile_results/run_vlm_plan_next_$(date +%Y%m%d_%H%M%S)" \
TEST_NUM=11 \
SEED=10000 \
TASK_NAME=place_burger_fries \
FORCED_BOUNDARY_S=2.17 \
MAX_FORCED_BOUNDARIES=1 \
STOP_AFTER_MEASUREMENT_HANDOFF=False \
HL_PLANNER_MODE=vlm_plan_next \
HL_PROMPT_MODE=completion_status \
HL_FORCE_VISUAL_ON_DECISION=True \
HL_RECENT_FRAMES_LENGTH=1 \
HL_VLM_MAX_IMAGES=1 \
GEMMA4_SERVER_URL=http://127.0.0.1:8003 \
bash scripts/eval_robotwin_burger_fries_forced_gemma4_12b_status.sh
```

### 5.9 Run `vlm_plan_all`

`vlm_plan_all` generates the full subtask list at episode start. At the boundary, the controller only runs the completion verifier and consumes the next cached subtask if the current one is completed.

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"

LOG_DIR="$WORK_ROOT/profile_results/run_vlm_plan_all_$(date +%Y%m%d_%H%M%S)" \
TEST_NUM=11 \
SEED=10000 \
TASK_NAME=place_burger_fries \
FORCED_BOUNDARY_S=2.17 \
MAX_FORCED_BOUNDARIES=1 \
STOP_AFTER_MEASUREMENT_HANDOFF=False \
HL_PLANNER_MODE=vlm_plan_all \
HL_PROMPT_MODE=completion_status \
HL_FORCE_VISUAL_ON_DECISION=True \
HL_RECENT_FRAMES_LENGTH=1 \
HL_VLM_MAX_IMAGES=1 \
GEMMA4_SERVER_URL=http://127.0.0.1:8003 \
bash scripts/eval_robotwin_burger_fries_forced_gemma4_12b_status.sh
```

The two commands use the same RoboTwin task, 2.17s forced boundary, one current head-camera image, and completion verifier. They differ only in when the next subtask is planned.

### 5.10 Run the Two-Subtask `stack_blocks_three` Fixed-Boundary Verifier

This command reproduces the issue in Section 3.3: at a fixed time, the controller sends the current head-camera frame to Gemma4 and asks whether the first subtask is completed.

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"

PHASES='Place the red block at the center.|||Stack the green block on the red block, then stack the blue block on the green block.'

LOG_DIR="$WORK_ROOT/profile_results/run_stack_blocks_fixed_verifier_$(date +%Y%m%d_%H%M%S)" \
TEST_NUM=2 \
SEED=10000 \
TASK_NAME=stack_blocks_three \
HANDOFF_TRIGGER=forced_boundary \
FORCED_BOUNDARY_S=3.00 \
MAX_FORCED_BOUNDARIES=1 \
HL_BACKEND=gemma4_vllm \
HL_PLANNER_MODE=profile_all \
HL_PROMPT_MODE=completion_status \
HL_PHASE_PROMPTS_JSON="$PHASES" \
HL_INCLUDE_PHASE_PROFILE_HINTS=True \
HL_FORCE_VISUAL_ON_DECISION=True \
HL_RECENT_FRAMES_LENGTH=1 \
HL_VLM_MAX_IMAGES=1 \
GEMMA4_SERVER_URL=http://127.0.0.1:8003 \
ROBOTWIN_EVAL_VIDEO_LOG=True \
LOW_LEVEL_TRACE_EVERY_STEP=True \
bash scripts/eval_robotwin_burger_fries_forced_gemma4_12b_status.sh
```

Important logs:

```text
hierarchical_decisions.jsonl      # high-level prompt / response / same-switch / timing
low_level_instruction_trace.jsonl # low-level instruction received by real model forward
eval_results/.../*.mp4            # rollout video, if ROBOTWIN_EVAL_VIDEO_LOG=True
```

### 5.11 Results and Log Locations

Each evaluation writes these files under `LOG_DIR`:

```text
hierarchical.log
hierarchical_decisions.jsonl
```

RoboTwin episode results are written to:

```text
$WORK_ROOT/RoboTwin/eval_result/
```

If `ROBOTWIN_EVAL_RESULT_ROOT` is set, episode results are written there instead.

The default scripts run 11 rollouts. Episode 0 is treated as an extra runtime warm-up; formal latency statistics use episodes 1-10.

### 5.12 Optional: Multi-Task Run

To re-run Gemma4 on six RoboTwin tasks, use the multi-task script and set `HL_PLANNER_MODE` to `vlm_plan_next` or `vlm_plan_all`. This runs the current `completion_status` / single-frame verifier path. Results should be summarized separately before being included in a report.

```bash
export WORK_ROOT=/mnt/cpfs/epic-user/niumuchun-20260612/high_level_memory
cd "$WORK_ROOT/Hy-Embodied-0.5-VLA"

OUT_ROOT="$WORK_ROOT/profile_results/run_multitask_gemma4_$(date +%Y%m%d_%H%M%S)" \
TEST_NUM=11 \
SEED=10000 \
HL_PLANNER_MODE=vlm_plan_next \
GEMMA4_SERVER_URL=http://127.0.0.1:8003 \
bash scripts/eval_robotwin_multitask_gemma4_12b_status_n10.sh
```

## 6. Ongoing Work

- Consolidate action-cache statistics for the latest `next/all`, fixed-boundary, and `stack_blocks_three` experiments.
- Improve verifier completion criteria, especially the `stack_blocks_three` single-frame false negatives.
- Test multi-frame verifier settings such as `HL_RECENT_FRAMES_LENGTH=3` or `HL_VISUAL_CONTEXT_MODE=unseen_memory`.
- Continue tuning planner prompts so the model outputs object-level, executable-order subtasks instead of overly fine-grained or dependency-invalid instructions.
- Build a formal demo package with rollout video, high-level prompt / response, low-level instruction trace, and action-cache switch records.

## 7. Notes

- This README only covers the RoboTwin handoff line of work.
- It no longer includes the original Hy-VLA paper-style introduction or original benchmark results.
- Current results include latency, planner/debug behavior, and some full-task success observations. Full-task success alone does not prove that the high-level subtask causally controlled the low-level policy, because Hy-VLA may have strong visual / task priors in some tasks.
- The next focus is async execution, same/switch verification, completion detection, and demo visualization.
