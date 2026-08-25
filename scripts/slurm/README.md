# Draco dual-system Slurm launcher

`draco_dual_system.sbatch` runs the decoupled high/low policy stack on one
Draco compute node. It never starts model or simulator processes on a login
node.

The four-process topology is:

1. SGLang serving the high-level Qwen2.5-VL checkpoint on the first allocated
   GPU;
2. the CPU-only high-level facade;
3. the stateful HyVLA low-level server on the second allocated GPU;
4. either the CPU-only real HTTP probe, or the official RoboTwin client on the
   third allocated GPU. The third GPU is idle during the fast probe.

All services bind to dynamically selected `127.0.0.1` ports. The selected
URLs, commits, and low-level configuration are recorded in
`$RUN_ROOT/topology.json`.

## Fixed Draco layout

The launcher expects these existing paths below `$STORAGE`:

| Purpose | Path |
| --- | --- |
| Code | `src/highlow-handoff` |
| High-level model | `models/highlow/Qwen2.5-VL-3B-Instruct` |
| Low-level model | `models/highlow/Hy-Embodied-0.5-VLA-RoboTwin` |
| SGLang Python | `miniforge3/envs/highlow-sglang/bin/python` |
| Framework runtime Python | `miniforge3/envs/highlow-runtime/bin/python` |
| HyVLA Python | `venvs/highlow-hyvla/bin/python` |
| RoboTwin Python | `miniforge3/envs/robotwin/bin/python` |
| CUDA compiler toolkit | `tools/cuda-12.8` |
| CUDA forward-compat libraries | `tools/cuda-compat-12.8/cuda-compat` |
| Pinned RoboTwin tree | `runs/highlow-handoff/robotwin_runtime/RoboTwin` |

The prepared Draco environments pin the serving-critical packages as follows:

| Environment | Serving-critical versions |
| --- | --- |
| `highlow-sglang` | Python 3.10, SGLang 0.4.10.post2, PyTorch 2.7.1+cu128 |
| `highlow-runtime` | Python 3.11, highlow-runtime 0.1.0, uvicorn 0.52 |
| `highlow-hyvla` | Python 3.10, PyTorch 2.7.1+cu128, transformers 4.57.0, flash-attn 2.7.4.post1 |
| `robotwin` | Python 3.10, PyTorch 2.4.1+cu121, SAPIEN 3.0.0b1 |

The `flash-attn` wheel in the HyVLA environment is compiled on Draco for A100
(`sm_80`) against the cluster's GLIBC. Do not replace it with the upstream
prebuilt Linux wheel: that artifact requires a newer GLIBC than Draco provides.

Draco's host driver is older than the CUDA 12.8 SGLang wheel stack. The
launcher therefore prepends the installed CUDA 12.8 forward-compatibility
directory only to the SGLang process. It deliberately does not expose those
driver libraries to HyVLA or RoboTwin/SAPIEN. The toolkit installation includes
the conventional `lib64` link required by FlashInfer's first-run JIT build.

RoboTwin is required to be at commit
`c3ddfa8b97d5519efa828b075999bd0006778e5e`. The job exits before model
loading if the checkout or its pinned `test_num=100` invariant differs. The
bounded harness changes the rollout count in its importing process; it does
not edit `script/eval_policy.py`.

Provision the smoke task configuration once on the login node. This adds an
untracked configuration symlink; it does not alter any official simulator
source file:

```bash
REPO="$STORAGE/src/highlow-handoff"
ROBOTWIN="$STORAGE/runs/highlow-handoff/robotwin_runtime/RoboTwin"
test ! -e "$ROBOTWIN/task_config/demo_highlow_smoke.yml"
ln -s \
  "$REPO/robotwin_dual_system/demo_highlow_smoke.yml" \
  "$ROBOTWIN/task_config/demo_highlow_smoke.yml"
test "$(readlink -f "$ROBOTWIN/task_config/demo_highlow_smoke.yml")" = \
  "$(readlink -f "$REPO/robotwin_dual_system/demo_highlow_smoke.yml")"
```

The batch launcher never creates or changes this shared symlink. In simulator
mode it fails closed unless the link resolves to the configuration in the
checked-out highlow repository.

Before submitting, the environments must satisfy these imports:

```bash
env CUDA_HOME="$STORAGE/tools/cuda-12.8" \
  PATH="$STORAGE/miniforge3/envs/highlow-sglang/bin:$STORAGE/tools/cuda-12.8/bin:$PATH" \
  LD_LIBRARY_PATH="$STORAGE/tools/cuda-compat-12.8/cuda-compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$STORAGE/miniforge3/envs/highlow-sglang/bin/python" -c \
  'import sglang, sgl_kernel.common_ops, torch'
PYTHONPATH="$STORAGE/src/highlow-handoff" \
  "$STORAGE/miniforge3/envs/highlow-runtime/bin/python" -c \
  'import highlow, numpy, PIL, uvicorn'
PYTHONPATH="$STORAGE/src/highlow-handoff" \
  "$STORAGE/venvs/highlow-hyvla/bin/python" -c \
  'import highlow, hy_vla, robotwin_eval, torch, uvicorn'
PYTHONPATH="$STORAGE/src/highlow-handoff" \
  "$STORAGE/miniforge3/envs/robotwin/bin/python" -c \
  'import highlow, robotwin_dual_system, sapien, torch'
```

Do not install packages or download checkpoints from inside the batch job.
The launcher enables offline Hugging Face mode. It also places Hugging Face,
Torch, XDG, uv, matplotlib, Triton, FlashInfer, CUDA, and temporary caches below
`$RUN_ROOT`; it neither changes nor uses `HOME` for runtime state. SGLang is the
one exception for transient IPC socket names: Linux limits Unix-domain socket
paths, so that process gets `/tmp/highlow-sglang-$SLURM_JOB_ID` as `TMPDIR`.
The process-group cleanup removes this node-local directory after SGLang exits;
no persistent artifact is stored there.

## Fast real-HTTP probe

The default mode starts SGLang and both inference services, runs
`scripts/probe_dual_system_http.py`, and then exits successfully. The EXIT trap
terminates the three server process groups; it does not wait indefinitely.

```bash
cd "$STORAGE/src/highlow-handoff"
SLURM_LOG_ROOT="$STORAGE/runs/highlow-handoff/slurm"
mkdir -p "$SLURM_LOG_ROOT"
sbatch \
  --output="$SLURM_LOG_ROOT/highlow-%j.out" \
  --error="$SLURM_LOG_ROOT/highlow-%j.err" \
  --export=ALL,STORAGE="$STORAGE",RUN_ROBOTWIN=0 \
  scripts/slurm/draco_dual_system.sbatch
```

This mode creates `probe_result.json`. It exercises TCP/HTTP serialization and
the episode lifecycle, rather than using in-process clients.

## One official RoboTwin rollout

Set `RUN_ROBOTWIN=1` to replace the probe with the bounded simulator harness:

```bash
cd "$STORAGE/src/highlow-handoff"
SLURM_LOG_ROOT="$STORAGE/runs/highlow-handoff/slurm"
mkdir -p "$SLURM_LOG_ROOT"
sbatch \
  --output="$SLURM_LOG_ROOT/highlow-%j.out" \
  --error="$SLURM_LOG_ROOT/highlow-%j.err" \
  --export=ALL,STORAGE="$STORAGE",RUN_ROBOTWIN=1,TASK_NAME=place_burger_fries,TASK_CONFIG=demo_highlow_smoke,INSTRUCTION_TYPE=unseen,SEED=0 \
  scripts/slurm/draco_dual_system.sbatch
```

The launcher always passes `--test-num 1` to
`scripts/run_robotwin_limited_eval.py`. The generated policy configuration in
`$RUN_ROOT/robotwin_dual_system.runtime.json` points the unchanged official
simulator to the job's dynamic high- and low-level URLs. Treat
`robotwin_result.json` as authoritative: the pinned evaluator's own
`_result.txt` keeps its upstream hard-coded denominator of 100 even when the
external harness safely limits execution to one rollout.

The low-level configuration is deliberately fixed to the real checkpoint's
evaluation semantics:

```text
blend_mode=rel_abs
exc_action_size=7
img_history_size=6
img_history_interval=5
umi_coord_frame=false
umi_gripper_space=false
```

The launcher verifies that `norm_stats.pkl` contains both absolute-action
statistics before loading HyVLA.

## Readiness and outputs

An HTTP 200 response alone is not accepted. Startup performs all of these
checks:

- SGLang `/v1/models` must be valid JSON containing the exact served model ID
  `highlow-qwen25-vl-3b`;
- the high-level `/health` JSON state must be `ready`, followed by exact
  backend/model/protocol capability checks;
- the low-level `/health` JSON state must be `ready`, `backend_open` must be
  true, and capabilities must identify the single-worker HyVLA backend.

Artifacts are retained at:

```text
$STORAGE/runs/highlow-handoff/jobs/$SLURM_JOB_ID/
├── READY
├── topology.json
├── robotwin_dual_system.runtime.json
├── probe_result.json or robotwin_result.json
├── cache/
└── logs/
    ├── launcher.log
    ├── sglang.log
    ├── high-facade.log
    ├── low-hyvla.log
    └── http-probe.log or robotwin.log
```

The commands above override `#SBATCH --output` and `--error` so Slurm's wrapper
logs do not appear as untracked files in the repository. Without those CLI
options they are created in the submission directory because Slurm does not
expand `$STORAGE` inside `#SBATCH` lines. Use `topology.json` and the per-process
logs for the authoritative run record.

## Operational notes

- The low-level server must remain at `--workers 1`; its episode and HyVLA
  action-cache state are process-local.
- CUDA device tokens come from the current Slurm allocation. The launcher
  understands the Draco `gpu:` prefix and validates that every selected
  environment sees one logical device. It never searches for or guesses a
  physical GPU outside the allocation.
- The prepared HyVLA environment is A100-specific because its locally compiled
  `flash-attn` wheel contains `sm_80` kernels. The launcher rejects a different
  compute capability before model loading instead of failing later in a CUDA
  kernel call.
- CUDA visibility isolates SGLang, HyVLA, and cuRobo. SAPIEN's Vulkan renderer
  can enumerate devices differently, so the bounded harness additionally
  replaces SAPIEN 3's legacy no-argument compatibility wrapper in-process and
  constructs the renderer with `Device("cuda:0")`. It requires exactly one
  visible CUDA device, verifies that it is render-capable, and asserts that
  the created scene's actual renderer has the same logical CUDA index and PCI
  address. It records both devices and the renderer summary in
  `robotwin_result.json`. The official simulator files remain untouched.
- Low-level observations currently cross the network as JSON. Watch request
  size and latency in the first real run; the ASGI adapter has a 32 MiB request
  limit.
- The three sockets are selected together and are unique within the job. A
  different process could theoretically claim an ephemeral port between
  selection and bind; in that case the owning service exits and readiness
  fails with its preserved log instead of connecting to an unrelated service.
- `SIGTERM`, `SIGINT`, normal success, and errors all trigger process-group
  cleanup. Logs, results, generated configuration, and caches remain intact.
