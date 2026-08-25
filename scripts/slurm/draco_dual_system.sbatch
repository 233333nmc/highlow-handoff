#!/usr/bin/env bash
#SBATCH --job-name=highlow-dual-system
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=batch_singlenode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:3
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@90
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# One-node client/server smoke test for the decoupled runtime:
#
#   allocated GPU 0 -> SGLang high-level model
#   CPU-only        -> high-level facade
#   allocated GPU 1 -> HyVLA low-level service
#   allocated GPU 2 -> optional RoboTwin client/simulator (idle in probe mode)
#
# RUN_ROBOTWIN=0 runs the real-HTTP probe and exits. RUN_ROBOTWIN=1 runs the
# bounded official RoboTwin harness. In both modes EXIT cleanup stops every
# process group while retaining all artifacts below RUN_ROOT.

set -Eeuo pipefail
umask 027

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run inside an sbatch allocation." >&2
  exit 2
fi
if [[ ! "$SLURM_JOB_ID" =~ ^[0-9]+([_.][0-9]+)*$ ]]; then
  echo "SLURM_JOB_ID contains unsupported path characters: $SLURM_JOB_ID" >&2
  exit 2
fi
if [[ -z "${STORAGE:-}" ]]; then
  echo "STORAGE must be exported when the job is submitted." >&2
  exit 2
fi

readonly REPO="$STORAGE/src/highlow-handoff"
readonly HIGH_MODEL="$STORAGE/models/highlow/Qwen2.5-VL-3B-Instruct"
readonly LOW_MODEL="$STORAGE/models/highlow/Hy-Embodied-0.5-VLA-RoboTwin"
readonly LOW_NORM="$LOW_MODEL/norm_stats.pkl"
readonly ROBOTWIN_ROOT="$STORAGE/runs/highlow-handoff/robotwin_runtime/RoboTwin"
readonly SMOKE_TASK_CONFIG_SOURCE="$REPO/robotwin_dual_system/demo_highlow_smoke.yml"
readonly SMOKE_TASK_CONFIG_LINK="$ROBOTWIN_ROOT/task_config/demo_highlow_smoke.yml"

readonly SGLANG_PY="$STORAGE/miniforge3/envs/highlow-sglang/bin/python"
readonly RUNTIME_PY="$STORAGE/miniforge3/envs/highlow-runtime/bin/python"
readonly LOW_PY="$STORAGE/venvs/highlow-hyvla/bin/python"
readonly ROBOTWIN_PY="$STORAGE/miniforge3/envs/robotwin/bin/python"
readonly CUDA_TOOLKIT_DIR="$STORAGE/tools/cuda-12.8"
readonly CUDA_COMPAT_DIR="$STORAGE/tools/cuda-compat-12.8/cuda-compat"
readonly SGLANG_LD_LIBRARY_PATH="$CUDA_COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
readonly SGLANG_PATH="$(dirname "$SGLANG_PY"):$CUDA_TOOLKIT_DIR/bin:$PATH"

readonly HIGH_SERVED_MODEL="highlow-qwen25-vl-3b"
readonly ROBOTWIN_PINNED_COMMIT="c3ddfa8b97d5519efa828b075999bd0006778e5e"
readonly RUN_ROOT="$STORAGE/runs/highlow-handoff/jobs/$SLURM_JOB_ID"
readonly LOG_ROOT="$RUN_ROOT/logs"
readonly CACHE_ROOT="$RUN_ROOT/cache"
readonly SGLANG_IPC_TMP="/tmp/highlow-sglang-$SLURM_JOB_ID"

mkdir -p \
  "$LOG_ROOT" \
  "$RUN_ROOT/tmp" \
  "$CACHE_ROOT/huggingface/hub" \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/xdg" \
  "$CACHE_ROOT/uv" \
  "$CACHE_ROOT/matplotlib" \
  "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/cuda" \
  "$CACHE_ROOT/flashinfer" \
  "$CACHE_ROOT/torch-extensions" \
  "$CACHE_ROOT/numba" \
  "$CACHE_ROOT/pip" \
  "$CACHE_ROOT/wandb"

# Keep all mutable caches on STORAGE. Deliberately do not assign, shadow, or
# derive any path from HOME.
export TMPDIR="$RUN_ROOT/tmp"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$CACHE_ROOT/huggingface/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export FLASHINFER_WORKSPACE_DIR="$CACHE_ROOT/flashinfer"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch-extensions"
export NUMBA_CACHE_DIR="$CACHE_ROOT/numba"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export WANDB_DIR="$CACHE_ROOT/wandb"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb/cache"
export WANDB_CONFIG_DIR="$CACHE_ROOT/wandb/config"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:$ROBOTWIN_ROOT:$ROBOTWIN_ROOT/script:$ROBOTWIN_ROOT/description/utils${PYTHONPATH:+:$PYTHONPATH}"

readonly LAUNCHER_LOG="$LOG_ROOT/launcher.log"
touch "$LAUNCHER_LOG"

log() {
  local timestamp
  timestamp="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  printf '[%s] %s\n' "$timestamp" "$*" | tee -a "$LAUNCHER_LOG"
}

die() {
  log "ERROR: $*"
  exit 1
}

case "${RUN_ROBOTWIN:-0}" in
  0 | 1) ;;
  *) die "RUN_ROBOTWIN must be 0 or 1" ;;
esac

for executable in curl setsid git; do
  command -v "$executable" >/dev/null 2>&1 || die "missing executable: $executable"
done
for python_path in "$SGLANG_PY" "$RUNTIME_PY" "$LOW_PY" "$ROBOTWIN_PY"; do
  [[ -x "$python_path" ]] || die "missing Python interpreter: $python_path"
done
for required_path in \
  "$REPO/highlow" \
  "$REPO/scripts/probe_dual_system_http.py" \
  "$REPO/scripts/run_robotwin_limited_eval.py" \
  "$SMOKE_TASK_CONFIG_SOURCE" \
  "$HIGH_MODEL/config.json" \
  "$LOW_MODEL/config.json" \
  "$LOW_NORM" \
  "$(dirname "$SGLANG_PY")/ninja" \
  "$CUDA_TOOLKIT_DIR/bin/nvcc" \
  "$CUDA_TOOLKIT_DIR/lib64/libcudart.so" \
  "$CUDA_COMPAT_DIR/libcuda.so.1" \
  "$ROBOTWIN_ROOT/script/eval_policy.py"; do
  [[ -e "$required_path" ]] || die "missing required path: $required_path"
done

# Fail early on incomplete environments rather than waiting through model
# startup. The framework runtime intentionally includes robotwin-codec because
# the real-HTTP probe constructs NumPy observations and JPEG high-level frames.
env CUDA_HOME="$CUDA_TOOLKIT_DIR" \
  PATH="$SGLANG_PATH" \
  LD_LIBRARY_PATH="$SGLANG_LD_LIBRARY_PATH" \
  "$SGLANG_PY" -c 'import sglang, sgl_kernel.common_ops, torch' \
  >>"$LAUNCHER_LOG" 2>&1 || \
  die "highlow-sglang environment is incomplete"
"$RUNTIME_PY" -c 'import highlow, numpy, PIL, uvicorn' >>"$LAUNCHER_LOG" 2>&1 || \
  die "highlow-runtime environment is incomplete"
"$LOW_PY" -c 'import highlow, hy_vla, robotwin_eval, torch, uvicorn' \
  >>"$LAUNCHER_LOG" 2>&1 || die "highlow-hyvla environment is incomplete"
if [[ "${RUN_ROBOTWIN:-0}" == 1 ]]; then
  "$ROBOTWIN_PY" -c 'import highlow, robotwin_dual_system, sapien, torch' \
    >>"$LAUNCHER_LOG" 2>&1 || die "RoboTwin environment is incomplete"

  [[ -L "$SMOKE_TASK_CONFIG_LINK" ]] || \
    die "provisioning symlink is missing: $SMOKE_TASK_CONFIG_LINK"
  smoke_source_real="$("$RUNTIME_PY" -c \
    'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' \
    "$SMOKE_TASK_CONFIG_SOURCE")"
  smoke_link_real="$("$RUNTIME_PY" -c \
    'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' \
    "$SMOKE_TASK_CONFIG_LINK")"
  [[ "$smoke_link_real" == "$smoke_source_real" ]] || \
    die "smoke task config symlink does not resolve to $SMOKE_TASK_CONFIG_SOURCE"
fi

REPO_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
readonly REPO_COMMIT
ROBOTWIN_COMMIT="$(git -C "$ROBOTWIN_ROOT" rev-parse HEAD)"
readonly ROBOTWIN_COMMIT
[[ "$ROBOTWIN_COMMIT" == "$ROBOTWIN_PINNED_COMMIT" ]] || \
  die "RoboTwin must be pinned at $ROBOTWIN_PINNED_COMMIT (found $ROBOTWIN_COMMIT)"
grep -Eq 'test_num[[:space:]]*=[[:space:]]*100' \
  "$ROBOTWIN_ROOT/script/eval_policy.py" || \
  die "pinned RoboTwin eval_policy.py invariant test_num=100 was not found"

# Parse a comma-separated CUDA device list. SLURM_JOB_GPUS on Draco may use
# tokens such as gpu:0,gpu:1,gpu:2, while CUDA_VISIBLE_DEVICES may use numeric,
# GPU UUID, or MIG UUID tokens.
PARSED_GPUS=()
parse_gpu_list() {
  local raw="$1"
  local token
  local -a candidates=()
  PARSED_GPUS=()
  raw="${raw//[[:space:]]/}"
  IFS=',' read -r -a candidates <<< "$raw"
  for token in "${candidates[@]}"; do
    token="${token#gpu:}"
    if [[ "$token" =~ ^[0-9]+$ || "$token" == GPU-* || "$token" == MIG-* ]]; then
      PARSED_GPUS+=("$token")
    else
      PARSED_GPUS=()
      return 1
    fi
  done
  [[ "${#PARSED_GPUS[@]}" -ge 3 ]]
}

GPU_SOURCE=""
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && parse_gpu_list "$CUDA_VISIBLE_DEVICES"; then
  GPU_SOURCE="CUDA_VISIBLE_DEVICES"
elif [[ -n "${SLURM_JOB_GPUS:-}" ]] && parse_gpu_list "$SLURM_JOB_GPUS"; then
  GPU_SOURCE="SLURM_JOB_GPUS"
else
  # Some Draco Slurm configurations expose the allocation through a device
  # cgroup but give SLURM_JOB_GPUS a range syntax. Only use logical 0,1,2 when
  # Slurm itself confirms that this job owns at least three visible GPUs.
  allocation_count=""
  if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ([0-9]+)$ ]]; then
    allocation_count="${BASH_REMATCH[1]}"
  elif [[ "${SLURM_GPUS:-}" =~ ^[0-9]+$ ]]; then
    allocation_count="$SLURM_GPUS"
  fi
  if [[ -n "$allocation_count" && "$allocation_count" -ge 3 ]]; then
    PARSED_GPUS=(0 1 2)
    GPU_SOURCE="job-local logical CUDA indices"
  else
    die "cannot resolve three GPUs from this Slurm allocation; refusing to guess physical devices"
  fi
fi

readonly SGLANG_GPU="${PARSED_GPUS[0]}"
readonly LOW_GPU="${PARSED_GPUS[1]}"
readonly ROBOTWIN_GPU="${PARSED_GPUS[2]}"
[[ "$SGLANG_GPU" != "$LOW_GPU" && \
   "$SGLANG_GPU" != "$ROBOTWIN_GPU" && \
   "$LOW_GPU" != "$ROBOTWIN_GPU" ]] || die "resolved GPU tokens are not unique"
log "GPU source: $GPU_SOURCE; roles: sglang=$SGLANG_GPU low=$LOW_GPU robotwin=$ROBOTWIN_GPU"

# CUDA_VISIBLE_DEVICES can differ between Slurm installations. Validate each
# resolved token inside the exact Python environment that will consume it.
check_single_cuda_device() {
  local role="$1" python_path="$2" gpu_token="$3"
  env CUDA_VISIBLE_DEVICES="$gpu_token" "$python_path" - "$role" <<'PY'
import sys
import torch

count = torch.cuda.device_count()
if count != 1:
    raise SystemExit(f"{sys.argv[1]} expected one CUDA device, found {count}")
capability = torch.cuda.get_device_capability(0)
if sys.argv[1] == "low" and capability != (8, 0):
    raise SystemExit(
        "the prepared HyVLA flash-attn wheel is A100/sm_80-only; "
        f"received compute capability {capability}"
    )
print(
    f"{sys.argv[1]} cuda:0={torch.cuda.get_device_name(0)} "
    f"compute_capability={capability}"
)
PY
}
check_single_cuda_device sglang "$SGLANG_PY" "$SGLANG_GPU" \
  >>"$LAUNCHER_LOG" 2>&1 || die "SGLang GPU isolation check failed"
check_single_cuda_device low "$LOW_PY" "$LOW_GPU" \
  >>"$LAUNCHER_LOG" 2>&1 || die "low-level GPU isolation check failed"
if [[ "${RUN_ROBOTWIN:-0}" == 1 ]]; then
  check_single_cuda_device robotwin "$ROBOTWIN_PY" "$ROBOTWIN_GPU" \
    >>"$LAUNCHER_LOG" 2>&1 || die "RoboTwin GPU isolation check failed"
fi

# rel_abs is only valid with the absolute-action statistics. Fail before the
# expensive model startup if the checkpoint is not the expected artifact.
"$LOW_PY" - "$LOW_NORM" >>"$LAUNCHER_LOG" 2>&1 <<'PY'
import pickle
import sys

with open(sys.argv[1], "rb") as handle:
    stats = pickle.load(handle)
required = {
    "qpos_mean",
    "qpos_std",
    "action_mean",
    "action_std",
    "action_mean_abs",
    "action_std_abs",
}
missing = sorted(required.difference(stats))
if missing:
    raise SystemExit(f"rel_abs norm statistics are missing: {missing}")
print("HyVLA rel_abs normalization statistics are present")
PY

# Bind three sockets simultaneously so the selected loopback ports are unique
# within this job. The processes bind them immediately after this preflight.
read -r SGLANG_PORT HIGH_PORT LOW_PORT < <("$RUNTIME_PY" <<'PY'
import socket

sockets = []
try:
    for _ in range(3):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sockets.append(sock)
    print(*(sock.getsockname()[1] for sock in sockets))
finally:
    for sock in sockets:
        sock.close()
PY
)
[[ "$SGLANG_PORT" =~ ^[0-9]+$ && \
   "$HIGH_PORT" =~ ^[0-9]+$ && \
   "$LOW_PORT" =~ ^[0-9]+$ ]] || die "dynamic port allocation failed"
[[ "$SGLANG_PORT" != "$HIGH_PORT" && \
   "$SGLANG_PORT" != "$LOW_PORT" && \
   "$HIGH_PORT" != "$LOW_PORT" ]] || die "dynamic ports are not unique"
readonly SGLANG_PORT HIGH_PORT LOW_PORT

readonly SGLANG_URL="http://127.0.0.1:$SGLANG_PORT"
readonly HIGH_URL="http://127.0.0.1:$HIGH_PORT"
readonly LOW_URL="http://127.0.0.1:$LOW_PORT"
log "endpoints: sglang=$SGLANG_URL high=$HIGH_URL low=$LOW_URL"

readonly POLICY_CONFIG="$RUN_ROOT/robotwin_dual_system.runtime.json"
"$RUNTIME_PY" - \
  "$POLICY_CONFIG" "$HIGH_URL" "$LOW_URL" "$ROBOTWIN_COMMIT" <<'PY'
import json
import pathlib
import sys

path, high_url, low_url, simulator_version = sys.argv[1:]
payload = {
    "high_level_url": high_url,
    "low_level_url": low_url,
    "high_level_timeout_s": 180,
    "low_level_timeout_s": 180,
    "verifier_interval_steps": 100,
    "source_fps": 25,
    "action_type": "ee",
    "simulator_version": simulator_version,
    "episode_metadata": {"launcher": "draco_dual_system.sbatch"},
}
pathlib.Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

LOW_CONFIG="$("$RUNTIME_PY" - "$LOW_MODEL" "$LOW_NORM" <<'PY'
import json
import sys

print(json.dumps({
    "policy_config": {
        "ckpt_path": sys.argv[1],
        "norm_path": sys.argv[2],
        "blend_mode": "rel_abs",
        "exc_action_size": 7,
        "img_history_size": 6,
        "img_history_interval": 5,
        "umi_coord_frame": False,
        "umi_gripper_space": False,
    }
}, separators=(",", ":")))
PY
)"
readonly LOW_CONFIG

"$RUNTIME_PY" - \
  "$RUN_ROOT/topology.json" "$SLURM_JOB_ID" "$(hostname)" \
  "$REPO_COMMIT" "$ROBOTWIN_COMMIT" \
  "$SGLANG_URL" "$HIGH_URL" "$LOW_URL" "$LOW_CONFIG" <<'PY'
import json
import pathlib
import sys

(path, job_id, node, repo_commit, robotwin_commit,
 sglang_url, high_url, low_url, low_config) = sys.argv[1:]
payload = {
    "job_id": job_id,
    "node": node,
    "repo_commit": repo_commit,
    "robotwin_commit": robotwin_commit,
    "endpoints": {
        "sglang": sglang_url,
        "high_level": high_url,
        "low_level": low_url,
    },
    "low_level_config": json.loads(low_config),
}
pathlib.Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

declare -a PROCESS_PIDS=()
declare -a PROCESS_NAMES=()
declare -a PROCESS_LOGS=()
LAST_PID=""
SGLANG_IPC_OWNED=0

start_process() {
  local name="$1" gpu_token="$2" process_log="$3"
  shift 3
  : >"$process_log"
  log "starting $name (CUDA_VISIBLE_DEVICES=${gpu_token:-hidden}, log=$process_log)"
  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu_token" \
    "$@" >>"$process_log" 2>&1 &
  LAST_PID=$!
  PROCESS_PIDS+=("$LAST_PID")
  PROCESS_NAMES+=("$name")
  PROCESS_LOGS+=("$process_log")
}

cleanup() {
  local rc=$?
  local index deadline alive
  trap - EXIT INT TERM HUP
  set +e

  log "cleanup: stopping ${#PROCESS_PIDS[@]} process group(s); artifacts remain in $RUN_ROOT"
  for ((index=${#PROCESS_PIDS[@]} - 1; index >= 0; index--)); do
    kill -TERM -- "-${PROCESS_PIDS[index]}" 2>/dev/null
  done

  deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    alive=0
    for pid in "${PROCESS_PIDS[@]}"; do
      if kill -0 -- "-$pid" 2>/dev/null; then
        alive=1
      fi
    done
    (( alive == 0 )) && break
    sleep 1
  done

  for ((index=${#PROCESS_PIDS[@]} - 1; index >= 0; index--)); do
    kill -KILL -- "-${PROCESS_PIDS[index]}" 2>/dev/null
    wait "${PROCESS_PIDS[index]}" 2>/dev/null
  done
  if [[ "$SGLANG_IPC_OWNED" == 1 && -d "$SGLANG_IPC_TMP" ]]; then
    find "$SGLANG_IPC_TMP" -mindepth 1 -depth -delete 2>/dev/null
    rmdir "$SGLANG_IPC_TMP" 2>/dev/null || \
      log "warning: could not remove SGLang IPC directory $SGLANG_IPC_TMP"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

[[ ! -e "$SGLANG_IPC_TMP" ]] || \
  die "SGLang IPC directory already exists: $SGLANG_IPC_TMP"
mkdir -m 700 "$SGLANG_IPC_TMP"
SGLANG_IPC_OWNED=1

start_process \
  sglang "$SGLANG_GPU" "$LOG_ROOT/sglang.log" \
  env TMPDIR="$SGLANG_IPC_TMP" \
  CUDA_HOME="$CUDA_TOOLKIT_DIR" \
  PATH="$SGLANG_PATH" \
  LD_LIBRARY_PATH="$SGLANG_LD_LIBRARY_PATH" \
  MAX_JOBS=8 \
  NVCC_THREADS=1 \
  "$SGLANG_PY" -u -m sglang.launch_server \
    --model-path "$HIGH_MODEL" \
    --served-model-name "$HIGH_SERVED_MODEL" \
    --host 127.0.0.1 \
    --port "$SGLANG_PORT" \
    --tp-size 1 \
    --trust-remote-code
readonly SGLANG_PID="$LAST_PID"

export HIGHLOW_LOW_LEVEL_FACTORY="highlow.compositions.robotwin_hyvla:create_robotwin_hyvla_backend"
export HIGHLOW_LOW_LEVEL_CONFIG="$LOW_CONFIG"
start_process \
  low-hyvla "$LOW_GPU" "$LOG_ROOT/low-hyvla.log" \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$LOW_PY" -u -m uvicorn \
    highlow.inference.low_level:create_policy_app_from_env \
    --factory \
    --host 127.0.0.1 \
    --port "$LOW_PORT" \
    --workers 1 \
    --lifespan on \
    --log-level info
readonly LOW_PID="$LAST_PID"

validate_json() {
  local kind="$1" expected="${2:-}"
  "$RUNTIME_PY" -c '
import json
import sys

kind, expected = sys.argv[1:3]
obj = json.load(sys.stdin)

def require(condition, message):
    if not condition:
        raise ValueError(message)

require(type(obj) is dict, "response must be a JSON object")

if kind == "models":
    require(obj.get("object") == "list", "models.object must be list")
    data = obj.get("data")
    require(type(data) is list, "models.data must be an array")
    require(
        any(type(item) is dict and item.get("id") == expected for item in data),
        f"served model {expected!r} is absent",
    )
elif kind == "high-health":
    require(obj.get("state") == "ready", "high-level state is not ready")
    require(
        obj.get("component") == "high-level-service",
        "unexpected high-level component",
    )
elif kind == "high-capabilities":
    require(obj.get("component") == "high-level-service", "unexpected component")
    require(obj.get("backend") == "sglang-openai", "unexpected high backend")
    require(obj.get("model") == expected, "unexpected high model")
    require(
        obj.get("protocol") == "openai-chat-completions",
        "unexpected high protocol",
    )
    require(obj.get("supports_images") is True, "images must be supported")
    require(
        obj.get("supports_structured_output") is True,
        "structured output must be supported",
    )
elif kind == "low-health":
    require(obj.get("state") == "ready", "low-level state is not ready")
    require(obj.get("service") == "low-level-service", "unexpected low service")
    details = obj.get("details")
    require(type(details) is dict, "low health details must be an object")
    require(details.get("backend_id") == "hyvla", "unexpected low backend")
    require(details.get("backend_open") is True, "low backend is not open")
elif kind == "low-capabilities":
    require(obj.get("component") == "low-level-service", "unexpected component")
    manifest = obj.get("manifest")
    require(type(manifest) is dict, "low manifest must be an object")
    require(manifest.get("policy_id") == "hyvla", "unexpected policy ID")
    require(manifest.get("backend_id") == "hyvla", "unexpected backend ID")
    require(
        manifest.get("max_concurrent_episodes") == 1,
        "HyVLA must advertise one concurrent episode",
    )
else:
    raise AssertionError(f"unknown validation kind: {kind}")
' "$kind" "$expected"
}

wait_for_json() {
  local name="$1" pid="$2" process_log="$3"
  local url="$4" kind="$5" expected="$6" timeout_s="$7"
  local deadline=$((SECONDS + timeout_s))
  local body=""

  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "$name exited during readiness"
      tail -n 100 "$process_log" | tee -a "$LAUNCHER_LOG" >&2
      return 1
    fi
    if body="$(curl -fsS --connect-timeout 2 --max-time 5 "$url" 2>/dev/null)" &&
       printf '%s' "$body" | validate_json "$kind" "$expected" >/dev/null 2>&1; then
      log "$name ready: $url"
      return 0
    fi
    sleep 2
  done

  log "$name readiness timed out at $url; last body: ${body:0:1000}"
  tail -n 100 "$process_log" | tee -a "$LAUNCHER_LOG" >&2
  return 1
}

wait_for_json \
  sglang "$SGLANG_PID" "$LOG_ROOT/sglang.log" \
  "$SGLANG_URL/v1/models" models "$HIGH_SERVED_MODEL" \
  "${SGLANG_READY_TIMEOUT_S:-1800}" || die "SGLang did not become ready"

export HIGHLOW_SGLANG_BASE_URL="$SGLANG_URL"
export HIGHLOW_SGLANG_MODEL="$HIGH_SERVED_MODEL"
export HIGHLOW_SGLANG_TIMEOUT_S="${HIGH_LEVEL_TIMEOUT_S:-180}"
start_process \
  high-facade "" "$LOG_ROOT/high-facade.log" \
  "$RUNTIME_PY" -u -m uvicorn \
    highlow.inference.high_level:create_sglang_app_from_env \
    --factory \
    --host 127.0.0.1 \
    --port "$HIGH_PORT" \
    --workers 1 \
    --lifespan on \
    --log-level info
readonly HIGH_PID="$LAST_PID"

wait_for_json \
  high-facade "$HIGH_PID" "$LOG_ROOT/high-facade.log" \
  "$HIGH_URL/health" high-health "" \
  "${FACADE_READY_TIMEOUT_S:-300}" || die "high-level facade did not become ready"
wait_for_json \
  low-hyvla "$LOW_PID" "$LOG_ROOT/low-hyvla.log" \
  "$LOW_URL/health" low-health "" \
  "${LOW_READY_TIMEOUT_S:-1800}" || die "low-level service did not become ready"

curl -fsS --connect-timeout 2 --max-time 10 "$HIGH_URL/capabilities" |
  validate_json high-capabilities "$HIGH_SERVED_MODEL" || die "invalid high-level capabilities"
curl -fsS --connect-timeout 2 --max-time 10 "$LOW_URL/capabilities" |
  validate_json low-capabilities "" || die "invalid low-level capabilities"
touch "$RUN_ROOT/READY"
log "all inference services passed strict readiness"

readonly SERVER_PIDS=("$SGLANG_PID" "$HIGH_PID" "$LOW_PID")

wait_for_client() {
  local client_name="$1" client_pid="$2" client_log="$3" timeout_s="$4"
  local deadline=$((SECONDS + timeout_s))
  local server_pid rc

  while kill -0 "$client_pid" 2>/dev/null; do
    for server_pid in "${SERVER_PIDS[@]}"; do
      if ! kill -0 "$server_pid" 2>/dev/null; then
        log "an inference server exited while $client_name was running"
        tail -n 100 "$client_log" | tee -a "$LAUNCHER_LOG" >&2
        return 1
      fi
    done
    if (( SECONDS >= deadline )); then
      log "$client_name timed out after ${timeout_s}s"
      kill -TERM -- "-$client_pid" 2>/dev/null || true
      return 1
    fi
    sleep 2
  done

  set +e
  wait "$client_pid"
  rc=$?
  set -e
  if (( rc != 0 )); then
    log "$client_name exited with status $rc"
    tail -n 150 "$client_log" | tee -a "$LAUNCHER_LOG" >&2
    return "$rc"
  fi
  return 0
}

validate_result_object() {
  local result_path="$1" result_kind="$2"
  "$RUNTIME_PY" - "$result_path" "$result_kind" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
result_kind = sys.argv[2]
if not path.is_file():
    raise SystemExit(f"missing result JSON: {path}")
with path.open(encoding="utf-8") as handle:
    result = json.load(handle)
if type(result) is not dict:
    raise SystemExit(f"result JSON must be an object: {path}")
if result.get("status") != "completed":
    raise SystemExit(f"result status is not completed: {result.get('status')!r}")
if result_kind == "probe":
    if result.get("schema_version") != "highlow.dual-system-http-probe.v1":
        raise SystemExit("unexpected probe result schema")
    if result.get("action_dimension") != 16:
        raise SystemExit("probe did not return one 16-dimensional action")
elif result_kind == "robotwin":
    if result.get("schema_version") != "highlow.robotwin-limited-eval.v1":
        raise SystemExit("unexpected RoboTwin result schema")
    request = result.get("request")
    evaluation = result.get("evaluation")
    if type(request) is not dict or request.get("test_num") != 1:
        raise SystemExit("RoboTwin result did not request exactly one rollout")
    if type(evaluation) is not dict or evaluation.get("completed_test_num") != 1:
        raise SystemExit("RoboTwin result did not complete exactly one rollout")
    renderer = result.get("renderer")
    if type(renderer) is not dict:
        raise SystemExit("RoboTwin result is missing renderer verification")
    scene_device = renderer.get("scene_device")
    if type(scene_device) is not dict:
        raise SystemExit("RoboTwin result is missing the actual scene device")
    if renderer.get("logical_cuda_index") != 0 or renderer.get("can_render") is not True:
        raise SystemExit("requested SAPIEN renderer binding is invalid")
    if scene_device.get("logical_cuda_index") != 0 or scene_device.get("can_render") is not True:
        raise SystemExit("actual SAPIEN scene renderer binding is invalid")
    requested_pci = renderer.get("pci_string")
    actual_pci = scene_device.get("pci_string")
    if not requested_pci or str(requested_pci).lower() != str(actual_pci).lower():
        raise SystemExit("actual SAPIEN scene renderer PCI device does not match")
else:
    raise SystemExit(f"unknown result kind: {result_kind}")
print(json.dumps(result, sort_keys=True))
PY
}

if [[ "${RUN_ROBOTWIN:-0}" == 0 ]]; then
  readonly PROBE_RESULT="$RUN_ROOT/probe_result.json"
  start_process \
    http-probe "" "$LOG_ROOT/http-probe.log" \
    "$RUNTIME_PY" -u "$REPO/scripts/probe_dual_system_http.py" \
      --high-url "$HIGH_URL" \
      --low-url "$LOW_URL" \
      --timeout-s "${PROBE_TIMEOUT_S:-300}" \
      --output "$PROBE_RESULT"
  readonly PROBE_PID="$LAST_PID"
  wait_for_client \
    http-probe "$PROBE_PID" "$LOG_ROOT/http-probe.log" \
    "${PROBE_PROCESS_TIMEOUT_S:-900}" || die "real-HTTP probe failed"
  validate_result_object "$PROBE_RESULT" probe >>"$LAUNCHER_LOG" || \
    die "probe result is not valid JSON"
  log "real-HTTP probe completed successfully"
else
  readonly ROBOTWIN_RESULT="$RUN_ROOT/robotwin_result.json"
  readonly TASK_NAME="${TASK_NAME:-place_burger_fries}"
  readonly TASK_CONFIG="${TASK_CONFIG:-demo_highlow_smoke}"
  readonly INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
  readonly SEED="${SEED:-0}"

  start_process \
    robotwin "$ROBOTWIN_GPU" "$LOG_ROOT/robotwin.log" \
    env HIGHLOW_ROBOTWIN_EXPECTED_COMMIT="$ROBOTWIN_COMMIT" \
    "$ROBOTWIN_PY" -u "$REPO/scripts/run_robotwin_limited_eval.py" \
      --robotwin-root "$ROBOTWIN_ROOT" \
      --config "$POLICY_CONFIG" \
      --task-name "$TASK_NAME" \
      --task-config "$TASK_CONFIG" \
      --instruction-type "$INSTRUCTION_TYPE" \
      --seed "$SEED" \
      --test-num 1 \
      --render-device cuda:0 \
      --result-json "$ROBOTWIN_RESULT"
  readonly ROBOTWIN_PID="$LAST_PID"
  wait_for_client \
    robotwin "$ROBOTWIN_PID" "$LOG_ROOT/robotwin.log" \
    "${ROBOTWIN_PROCESS_TIMEOUT_S:-10800}" || die "bounded RoboTwin evaluation failed"
  validate_result_object "$ROBOTWIN_RESULT" robotwin >>"$LAUNCHER_LOG" || \
    die "RoboTwin result is not valid JSON"
  log "bounded RoboTwin evaluation completed successfully"
fi

log "job completed successfully; EXIT cleanup will now stop the inference servers"
