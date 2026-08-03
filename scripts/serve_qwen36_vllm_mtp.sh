#!/usr/bin/env bash
set -euo pipefail

# Serve the public Qwen3.6-27B VLM as the high-level planner. The model
# remains in the shared read-only model directory; vLLM loads it directly.
#
# Qwen3.6 uses native MTP through vLLM speculative decoding. We use the
# external launcher mode with LOCAL_RANK=1 so the service uses the second
# platform-assigned logical GPU without overriding CUDA_VISIBLE_DEVICES.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"
USER_ROOT="$(cd "${WORK_ROOT}/.." && pwd)"

MODEL_PATH=${QWEN36_MODEL_PATH:-/mnt/cpfs/public_data/public_model/Qwen/Qwen3.6-27B}
MODEL_NAME=${QWEN36_MODEL_NAME:-qwen3.6-27b-mtp}
HOST=${QWEN36_HOST:-127.0.0.1}
PORT=${QWEN36_PORT:-8001}
LOCAL_GPU=${QWEN36_LOCAL_GPU:-1}
GPU_MEMORY_UTILIZATION=${QWEN36_GPU_MEMORY_UTILIZATION:-0.80}
MAX_MODEL_LEN=${QWEN36_MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${QWEN36_MAX_NUM_SEQS:-1}
# The public local checkpoint is BF16. vLLM 0.19.1 cannot apply online FP8
# quantization to this Qwen3.6 VLM's 4304-wide visual layer. Leave this empty
# for the checkpoint's native BF16 dtype; set it explicitly for a compatible
# pre-quantized FP8 checkpoint or a future fixed vLLM release.
QUANTIZATION=${QWEN36_QUANTIZATION:-}
MTP_SPECULATIVE_TOKENS=${QWEN36_MTP_SPECULATIVE_TOKENS:-1}
# Keep this path short: vLLM's ZeroMQ IPC socket path must fit within
# sockaddr_un.sun_path, which is only 107 bytes on this platform.
RUNTIME_ROOT=${QWEN36_RUNTIME_ROOT:-"${USER_ROOT}/rt/q36"}

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Qwen3.6 model directory does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ "${LOCAL_GPU}" != "1" ]]; then
  echo "This high-level service is designed for logical GPU 1; got ${LOCAL_GPU}." >&2
  exit 1
fi

mkdir -p \
  "${RUNTIME_ROOT}/scratch" \
  "${RUNTIME_ROOT}/inductor" \
  "${RUNTIME_ROOT}/xdg" \
  "${RUNTIME_ROOT}/matplotlib" \
  "${RUNTIME_ROOT}/rpc"

# Do not set CUDA_VISIBLE_DEVICES here. The platform has already assigned
# the visible GPUs. External-launcher rank selects cuda:1 from that list.
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK="${LOCAL_GPU}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${QWEN36_MASTER_PORT:-29531}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TMPDIR="${RUNTIME_ROOT}/scratch"
export TORCHINDUCTOR_CACHE_DIR="${RUNTIME_ROOT}/inductor"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/xdg"
export MPLCONFIGDIR="${RUNTIME_ROOT}/matplotlib"
export VLLM_RPC_BASE_PATH="${RUNTIME_ROOT}/rpc"
# This is a vLLM 0.19.1 compatibility shim for the platform's UUID-form
# CUDA_VISIBLE_DEVICES. It is inherited by vLLM's model-inspection subprocess.
export PYTHONPATH="${SCRIPT_DIR}/vllm_uuid_sitecustomize${PYTHONPATH:+:${PYTHONPATH}}"

SPECULATIVE_CONFIG=$(printf '{"method":"mtp","num_speculative_tokens":%s}' "${MTP_SPECULATIVE_TOKENS}")

echo "Qwen3.6 model       : ${MODEL_PATH}"
echo "vLLM endpoint        : http://${HOST}:${PORT}/v1"
echo "Logical GPU           : cuda:${LOCAL_GPU}"
echo "Quantization          : ${QUANTIZATION:-native checkpoint BF16}"
echo "MTP speculative tokens: ${MTP_SPECULATIVE_TOKENS}"
echo "Runtime cache root    : ${RUNTIME_ROOT}"

VLLM_ARGS=(
  serve "${MODEL_PATH}"
  --served-model-name "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size 1
  --distributed-executor-backend external_launcher
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --mamba-cache-mode align
  --speculative-config "${SPECULATIVE_CONFIG}"
  --default-chat-template-kwargs '{"enable_thinking": false}'
)
if [[ -n "${QUANTIZATION}" && "${QUANTIZATION}" != "none" ]]; then
  VLLM_ARGS+=(--quantization "${QUANTIZATION}")
fi

exec /usr/local/bin/python3 "${SCRIPT_DIR}/vllm_uuid_compat.py" "${VLLM_ARGS[@]}"
