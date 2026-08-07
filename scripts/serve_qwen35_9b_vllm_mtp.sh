#!/usr/bin/env bash
set -euo pipefail

# Serve Qwen3.5-9B as the high-level planner through vLLM.
# The official Qwen3.5 recipe uses Qwen's native MTP speculative decoding via
# method=qwen3_next_mtp. Keep all runtime cache paths under the user's CPFS
# directory and do not override CUDA_VISIBLE_DEVICES.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"
USER_ROOT="$(cd "${WORK_ROOT}/.." && pwd)"

MODEL_PATH=${QWEN35_MODEL_PATH:-"${WORK_ROOT}/checkpoints/Qwen/Qwen3.5-9B"}
MODEL_NAME=${QWEN35_MODEL_NAME:-qwen3.5-9b-mtp}
HOST=${QWEN35_HOST:-127.0.0.1}
PORT=${QWEN35_PORT:-8001}
LOCAL_GPU=${QWEN35_LOCAL_GPU:-1}
GPU_MEMORY_UTILIZATION=${QWEN35_GPU_MEMORY_UTILIZATION:-0.80}
MAX_MODEL_LEN=${QWEN35_MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${QWEN35_MAX_NUM_SEQS:-1}
MTP_SPECULATIVE_TOKENS=${QWEN35_MTP_SPECULATIVE_TOKENS:-2}
RUNTIME_ROOT=${QWEN35_RUNTIME_ROOT:-"${USER_ROOT}/rt/q35"}

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Qwen3.5 model directory does not exist: ${MODEL_PATH}" >&2
  echo "Set QWEN35_MODEL_PATH to the downloaded Qwen/Qwen3.5-9B checkpoint." >&2
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

export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK="${LOCAL_GPU}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${QWEN35_MASTER_PORT:-29532}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TMPDIR="${RUNTIME_ROOT}/scratch"
export TORCHINDUCTOR_CACHE_DIR="${RUNTIME_ROOT}/inductor"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/xdg"
export MPLCONFIGDIR="${RUNTIME_ROOT}/matplotlib"
export VLLM_RPC_BASE_PATH="${RUNTIME_ROOT}/rpc"
export PYTHONPATH="${SCRIPT_DIR}/vllm_uuid_sitecustomize${PYTHONPATH:+:${PYTHONPATH}}"

SPECULATIVE_CONFIG=$(printf '{"method":"qwen3_next_mtp","num_speculative_tokens":%s}' "${MTP_SPECULATIVE_TOKENS}")

echo "Qwen3.5 model       : ${MODEL_PATH}"
echo "vLLM endpoint        : http://${HOST}:${PORT}/v1"
echo "Logical GPU          : cuda:${LOCAL_GPU}"
echo "MTP speculative toks : ${MTP_SPECULATIVE_TOKENS}"
echo "Runtime cache root   : ${RUNTIME_ROOT}"

exec /usr/local/bin/python3 "${SCRIPT_DIR}/vllm_uuid_compat.py" \
  serve "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --distributed-executor-backend external_launcher \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --mamba-cache-mode align \
  --speculative-config "${SPECULATIVE_CONFIG}" \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
