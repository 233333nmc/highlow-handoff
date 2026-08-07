#!/usr/bin/env bash
set -euo pipefail

# Serve Gemma 4 12B through a minimal OpenAI-compatible API using
# Transformers. Use this when vLLM does not yet support the Gemma 4 unified
# architecture end to end.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"
USER_ROOT="$(cd "${WORK_ROOT}/.." && pwd)"

MODEL_PATH=${GEMMA4_MODEL_PATH:-"${WORK_ROOT}/checkpoints/google/gemma-4-12B-it"}
MODEL_NAME=${GEMMA4_MODEL_NAME:-gemma-4-12B-it}
ASSISTANT_MODEL_PATH=${GEMMA4_ASSISTANT_MODEL_PATH:-}
ASSISTANT_DEVICE=${GEMMA4_ASSISTANT_DEVICE:-cuda:0}
NUM_ASSISTANT_TOKENS=${GEMMA4_NUM_ASSISTANT_TOKENS:-4}
NUM_ASSISTANT_TOKENS_SCHEDULE=${GEMMA4_NUM_ASSISTANT_TOKENS_SCHEDULE:-heuristic}
PYTHON_BIN=${GEMMA4_TRANSFORMERS_PYTHON:-/usr/local/bin/python3}
HOST=${GEMMA4_HOST:-127.0.0.1}
PORT=${GEMMA4_PORT:-8002}
DEVICE=${GEMMA4_DEVICE:-cuda:1}
MAX_NEW_TOKENS=${GEMMA4_MAX_NEW_TOKENS:-64}
MAX_NEW_TOKENS_CAP=${GEMMA4_MAX_NEW_TOKENS_CAP:-256}
RUNTIME_ROOT=${GEMMA4_TRANSFORMERS_RUNTIME_ROOT:-"${USER_ROOT}/rt/g4_transformers"}
TRANSFORMERS_PATH=${GEMMA4_TRANSFORMERS_PATH:-"${WORK_ROOT}/.cache/uv/archive-v0/uqBGzfMtPGM3KZoA"}

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Gemma4 model directory does not exist: ${MODEL_PATH}" >&2
  echo "Run scripts/download_gemma4_12b.sh first, or set GEMMA4_MODEL_PATH." >&2
  exit 1
fi

if [[ ! -d "${TRANSFORMERS_PATH}/transformers" ]]; then
  echo "Transformers path does not contain a transformers package: ${TRANSFORMERS_PATH}" >&2
  echo "Set GEMMA4_TRANSFORMERS_PATH to a Transformers version that supports gemma4_unified." >&2
  exit 1
fi

mkdir -p \
  "${RUNTIME_ROOT}/scratch" \
  "${RUNTIME_ROOT}/hf" \
  "${RUNTIME_ROOT}/xdg" \
  "${RUNTIME_ROOT}/matplotlib" \
  "${RUNTIME_ROOT}/inductor"

export TMPDIR="${RUNTIME_ROOT}/scratch"
export HF_HOME="${RUNTIME_ROOT}/hf"
export HF_HUB_CACHE="${RUNTIME_ROOT}/hf/hub"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/xdg"
export MPLCONFIGDIR="${RUNTIME_ROOT}/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="${RUNTIME_ROOT}/inductor"
export PYTHONPATH="${TRANSFORMERS_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Gemma4 model       : ${MODEL_PATH}"
echo "API endpoint       : http://${HOST}:${PORT}/v1"
echo "Device             : ${DEVICE}"
if [[ -n "${ASSISTANT_MODEL_PATH}" ]]; then
  echo "Assistant model    : ${ASSISTANT_MODEL_PATH}"
  echo "Assistant device   : ${ASSISTANT_DEVICE}"
  echo "Assistant tokens   : ${NUM_ASSISTANT_TOKENS} (${NUM_ASSISTANT_TOKENS_SCHEDULE})"
fi
echo "Python             : ${PYTHON_BIN}"
echo "Transformers path  : ${TRANSFORMERS_PATH}"
echo "Runtime cache root : ${RUNTIME_ROOT}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/serve_gemma4_12b_transformers_api.py" \
  --model-path "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --assistant-model-path "${ASSISTANT_MODEL_PATH}" \
  --assistant-device "${ASSISTANT_DEVICE}" \
  --num-assistant-tokens "${NUM_ASSISTANT_TOKENS}" \
  --num-assistant-tokens-schedule "${NUM_ASSISTANT_TOKENS_SCHEDULE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-new-tokens-cap "${MAX_NEW_TOKENS_CAP}"
