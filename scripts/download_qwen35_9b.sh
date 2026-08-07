#!/usr/bin/env bash
set -euo pipefail

# Download Qwen/Qwen3.5-9B into the project checkpoint directory.
# All cache/temp paths are kept under the user's CPFS workspace.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"
USER_ROOT="$(cd "${WORK_ROOT}/.." && pwd)"

REPO_ID=${QWEN35_REPO_ID:-Qwen/Qwen3.5-9B}
MODEL_DIR=${QWEN35_MODEL_PATH:-"${WORK_ROOT}/checkpoints/Qwen/Qwen3.5-9B"}
RUNTIME_ROOT=${QWEN35_DOWNLOAD_RUNTIME_ROOT:-"${USER_ROOT}/rt/q35_download"}
MAX_WORKERS=${QWEN35_DOWNLOAD_MAX_WORKERS:-1}

mkdir -p \
  "${MODEL_DIR}" \
  "${RUNTIME_ROOT}/scratch" \
  "${RUNTIME_ROOT}/hf_home" \
  "${RUNTIME_ROOT}/hf_hub"

export TMPDIR="${RUNTIME_ROOT}/scratch"
export HF_HOME="${RUNTIME_ROOT}/hf_home"
export HF_HUB_CACHE="${RUNTIME_ROOT}/hf_hub"

echo "Repo id      : ${REPO_ID}"
echo "Model target : ${MODEL_DIR}"
echo "HF cache     : ${HF_HUB_CACHE}"
echo "Temp dir     : ${TMPDIR}"
echo "Max workers  : ${MAX_WORKERS}"

hf download "${REPO_ID}" \
  --repo-type model \
  --local-dir "${MODEL_DIR}" \
  --max-workers "${MAX_WORKERS}"
