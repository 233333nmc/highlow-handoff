#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"

TEST1_ROOT="${WORK_ROOT}/profile_results/gemma_fixed_next_12tasks_n5_20260818"
TEST2_ROOT="${WORK_ROOT}/profile_results/gemma_fixed_next_test2"
CALIBRATION_FILE="${TEST1_ROOT}/boundary_calibration.tsv"

if [[ ! -f "${CALIBRATION_FILE}" ]]; then
  echo "Missing test1 calibration file: ${CALIBRATION_FILE}" >&2
  exit 2
fi

cd "${HY_VLA_DIR}"
RUN_ROOT="${TEST2_ROOT}" \
BOUNDARY_CALIBRATION_FILE="${CALIBRATION_FILE}" \
CALIBRATE_BOUNDARIES=False \
TEST_NUM="${TEST_NUM:-5}" \
ONE_OBJECT_ONLY=True \
bash "${SCRIPT_DIR}/run_gemma_fixed_next_12tasks_n5.sh"
