#!/usr/bin/env bash
set -euo pipefail

# Smoke-test the asynchronous high-level planner path with Gemma 4 12B.
# Start `serve_gemma4_12b_vllm.sh` (compat wrapper) or
# `serve_gemma4_12b_transformers_api.sh` first.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HY_VLA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${HY_VLA_DIR}/.." && pwd)"

export TASK_NAME=${TASK_NAME:-place_burger_fries}
export TEST_NUM=${TEST_NUM:-3}
export HANDOFF_TRIGGER=${HANDOFF_TRIGGER:-forced_boundary}
export FORCED_BOUNDARY_S=${FORCED_BOUNDARY_S:-2.0}
export MAX_FORCED_BOUNDARIES=${MAX_FORCED_BOUNDARIES:-10}
export STOP_AFTER_MEASUREMENT_HANDOFF=${STOP_AFTER_MEASUREMENT_HANDOFF:-False}

export HL_BACKEND=${HL_BACKEND:-gemma4_vllm}
export HL_PROMPT_MODE=${HL_PROMPT_MODE:-completion_status}
export HL_PLANNER_MODE=${HL_PLANNER_MODE:-vlm_plan_next}
export HL_ASYNC_ENABLED=${HL_ASYNC_ENABLED:-True}
export HL_ASYNC_DROP_STALE_S=${HL_ASYNC_DROP_STALE_S:-0.0}
export HL_VLM_MAX_IMAGES=${HL_VLM_MAX_IMAGES:-1}
export HL_VLM_MAX_NEW_TOKENS=${HL_VLM_MAX_NEW_TOKENS:-48}

export GEMMA4_SERVER_URL=${GEMMA4_SERVER_URL:-http://127.0.0.1:8002}
export GEMMA4_MODEL_NAME=${GEMMA4_MODEL_NAME:-gemma-4-12B-it}
export GEMMA4_REQUEST_TIMEOUT_S=${GEMMA4_REQUEST_TIMEOUT_S:-120}
export GEMMA4_JPEG_QUALITY=${GEMMA4_JPEG_QUALITY:-90}
export GEMMA4_IMAGE_SELECTION=${GEMMA4_IMAGE_SELECTION:-latest}
export GEMMA4_IMAGE_MAX_SIDE=${GEMMA4_IMAGE_MAX_SIDE:-0}
export GEMMA4_OUTPUT_SCHEMA=${GEMMA4_OUTPUT_SCHEMA:-completion_status}
export GEMMA4_WAIT_FOR_SERVER_S=${GEMMA4_WAIT_FOR_SERVER_S:-300}

if [[ -z "${LOG_DIR:-}" ]]; then
  export LOG_DIR="${WORK_ROOT}/profile_results/robotwin_${TASK_NAME}/async_gemma4_12b_$(date +%Y%m%d_%H%M%S)"
fi

wait_for_gemma4_server() {
  local server_url="$1"
  local timeout_s="$2"
  /usr/local/bin/python3 - "$server_url" "$timeout_s" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

server_url = sys.argv[1].rstrip("/")
timeout_s = max(0.0, float(sys.argv[2]))
if server_url.endswith("/v1/chat/completions"):
    models_url = server_url[: -len("/chat/completions")] + "/models"
elif server_url.endswith("/v1"):
    models_url = server_url + "/models"
else:
    models_url = server_url + "/v1/models"

deadline = time.monotonic() + timeout_s
last_error = None
while True:
    try:
        with urllib.request.urlopen(models_url, timeout=5.0) as response:
            if 200 <= response.status < 300:
                body = response.read(65536).decode("utf-8", errors="replace")
                data = json.loads(body)
                model_ids = [
                    item.get("id")
                    for item in data.get("data", [])
                    if isinstance(item, dict)
                ]
                print(
                    "Gemma4 vLLM server ready: "
                    + (", ".join(str(model_id) for model_id in model_ids) or models_url)
                )
                sys.exit(0)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        last_error = exc

    if time.monotonic() >= deadline:
        print(
            f"Gemma4 vLLM server was not ready at {models_url}: {last_error}",
            file=sys.stderr,
        )
        sys.exit(1)
    time.sleep(2.0)
PY
}

if [[ "${GEMMA4_WAIT_FOR_SERVER_S}" != "0" ]]; then
  echo "Waiting for Gemma4 vLLM server at ${GEMMA4_SERVER_URL} (${GEMMA4_WAIT_FOR_SERVER_S}s timeout)"
  if ! wait_for_gemma4_server "${GEMMA4_SERVER_URL}" "${GEMMA4_WAIT_FOR_SERVER_S}"; then
    echo "Gemma4 vLLM server is not reachable." >&2
    echo "Start it in another terminal first:" >&2
    echo "  cd ${HY_VLA_DIR}" >&2
    echo "  bash scripts/serve_gemma4_12b_vllm.sh" >&2
    echo "Then rerun this eval script, or set GEMMA4_SERVER_URL to the active server." >&2
    exit 1
  fi
fi

exec bash "${SCRIPT_DIR}/eval_robotwin_burger_fries_hierarchical.sh"
