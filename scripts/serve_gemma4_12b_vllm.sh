#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper for the Gemma 4 12B server.
# The original vLLM path is not usable with the current stack, so this
# forwards to the Transformers-based OpenAI-compatible server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/serve_gemma4_12b_transformers_api.sh" "$@"
