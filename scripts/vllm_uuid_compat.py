"""Entrypoint for vLLM with project-local CUDA UUID compatibility enabled.

``serve_qwen36_vllm_mtp.sh`` prepends ``vllm_uuid_sitecustomize`` to
``PYTHONPATH``. Python imports that directory's ``sitecustomize`` module in
both the API server and vLLM's model-inspection subprocess before vLLM loads.
"""

from __future__ import annotations


def main() -> None:
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
