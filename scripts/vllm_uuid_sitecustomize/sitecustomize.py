"""Resolve UUID-form CUDA_VISIBLE_DEVICES values for vLLM 0.19.1.

The platform lease exposes GPU UUIDs, which PyTorch supports directly. vLLM
0.19.1 incorrectly coerces those UUIDs to integers when querying NVML. This
module is imported automatically in the API server and vLLM subprocesses via
the project-local PYTHONPATH entry set by the service script.
"""

from __future__ import annotations

import os


def _uuid_to_nvml_index(device_uuid: str) -> int:
    from vllm.utils.import_utils import import_pynvml

    pynvml = import_pynvml()
    target = device_uuid.strip().lower()
    pynvml.nvmlInit()
    try:
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            current = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(current, bytes):
                current = current.decode("utf-8")
            if str(current).lower() == target:
                return index
    finally:
        pynvml.nvmlShutdown()
    raise RuntimeError(
        "vLLM could not map CUDA_VISIBLE_DEVICES UUID "
        f"{device_uuid!r} to an NVML device."
    )


def _install_uuid_compatibility() -> None:
    from vllm.platforms import current_platform

    platform_cls = type(current_platform)
    original_resolver = current_platform.device_id_to_physical_device_id
    device_env = current_platform.device_control_env_var

    @classmethod
    def resolve_visible_device(cls, device_id: int) -> int:
        visible = os.environ.get(device_env, "")
        if not visible:
            return original_resolver(device_id)

        device_token = visible.split(",")[device_id].strip()
        if device_token.lower().startswith(("gpu-", "mig-")):
            return _uuid_to_nvml_index(device_token)
        return original_resolver(device_id)

    platform_cls.device_id_to_physical_device_id = resolve_visible_device


def _install_tcp_zmq_compatibility() -> None:
    """Use loopback TCP for vLLM local ZMQ channels instead of IPC sockets.

    vLLM normally uses Unix-domain IPC sockets for colocated engine/client
    communication. CPFS supports normal files but not these socket binds, and
    `/tmp` is intentionally not used in this workspace. Loopback TCP keeps the
    channel local without requiring a socket-capable filesystem path.
    """

    from vllm.utils import network_utils
    from vllm.v1 import utils as v1_utils

    def get_open_zmq_tcp_path() -> str:
        return network_utils.get_tcp_uri("127.0.0.1", network_utils.get_open_port())

    def get_engine_client_zmq_addr(
        local_only: bool, host: str, port: int = 0
    ) -> str:
        del local_only
        return network_utils.get_tcp_uri(host, port or network_utils.get_open_port())

    network_utils.get_open_zmq_ipc_path = get_open_zmq_tcp_path
    v1_utils.get_open_zmq_ipc_path = get_open_zmq_tcp_path
    v1_utils.get_engine_client_zmq_addr = get_engine_client_zmq_addr

    # These modules import the function object at module import time. Import
    # and patch them here so later engine setup always sees the TCP variant.
    from vllm.v1.engine import coordinator as engine_coordinator
    from vllm.v1.engine import utils as engine_utils

    engine_coordinator.get_engine_client_zmq_addr = get_engine_client_zmq_addr
    engine_utils.get_engine_client_zmq_addr = get_engine_client_zmq_addr
    engine_utils.get_open_zmq_ipc_path = get_open_zmq_tcp_path


_install_uuid_compatibility()
_install_tcp_zmq_compatibility()
