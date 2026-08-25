"""Thin four-hook shim consumed by RoboTwin's unchanged policy loader.

RoboTwin owns episode iteration and observation production.  Its existing
policy contract also requires ``eval`` to submit the returned action, so the
single ``TASK_ENV.take_action`` call below is deliberately kept at this outer
integration edge; neither the generic bridge nor its simulator adapter drives
the environment.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from typing import Any, Mapping

from highlow.adapters import RoboTwinAdapter, RoboTwinPolicyHook, create_robotwin_bridge
from highlow.clients import JsonHttpInferenceClient


def _positive_float(value: Any, field_name: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None or value == "" or int(value) == 0:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{field_name} must be positive or zero to disable")
    return parsed


def _headers(value: Any, field_name: str) -> Mapping[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return {str(key): str(item) for key, item in value.items()}


def _http_client(
    config: Mapping[str, Any],
    *,
    prefix: str,
) -> JsonHttpInferenceClient:
    url_key = f"{prefix}_url"
    try:
        base_url = str(config[url_key]).strip()
    except KeyError as exc:
        raise ValueError(f"{url_key} is required") from exc
    if not base_url:
        raise ValueError(f"{url_key} must not be empty")

    common = {
        "timeout_s": _positive_float(
            config.get(f"{prefix}_timeout_s", 120.0),
            f"{prefix}_timeout_s",
        ),
        "headers": _headers(
            config.get(f"{prefix}_headers"),
            f"{prefix}_headers",
        ),
    }
    if prefix == "high_level":
        return JsonHttpInferenceClient(
            base_url,
            infer_endpoint="/v1/high-level/infer",
            health_endpoint="/health",
            capabilities_endpoint="/capabilities",
            **common,
        )
    return JsonHttpInferenceClient(
        base_url,
        infer_endpoint="/v1/infer",
        open_context_endpoint="/v1/episodes/open",
        reset_context_endpoint="/v1/episodes/reset",
        close_context_endpoint="/v1/episodes/close",
        health_endpoint="/health",
        capabilities_endpoint="/capabilities",
        **common,
    )


@dataclass(slots=True)
class _RemoteDualSystemPolicy:
    hook: RoboTwinPolicyHook
    task_id: str = ""
    seed: int | None = None
    simulator_version: str = ""
    source_fps: float = 25.0
    action_type: str = "ee"
    episode_metadata: Mapping[str, Any] = field(default_factory=dict)
    step_id: int = 0

    def start_if_needed(self, task_env: Any) -> None:
        if self.hook.ready:
            return
        get_instruction = getattr(task_env, "get_instruction", None)
        if not callable(get_instruction):
            raise TypeError("RoboTwin TASK_ENV must provide get_instruction()")
        instruction = str(get_instruction()).strip()
        if not instruction:
            raise ValueError("RoboTwin task instruction must not be empty")
        task_id = self.task_id or str(
            getattr(task_env, "task_name", type(task_env).__name__)
        ).strip()
        self.hook.start_episode(
            task_id or "robotwin-task",
            instruction,
            seed=self.seed,
            simulator_version=self.simulator_version,
            metadata=self.episode_metadata,
        )

    def close(self) -> None:
        """Close the current remote context, if any."""

        self.hook.finish()
        self.step_id = 0


def _best_effort_process_exit_close(model: _RemoteDualSystemPolicy) -> None:
    try:
        model.close()
    except Exception:
        # Interpreter shutdown cannot surface an actionable retry. Stateful
        # deployments should additionally enforce an episode lease/TTL for
        # clients terminated by SIGKILL or machine failure.
        pass


def get_model(usr_args: Mapping[str, Any]) -> _RemoteDualSystemPolicy:
    """Build remote inference clients without loading any model locally."""

    if not isinstance(usr_args, Mapping):
        raise TypeError("usr_args must be a mapping")
    # Private injection points keep the official shim easy to test in-process;
    # ordinary YAML configuration supplies URLs instead.
    high_client = usr_args.get("_high_client")
    low_client = usr_args.get("_low_client")
    if high_client is None:
        high_client = _http_client(usr_args, prefix="high_level")
    if low_client is None:
        low_client = _http_client(usr_args, prefix="low_level")

    interval = _optional_positive_int(
        usr_args.get("verifier_interval_steps"),
        "verifier_interval_steps",
    )
    adapter = RoboTwinAdapter()
    hook = RoboTwinPolicyHook(
        adapter,
        create_robotwin_bridge(
            high_client,
            low_client,
            verifier_interval_steps=interval,
        ),
    )
    raw_metadata = usr_args.get("episode_metadata", {})
    if not isinstance(raw_metadata, Mapping):
        raise TypeError("episode_metadata must be a mapping")
    seed_value = usr_args.get("seed")
    model = _RemoteDualSystemPolicy(
        hook=hook,
        task_id=str(usr_args.get("task_name", "")).strip(),
        seed=None if seed_value is None else int(seed_value),
        simulator_version=str(usr_args.get("simulator_version", "")),
        source_fps=_positive_float(usr_args.get("source_fps", 25.0), "source_fps"),
        action_type=str(usr_args.get("action_type", "ee")),
        episode_metadata={
            "integration": "robotwin_dual_system",
            **dict(raw_metadata),
        },
    )
    atexit.register(_best_effort_process_exit_close, model)
    return model


def eval(  # noqa: A001 - name is fixed by RoboTwin's policy protocol
    TASK_ENV: Any,
    model: _RemoteDualSystemPolicy,
    observation: Mapping[str, Any],
) -> None:
    """Run one remote dual-policy tick, then submit its native action."""

    if not isinstance(model, _RemoteDualSystemPolicy):
        raise TypeError("model must be created by get_model()")
    model.start_if_needed(TASK_ENV)
    action = model.hook.act(
        observation,
        model.step_id,
        model.step_id / model.source_fps,
    )
    # JSON transports restore arrays as lists. The official RoboTwin policy
    # contract historically submits a float32 NumPy action to TASK_ENV.
    import numpy as np

    action = np.asarray(action, dtype=np.float32)
    take_action = getattr(TASK_ENV, "take_action", None)
    if not callable(take_action):
        raise TypeError("RoboTwin TASK_ENV must provide take_action()")
    take_action(action, action_type=model.action_type)
    model.step_id += 1


def reset_model(model: _RemoteDualSystemPolicy) -> str:
    """Close inference episode state; RoboTwin remains owner of env reset."""

    if not isinstance(model, _RemoteDualSystemPolicy):
        raise TypeError("model must be created by get_model()")
    model.close()
    return "remote dual-system policy reset"


def encode_obs(observation: Mapping[str, Any], instruction: str) -> Any:
    """Offline helper exposing the canonical RoboTwin observation envelope."""

    adapter = RoboTwinAdapter()
    context = adapter.create_episode_context("offline", instruction)
    return adapter.encode_observation(
        context,
        observation,
        step_id=0,
        timestamp_s=0.0,
    )


__all__ = ["encode_obs", "eval", "get_model", "reset_model"]
