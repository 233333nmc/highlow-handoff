"""Deployment composition for RoboTwin's existing HyVLA wrapper."""

from __future__ import annotations

from typing import Any, Mapping


def build_official_hyvla_policy(config: Mapping[str, Any]) -> Any:
    """Construct the existing wrapper without importing it during discovery."""

    from robotwin_eval.policy_wrapper import build_policy

    return build_policy(dict(config))


def create_robotwin_hyvla_backend(
    policy_config: Mapping[str, Any] | None = None,
    **backend_kwargs: Any,
) -> Any:
    """Inject the official wrapper into the generic low-level HyVLA plugin."""

    from highlow.inference.low_level.plugins.hyvla import HyVLABackend

    return HyVLABackend(
        policy_config,
        policy_factory=build_official_hyvla_policy,
        **backend_kwargs,
    )


__all__ = ["build_official_hyvla_policy", "create_robotwin_hyvla_backend"]
