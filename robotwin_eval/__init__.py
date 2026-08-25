# coding=utf-8
# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hy-VLA RoboTwin evaluation package.

Drop this directory under ``robotwin/policy/hy_vla/`` (or any other
RoboTwin policy slot) and point ``--config`` at ``deploy_policy.yml``.
RoboTwin imports the four hooks from ``deploy_policy``:
``encode_obs``, ``get_model``, ``eval``, ``reset_model``.
Imports are lazy so model-serving composition can load ``policy_wrapper``
without first importing the legacy simulator hook and hierarchical stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "encode_obs",
    "eval",
    "get_model",
    "reset_model",
    "HyVLAPolicyWrapper",
    "build_policy",
]


def __getattr__(name: str) -> Any:
    if name in {"encode_obs", "eval", "get_model", "reset_model"}:
        deploy_policy = import_module(".deploy_policy", __name__)
        return getattr(deploy_policy, name)
    if name in {"HyVLAPolicyWrapper", "build_policy"}:
        policy_wrapper = import_module(".policy_wrapper", __name__)
        return getattr(policy_wrapper, name)
    raise AttributeError(name)
