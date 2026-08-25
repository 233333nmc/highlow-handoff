#!/usr/bin/env python3
"""Run the pinned official RoboTwin evaluator with a fail-closed episode cap.

The official source remains untouched.  This launcher imports its
``script/eval_policy.py`` as an external module and replaces that module's
``eval_policy`` reference only for the lifetime of this process.  The wrapper
verifies the pinned checkout and the upstream ``test_num=100`` contract before
forcing the requested number of episodes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from types import ModuleType
from typing import Any, Callable, Mapping
import uuid


EXPECTED_ROBOTWIN_COMMIT = "c3ddfa8b97d5519efa828b075999bd0006778e5e"
UPSTREAM_TEST_NUM = 100
RESULT_SCHEMA_VERSION = "highlow.robotwin-limited-eval.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_CUDA_DEVICE_PATTERN = re.compile(r"^cuda:([0-9]+)$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _validate_pinned_checkout(root: Path) -> tuple[Path, dict[str, str]]:
    eval_policy_path = root / "script" / "eval_policy.py"
    if not eval_policy_path.is_file():
        raise FileNotFoundError(
            f"pinned evaluator does not exist: {eval_policy_path}"
        )
    commit = _git(root, "rev-parse", "HEAD")
    if commit != EXPECTED_ROBOTWIN_COMMIT:
        raise RuntimeError(
            "RoboTwin checkout is not the pinned revision: "
            f"expected {EXPECTED_ROBOTWIN_COMMIT}, found {commit}"
        )
    tracked_changes = _git(root, "status", "--short", "--untracked-files=no")
    if tracked_changes:
        raise RuntimeError(
            "pinned RoboTwin checkout has tracked worktree modifications: "
            f"{tracked_changes}"
        )
    expected_blob = _git(root, "rev-parse", "HEAD:script/eval_policy.py")
    worktree_blob = _git(root, "hash-object", str(eval_policy_path))
    if worktree_blob != expected_blob:
        raise RuntimeError(
            "pinned script/eval_policy.py has tracked worktree modifications"
        )
    digest = hashlib.sha256(eval_policy_path.read_bytes()).hexdigest()
    return eval_policy_path, {
        "commit": commit,
        "eval_policy_git_blob": expected_blob,
        "eval_policy_sha256": digest,
    }


def _validate_name(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized or _NAME_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            f"{field_name} must contain only letters, digits, and '_' characters"
        )
    return normalized


def _load_official_module(eval_policy_path: Path) -> ModuleType:
    module_name = "_highlow_pinned_robotwin_eval_policy"
    spec = importlib.util.spec_from_file_location(module_name, eval_policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create an import spec for {eval_policy_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _parse_official_config(module: ModuleType, config_path: Path) -> dict[str, Any]:
    parse_config = getattr(module, "parse_args_and_config", None)
    if not callable(parse_config):
        raise RuntimeError("pinned evaluator does not expose parse_args_and_config()")
    original_argv = sys.argv
    sys.argv = [str(getattr(module, "__file__", "eval_policy.py")), "--config", str(config_path)]
    try:
        value = parse_config()
    finally:
        sys.argv = original_argv
    if not isinstance(value, Mapping):
        raise RuntimeError("official config parser did not return a mapping")
    parsed = dict(value)
    for field_name in ("high_level_url", "low_level_url"):
        field_value = str(parsed.get(field_name, "")).strip()
        if not field_value.startswith(("http://", "https://")):
            raise RuntimeError(
                f"deploy policy config must provide an absolute {field_name}"
            )
    return parsed


def _validate_smoke_task_config(module: ModuleType, config_path: Path) -> None:
    yaml_module = getattr(module, "yaml", None)
    safe_load = getattr(yaml_module, "safe_load", None)
    if not callable(safe_load):
        raise RuntimeError("pinned evaluator does not expose yaml.safe_load()")
    with config_path.open("r", encoding="utf-8") as stream:
        value = safe_load(stream)
    if not isinstance(value, Mapping):
        raise RuntimeError("RoboTwin task config must be a YAML mapping")
    for field_name in ("collect_data", "eval_video_log"):
        if value.get(field_name) is not False:
            raise RuntimeError(
                f"smoke task config must set {field_name}: false; "
                f"found {value.get(field_name)!r} in {config_path}"
            )


class _EvaluationCapture:
    def __init__(self, original: Callable[..., Any], forced_test_num: int) -> None:
        self.original = original
        self.forced_test_num = forced_test_num
        self.signature = inspect.signature(original)
        try:
            test_parameter = self.signature.parameters["test_num"]
        except KeyError as exc:
            raise RuntimeError(
                "pinned eval_policy() no longer has a test_num parameter"
            ) from exc
        if test_parameter.default != UPSTREAM_TEST_NUM:
            raise RuntimeError(
                "pinned eval_policy() default changed: expected test_num=100, "
                f"found {test_parameter.default!r}"
            )
        self.calls: list[dict[str, Any]] = []

    def callable(self) -> Callable[..., Any]:
        @functools.wraps(self.original)
        def limited(*args: Any, **kwargs: Any) -> Any:
            if self.calls:
                raise RuntimeError("official main() invoked eval_policy() more than once")
            bound = self.signature.bind(*args, **kwargs)
            bound.apply_defaults()
            upstream_requested = bound.arguments.get("test_num")
            if upstream_requested != UPSTREAM_TEST_NUM:
                raise RuntimeError(
                    "official main() no longer requests test_num=100; "
                    f"received {upstream_requested!r}"
                )
            task_env = next(
                (
                    value
                    for name, value in bound.arguments.items()
                    if name.lower() == "task_env"
                ),
                None,
            )
            if task_env is None:
                raise RuntimeError("cannot locate TASK_ENV in eval_policy() arguments")
            bound.arguments["test_num"] = self.forced_test_num
            result = self.original(*bound.args, **bound.kwargs)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError(
                    "pinned eval_policy() must return (next_seed, success_count)"
                )
            next_seed, returned_successes = result
            completed = getattr(task_env, "test_num", None)
            environment_successes = getattr(task_env, "suc", None)
            if completed != self.forced_test_num:
                raise RuntimeError(
                    "official evaluator completed an unexpected number of episodes: "
                    f"expected {self.forced_test_num}, found {completed!r}"
                )
            if returned_successes != environment_successes:
                raise RuntimeError(
                    "eval_policy() return value disagrees with TASK_ENV.suc"
                )
            if (
                isinstance(returned_successes, bool)
                or not isinstance(returned_successes, (int, float))
                or not math.isfinite(float(returned_successes))
                or float(returned_successes) != int(returned_successes)
                or not 0 <= int(returned_successes) <= self.forced_test_num
            ):
                raise RuntimeError(
                    f"invalid success count returned by evaluator: {returned_successes!r}"
                )
            record = {
                "upstream_requested_test_num": int(upstream_requested),
                "forced_test_num": self.forced_test_num,
                "completed_test_num": int(completed),
                "success_count": int(returned_successes),
                "success_rate": int(returned_successes) / self.forced_test_num,
                "next_seed": int(next_seed),
            }
            self.calls.append(record)
            return result

        return limited


def _run_sapien_test() -> Any:
    try:
        test_render = importlib.import_module("test_render")
        sapien_test = getattr(test_render, "Sapien_TEST")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("pinned RoboTwin does not expose test_render.Sapien_TEST") from exc
    if not callable(sapien_test):
        raise RuntimeError("test_render.Sapien_TEST is not callable")
    try:
        probe = sapien_test()
    except SystemExit as exc:
        # The pinned implementation calls exit() from a broad exception block,
        # which otherwise reports status 0. Convert it into a hard failure.
        raise RuntimeError("Sapien_TEST terminated the process") from exc
    if probe is None:
        raise RuntimeError("Sapien_TEST did not construct an environment")
    return probe


def _verify_scene_render_device(
    probe: Any,
    renderer_binding: dict[str, Any],
) -> None:
    try:
        actual_device = probe.scene.render_system.device
        actual_pci = actual_device.pci_string
        actual_cuda_id = actual_device.cuda_id
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "Sapien_TEST did not expose its actual scene render device"
        ) from exc

    expected_pci = renderer_binding.get("pci_string")
    expected_cuda_id = renderer_binding.get("logical_cuda_index")
    if not expected_pci or not actual_pci:
        raise RuntimeError("SAPIEN render-device PCI address must not be empty")
    if str(actual_pci).lower() != str(expected_pci).lower():
        raise RuntimeError(
            "SAPIEN scene renderer used the wrong PCI device: "
            f"expected {expected_pci}, found {actual_pci}"
        )
    if actual_cuda_id != expected_cuda_id:
        raise RuntimeError(
            "SAPIEN scene renderer used the wrong logical CUDA device: "
            f"expected {expected_cuda_id}, found {actual_cuda_id}"
        )
    if not actual_device.is_cuda() or not actual_device.can_render():
        raise RuntimeError("SAPIEN scene renderer device is not CUDA render-capable")

    renderer_binding["scene_device"] = {
        "name": str(actual_device.name),
        "logical_cuda_index": int(actual_cuda_id),
        "pci_string": str(actual_pci),
        "can_render": True,
    }


def _bind_sapien_renderer(device_alias: str) -> dict[str, Any]:
    """Bind RoboTwin's legacy no-argument renderer to one CUDA device.

    RoboTwin imports ``sapien.core`` and constructs ``SapienRenderer()``
    without arguments.  SAPIEN 3's compatibility wrapper discards renderer
    keyword arguments, so the binding must happen before importing the
    official evaluator.  Replacing the wrapper class in this process leaves
    the pinned simulator source untouched while preserving its public methods.
    """

    normalized = device_alias.strip().lower()
    match = _CUDA_DEVICE_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError("render-device must use the form cuda:<logical-index>")

    import sapien
    import sapien.core as sapien_core
    import torch

    logical_index = int(match.group(1))
    visible_count = torch.cuda.device_count()
    if visible_count != 1:
        raise RuntimeError(
            "the bounded RoboTwin process must see exactly one CUDA device; "
            f"found {visible_count}"
        )
    if logical_index != 0:
        raise RuntimeError(
            "the isolated RoboTwin process must bind its sole logical device as cuda:0"
        )

    selected_device = sapien.Device(normalized)
    if not selected_device.is_cuda() or not selected_device.can_render():
        raise RuntimeError(f"SAPIEN device cannot render: {selected_device!r}")
    if selected_device.cuda_id != logical_index:
        raise RuntimeError(
            "SAPIEN resolved a different CUDA device: "
            f"expected {logical_index}, found {selected_device.cuda_id}"
        )

    wrapper_renderer = sapien.SapienRenderer
    if wrapper_renderer is not sapien_core.SapienRenderer:
        raise RuntimeError("sapien and sapien.core expose different renderer wrappers")
    raw_renderer = sapien.render.SapienRenderer

    class BoundSapienRenderer(wrapper_renderer):
        def __init__(self, **_ignored: Any) -> None:
            raw_renderer.__init__(self, selected_device)

    BoundSapienRenderer.__name__ = "BoundSapienRenderer"
    sapien.SapienRenderer = BoundSapienRenderer
    sapien_core.SapienRenderer = BoundSapienRenderer
    return {
        "requested_alias": normalized,
        "logical_cuda_index": logical_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "name": str(selected_device.name),
        "pci_string": selected_device.pci_string,
        "can_render": True,
    }


def run_limited_eval(args: argparse.Namespace) -> dict[str, Any]:
    robotwin_root = args.robotwin_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not robotwin_root.is_dir():
        raise NotADirectoryError(f"RoboTwin root does not exist: {robotwin_root}")
    if not config_path.is_file():
        raise FileNotFoundError(f"deploy policy config does not exist: {config_path}")
    task_name = _validate_name(args.task_name, "task-name")
    task_config = _validate_name(args.task_config, "task-config")
    task_config_path = robotwin_root / "task_config" / f"{task_config}.yml"
    if not task_config_path.is_file():
        raise FileNotFoundError(
            "task config must be installed in the pinned RoboTwin checkout: "
            f"{task_config_path}"
        )

    eval_policy_path, revision = _validate_pinned_checkout(robotwin_root)
    search_paths = (
        REPOSITORY_ROOT,
        robotwin_root,
        robotwin_root / "script",
        robotwin_root / "policy",
        robotwin_root / "description" / "utils",
    )
    for search_path in reversed(search_paths):
        value = str(search_path)
        if value not in sys.path:
            sys.path.insert(0, value)

    original_cwd = Path.cwd()
    run_tag = re.sub(
        r"[^A-Za-z0-9_-]",
        "-",
        os.environ.get("SLURM_JOB_ID", f"pid-{os.getpid()}"),
    )
    ckpt_setting = f"highlow-http-smoke-{run_tag}-{uuid.uuid4().hex[:8]}"
    try:
        os.chdir(robotwin_root)
        renderer_binding = _bind_sapien_renderer(args.render_device)
        module = _load_official_module(eval_policy_path)
        original_eval_policy = getattr(module, "eval_policy", None)
        official_main = getattr(module, "main", None)
        if not callable(original_eval_policy) or not callable(official_main):
            raise RuntimeError("pinned evaluator must expose main() and eval_policy()")
        capture = _EvaluationCapture(original_eval_policy, args.test_num)
        module.eval_policy = capture.callable()

        _validate_smoke_task_config(module, task_config_path)
        sapien_probe = _run_sapien_test()
        _verify_scene_render_device(sapien_probe, renderer_binding)
        import sapien

        renderer_binding["summary"] = str(sapien.render.get_device_summary())
        usr_args = _parse_official_config(module, config_path)
        # Policy/model identifiers are metadata in this remote deployment; no
        # checkpoint is loaded inside the simulator process.
        usr_args.pop("eval_num_episodes", None)
        usr_args.update(
            {
                "task_name": task_name,
                "task_config": task_config,
                "instruction_type": args.instruction_type,
                "seed": args.seed,
                "policy_name": "robotwin_dual_system",
                "ckpt_setting": ckpt_setting,
            }
        )
        official_main(usr_args)
        if len(capture.calls) != 1:
            raise RuntimeError(
                "official main() returned without exactly one eval_policy() invocation"
            )
    finally:
        os.chdir(original_cwd)

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed",
        "authoritative": True,
        "robotwin": {
            "root": str(robotwin_root),
            "eval_policy": str(eval_policy_path),
            **revision,
        },
        "request": {
            "config": str(config_path),
            "task_name": task_name,
            "task_config": task_config,
            "task_config_path": str(task_config_path),
            "instruction_type": args.instruction_type,
            "seed": args.seed,
            "test_num": args.test_num,
            "render_device": args.render_device,
            "official_ckpt_setting": ckpt_setting,
        },
        "renderer": renderer_binding,
        "evaluation": capture.calls[0],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="robotwin_dual_system/deploy_policy.yml",
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument(
        "--instruction-type",
        choices=("seen", "unseen"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--test-num", type=_positive_int, default=1)
    parser.add_argument(
        "--render-device",
        required=True,
        help="Isolated SAPIEN CUDA device alias; the launcher passes cuda:0",
    )
    parser.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.result_json = args.result_json.expanduser().resolve()
    started_at = _utc_now()
    started = time.perf_counter()
    try:
        payload = run_limited_eval(args)
    except (Exception, SystemExit) as exc:
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "failed",
            "authoritative": True,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_s": time.perf_counter() - started,
            "request": {
                "robotwin_root": str(args.robotwin_root),
                "config": str(args.config),
                "task_name": args.task_name,
                "task_config": args.task_config,
                "instruction_type": args.instruction_type,
                "seed": args.seed,
                "test_num": args.test_num,
                "render_device": args.render_device,
            },
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        _write_json(args.result_json, payload)
        print(
            f"RoboTwin limited evaluation failed; result written to {args.result_json}",
            file=sys.stderr,
        )
        return 1

    payload["started_at"] = started_at
    payload["finished_at"] = _utc_now()
    payload["elapsed_s"] = time.perf_counter() - started
    _write_json(args.result_json, payload)
    print(f"RoboTwin limited evaluation passed; result written to {args.result_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
