from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # Core/service tests intentionally need no NumPy.
    np = None

from highlow.contracts import (
    ActionEnvelope,
    EpisodeContext,
    EpisodeResult,
    Modality,
    ObservationEnvelope,
    RequestContext,
)
from highlow.inference.low_level import (
    BackendRegistrationError,
    BackendRegistry,
    ContextMode,
    LifecycleError,
    LowLevelInferenceRequest,
    LowLevelService,
    PolicyManifest,
    PolicyPrediction,
    PredictionContractError,
    PredictionMode,
    TemporalMode,
    UnknownBackendError,
    create_app,
    create_policy_app_from_env,
)
from highlow.inference.common import JsonAsgiApp
from highlow.inference.low_level.plugins.hyvla import (
    END_EFFECTOR_STATE_MODALITY,
    HEAD_RGB_MODALITY,
    HYVLA_MANIFEST,
    LEFT_RGB_MODALITY,
    RGB_SCHEMA_ID,
    RIGHT_RGB_MODALITY,
    STATE_ACTION_SCHEMA_ID,
    HyVLABackend,
    HyVLAInputAdapter,
    register_hyvla,
)


def _observation(episode_id: str, step_id: int = 0) -> ObservationEnvelope:
    if np is None:
        image = [
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
        ]
        state = list(range(16))
    else:
        image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        state = np.arange(16, dtype=np.float32)
    modalities = {
        name: Modality(
            name=name,
            schema_id=RGB_SCHEMA_ID,
            data=image.copy() if hasattr(image, "copy") else image,
        )
        for name in (HEAD_RGB_MODALITY, LEFT_RGB_MODALITY, RIGHT_RGB_MODALITY)
    }
    modalities[END_EFFECTOR_STATE_MODALITY] = Modality(
        name=END_EFFECTOR_STATE_MODALITY,
        schema_id=STATE_ACTION_SCHEMA_ID,
        data=state,
    )
    return ObservationEnvelope(
        episode_id=episode_id,
        step_id=step_id,
        timestamp_s=float(step_id),
        modalities=modalities,
    )


class RecordingBackend:
    def __init__(self, max_concurrent_episodes: int = 1) -> None:
        self.manifest = PolicyManifest(
            policy_id="recording",
            policy_version="1",
            action_schema_ids=(STATE_ACTION_SCHEMA_ID,),
            max_concurrent_episodes=max_concurrent_episodes,
        )
        self.calls: list[object] = []
        self.episodes: set[str] = set()
        self.results: list[object] = []

    def open(self) -> None:
        self.calls.append("open")

    def begin_episode(self, episode: EpisodeContext) -> None:
        self.calls.append(("begin", episode.episode_id))
        self.episodes.add(episode.episode_id)

    def infer(self, request: LowLevelInferenceRequest) -> PolicyPrediction:
        self.calls.append(("infer", request.episode_id, request.step_id))
        return PolicyPrediction(
            action=ActionEnvelope(
                episode_id=request.episode_id,
                step_id=request.step_id,
                schema_id=STATE_ACTION_SCHEMA_ID,
                data=[request.step_id],
            )
        )

    def end_episode(self, episode: EpisodeContext, result=None) -> None:
        self.calls.append(("end", episode.episode_id))
        self.results.append(result)
        self.episodes.remove(episode.episode_id)

    def close(self) -> None:
        self.calls.append("close")


class PolicyManifestTests(unittest.TestCase):
    def test_manifest_has_explicit_semantics_and_frozen_metadata(self) -> None:
        manifest = PolicyManifest(
            policy_id="example",
            policy_version="2",
            context_mode="stateless",
            temporal_mode="history",
            prediction_mode="action_chunk",
            capabilities={"streaming": True},
        )
        self.assertEqual(manifest.backend_id, "example")
        self.assertIs(manifest.context_mode, ContextMode.STATELESS)
        self.assertIs(manifest.temporal_mode, TemporalMode.HISTORY)
        self.assertIs(manifest.prediction_mode, PredictionMode.ACTION_CHUNK)
        with self.assertRaises(TypeError):
            manifest.capabilities["streaming"] = False
        with self.assertRaises(FrozenInstanceError):
            manifest.policy_id = "changed"

    def test_manifest_rejects_ambiguous_modalities(self) -> None:
        with self.assertRaises(ValueError):
            PolicyManifest(
                policy_id="bad",
                policy_version="1",
                required_modalities=("camera",),
                optional_modalities=("camera",),
            )


class BackendRegistryTests(unittest.TestCase):
    def test_register_create_duplicate_and_unregister(self) -> None:
        registry = BackendRegistry()
        factory = lambda **kwargs: RecordingBackend(**kwargs)
        registry.register(" RECORDING ", factory)
        self.assertEqual(registry.names(), ("recording",))
        backend = registry.create("recording", max_concurrent_episodes=2)
        self.assertEqual(backend.manifest.max_concurrent_episodes, 2)
        with self.assertRaises(BackendRegistrationError):
            registry.register("recording", factory)
        registry.unregister("recording")
        with self.assertRaises(UnknownBackendError):
            registry.create("recording")

    def test_hyvla_registration_remains_opt_in(self) -> None:
        registry = BackendRegistry()
        register_hyvla(registry)
        backend = registry.create("hyvla", policy_factory=lambda config: FakePolicy())
        self.assertIsInstance(backend, HyVLABackend)


class LowLevelServiceTests(unittest.TestCase):
    def test_service_routes_multiple_episodes_up_to_manifest_limit(self) -> None:
        backend = RecordingBackend(max_concurrent_episodes=2)
        service = LowLevelService(backend)
        first = EpisodeContext(task_id="a", instruction="first", episode_id="ep-a")
        second = EpisodeContext(task_id="b", instruction="second", episode_id="ep-b")
        first_context = service.begin_episode(first)
        second_context = service.begin_episode(second)

        first_prediction = service.infer(
            LowLevelInferenceRequest(_observation("ep-a", step_id=3))
        )
        second_prediction = second_context.infer(
            LowLevelInferenceRequest(_observation("ep-b", step_id=4))
        )
        self.assertEqual(first_prediction.action.data, [3])
        self.assertEqual(second_prediction.action.data, [4])
        self.assertEqual(set(service.active_contexts), {"ep-a", "ep-b"})

        with self.assertRaises(LifecycleError):
            service.begin_episode(
                EpisodeContext(task_id="c", instruction="third", episode_id="ep-c")
            )
        with self.assertRaises(LifecycleError):
            service.end_episode()

        first_context.finish()
        service.reset_context("ep-b")
        service.end_episode(episode_id="ep-b")
        service.close()
        self.assertEqual(backend.calls.count("open"), 1)
        self.assertEqual(backend.calls[-1], "close")

    def test_service_enforces_episode_and_api_contracts(self) -> None:
        service = LowLevelService(RecordingBackend())
        context = service.begin_episode(
            EpisodeContext(task_id="task", instruction="do it", episode_id="expected")
        )
        with self.assertRaises(LifecycleError):
            service.infer(LowLevelInferenceRequest(_observation("other")))
        with self.assertRaises(PredictionContractError):
            context.infer(
                LowLevelInferenceRequest(
                    _observation("expected"),
                    context=RequestContext(api_version="v999"),
                )
            )
        context.finish()
        with self.assertRaises(LifecycleError):
            context.infer(LowLevelInferenceRequest(_observation("expected")))
        service.close()

    def test_episode_open_infer_and_close_are_retry_safe(self) -> None:
        backend = RecordingBackend()
        service = LowLevelService(backend)
        episode = EpisodeContext(
            task_id="task", instruction="do it", episode_id="retry-episode"
        )
        first_context = service.open_context(episode)
        self.assertIs(service.open_context(episode), first_context)
        request = LowLevelInferenceRequest(_observation("retry-episode", 7))
        first = service.infer(request)
        replay = service.infer(request)
        self.assertIs(replay, first)
        self.assertEqual(
            [call for call in backend.calls if isinstance(call, tuple) and call[0] == "infer"],
            [("infer", "retry-episode", 7)],
        )
        service.close_context("retry-episode")
        service.close_context("retry-episode")
        service.close()

    def test_failed_close_keeps_context_as_non_inferable_retry_tombstone(self) -> None:
        class FlakyEndBackend(RecordingBackend):
            def __init__(self) -> None:
                super().__init__()
                self.end_attempts = 0

            def end_episode(self, episode, result=None) -> None:
                self.end_attempts += 1
                if self.end_attempts == 1:
                    raise TimeoutError("transient teardown failure")
                super().end_episode(episode, result)

        backend = FlakyEndBackend()
        service = LowLevelService(backend)
        episode = EpisodeContext(
            task_id="task", instruction="do it", episode_id="close-tombstone"
        )
        service.open_context(episode)
        result = EpisodeResult(
            episode_id=episode.episode_id,
            success=True,
            terminated=True,
            step_count=7,
        )

        with self.assertRaises(TimeoutError):
            service.close_context(episode.episode_id, result)
        self.assertIn(episode.episode_id, service.active_contexts)
        with self.assertRaises(LifecycleError):
            service.infer(LowLevelInferenceRequest(_observation(episode.episode_id)))

        service.close_context(episode.episode_id)
        self.assertEqual(backend.end_attempts, 2)
        self.assertEqual(backend.results, [result])
        self.assertNotIn(episode.episode_id, service.active_contexts)
        service.close()

    def test_reset_retry_resumes_begin_phase_without_repeating_end(self) -> None:
        class FlakyBeginBackend(RecordingBackend):
            def __init__(self) -> None:
                super().__init__()
                self.begin_attempts = 0

            def begin_episode(self, episode) -> None:
                self.begin_attempts += 1
                # Initial begin succeeds; the first post-reset begin fails.
                if self.begin_attempts == 2:
                    raise TimeoutError("transient begin failure")
                super().begin_episode(episode)

        backend = FlakyBeginBackend()
        service = LowLevelService(backend)
        episode = EpisodeContext(
            task_id="task", instruction="do it", episode_id="reset-phases"
        )
        service.open_context(episode)

        with self.assertRaises(TimeoutError):
            service.reset_context(episode.episode_id)
        with self.assertRaises(LifecycleError):
            service.infer(LowLevelInferenceRequest(_observation(episode.episode_id)))
        self.assertEqual(
            [call for call in backend.calls if call == ("end", episode.episode_id)],
            [("end", episode.episode_id)],
        )

        service.reset_context(episode.episode_id)
        self.assertEqual(backend.begin_attempts, 3)
        self.assertEqual(
            [call for call in backend.calls if call == ("end", episode.episode_id)],
            [("end", episode.episode_id)],
        )
        prediction = service.infer(
            LowLevelInferenceRequest(_observation(episode.episode_id))
        )
        self.assertEqual(prediction.action.step_id, 0)
        service.close()

    def test_distinct_episodes_can_infer_concurrently_when_advertised(self) -> None:
        barrier = threading.Barrier(2, timeout=2)

        class ConcurrentBackend(RecordingBackend):
            def __init__(self) -> None:
                super().__init__(max_concurrent_episodes=2)

            def infer(self, request):
                barrier.wait()
                return super().infer(request)

        backend = ConcurrentBackend()
        service = LowLevelService(backend)
        episode_ids = ("concurrent-a", "concurrent-b")
        for episode_id in episode_ids:
            service.open_context(
                EpisodeContext(
                    task_id="task",
                    instruction="do it",
                    episode_id=episode_id,
                )
            )

        errors = []

        def run(episode_id):
            try:
                service.infer(LowLevelInferenceRequest(_observation(episode_id)))
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=run, args=(episode_id,))
            for episode_id in episode_ids
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)

        self.assertFalse(errors)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        for episode_id in episode_ids:
            service.close_context(episode_id)
        service.close()

    def test_manifest_required_modalities_are_checked_before_backend(self) -> None:
        backend = RecordingBackend()
        backend.manifest = PolicyManifest(
            policy_id="recording",
            policy_version="1",
            required_modalities=("required.camera",),
        )
        service = LowLevelService(backend)
        service.open_context(
            EpisodeContext(task_id="task", instruction="do it", episode_id="missing")
        )
        with self.assertRaises(PredictionContractError):
            service.infer(LowLevelInferenceRequest(_observation("missing")))
        self.assertFalse(any(call[0] == "infer" for call in backend.calls if isinstance(call, tuple)))
        service.close()


class FakePolicy:
    def __init__(self) -> None:
        self.reset_count = 0
        self.closed = False
        self.invalidations = 0
        self.batches: list[dict[str, object]] = []
        self.last_action_used_cache = False
        self.last_model_forward_ms = 12.5
        self.last_action_call_ms = 13.0

    def reset(self) -> str:
        self.reset_count += 1
        return "reset"

    def invalidate_action_cache(self) -> int:
        self.invalidations += 1
        return 7

    def get_action(self, batch):
        self.batches.append(batch)
        return list(range(16))

    def close(self) -> None:
        self.closed = True


class HyVLAPluginTests(unittest.TestCase):
    def test_plugin_import_does_not_load_model_frameworks(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        script = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.') or name == 'robotwin_eval' or name.startswith('robotwin_eval.'):
        raise AssertionError('eager heavy import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import highlow.inference.low_level
import highlow.inference.low_level.plugins.hyvla
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    @unittest.skipIf(np is None, "NumPy is an optional runtime dependency")
    def test_input_adapter_builds_current_hyvla_batch_contract(self) -> None:
        batch = HyVLAInputAdapter().build_batch(_observation("ep"), "pick object")
        expected_keys = {
            "observation.images.top_head",
            "observation.images.hand_left",
            "observation.images.hand_right",
            "observation.state",
            "task",
            "raw_images.top_head",
            "raw_images.hand_left",
            "raw_images.hand_right",
        }
        self.assertEqual(set(batch), expected_keys)
        self.assertEqual(batch["observation.images.top_head"].shape, (1, 3, 2, 3))
        self.assertEqual(batch["observation.images.top_head"].dtype, np.float32)
        self.assertEqual(batch["observation.state"].shape, (1, 32))
        np.testing.assert_array_equal(
            batch["observation.state"][0, :16], np.arange(16, dtype=np.float32)
        )
        np.testing.assert_array_equal(batch["observation.state"][0, 16:], 0.0)
        self.assertEqual(batch["task"], ["pick object"])

    @unittest.skipIf(np is None, "NumPy is an optional runtime dependency")
    def test_input_adapter_converts_valid_json_lists_but_keeps_arrays_strict(self) -> None:
        observation = _observation("json")
        modalities = {
            name: Modality(
                name=name,
                schema_id=modality.schema_id,
                data=modality.data.tolist()
                if hasattr(modality.data, "tolist")
                else modality.data,
            )
            for name, modality in observation.modalities.items()
        }
        converted = HyVLAInputAdapter().build_batch(
            ObservationEnvelope(
                episode_id="json",
                step_id=0,
                timestamp_s=0.0,
                modalities=modalities,
            ),
            "instruction",
        )
        self.assertEqual(converted["raw_images.top_head"].dtype, np.uint8)

        bad_modalities = dict(observation.modalities)
        bad_modalities[HEAD_RGB_MODALITY] = Modality(
            name=HEAD_RGB_MODALITY,
            schema_id=RGB_SCHEMA_ID,
            data=np.zeros((2, 3, 3), dtype=np.float32),
        )
        with self.assertRaises(PredictionContractError):
            HyVLAInputAdapter().build_batch(
                ObservationEnvelope(
                    episode_id="json",
                    step_id=0,
                    timestamp_s=0.0,
                    modalities=bad_modalities,
                ),
                "instruction",
            )

    def test_injected_factory_runs_only_on_open_and_supports_instruction_handoff(self) -> None:
        policies: list[FakePolicy] = []
        factory_configs: list[dict[str, object]] = []

        def factory(config):
            factory_configs.append(config)
            policy = FakePolicy()
            policies.append(policy)
            return policy

        backend = HyVLABackend(
            {"ckpt_path": "unused-in-test"},
            policy_factory=factory,
            input_adapter=FakeInputAdapter(),
        )
        self.assertEqual(factory_configs, [])
        service = LowLevelService(backend)
        context = service.begin_episode(
            EpisodeContext(task_id="task", instruction="first", episode_id="ep")
        )
        self.assertEqual(factory_configs, [{"ckpt_path": "unused-in-test"}])

        first = context.infer(
            LowLevelInferenceRequest(
                _observation("ep", 0),
                context=RequestContext(request_id="r1", trace_id="trace"),
            )
        )
        second = service.infer(
            LowLevelInferenceRequest(
                _observation("ep", 1), conditioning={"instruction": "second"}
            )
        )
        self.assertEqual(first.action.schema_id, STATE_ACTION_SCHEMA_ID)
        self.assertEqual(first.action.metadata["request_id"], "r1")
        self.assertFalse(first.metadata["instruction_changed"])
        self.assertTrue(second.metadata["instruction_changed"])
        self.assertEqual(second.metadata["stale_action_count"], 7)
        self.assertEqual(policies[0].invalidations, 1)
        self.assertEqual(policies[0].batches[0]["task"], ["first"])
        self.assertEqual(policies[0].batches[1]["task"], ["second"])
        context.finish()
        service.close()
        self.assertEqual(policies[0].reset_count, 2)
        self.assertTrue(policies[0].closed)
        self.assertIs(HYVLA_MANIFEST.context_mode, ContextMode.EPISODE)
        self.assertIs(HYVLA_MANIFEST.temporal_mode, TemporalMode.HISTORY)
        self.assertIs(HYVLA_MANIFEST.prediction_mode, PredictionMode.SINGLE_ACTION)

    def test_hyvla_reset_recovers_when_policy_teardown_raises(self) -> None:
        class FlakyResetPolicy(FakePolicy):
            def reset(self):
                self.reset_count += 1
                if self.reset_count == 2:
                    raise TimeoutError("policy reset failed after teardown started")
                return "reset"

        policy = FlakyResetPolicy()
        service = LowLevelService(
            HyVLABackend(
                policy_factory=lambda config: policy,
                input_adapter=FakeInputAdapter(),
            )
        )
        episode = EpisodeContext(
            task_id="task", instruction="do it", episode_id="hyvla-reset-retry"
        )
        service.open_context(episode)

        with self.assertRaises(TimeoutError):
            service.reset_context(episode.episode_id)
        service.reset_context(episode.episode_id)

        prediction = service.infer(
            LowLevelInferenceRequest(_observation(episode.episode_id))
        )
        self.assertEqual(prediction.action.episode_id, episode.episode_id)
        self.assertEqual(policy.reset_count, 3)
        service.close()


class FakeInputAdapter:
    def build_batch(self, observation, instruction):
        return {"task": [instruction], "step_id": observation.step_id}


def _asgi_request(app, method: str, path: str, payload=None):
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    received = False
    messages = []

    async def receive():
        nonlocal received
        if received:
            return {"type": "http.request", "body": b"", "more_body": False}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run(
        app(
            {"type": "http", "method": method, "path": path},
            receive,
            send,
        )
    )
    status = messages[0]["status"]
    response = json.loads(messages[1]["body"])
    return status, response


class LowLevelApiTests(unittest.TestCase):
    def test_health_capabilities_context_and_infer_routes(self) -> None:
        backend = RecordingBackend(max_concurrent_episodes=2)
        service = LowLevelService(backend)
        app = create_app(service)

        status, health = _asgi_request(app, "GET", "/v1/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["state"], "starting")

        status, opened = _asgi_request(
            app,
            "POST",
            "/v1/episodes/open",
            {"task_id": "task", "instruction": "do it", "episode_id": "api-ep"},
        )
        self.assertEqual((status, opened["state"]), (200, "open"))

        status, health = _asgi_request(app, "GET", "/health")
        self.assertEqual((status, health["state"]), (200, "ready"))

        status, capabilities = _asgi_request(app, "GET", "/v1/capabilities")
        self.assertEqual(status, 200)
        self.assertEqual(capabilities["manifest"]["backend_id"], "recording")
        self.assertEqual(capabilities["active_episode_ids"], ["api-ep"])

        status, prediction = _asgi_request(
            app,
            "POST",
            "/v1/infer",
            {
                "observation": {
                    "episode_id": "api-ep",
                    "step_id": 2,
                    "timestamp_s": 1.0,
                    "modalities": {},
                },
                "conditioning": {"instruction": "new instruction"},
                "context": {"request_id": "api-request", "trace_id": "trace"},
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(prediction["action"]["data"], [2])

        status, reset = _asgi_request(
            app, "POST", "/v1/episodes/reset", {"episode_id": "api-ep"}
        )
        self.assertEqual((status, reset["reset"]), (200, True))
        status, closed = _asgi_request(
            app,
            "POST",
            "/v1/episodes/close",
            {
                "episode_id": "api-ep",
                "success": True,
                "terminated": True,
                "step_count": 3,
            },
        )
        self.assertEqual((status, closed["state"]), (200, "closed"))
        self.assertTrue(backend.results[-1].success)

        status, closed = _asgi_request(
            app, "POST", "/v1/episodes/close", {"episode_id": "api-ep"}
        )
        self.assertEqual((status, closed["state"]), (200, "closed"))

    def test_environment_factory_loads_arbitrary_backend_factory(self) -> None:
        app = create_policy_app_from_env(
            {
                "HIGHLOW_LOW_LEVEL_FACTORY": (
                    "tests.test_highlow_low_level:create_recording_backend"
                ),
                "HIGHLOW_LOW_LEVEL_CONFIG": '{"max_concurrent_episodes": 2}',
            }
        )
        self.assertIsInstance(app, JsonAsgiApp)


def create_recording_backend(max_concurrent_episodes: int = 1):
    return RecordingBackend(max_concurrent_episodes=max_concurrent_episodes)


if __name__ == "__main__":
    unittest.main()
