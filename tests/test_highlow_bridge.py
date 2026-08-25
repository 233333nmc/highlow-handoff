from __future__ import annotations

import asyncio
import importlib
from io import BytesIO
import json
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import urlsplit

from highlow.adapters import (
    EE_ACTION_SCHEMA_ID,
    END_EFFECTOR_STATE,
    HEAD_RGB,
    LEFT_RGB,
    RIGHT_RGB,
    RoboTwinAdapter,
    RoboTwinPolicyHook,
    create_robotwin_bridge,
    robotwin_high_level_request,
)
from highlow.bridge import (
    BridgeTurn,
    DualSystemBridge,
    SimulatorAdapter,
    low_level_request,
    periodic_high_level_trigger,
    planner_request,
    planner_then_verifier_request,
    verifier_request,
)
from highlow.clients import (
    InProcessHighLevelClient,
    InProcessLowLevelClient,
    JsonHttpInferenceClient,
    JsonHttpTransportError,
)
from highlow.contracts import (
    ActionEnvelope,
    EpisodeResult,
    ObservationEnvelope,
    RequestContext,
)


def _robotwin_observation():
    return {
        "observation": {
            "head_camera": {"rgb": [["head"]]},
            "left_camera": {"rgb": [["left"]]},
            "right_camera": {"rgb": [["right"]]},
        },
        "endpose": {
            "left_endpose": [1, 2, 3, 1, 0, 0, 0],
            "left_gripper": 0.25,
            "right_endpose": [4, 5, 6, 1, 0, 0, 0],
            "right_gripper": 0.75,
        },
    }


def _asgi_transport(app):
    def transport(method, url, body, headers, timeout_s):
        del headers, timeout_s

        async def exchange():
            sent = []
            received = False

            async def receive():
                nonlocal received
                if received:
                    return {"type": "http.disconnect"}
                received = True
                return {
                    "type": "http.request",
                    "body": body or b"",
                    "more_body": False,
                }

            async def send(message):
                sent.append(message)

            await app(
                {
                    "type": "http",
                    "method": method,
                    "path": urlsplit(url).path,
                },
                receive,
                send,
            )
            if sent[0]["status"] >= 400:
                raise AssertionError(sent[1]["body"].decode("utf-8"))
            return sent[1]["body"]

        return asyncio.run(exchange())

    return transport


class _HighService:
    def __init__(self):
        self.events = []

    def open_context(self, context):
        self.events.append(("open", context.episode_id))

    def reset_context(self, context):
        self.events.append(("reset", context.episode_id))

    def close_context(self, context):
        self.events.append(("close", context.episode_id))

    def infer(self, request):
        step_id = int(request.structured_context["step_id"])
        self.events.append(("infer", step_id))
        if step_id == 1:
            return {"next_subtask": "pick the fries"}
        return None

    def capabilities(self):
        return {"planning": True}


class _LowService:
    def __init__(self):
        self.requests = []
        self.events = []

    def open_context(self, context):
        self.events.append(("open", context.episode_id))

    def reset_context(self, context):
        self.events.append(("reset", context.episode_id))

    def close_context(self, context):
        self.events.append(("close", context.episode_id))

    def infer(self, request):
        self.requests.append(request)
        return ActionEnvelope(
            episode_id=request.episode_id,
            step_id=request.observation.step_id,
            schema_id=EE_ACTION_SCHEMA_ID,
            data=tuple(range(16)),
        )

    def health(self):
        return {"status": "ready"}


class HighLowBridgeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = RoboTwinAdapter()

    def _episode(self, episode_id="ep-1"):
        return self.adapter.create_episode_context(
            "burger_fries", "assemble the meal", episode_id=episode_id, seed=3
        )

    def _observation(self, episode, step_id=0):
        return self.adapter.encode_observation(
            episode,
            _robotwin_observation(),
            step_id=step_id,
            timestamp_s=step_id / 25,
        )

    def test_robotwin_adapter_is_conversion_only_and_canonical(self):
        self.assertIsInstance(self.adapter, SimulatorAdapter)
        for forbidden in ("reset", "step", "take_action"):
            self.assertFalse(hasattr(self.adapter, forbidden))
        envelope = self._observation(self._episode())
        self.assertEqual(
            set(envelope.modalities),
            {HEAD_RGB, LEFT_RGB, RIGHT_RGB, END_EFFECTOR_STATE},
        )
        self.assertEqual(
            envelope.require(HEAD_RGB).schema_id, "image.rgb.uint8.hwc@1"
        )
        state = envelope.require(END_EFFECTOR_STATE)
        self.assertEqual(state.schema_id, EE_ACTION_SCHEMA_ID)
        self.assertEqual(len(state.data), 16)
        native = tuple(range(16))
        self.assertIs(
            self.adapter.decode_action(
                ActionEnvelope("ep-1", 0, EE_ACTION_SCHEMA_ID, native)
            ),
            native,
        )

    def test_bridge_trigger_handoff_and_lifecycle(self):
        episode = self._episode("ep-bridge")
        high, low = _HighService(), _LowService()
        bridge = DualSystemBridge(
            InProcessHighLevelClient(high),
            InProcessLowLevelClient(low),
            high_level_trigger=lambda turn: turn.observation.step_id in (0, 1),
        )
        bridge.open_context(episode)
        for step_id in range(3):
            action = bridge.infer(
                self._observation(episode, step_id),
                request_context=RequestContext(request_id=f"request-{step_id}"),
            )
            self.assertEqual(action.step_id, step_id)
        self.assertEqual(
            [event for event in high.events if event[0] == "infer"],
            [("infer", 0), ("infer", 1)],
        )
        self.assertEqual(
            [request.conditioning["instruction"] for request in low.requests],
            ["assemble the meal", "pick the fries", "pick the fries"],
        )
        self.assertEqual(
            [request.conditioning["instruction_revision"] for request in low.requests],
            [0, 1, 1],
        )
        bridge.reset_context(episode)
        bridge.close_context(episode)
        self.assertEqual(high.events[-2:], [("reset", "ep-bridge"), ("close", "ep-bridge")])
        self.assertEqual(low.events[-2:], [("reset", "ep-bridge"), ("close", "ep-bridge")])

    def test_default_trigger_only_initial_step(self):
        episode, high = self._episode("ep-trigger"), _HighService()
        bridge = DualSystemBridge(
            InProcessHighLevelClient(high), InProcessLowLevelClient(_LowService())
        )
        bridge.open_context(episode)
        bridge.infer(self._observation(episode, 0))
        bridge.infer(self._observation(episode, 1))
        self.assertEqual(
            [event for event in high.events if event[0] == "infer"], [("infer", 0)]
        )

    def test_typed_planner_and_verifier_outputs(self):
        from highlow.inference.high_level import (
            PlannerOutput,
            VerifierDecision,
            VerifierOutput,
            VerifierStatus,
        )

        episode = self._episode("ep-typed")

        class High:
            outputs = [
                {"output": PlannerOutput(("first subtask",))},
                {
                    "output": VerifierOutput(
                        VerifierDecision.SAME,
                        VerifierStatus.RUNNING,
                        next_subtask="",
                    )
                },
                {
                    "output": VerifierOutput(
                        VerifierDecision.SWITCH,
                        VerifierStatus.COMPLETED,
                        next_subtask="second subtask",
                    )
                },
            ]

            def infer(self, request):
                return self.outputs.pop(0)

        low = _LowService()
        bridge = DualSystemBridge(
            InProcessHighLevelClient(High()),
            InProcessLowLevelClient(low),
            high_level_trigger=lambda turn: True,
        )
        bridge.open_context(episode)
        for step in range(3):
            bridge.infer(self._observation(episode, step))
        self.assertEqual(
            [item.conditioning["instruction"] for item in low.requests],
            ["first subtask", "first subtask", "second subtask"],
        )

    def test_episode_locks_allow_cross_episode_concurrency(self):
        barrier = threading.Barrier(2, timeout=2)

        class BlockingHigh:
            def infer(self, request):
                barrier.wait()
                return None

        bridge = DualSystemBridge(
            InProcessHighLevelClient(BlockingHigh()),
            InProcessLowLevelClient(_LowService()),
        )
        episodes = [self._episode(f"ep-{index}") for index in range(2)]
        for episode in episodes:
            bridge.open_context(episode)
        errors = []

        def run(episode):
            try:
                bridge.infer(self._observation(episode))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(episode,)) for episode in episodes]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
        self.assertFalse(errors)
        self.assertTrue(all(not thread.is_alive() for thread in threads))

    def test_service_factories_build_real_contracts(self):
        from highlow.inference.high_level import HighLevelRequest
        from highlow.inference.low_level import LowLevelInferenceRequest

        episode = self._episode("ep-factory")
        turn = BridgeTurn(
            RequestContext(request_id="factory"),
            episode,
            self._observation(episode),
            "pick item",
            2,
        )
        plan = planner_request(turn)
        verify = verifier_request(turn)
        low = low_level_request(turn, None)
        self.assertIsInstance(plan, HighLevelRequest)
        self.assertEqual(plan.operation.value, "planner")
        self.assertIsInstance(verify, HighLevelRequest)
        self.assertEqual(verify.structured_context["current_subtask"], "pick item")
        self.assertIsInstance(low, LowLevelInferenceRequest)
        self.assertEqual(low.conditioning["instruction_revision"], 2)
        later_turn = BridgeTurn(
            turn.request_context,
            turn.episode_context,
            turn.observation,
            turn.instruction,
            turn.instruction_revision,
            high_level_call_count=1,
        )
        self.assertEqual(
            planner_then_verifier_request(later_turn).operation.value, "verifier"
        )
        trigger = periodic_high_level_trigger(4)
        self.assertTrue(trigger(turn))

    def test_default_factories_compose_with_bundled_services(self):
        from highlow.inference.high_level import (
            BackendInferenceResponse,
            HighLevelService,
        )
        from highlow.inference.low_level import LowLevelService, PolicyManifest

        class HighBackend:
            def infer(self, request):
                return BackendInferenceResponse(
                    raw_text='{"subtasks":["default subtask"]}',
                    structured_output={"subtasks": ["default subtask"]},
                    backend="test",
                    model="test",
                )

        class LowBackend:
            manifest = PolicyManifest(
                policy_id="default-compose",
                policy_version="1",
                action_schema_ids=(EE_ACTION_SCHEMA_ID,),
            )

            def open(self):
                pass

            def begin_episode(self, episode):
                pass

            def infer(self, request):
                self.last_request = request
                return ActionEnvelope(
                    request.episode_id,
                    request.step_id,
                    EE_ACTION_SCHEMA_ID,
                    tuple(range(16)),
                )

            def end_episode(self, episode, result=None):
                pass

            def close(self):
                pass

        low_backend = LowBackend()
        low_service = LowLevelService(low_backend)
        bridge = DualSystemBridge(
            InProcessHighLevelClient(HighLevelService(HighBackend())),
            InProcessLowLevelClient(low_service),
        )
        episode = self._episode("default-contracts")

        bridge.open_context(episode)
        action = bridge.infer(self._observation(episode))

        self.assertEqual(action.step_id, 0)
        self.assertEqual(
            low_backend.last_request.conditioning["instruction"],
            "default subtask",
        )
        bridge.close_context(episode)
        low_service.close()

    def test_policy_hook_never_drives_environment(self):
        episode_id = "ep-hook"
        bridge = DualSystemBridge(
            InProcessHighLevelClient(_HighService()),
            InProcessLowLevelClient(_LowService()),
        )
        hook = RoboTwinPolicyHook(self.adapter, bridge)
        hook.start_episode("task", "do it", episode_id=episode_id)
        native_action = hook.act(_robotwin_observation(), 0, 0.0)
        self.assertEqual(tuple(native_action), tuple(range(16)))
        hook.reset()
        hook.finish()
        self.assertIsNone(hook.context)
        for forbidden in ("step", "take_action"):
            self.assertFalse(hasattr(hook, forbidden))

    def test_json_http_client_encodes_and_decodes_without_service_schema(self):
        calls = []

        def transport(method, url, body, headers, timeout_s):
            calls.append((method, url, body, dict(headers), timeout_s))
            payload = {} if body is None else json.loads(body)
            return {"echo": payload, "path": url.rsplit("/", 1)[-1]}

        client = JsonHttpInferenceClient(
            "http://inference.local/api/",
            infer_endpoint="v1/infer",
            open_context_endpoint="v1/open",
            response_decoder=lambda payload: payload["echo"],
            transport=transport,
            timeout_s=2.5,
            headers={"Authorization": "Bearer test"},
        )
        self.assertEqual(client.infer({"value": 3}), {"value": 3})
        episode = self._episode("ep-http")
        opened = client.open_context(episode)
        self.assertEqual(opened["episode_id"], "ep-http")
        self.assertIsNone(client.reset_context(episode))
        self.assertEqual(calls[0][0:2], ("POST", "http://inference.local/api/v1/infer"))
        self.assertEqual(calls[0][4], 2.5)
        self.assertEqual(calls[0][3]["Authorization"], "Bearer test")

    def test_json_http_default_decoder_restores_binary_envelope(self):
        client = JsonHttpInferenceClient(
            "http://inference.local",
            transport=lambda *args: {"encoding": "base64", "data": "AP8="},
        )
        self.assertEqual(client.infer({"value": 1}), b"\x00\xff")

    def test_json_http_error_preserves_service_response_body(self):
        body = (
            b'{"error":{"code":"HIGH_LEVEL_INFERENCE_ERROR",'
            b'"message":"next mode must return exactly one subtask"}}'
        )
        error = HTTPError(
            "http://inference.local/v1/infer",
            422,
            "Unprocessable Entity",
            hdrs=None,
            fp=BytesIO(body),
        )
        client = JsonHttpInferenceClient("http://inference.local")

        with patch(
            "highlow.clients.json_http.urlopen", side_effect=error
        ), self.assertRaises(JsonHttpTransportError) as raised:
            client.infer({"value": 1})

        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.response_body, body)
        self.assertIn("HIGH_LEVEL_INFERENCE_ERROR", str(raised.exception))
        self.assertIn("exactly one subtask", str(raised.exception))

    def test_default_trigger_uses_first_call_not_step_zero(self):
        episode, high = self._episode("ep-nonzero"), _HighService()
        bridge = DualSystemBridge(
            InProcessHighLevelClient(high), InProcessLowLevelClient(_LowService())
        )
        bridge.open_context(episode)
        bridge.infer(self._observation(episode, 5))
        bridge.infer(self._observation(episode, 6))
        self.assertEqual(
            [event for event in high.events if event[0] == "infer"], [("infer", 5)]
        )

    def test_high_level_response_correlation_is_enforced(self):
        episode = self._episode("expected")

        class WrongHigh:
            def infer(self, request):
                return {"episode_id": "wrong", "next_subtask": "unsafe"}

        low = _LowService()
        bridge = DualSystemBridge(
            InProcessHighLevelClient(WrongHigh()), InProcessLowLevelClient(low)
        )
        bridge.open_context(episode)
        with self.assertRaises(ValueError):
            bridge.infer(self._observation(episode))
        self.assertEqual(low.requests, [])

    def test_lost_low_response_reuses_high_decision_and_action(self):
        episode = self._episode("ep-retry")

        class ChangingHigh:
            def __init__(self):
                self.calls = 0

            def infer(self, request):
                self.calls += 1
                return {"next_subtask": f"decision-{self.calls}"}

        class LostResponseLow(_LowService):
            def __init__(self):
                super().__init__()
                self.lost = False

            def infer(self, request):
                action = super().infer(request)
                if not self.lost:
                    self.lost = True
                    raise TimeoutError("response was lost after execution")
                return action

        high, low = ChangingHigh(), LostResponseLow()
        bridge = DualSystemBridge(
            InProcessHighLevelClient(high), InProcessLowLevelClient(low)
        )
        bridge.open_context(episode)
        observation = self._observation(episode)
        with self.assertRaises(TimeoutError):
            bridge.infer(observation)
        action = bridge.infer(observation)
        replay = bridge.infer(observation)
        self.assertIs(replay, action)
        self.assertEqual(high.calls, 1)
        self.assertEqual(
            [request.conditioning["instruction"] for request in low.requests],
            ["decision-1", "decision-1"],
        )
        self.assertEqual(
            low.requests[0].request_context.request_id,
            low.requests[1].request_context.request_id,
        )

    def test_close_failure_retains_retryable_tombstone(self):
        episode = self._episode("ep-close-retry")

        class FlakyCloseLow(_LowService):
            def __init__(self):
                super().__init__()
                self.failures = 1

            def close_context(self, context):
                self.events.append(("close", context.episode_id))
                if self.failures:
                    self.failures -= 1
                    raise TimeoutError("close response lost")

        low = FlakyCloseLow()
        bridge = DualSystemBridge(
            InProcessHighLevelClient(_HighService()), InProcessLowLevelClient(low)
        )
        bridge.open_context(episode)
        with self.assertRaises(TimeoutError):
            bridge.close_context(episode)
        with self.assertRaises(RuntimeError):
            bridge.infer(self._observation(episode))
        bridge.close_context(episode)
        self.assertEqual(
            [event for event in low.events if event[0] == "close"],
            [("close", "ep-close-retry"), ("close", "ep-close-retry")],
        )

    def test_close_retry_preserves_the_original_episode_result(self):
        episode = self._episode("ep-close-result")

        class LostBeforeCloseLow(_LowService):
            def __init__(self):
                super().__init__()
                self.failures = 1
                self.close_values = []

            def close_context(self, context):
                if self.failures:
                    self.failures -= 1
                    raise TimeoutError("request did not reach low-level service")
                self.close_values.append(context)

        low = LostBeforeCloseLow()
        hook = RoboTwinPolicyHook(
            self.adapter,
            DualSystemBridge(
                InProcessHighLevelClient(_HighService()),
                InProcessLowLevelClient(low),
            ),
        )
        hook.start_episode("task", "do it", episode_id=episode.episode_id)
        result = EpisodeResult(
            episode_id=episode.episode_id,
            success=True,
            terminated=True,
            step_count=9,
            metrics={"score": 1.0},
        )

        with self.assertRaises(TimeoutError):
            hook.finish(result)
        hook.finish()

        self.assertEqual(low.close_values, [result])
        self.assertIsNone(hook.context)

    def test_open_response_loss_reuses_auto_episode_id(self):
        class LostOpenLow(_LowService):
            def __init__(self):
                super().__init__()
                self.opened = set()
                self.failures = 1

            def open_context(self, context):
                self.events.append(("open", context.episode_id))
                self.opened.add(context.episode_id)
                if self.failures:
                    self.failures -= 1
                    raise TimeoutError("open response was lost")

        high, low = _HighService(), LostOpenLow()
        bridge = DualSystemBridge(
            InProcessHighLevelClient(high), InProcessLowLevelClient(low)
        )
        hook = RoboTwinPolicyHook(self.adapter, bridge)

        with self.assertRaises(TimeoutError):
            hook.start_episode("task", "do it")
        retained = hook.context
        self.assertIsNotNone(retained)
        self.assertFalse(hook.ready)
        with self.assertRaises(RuntimeError):
            hook.act(_robotwin_observation(), 0, 0.0)

        retried = hook.start_episode("task", "do it")
        self.assertIs(retried, retained)
        self.assertTrue(hook.ready)
        self.assertEqual(low.opened, {retained.episode_id})
        self.assertEqual(
            [event for event in low.events if event[0] == "open"],
            [("open", retained.episode_id), ("open", retained.episode_id)],
        )
        # High-level opening completed before the low response was lost, so
        # the bridge resumes at the low phase instead of reopening both sides.
        self.assertEqual(
            [event for event in high.events if event[0] == "open"],
            [("open", retained.episode_id)],
        )
        hook.finish()
        self.assertIsNone(hook.context)

    def test_partial_reset_is_non_inferable_and_resumes_failed_side(self):
        class FlakyResetLow(_LowService):
            def __init__(self):
                super().__init__()
                self.failures = 1

            def reset_context(self, context):
                self.events.append(("reset", context.episode_id))
                if self.failures:
                    self.failures -= 1
                    raise TimeoutError("reset response was lost")

        episode = self._episode("ep-reset-retry")
        high, low = _HighService(), FlakyResetLow()
        bridge = DualSystemBridge(
            InProcessHighLevelClient(high), InProcessLowLevelClient(low)
        )
        bridge.open_context(episode)

        with self.assertRaises(TimeoutError):
            bridge.reset_context(episode)
        with self.assertRaises(RuntimeError):
            bridge.infer(self._observation(episode))
        bridge.reset_context(episode)
        self.assertEqual(
            [event for event in high.events if event[0] == "reset"],
            [("reset", episode.episode_id)],
        )
        self.assertEqual(
            [event for event in low.events if event[0] == "reset"],
            [("reset", episode.episode_id), ("reset", episode.episode_id)],
        )
        self.assertEqual(bridge.infer(self._observation(episode)).step_id, 0)
        bridge.close_context(episode)

    def test_ready_made_robotwin_factory_encodes_visual_frames(self):
        try:
            import numpy as np
            import PIL  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("NumPy and Pillow are optional image codec dependencies")
        native = _robotwin_observation()
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        for camera in ("head_camera", "left_camera", "right_camera"):
            native["observation"][camera]["rgb"] = image
        episode = self._episode("ep-images")
        turn = BridgeTurn(
            RequestContext(request_id="images"),
            episode,
            self.adapter.encode_observation(
                episode, native, step_id=0, timestamp_s=0.0
            ),
            "instruction",
            0,
        )
        request = robotwin_high_level_request(turn)
        self.assertEqual(request.operation.value, "planner")
        self.assertEqual(len(request.frames), 3)
        self.assertTrue(all(frame.data.startswith(b"\xff\xd8") for frame in request.frames))
        verify_turn = BridgeTurn(
            turn.request_context,
            turn.episode_context,
            turn.observation,
            "pick object",
            1,
            high_level_call_count=1,
        )
        self.assertEqual(
            robotwin_high_level_request(verify_turn).operation.value, "verifier"
        )

    def test_full_http_high_low_robotwin_composition(self):
        try:
            import numpy as np
            import PIL  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("NumPy and Pillow are optional image codec dependencies")

        from highlow.inference.high_level import (
            BackendInferenceResponse,
            HighLevelService,
            create_app as create_high_app,
        )
        from highlow.inference.low_level import (
            LowLevelService,
            PolicyManifest,
            create_app as create_low_app,
        )

        class HighBackend:
            def __init__(self):
                self.requests = []

            def infer(self, request):
                self.requests.append(request)
                return BackendInferenceResponse(
                    raw_text='{"subtasks":["pick the object"]}',
                    structured_output={"subtasks": ["pick the object"]},
                    backend="fake",
                    model="fake-vlm",
                )

        class LowBackend:
            manifest = PolicyManifest(
                policy_id="e2e",
                policy_version="1",
                action_schema_ids=(EE_ACTION_SCHEMA_ID,),
            )

            def __init__(self):
                self.requests = []
                self.results = []

            def open(self):
                pass

            def begin_episode(self, episode):
                pass

            def infer(self, request):
                self.requests.append(request)
                return ActionEnvelope(
                    request.episode_id,
                    request.step_id,
                    EE_ACTION_SCHEMA_ID,
                    list(range(16)),
                )

            def end_episode(self, episode, result=None):
                self.results.append(result)

            def close(self):
                pass

        high_backend, low_backend = HighBackend(), LowBackend()
        high_app = create_high_app(HighLevelService(high_backend))
        low_service = LowLevelService(low_backend)
        low_app = create_low_app(low_service)
        high_client = JsonHttpInferenceClient(
            "http://high",
            infer_endpoint="/v1/high-level/infer",
            health_endpoint="/health",
            capabilities_endpoint="/capabilities",
            transport=_asgi_transport(high_app),
        )
        low_client = JsonHttpInferenceClient(
            "http://low",
            infer_endpoint="/v1/infer",
            open_context_endpoint="/v1/episodes/open",
            reset_context_endpoint="/v1/episodes/reset",
            close_context_endpoint="/v1/episodes/close",
            health_endpoint="/health",
            capabilities_endpoint="/capabilities",
            transport=_asgi_transport(low_app),
        )
        bridge = create_robotwin_bridge(high_client, low_client)
        hook = RoboTwinPolicyHook(self.adapter, bridge)
        episode = hook.start_episode("task", "complete task", episode_id="http-e2e")
        native = _robotwin_observation()
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        for camera in ("head_camera", "left_camera", "right_camera"):
            native["observation"][camera]["rgb"] = image
        action = hook.act(native, 0, 0.0)
        self.assertEqual(tuple(action), tuple(range(16)))
        self.assertEqual(len(high_backend.requests[0].frames), 3)
        self.assertEqual(
            low_backend.requests[0].conditioning["instruction"], "pick the object"
        )
        hook.finish(
            EpisodeResult(
                episode_id=episode.episode_id,
                success=True,
                terminated=True,
                step_count=1,
            )
        )
        self.assertTrue(low_backend.results[-1].success)

        # A string close is normalized back to the retained EpisodeContext,
        # so the generic HTTP client still sends the object-shaped wire form.
        string_context = self._episode("http-close-string")
        bridge.open_context(string_context)
        bridge.close_context(string_context.episode_id)
        low_service.close()

    def test_official_robotwin_four_hook_shim_uses_remote_runtime(self):
        try:
            import numpy as np
            import PIL  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("NumPy and Pillow are optional image codec dependencies")

        from robotwin_dual_system import (
            encode_obs,
            eval as policy_eval,
            get_model,
            reset_model,
        )

        class TaskEnv:
            task_name = "burger_fries"

            def __init__(self):
                self.actions = []

            def get_instruction(self):
                return "assemble the meal"

            def take_action(self, action, *, action_type):
                self.actions.append((action, action_type))

        high, low = _HighService(), _LowService()
        model = get_model(
            {
                "task_name": "burger_fries",
                "_high_client": InProcessHighLevelClient(high),
                "_low_client": InProcessLowLevelClient(low),
                "verifier_interval_steps": 0,
            }
        )
        env = TaskEnv()
        native = _robotwin_observation()
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        for camera in ("head_camera", "left_camera", "right_camera"):
            native["observation"][camera]["rgb"] = image

        envelope = encode_obs(native, "offline instruction")
        self.assertIsInstance(envelope, ObservationEnvelope)
        policy_eval(env, model, native)
        first_episode_id = model.hook.context.episode_id
        self.assertEqual(tuple(env.actions[0][0]), tuple(range(16)))
        self.assertEqual(env.actions[0][1], "ee")

        self.assertEqual(reset_model(model), "remote dual-system policy reset")
        self.assertIsNone(model.hook.context)
        policy_eval(env, model, native)
        self.assertNotEqual(model.hook.context.episode_id, first_episode_id)
        reset_model(model)

    def test_official_shim_retries_an_incomplete_remote_open(self):
        try:
            import numpy as np
            import PIL  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("NumPy and Pillow are optional image codec dependencies")

        from robotwin_dual_system import (
            eval as policy_eval,
            get_model,
            reset_model,
        )

        class LostOpenLow(_LowService):
            def __init__(self):
                super().__init__()
                self.failures = 1

            def open_context(self, context):
                self.events.append(("open", context.episode_id))
                if self.failures:
                    self.failures -= 1
                    raise TimeoutError("open response was lost")

        class TaskEnv:
            task_name = "task"

            def __init__(self):
                self.actions = []

            def get_instruction(self):
                return "do it"

            def take_action(self, action, *, action_type):
                self.actions.append((action, action_type))

        low = LostOpenLow()
        model = get_model(
            {
                "_high_client": InProcessHighLevelClient(_HighService()),
                "_low_client": InProcessLowLevelClient(low),
            }
        )
        env = TaskEnv()
        native = _robotwin_observation()
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        for camera in ("head_camera", "left_camera", "right_camera"):
            native["observation"][camera]["rgb"] = image

        with self.assertRaises(TimeoutError):
            policy_eval(env, model, native)
        retained_id = model.hook.context.episode_id
        policy_eval(env, model, native)

        self.assertEqual(model.hook.context.episode_id, retained_id)
        self.assertEqual(len(env.actions), 1)
        self.assertEqual(
            [event for event in low.events if event[0] == "open"],
            [("open", retained_id), ("open", retained_id)],
        )
        reset_model(model)

    def test_official_shim_registers_final_episode_cleanup(self):
        import robotwin_dual_system.deploy_policy as shim

        class TaskEnv:
            task_name = "task"

            def get_instruction(self):
                return "do it"

        low = _LowService()
        registrations = []
        with patch.object(
            shim.atexit,
            "register",
            side_effect=lambda callback, *args: registrations.append(
                (callback, args)
            ),
        ):
            model = shim.get_model(
                {
                    "_high_client": InProcessHighLevelClient(_HighService()),
                    "_low_client": InProcessLowLevelClient(low),
                }
            )
        model.start_if_needed(TaskEnv())
        episode_id = model.hook.context.episode_id

        self.assertEqual(len(registrations), 1)
        callback, args = registrations[0]
        callback(*args)

        self.assertIsNone(model.hook.context)
        self.assertIn(("close", episode_id), low.events)

    def test_imports_are_framework_free(self):
        before = ("torch" in sys.modules, "hy_vla" in sys.modules)
        for name in (
            "highlow.bridge.protocols",
            "highlow.bridge.dual_system",
            "highlow.clients.in_process",
            "highlow.clients.json_http",
            "highlow.adapters.robotwin",
            "highlow.adapters.robotwin_hook",
            "robotwin_dual_system",
        ):
            importlib.import_module(name)
        self.assertEqual(("torch" in sys.modules, "hy_vla" in sys.modules), before)


if __name__ == "__main__":
    unittest.main()
