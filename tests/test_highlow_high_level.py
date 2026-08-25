from __future__ import annotations

import asyncio
import base64
import json
import unittest

from highlow.contracts import EpisodeContext, RequestContext
from highlow.inference.high_level import (
    BackendInferenceResponse,
    BackendTransportError,
    HealthState,
    HealthStatus,
    HighLevelCapabilities,
    HighLevelFrame,
    HighLevelOperation,
    HighLevelRequest,
    HighLevelService,
    HighLevelValidationError,
    HttpResponse,
    PLANNER_PROMPT_V1,
    PLANNER_SCHEMA_V1,
    PlannerOutput,
    SGLangOpenAIBackend,
    UnsupportedVersionError,
    VERIFIER_PROMPT_V1,
    VERIFIER_SCHEMA_V1,
    VerifierDecision,
    VerifierOutput,
    VerifierStatus,
    create_app,
    create_sglang_app_from_env,
)
from highlow.inference.common import JsonAsgiApp, to_jsonable


class FakeBackend:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def infer(self, request):
        self.requests.append(request)
        raw = json.dumps(self.payload)
        return BackendInferenceResponse(
            raw_text=raw,
            structured_output=self.payload,
            backend="fake",
            model="fake-model",
            finish_reason="stop",
            usage={"total_tokens": 7},
            timing_ms={"model": 3.5},
        )

    def capabilities(self):
        return HighLevelCapabilities(
            component="fake-backend",
            backend="fake",
            model="fake-model",
            protocol="in-process",
            operations=("planner", "verifier"),
        )

    def health(self):
        return HealthStatus(
            state=HealthState.READY,
            component="fake-backend",
            checked_at_s=1.0,
            detail="ok",
        )


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def send(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected transport request")
        return self.responses.pop(0)


def request_context(request_id="req-1"):
    return RequestContext(request_id=request_id, trace_id="trace-1")


def call_asgi(app, method, path, payload=None):
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")

    async def exchange():
        sent = []
        incoming = [
            {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }
        ]

        async def receive():
            return incoming.pop(0)

        async def send(message):
            sent.append(message)

        await app(
            {
                "type": "http",
                "method": method,
                "path": path,
                "headers": [],
            },
            receive,
            send,
        )
        status = sent[0]["status"]
        response_body = json.loads(sent[1]["body"])
        return status, response_body

    return asyncio.run(exchange())


class HighLevelContractsTest(unittest.TestCase):
    def test_request_can_derive_episode_fields(self):
        episode = EpisodeContext(
            task_id="task",
            instruction="place the object",
            episode_id="episode-1",
        )
        request = HighLevelRequest(
            request_context=request_context(),
            operation="plan",
            episode_context=episode,
        )
        self.assertEqual(request.operation, HighLevelOperation.PLANNER)
        self.assertEqual(request.episode_id, "episode-1")
        self.assertEqual(request.task_instruction, "place the object")

    def test_episode_id_mismatch_is_rejected(self):
        episode = EpisodeContext(
            task_id="task",
            instruction="do it",
            episode_id="episode-a",
        )
        with self.assertRaises(ValueError):
            HighLevelRequest(
                request_context=request_context(),
                operation="planner",
                episode_id="episode-b",
                episode_context=episode,
            )


class HighLevelServiceTest(unittest.TestCase):
    def test_planner_v1_renders_and_validates_one_next_subtask(self):
        backend = FakeBackend(
            {"subtasks": ["Place the object on the tray."], "confidence": 0.8}
        )
        service = HighLevelService(backend)
        response = service.infer(
            HighLevelRequest(
                request_context=request_context(),
                operation="planner",
                episode_id="episode-1",
                task_instruction="Put the object on the tray.",
                structured_context={
                    "planner_mode": "next",
                    "current_subtask": "",
                },
            )
        )

        self.assertIsInstance(response.output, PlannerOutput)
        self.assertEqual(response.output.subtasks, ("Place the object on the tray.",))
        self.assertEqual(response.prompt_version, PLANNER_PROMPT_V1)
        self.assertEqual(response.schema_version, PLANNER_SCHEMA_V1)
        backend_request = backend.requests[0]
        self.assertIn("Generate exactly one next subtask", backend_request.prompt)
        self.assertEqual(backend_request.schema_name, "high_level_planner_v1")
        self.assertEqual(backend_request.metadata["episode_id"], "episode-1")

    def test_verifier_v1_returns_typed_output(self):
        backend = FakeBackend(
            {
                "decision": "switch",
                "status": "completed",
                "next_subtask": "Place the fries on the tray.",
                "evidence": "The burger is on the tray.",
            }
        )
        service = HighLevelService(backend)
        response = service.infer(
            HighLevelRequest(
                request_context=request_context(),
                operation="verify",
                episode_id="episode-1",
                task_instruction="Place both foods on the tray.",
                structured_context={
                    "current_subtask": "Place the burger on the tray."
                },
                frames=(
                    HighLevelFrame(
                        data=b"jpeg bytes", frame_id=12, source="head_camera"
                    ),
                ),
            )
        )

        self.assertIsInstance(response.output, VerifierOutput)
        self.assertEqual(response.output.decision, VerifierDecision.SWITCH)
        self.assertEqual(response.output.status, VerifierStatus.COMPLETED)
        self.assertEqual(response.prompt_version, VERIFIER_PROMPT_V1)
        self.assertEqual(response.schema_version, VERIFIER_SCHEMA_V1)
        self.assertIn("frame_id\":12", backend.requests[0].prompt)

    def test_missing_verifier_current_subtask_fails_before_backend(self):
        backend = FakeBackend({})
        service = HighLevelService(backend)
        with self.assertRaises(HighLevelValidationError):
            service.infer(
                HighLevelRequest(
                    request_context=request_context(),
                    operation="verifier",
                    episode_id="episode-1",
                    task_instruction="Do the task.",
                )
            )
        self.assertEqual(backend.requests, [])

    def test_next_planner_rejects_multiple_subtasks(self):
        backend = FakeBackend({"subtasks": ["one", "two"]})
        service = HighLevelService(backend)
        with self.assertRaises(HighLevelValidationError):
            service.infer(
                HighLevelRequest(
                    request_context=request_context(),
                    operation="planner",
                    episode_id="episode-1",
                    task_instruction="Do the task.",
                )
            )

    def test_unsupported_version_fails_before_backend(self):
        backend = FakeBackend({"subtasks": ["one"]})
        service = HighLevelService(backend)
        with self.assertRaises(UnsupportedVersionError):
            service.infer(
                HighLevelRequest(
                    request_context=request_context(),
                    operation="planner",
                    episode_id="episode-1",
                    task_instruction="Do the task.",
                    prompt_version="planner.v99",
                )
            )
        self.assertEqual(backend.requests, [])

    def test_capabilities_and_health_include_backend(self):
        service = HighLevelService(FakeBackend({"subtasks": ["one"]}))
        capabilities = service.capabilities()
        self.assertEqual(capabilities.component, "high-level-service")
        self.assertEqual(
            capabilities.prompt_versions["planner"], (PLANNER_PROMPT_V1,)
        )
        self.assertEqual(service.health().state, HealthState.READY)


class HighLevelApiTest(unittest.TestCase):
    def test_health_capabilities_and_infer_routes(self):
        backend = FakeBackend(
            {"subtasks": ["Move to the next semantic phase."], "confidence": 0.9}
        )
        app = create_app(HighLevelService(backend))

        health_status, health = call_asgi(app, "GET", "/health")
        self.assertEqual(health_status, 200)
        self.assertEqual(health["state"], "ready")

        capabilities_status, capabilities = call_asgi(
            app, "GET", "/capabilities"
        )
        self.assertEqual(capabilities_status, 200)
        self.assertEqual(capabilities["operations"], ["planner", "verifier"])
        self.assertEqual(capabilities["prompt_versions"]["planner"], ["planner.v1"])

        infer_status, response = call_asgi(
            app,
            "POST",
            "/v1/high-level/infer",
            {
                "request_context": {
                    "request_id": "http-request-1",
                    "trace_id": "http-trace-1",
                },
                "episode_context": {
                    "task_id": "task-1",
                    "instruction": "Complete the task.",
                    "episode_id": "episode-http-1",
                },
                "operation": "planner",
                "structured_context": {"planner_mode": "next"},
                "frames": [
                    {
                        "data_url": "data:image/jpeg;base64,YWJj",
                        "mime_type": "image/jpeg",
                        "frame_id": 4,
                    }
                ],
            },
        )
        self.assertEqual(infer_status, 200)
        self.assertEqual(response["request_context"]["request_id"], "http-request-1")
        self.assertEqual(response["episode_id"], "episode-http-1")
        self.assertEqual(response["operation"], "planner")
        self.assertEqual(
            response["output"]["subtasks"], ["Move to the next semantic phase."]
        )
        self.assertEqual(backend.requests[0].frames[0].frame_id, 4)

    def test_infer_route_reports_invalid_frame(self):
        app = create_app(HighLevelService(FakeBackend({"subtasks": ["one"]})))
        status, response = call_asgi(
            app,
            "POST",
            "/v1/high-level/infer",
            {
                "request_context": {"request_id": "request-1"},
                "operation": "planner",
                "episode_id": "episode-1",
                "task_instruction": "Do the task.",
                "frames": [{"data": "not-a-url"}],
            },
        )
        self.assertEqual(status, 400)
        self.assertEqual(response["error"]["code"], "INVALID_REQUEST")

    def test_http_decoder_generates_request_id_and_accepts_encoded_bytes(self):
        backend = FakeBackend({"subtasks": ["next"]})
        app = create_app(HighLevelService(backend))
        status, response = call_asgi(
            app,
            "POST",
            "/v1/high-level/infer",
            {
                "request_context": {},
                "operation": "planner",
                "episode_id": "episode-bytes",
                "task_instruction": "Do it.",
                "frames": [
                    {
                        "data": to_jsonable(b"encoded-image"),
                        "mime_type": "image/jpeg",
                    }
                ],
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(response["request_context"]["request_id"])
        self.assertEqual(backend.requests[0].frames[0].data, b"encoded-image")

    def test_environment_factory_builds_asgi_app_without_contacting_sglang(self):
        app = create_sglang_app_from_env(
            {
                "HIGHLOW_SGLANG_BASE_URL": "http://sglang.internal:30000",
                "HIGHLOW_SGLANG_MODEL": "served-model",
            }
        )
        self.assertIsInstance(app, JsonAsgiApp)


class SGLangBackendTest(unittest.TestCase):
    def test_openai_payload_uses_images_json_schema_and_request_id(self):
        completion = {
            "id": "chatcmpl-1",
            "model": "served-model",
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"subtasks": ["Do the next phase."]})
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            "timing": {"generation_s": 0.25},
        }
        transport = FakeTransport(
            HttpResponse(status_code=200, body=json.dumps(completion).encode())
        )
        backend = SGLangOpenAIBackend(
            base_url="http://127.0.0.1:30000/v1",
            model="configured-model",
            api_key="secret",
            transport=transport,
        )
        service = HighLevelService(backend)
        response = service.infer(
            HighLevelRequest(
                request_context=request_context("request-123"),
                operation="planner",
                episode_id="episode-1",
                task_instruction="Do the task.",
                frames=(HighLevelFrame(data=b"abc", mime_type="image/jpeg"),),
            )
        )

        sent = transport.requests[0]
        self.assertEqual(sent.url, "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(sent.headers["Authorization"], "Bearer secret")
        payload = json.loads(sent.body)
        self.assertEqual(payload["user"], "request-123")
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        image_url = payload["messages"][0]["content"][0]["image_url"]["url"]
        self.assertEqual(
            image_url, "data:image/jpeg;base64," + base64.b64encode(b"abc").decode()
        )
        self.assertEqual(response.model, "served-model")
        self.assertEqual(response.timing_ms["server_generation_s"], 250.0)

    def test_health_is_non_throwing_and_capabilities_are_static(self):
        transport = FakeTransport(HttpResponse(status_code=200, body=b"ok"))
        backend = SGLangOpenAIBackend(
            base_url="http://localhost:30000",
            model="model",
            transport=transport,
        )
        self.assertEqual(backend.health().state, HealthState.READY)
        capabilities = backend.capabilities()
        self.assertEqual(capabilities.backend, "sglang-openai")
        self.assertTrue(capabilities.supports_structured_output)
        self.assertEqual(transport.requests[0].method, "GET")

    def test_http_failure_is_transport_error(self):
        transport = FakeTransport(HttpResponse(status_code=503, body=b"not ready"))
        backend = SGLangOpenAIBackend(
            base_url="http://localhost:30000",
            model="model",
            transport=transport,
        )
        service = HighLevelService(backend)
        with self.assertRaises(BackendTransportError):
            service.infer(
                HighLevelRequest(
                    request_context=request_context(),
                    operation="planner",
                    episode_id="episode-1",
                    task_instruction="Do the task.",
                )
            )

    def test_switch_requires_completed_status_and_next_subtask(self):
        for payload in (
            {
                "decision": "switch",
                "status": "running",
                "next_subtask": "next",
                "evidence": "not done",
            },
            {
                "decision": "switch",
                "status": "completed",
                "next_subtask": "",
                "evidence": "done",
            },
        ):
            service = HighLevelService(FakeBackend(payload))
            with self.assertRaises(HighLevelValidationError):
                service.infer(
                    HighLevelRequest(
                        request_context=request_context(),
                        operation="verifier",
                        episode_id="episode-1",
                        task_instruction="Do it.",
                        structured_context={"current_subtask": "first"},
                    )
                )


if __name__ == "__main__":
    unittest.main()
