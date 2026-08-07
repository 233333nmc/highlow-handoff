"""Unit tests for the Qwen3.6 vLLM high-level client."""

from __future__ import annotations

import unittest

import numpy as np

from hy_vla.hierarchical import FrameRecord, Gemma4VllmBackend, Qwen36VllmBackend


class _FakeResponse:
    text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "usage": {
                "prompt_tokens": 321,
                "completion_tokens": 12,
                "total_tokens": 333,
            },
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"current_subtask":"Place the french fries on the tray.",'
                            '"confidence":0.9}'
                        )
                    }
                }
            ]
        }


class _FakeMinimalResponse:
    text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 6,
                "total_tokens": 106,
            },
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"next_subtask":"Place the french fries on the tray."}'
                        )
                    }
                }
            ],
        }


class _FakeMinimalStatusResponse:
    text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "usage": {
                "prompt_tokens": 108,
                "completion_tokens": 9,
                "total_tokens": 117,
            },
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"action":"continue",'
                            '"next_subtask":"Place the burger on the tray.",'
                            '"subtask_status":"unfinished"}'
                        )
                    }
                }
            ],
        }


class _FakeCompletionStatusResponse:
    text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "usage": {
                "prompt_tokens": 96,
                "completion_tokens": 5,
                "total_tokens": 101,
            },
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"decision":"same","status":"running",'
                            '"next_subtask":"","evidence":"burger not placed yet"}'
                        )
                    }
                }
            ],
        }


class _FakeSubtaskPlanResponse:
    text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "usage": {
                "prompt_tokens": 88,
                "completion_tokens": 14,
                "total_tokens": 102,
            },
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"subtasks":["Move the burger onto the tray.",'
                            '"Move the fries onto the tray."],"confidence":0.8}'
                        )
                    }
                }
            ],
        }


class _FakeSession:
    def __init__(self, response: object | None = None) -> None:
        self.url: str | None = None
        self.payload: dict[str, object] | None = None
        self.timeout: float | None = None
        self.response = response or _FakeResponse()

    def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        timeout: float,
    ) -> _FakeResponse:
        self.url = url
        self.payload = json
        self.timeout = timeout
        return self.response


class Qwen36VllmBackendTest(unittest.TestCase):
    def test_predict_uses_image_chat_json_and_records_mtp_timing(self) -> None:
        backend = Qwen36VllmBackend(
            server_url="http://127.0.0.1:8001",
            model_name="qwen3.6-27b-mtp",
            max_images=3,
            max_new_tokens=64,
            warmup_calls=0,
        )
        session = _FakeSession()
        backend.session = session
        frame = FrameRecord(
            frame_id=50,
            timestamp_s=2.0,
            camera_images={
                "head_camera": np.zeros((8, 8, 3), dtype=np.uint8),
            },
        )

        prediction = backend.predict(
            prompt="Task: place_burger_fries\nExecution report: status=unknown",
            memory_frames=[],
            recent_frames=[frame],
            timestamp_s=2.2,
            phase_index=1,
            phase_prompt="Place the french fries on the tray.",
            execution_report=None,
        )

        self.assertEqual(
            "Place the french fries on the tray.",
            prediction.current_subtask,
        )
        self.assertTrue(prediction.parse_ok)
        self.assertEqual(1.0, prediction.timing_ms["mtp_enabled"])
        self.assertEqual(1.0, prediction.timing_ms["mtp_speculative_tokens"])
        self.assertEqual(1.0, prediction.timing_ms["usage_available"])
        self.assertEqual(321.0, prediction.timing_ms["usage_prompt_tokens"])
        self.assertEqual(12.0, prediction.timing_ms["usage_completion_tokens"])
        self.assertEqual(333.0, prediction.timing_ms["usage_total_tokens"])
        self.assertEqual(1.0, prediction.timing_ms["image_count"])

        self.assertEqual(
            "http://127.0.0.1:8001/v1/chat/completions",
            session.url,
        )
        self.assertEqual(120.0, session.timeout)
        assert session.payload is not None
        self.assertEqual("qwen3.6-27b-mtp", session.payload["model"])
        self.assertEqual(
            {"enable_thinking": False},
            session.payload["chat_template_kwargs"],
        )
        self.assertEqual(
            {"type": "json_object"},
            session.payload["response_format"],
        )
        message = session.payload["messages"][0]
        content = message["content"]
        self.assertEqual("image_url", content[0]["type"])
        self.assertTrue(
            content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )
        self.assertEqual("text", content[1]["type"])

    def test_qwen35_alias_uses_qwen35_model_name(self) -> None:
        backend = Qwen36VllmBackend(
            server_url="http://127.0.0.1:8001",
            model_name="qwen3.5-9b-mtp",
            max_images=1,
            max_new_tokens=48,
            output_schema="minimal_status",
            warmup_calls=0,
            backend_label="Qwen3.5 vLLM",
            mtp_speculative_tokens=2,
        )
        session = _FakeSession(response=_FakeMinimalStatusResponse())
        backend.session = session
        frame = FrameRecord(
            frame_id=25,
            timestamp_s=1.0,
            camera_images={
                "head_camera": np.zeros((8, 8, 3), dtype=np.uint8),
            },
        )

        prediction = backend.predict(
            prompt="Task: place_burger_fries",
            memory_frames=[],
            recent_frames=[frame],
            timestamp_s=1.0,
            phase_index=0,
            phase_prompt="Place the burger on the tray.",
            execution_report=None,
        )

        self.assertTrue(prediction.parse_ok)
        assert session.payload is not None
        self.assertEqual("qwen3.5-9b-mtp", session.payload["model"])
        self.assertEqual(48, session.payload["max_tokens"])
        self.assertEqual(
            {"enable_thinking": False},
            session.payload["chat_template_kwargs"],
        )
        self.assertEqual(2.0, prediction.timing_ms["mtp_speculative_tokens"])

    def test_latest_image_selection_records_compression_metrics(self) -> None:
        backend = Qwen36VllmBackend(
            server_url="http://127.0.0.1:8001",
            max_images=3,
            max_new_tokens=32,
            image_selection="latest",
            image_max_side=8,
            warmup_calls=0,
        )
        session = _FakeSession()
        backend.session = session
        frames = [
            FrameRecord(
                frame_id=frame_id,
                timestamp_s=float(index),
                camera_images={
                    "head_camera": np.full((8, 16, 3), index, dtype=np.uint8),
                },
            )
            for index, frame_id in enumerate([0, 25, 50])
        ]

        prediction = backend.predict(
            prompt="Task: place_burger_fries",
            memory_frames=[],
            recent_frames=frames,
            timestamp_s=2.2,
            phase_index=1,
            phase_prompt="Place the french fries on the tray.",
            execution_report=None,
        )

        self.assertIn("evidence_frame_ids=[50]", prediction.raw_text)
        self.assertEqual(1.0, prediction.timing_ms["image_count"])
        self.assertEqual(8.0, prediction.timing_ms["image_avg_width"])
        self.assertEqual(4.0, prediction.timing_ms["image_avg_height"])
        self.assertGreater(prediction.timing_ms["image_encoded_bytes"], 0.0)
        assert session.payload is not None
        content = session.payload["messages"][0]["content"]
        self.assertEqual("image_url", content[0]["type"])
        self.assertEqual("text", content[1]["type"])
        self.assertIn("image 1 -> recent position 3", content[1]["text"])

    def test_minimal_output_schema_accepts_next_subtask_only(self) -> None:
        backend = Qwen36VllmBackend(
            server_url="http://127.0.0.1:8001",
            max_images=1,
            output_schema="minimal",
            warmup_calls=0,
        )
        session = _FakeSession(response=_FakeMinimalResponse())
        backend.session = session
        frame = FrameRecord(
            frame_id=50,
            timestamp_s=2.0,
            camera_images={
                "head_camera": np.zeros((8, 8, 3), dtype=np.uint8),
            },
        )

        prediction = backend.predict(
            prompt="Task: place_burger_fries",
            memory_frames=[],
            recent_frames=[frame],
            timestamp_s=2.2,
            phase_index=1,
            phase_prompt="Place the french fries on the tray.",
            execution_report=None,
        )

        self.assertEqual(
            "Place the french fries on the tray.",
            prediction.current_subtask,
        )
        assert session.payload is not None
        content = session.payload["messages"][0]["content"]
        self.assertIn('"next_subtask": string', content[1]["text"])

    def test_minimal_status_output_schema_can_request_continue(self) -> None:
        backend = Qwen36VllmBackend(
            server_url="http://127.0.0.1:8001",
            max_images=1,
            output_schema="minimal_status",
            warmup_calls=0,
        )
        session = _FakeSession(response=_FakeMinimalStatusResponse())
        backend.session = session
        frame = FrameRecord(
            frame_id=50,
            timestamp_s=2.0,
            camera_images={
                "head_camera": np.zeros((8, 8, 3), dtype=np.uint8),
            },
        )

        prediction = backend.predict(
            prompt="Task: place_burger_fries",
            memory_frames=[],
            recent_frames=[frame],
            timestamp_s=2.2,
            phase_index=1,
            phase_prompt="Place the french fries on the tray.",
            execution_report=None,
        )

        self.assertEqual("continue", prediction.action)
        self.assertEqual("unfinished", prediction.subtask_status)
        self.assertEqual("Place the burger on the tray.", prediction.current_subtask)
        assert session.payload is not None
        content = session.payload["messages"][0]["content"]
        self.assertIn('"action": "continue" | "switch"', content[1]["text"])
        self.assertIn('"subtask_status"', content[1]["text"])
        self.assertNotIn("confidence", content[1]["text"])

    def test_completion_status_schema_verifies_only_current_subtask(self) -> None:
        backend = Qwen36VllmBackend(
            server_url="http://127.0.0.1:8001",
            max_images=1,
            output_schema="completion_status",
            warmup_calls=0,
        )
        session = _FakeSession(response=_FakeCompletionStatusResponse())
        backend.session = session
        frame = FrameRecord(
            frame_id=55,
            timestamp_s=2.2,
            camera_images={
                "head_camera": np.zeros((8, 8, 3), dtype=np.uint8),
            },
        )

        prediction = backend.predict(
            prompt="Current low-level subtask: Place the burger on the tray.",
            memory_frames=[],
            recent_frames=[frame],
            timestamp_s=2.2,
            phase_index=1,
            phase_prompt="Place the french fries on the tray.",
            execution_report=None,
        )

        self.assertEqual("continue", prediction.action)
        self.assertEqual("unfinished", prediction.subtask_status)
        self.assertEqual("running", prediction.high_level_status)
        self.assertEqual("burger not placed yet", prediction.evidence)
        self.assertEqual("", prediction.verifier_next_subtask)
        assert session.payload is not None
        content = session.payload["messages"][0]["content"]
        self.assertIn('"decision": "same" | "switch"', content[1]["text"])
        self.assertIn('"status": "running" | "completed"', content[1]["text"])
        self.assertIn('"next_subtask": string', content[1]["text"])
        self.assertIn('"evidence": string', content[1]["text"])
        self.assertNotIn("confidence", content[1]["text"])

    def test_subtask_plan_schema_allows_text_only_planning(self) -> None:
        backend = Qwen36VllmBackend(
            server_url="http://127.0.0.1:8001",
            max_images=1,
            output_schema="completion_status",
            warmup_calls=0,
        )
        session = _FakeSession(response=_FakeSubtaskPlanResponse())
        backend.session = session

        prediction = backend.predict(
            prompt="Task instruction: put the burger and fries on the tray.",
            memory_frames=[],
            recent_frames=[],
            timestamp_s=0.0,
            phase_index=0,
            phase_prompt="fallback",
            execution_report=None,
            output_schema="subtask_plan",
            allow_text_only=True,
        )

        self.assertTrue(prediction.parse_ok)
        self.assertEqual(
            ["Move the burger onto the tray.", "Move the fries onto the tray."],
            prediction.subtasks,
        )
        self.assertEqual("Move the burger onto the tray.", prediction.current_subtask)
        self.assertIn("text_only=true", prediction.raw_text)
        self.assertEqual(0.0, prediction.timing_ms["image_count"])
        assert session.payload is not None
        content = session.payload["messages"][0]["content"]
        self.assertEqual("text", content[0]["type"])
        self.assertIn('"subtasks": [string]', content[0]["text"])

    def test_gemma4_backend_omits_qwen_chat_template_kwargs(self) -> None:
        backend = Gemma4VllmBackend(
            server_url="http://127.0.0.1:8002",
            max_images=1,
            warmup_calls=0,
        )
        session = _FakeSession(response=_FakeMinimalStatusResponse())
        backend.session = session
        frame = FrameRecord(
            frame_id=50,
            timestamp_s=2.0,
            camera_images={
                "head_camera": np.zeros((8, 8, 3), dtype=np.uint8),
            },
        )

        prediction = backend.predict(
            prompt="Task: place_burger_fries",
            memory_frames=[],
            recent_frames=[frame],
            timestamp_s=2.2,
            phase_index=1,
            phase_prompt="Place the french fries on the tray.",
            execution_report=None,
        )

        self.assertEqual("continue", prediction.action)
        assert session.payload is not None
        self.assertEqual("gemma-4-12B-it", session.payload["model"])
        self.assertNotIn("chat_template_kwargs", session.payload)
        content = session.payload["messages"][0]["content"]
        self.assertIn('"action": "continue" | "switch"', content[1]["text"])

    def test_first_latest_selection_keeps_temporal_endpoints(self) -> None:
        backend = Qwen36VllmBackend(
            server_url="http://127.0.0.1:8001",
            max_images=2,
            image_selection="first_latest",
            warmup_calls=0,
        )
        frames = [
            FrameRecord(
                frame_id=frame_id,
                timestamp_s=float(index),
                camera_images={
                    "head_camera": np.full((8, 8, 3), index, dtype=np.uint8),
                },
            )
            for index, frame_id in enumerate([0, 25, 50])
        ]

        _images, frame_ids, recent_positions = backend._select_images(
            memory_frames=[],
            recent_frames=frames,
        )

        self.assertEqual([0, 50], frame_ids)
        self.assertEqual([1, 3], recent_positions)

    def test_diverse_selection_keeps_latest_and_visually_novel_frame(self) -> None:
        backend = Qwen36VllmBackend(
            server_url="http://127.0.0.1:8001",
            max_images=2,
            image_selection="diverse",
            warmup_calls=0,
        )
        frames = [
            FrameRecord(
                frame_id=0,
                timestamp_s=0.0,
                camera_images={
                    "head_camera": np.zeros((8, 8, 3), dtype=np.uint8),
                },
            ),
            FrameRecord(
                frame_id=25,
                timestamp_s=1.0,
                camera_images={
                    "head_camera": np.full((8, 8, 3), 128, dtype=np.uint8),
                },
            ),
            FrameRecord(
                frame_id=50,
                timestamp_s=2.0,
                camera_images={
                    "head_camera": np.full((8, 8, 3), 130, dtype=np.uint8),
                },
            ),
        ]

        _images, frame_ids, recent_positions = backend._select_images(
            memory_frames=[],
            recent_frames=frames,
        )

        self.assertEqual([0, 50], frame_ids)
        self.assertEqual([1, 3], recent_positions)

    def test_invalid_image_selection_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            Qwen36VllmBackend(
                server_url="http://127.0.0.1:8001",
                image_selection="middle",
                warmup_calls=0,
            )


if __name__ == "__main__":
    unittest.main()
