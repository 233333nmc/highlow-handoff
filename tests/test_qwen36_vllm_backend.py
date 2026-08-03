"""Unit tests for the Qwen3.6 vLLM high-level client."""

from __future__ import annotations

import unittest

import numpy as np

from hy_vla.hierarchical import FrameRecord, Qwen36VllmBackend


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
                            '"keyframe_positions":[1],"confidence":0.9}'
                        )
                    }
                }
            ]
        }


class _FakeSession:
    def __init__(self) -> None:
        self.url: str | None = None
        self.payload: dict[str, object] | None = None
        self.timeout: float | None = None

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
        return _FakeResponse()


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
        self.assertEqual([1], prediction.keyframe_positions)
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

    def test_invalid_image_selection_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            Qwen36VllmBackend(
                server_url="http://127.0.0.1:8001",
                image_selection="middle",
                warmup_calls=0,
            )


if __name__ == "__main__":
    unittest.main()
