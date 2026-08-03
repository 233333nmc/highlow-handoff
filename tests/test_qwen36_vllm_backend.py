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


if __name__ == "__main__":
    unittest.main()
