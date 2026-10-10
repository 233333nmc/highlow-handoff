"""Tests for dependency-free inference infrastructure."""

from __future__ import annotations

import asyncio
import base64
import json
import unittest

from highlow.inference.common import JsonAsgiApp, Route, from_jsonable, to_jsonable
from highlow.inference.common.health import HealthReport, HealthState


async def _call_asgi(app: JsonAsgiApp, method: str, path: str, body: object = None):
    sent: list[dict[str, object]] = []
    raw = b"" if body is None else json.dumps(body).encode("utf-8")
    received = False

    async def receive():
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": raw, "more_body": False}

    async def send(message):
        sent.append(message)

    await app(
        {"type": "http", "method": method, "path": path},
        receive,
        send,
    )
    status = sent[0]["status"]
    payload = json.loads(sent[1]["body"])
    return status, payload


class CommonInferenceTest(unittest.TestCase):
    def test_dataclass_and_enum_are_json_serializable(self) -> None:
        value = to_jsonable(HealthReport("low-level", HealthState.READY))

        self.assertEqual("ready", value["state"])
        self.assertEqual("low-level", value["service"])

    def test_bytes_have_an_explicit_round_trip(self) -> None:
        encoded = to_jsonable({"image": b"jpeg"})
        self.assertEqual(encoded["image"]["encoding"], "base64")
        self.assertEqual(
            encoded["image"]["data"], base64.b64encode(b"jpeg").decode("ascii")
        )
        self.assertEqual(from_jsonable(encoded), {"image": b"jpeg"})

    def test_asgi_app_dispatches_json_route(self) -> None:
        app = JsonAsgiApp(
            [Route("POST", "/v1/echo", lambda body: {"echo": body["value"]})]
        )

        status, payload = asyncio.run(
            _call_asgi(app, "POST", "/v1/echo", {"value": 7})
        )

        self.assertEqual(200, status)
        self.assertEqual({"echo": 7}, payload)

    def test_asgi_app_returns_structured_404(self) -> None:
        status, payload = asyncio.run(_call_asgi(JsonAsgiApp([]), "GET", "/missing"))

        self.assertEqual(404, status)
        self.assertEqual("NOT_FOUND", payload["error"]["code"])

    def test_asgi_lifespan_runs_startup_and_shutdown(self) -> None:
        events = []
        app = JsonAsgiApp(
            [],
            on_startup=lambda: events.append("startup"),
            on_shutdown=lambda: events.append("shutdown"),
        )

        async def run_lifespan():
            incoming = [
                {"type": "lifespan.startup"},
                {"type": "lifespan.shutdown"},
            ]
            sent = []

            async def receive():
                return incoming.pop(0)

            async def send(message):
                sent.append(message)

            await app({"type": "lifespan"}, receive, send)
            return sent

        sent = asyncio.run(run_lifespan())
        self.assertEqual(events, ["startup", "shutdown"])
        self.assertEqual(
            [message["type"] for message in sent],
            ["lifespan.startup.complete", "lifespan.shutdown.complete"],
        )


if __name__ == "__main__":
    unittest.main()
