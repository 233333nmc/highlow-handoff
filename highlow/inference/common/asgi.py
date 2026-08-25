"""A tiny JSON ASGI adapter so services are not tied to a web framework."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import json
import logging
from typing import Any, Awaitable, Callable, Mapping

from .errors import InferenceServiceError
from .serialization import to_jsonable


JsonHandler = Callable[[Mapping[str, Any]], Any | Awaitable[Any]]
LifecycleHandler = Callable[[], Any | Awaitable[Any]]


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Route:
    method: str
    path: str
    handler: JsonHandler


class JsonAsgiApp:
    """Minimal ASGI application for versioned JSON inference endpoints."""

    def __init__(
        self,
        routes: list[Route],
        *,
        max_body_bytes: int = 32 << 20,
        on_startup: LifecycleHandler | None = None,
        on_shutdown: LifecycleHandler | None = None,
    ) -> None:
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")
        route_map: dict[tuple[str, str], JsonHandler] = {}
        for route in routes:
            key = (route.method.upper(), route.path)
            if key in route_map:
                raise ValueError(f"duplicate ASGI route: {key[0]} {key[1]}")
            route_map[key] = route.handler
        self._routes = route_map
        self._max_body_bytes = int(max_body_bytes)
        self._on_startup = on_startup
        self._on_shutdown = on_shutdown

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope.get("type") != "http":
            raise RuntimeError("JsonAsgiApp only supports HTTP and lifespan scopes")
        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", "/"))
        handler = self._routes.get((method, path))
        if handler is None:
            await self._respond(send, 404, {"error": {"code": "NOT_FOUND", "message": path}})
            return
        try:
            payload = await self._read_json(receive) if method not in {"GET", "HEAD"} else {}
            if inspect.iscoroutinefunction(handler):
                result = handler(payload)
            else:
                # Model inference and stdlib HTTP clients are synchronous.
                # Keep them off the ASGI event loop so health checks and other
                # episodes remain responsive while a request is running.
                result = await asyncio.to_thread(handler, payload)
            if inspect.isawaitable(result):
                result = await result
            try:
                response_value = to_jsonable(result)
            except Exception:
                logger.exception("failed to serialize ASGI handler result")
                await self._respond(
                    send,
                    500,
                    {
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": "internal inference service error",
                            "retryable": False,
                        }
                    },
                )
                return
            await self._respond(send, 200, response_value)
        except InferenceServiceError as exc:
            await self._respond(send, exc.status_code, exc.to_dict())
        except (TypeError, ValueError, KeyError) as exc:
            await self._respond(
                send,
                400,
                {
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": str(exc),
                        "retryable": False,
                    }
                },
            )
        except Exception:
            logger.exception("unhandled inference service error")
            await self._respond(
                send,
                500,
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "internal inference service error",
                        "retryable": False,
                    }
                },
            )

    async def _lifespan(self, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            message_type = message.get("type")
            if message_type == "lifespan.startup":
                try:
                    await self._run_lifecycle_handler(self._on_startup)
                except Exception as exc:
                    logger.exception("inference service startup failed")
                    await send(
                        {
                            "type": "lifespan.startup.failed",
                            "message": str(exc),
                        }
                    )
                    return
                await send({"type": "lifespan.startup.complete"})
            elif message_type == "lifespan.shutdown":
                try:
                    await self._run_lifecycle_handler(self._on_shutdown)
                except Exception as exc:
                    logger.exception("inference service shutdown failed")
                    await send(
                        {
                            "type": "lifespan.shutdown.failed",
                            "message": str(exc),
                        }
                    )
                    return
                await send({"type": "lifespan.shutdown.complete"})
                return

    @staticmethod
    async def _run_lifecycle_handler(handler: LifecycleHandler | None) -> None:
        if handler is None:
            return
        if inspect.iscoroutinefunction(handler):
            result = handler()
        else:
            result = await asyncio.to_thread(handler)
        if inspect.isawaitable(result):
            await result

    async def _read_json(self, receive: Any) -> Mapping[str, Any]:
        body = bytearray()
        more = True
        while more:
            message = await receive()
            if message.get("type") == "http.disconnect":
                raise InferenceServiceError(
                    "CLIENT_DISCONNECTED",
                    "client disconnected before the request body completed",
                    status_code=400,
                )
            if message.get("type") != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > self._max_body_bytes:
                raise InferenceServiceError(
                    "PAYLOAD_TOO_LARGE",
                    f"request body exceeds {self._max_body_bytes} bytes",
                    status_code=413,
                )
            more = bool(message.get("more_body", False))
        if not body:
            return {}
        value = json.loads(bytes(body))
        if not isinstance(value, Mapping):
            raise ValueError("JSON request body must be an object")
        return value

    @staticmethod
    async def _respond(send: Any, status: int, value: Any) -> None:
        body = json.dumps(value, separators=(",", ":")).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": int(status),
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
