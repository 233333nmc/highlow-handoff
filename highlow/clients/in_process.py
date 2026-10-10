"""Zero-serialization clients for services hosted in the same process."""

from __future__ import annotations

import inspect
from typing import Any, Generic, TypeVar


ServiceT = TypeVar("ServiceT")


class InProcessInferenceClient(Generic[ServiceT]):
    """Delegate the service surface directly while preserving client shape."""

    def __init__(self, service: ServiceT) -> None:
        infer = getattr(service, "infer", None)
        if not callable(infer):
            raise TypeError("in-process service must provide infer(request)")
        self.service = service

    def infer(self, request: Any) -> Any:
        return self.service.infer(request)  # type: ignore[attr-defined]

    def open_context(self, context: Any) -> Any:
        method = getattr(self.service, "open_context", None)
        if callable(method):
            return method(self._context_argument(method, context))
        method = getattr(self.service, "begin_episode", None)
        if callable(method):
            return method(context)
        return None

    def reset_context(self, context: Any) -> Any:
        method = getattr(self.service, "reset_context", None)
        if callable(method):
            return method(self._context_argument(method, context))
        # Adapt LowLevelService's begin/end episode vocabulary without making
        # the bridge depend on that concrete service class.
        end = getattr(self.service, "end_episode", None)
        begin = getattr(self.service, "begin_episode", None)
        if callable(end) and callable(begin):
            end(episode_id=context.episode_id)
            return begin(context)
        return None

    def close_context(self, context: Any) -> Any:
        method = getattr(self.service, "close_context", None)
        if callable(method):
            argument = self._context_argument(method, context)
            if argument is not context and hasattr(context, "terminated"):
                return method(argument, context)
            return method(argument)
        end = getattr(self.service, "end_episode", None)
        if not callable(end):
            return None
        # EpisodeResult-like objects are accepted positionally by the generic
        # low-level service; EpisodeContext/string values identify a context.
        if hasattr(context, "terminated") and hasattr(context, "step_count"):
            return end(context)
        episode_id = context if isinstance(context, str) else context.episode_id
        return end(episode_id=episode_id)

    def capabilities(self) -> Any:
        result = self._optional("capabilities")
        return {} if result is None else result

    def health(self) -> Any:
        result = self._optional("health")
        return {"status": "ok", "transport": "in_process"} if result is None else result

    def _optional(self, method_name: str, *args: Any) -> Any:
        method = getattr(self.service, method_name, None)
        if method is None:
            return None
        if not callable(method):
            raise TypeError(
                f"{type(self.service).__name__}.{method_name} must be callable"
            )
        return method(*args)

    @staticmethod
    def _context_argument(method: Any, context: Any) -> Any:
        """Adapt episode objects to services whose port takes an episode id."""

        try:
            first = next(iter(inspect.signature(method).parameters.values()))
        except (StopIteration, TypeError, ValueError):
            return context
        if first.name in {"episode_id", "context_id", "session_id"}:
            return context if isinstance(context, str) else context.episode_id
        return context


class InProcessHighLevelClient(InProcessInferenceClient[ServiceT]):
    """Named in-process client for a high-level service."""


class InProcessLowLevelClient(InProcessInferenceClient[ServiceT]):
    """Named in-process client for a low-level service."""


__all__ = [
    "InProcessHighLevelClient",
    "InProcessInferenceClient",
    "InProcessLowLevelClient",
]
