"""Lifecycle enforcement around an arbitrary low-level backend."""

from __future__ import annotations

from threading import RLock
from types import MappingProxyType
from typing import Mapping

from highlow.contracts import ActionEnvelope, EpisodeContext, EpisodeResult

from .core import (
    Backend,
    LifecycleError,
    LowLevelInferenceRequest,
    PolicyManifest,
    PolicyPrediction,
    PredictionContractError,
)
from .registry import BackendRegistry, default_registry


class LowLevelEpisodeContext:
    """A handle bound to one active episode in a :class:`LowLevelService`."""

    def __init__(self, service: LowLevelService, episode: EpisodeContext) -> None:
        self._service = service
        self._episode = episode
        self._active = True
        self._closing = False
        self._resetting = False
        # ``ending`` means reset still has to finish teardown; ``beginning``
        # means teardown returned successfully and a retry must resume at
        # begin_episode instead of tearing down a second time.
        self._reset_phase: str | None = None
        self._close_result: EpisodeResult | None = None
        self._close_result_frozen = False
        self._lock = RLock()
        self._last_step_id = -1
        self._last_prediction: PolicyPrediction | None = None

    @property
    def episode(self) -> EpisodeContext:
        return self._episode

    @property
    def active(self) -> bool:
        return self._active

    def infer(self, request: LowLevelInferenceRequest) -> PolicyPrediction:
        if not self._active:
            raise LifecycleError("episode context is no longer active")
        return self._service._infer_for_context(self, request)

    def finish(self, result: EpisodeResult | None = None) -> None:
        if not self._active:
            return
        self._service._finish_context(self, result)

    close = finish

    def __enter__(self) -> LowLevelEpisodeContext:
        if not self._active:
            raise LifecycleError("episode context is no longer active")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.finish()


class LowLevelService:
    """Route episode contexts and validate an arbitrary backend boundary.

    Calls for the same episode are serialized.  Distinct episodes may execute
    concurrently only when the backend advertises a larger
    ``max_concurrent_episodes`` value.
    """

    def __init__(self, backend: Backend) -> None:
        if not isinstance(backend, Backend):
            raise TypeError("backend must implement the Backend protocol")
        self._backend = backend
        self._opened = False
        self._closed = False
        self._contexts: dict[str, LowLevelEpisodeContext] = {}
        self._lock = RLock()

    @classmethod
    def from_registry(
        cls,
        backend_id: str,
        *,
        registry: BackendRegistry = default_registry,
        **backend_kwargs: object,
    ) -> LowLevelService:
        return cls(registry.create(backend_id, **backend_kwargs))

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def manifest(self) -> PolicyManifest:
        return self._backend.manifest

    @property
    def is_open(self) -> bool:
        return self._opened and not self._closed

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def active_context(self) -> LowLevelEpisodeContext | None:
        """Return the sole context, or ``None`` when zero/multiple are active."""

        with self._lock:
            if len(self._contexts) == 1:
                return next(iter(self._contexts.values()))
            return None

    @property
    def active_contexts(self) -> Mapping[str, LowLevelEpisodeContext]:
        with self._lock:
            return MappingProxyType(dict(self._contexts))

    def open(self) -> LowLevelService:
        with self._lock:
            self._open_locked()
        return self

    def _open_locked(self) -> None:
        if self._closed:
            raise LifecycleError("low-level service is closed")
        if not self._opened:
            self._backend.open()
            self._opened = True

    def begin_episode(self, episode: EpisodeContext) -> LowLevelEpisodeContext:
        if not isinstance(episode, EpisodeContext):
            raise TypeError("episode must be an EpisodeContext")
        with self._lock:
            self._open_locked()
            episode_id = episode.episode_id
            existing = self._contexts.get(episode_id)
            if existing is not None:
                if existing._closing:
                    raise LifecycleError(f"episode {episode_id!r} is closing")
                if existing._resetting:
                    raise LifecycleError(f"episode {episode_id!r} has an incomplete reset")
                if existing._active and existing.episode == episode:
                    # Retrying an episode-open request after a lost response is
                    # safe and must not reset a stateful policy a second time.
                    return existing
                raise LifecycleError(f"episode {episode_id!r} is already active")
            limit = self.manifest.max_concurrent_episodes
            if len(self._contexts) >= limit:
                raise LifecycleError(
                    f"backend {self.manifest.backend_id!r} permits at most "
                    f"{limit} concurrent episode(s)"
                )
            self._backend.begin_episode(episode)
            context = LowLevelEpisodeContext(self, episode)
            self._contexts[episode_id] = context
            return context

    start_episode = begin_episode
    open_context = begin_episode
    context = begin_episode

    def infer(self, request: LowLevelInferenceRequest) -> PolicyPrediction:
        """Run inference in the active episode context."""

        if not isinstance(request, LowLevelInferenceRequest):
            raise TypeError("request must be a LowLevelInferenceRequest")
        with self._lock:
            context = self._contexts.get(request.episode_id)
            if context is None:
                raise LifecycleError(
                    f"episode {request.episode_id!r} has not been started"
                )
        return self._infer_for_context(context, request)

    def _infer_for_context(
        self,
        context: LowLevelEpisodeContext,
        request: LowLevelInferenceRequest,
    ) -> PolicyPrediction:
        if not isinstance(request, LowLevelInferenceRequest):
            raise TypeError("request must be a LowLevelInferenceRequest")
        # Only serialize requests within one stateful episode. Different episode
        # contexts may execute concurrently when the manifest permits them.
        with context._lock:
            with self._lock:
                episode_id = context.episode.episode_id
                if self._contexts.get(episode_id) is not context or not context._active:
                    raise LifecycleError("episode context is no longer active")
                if context._closing:
                    raise LifecycleError(f"episode {episode_id!r} is closing")
                if context._resetting:
                    raise LifecycleError(
                        f"episode {episode_id!r} has an incomplete reset"
                    )
                if request.episode_id != episode_id:
                    raise PredictionContractError(
                        f"observation episode {request.episode_id!r} does not match "
                        f"active episode {episode_id!r}"
                    )
                if request.context.api_version not in self.manifest.supported_api_versions:
                    raise PredictionContractError(
                        f"API version {request.context.api_version!r} is not supported by "
                        f"backend {self.manifest.backend_id!r}"
                    )

                observation = request.observation
                missing = sorted(
                    set(self.manifest.required_modalities) - set(observation.modalities)
                )
                if missing:
                    raise PredictionContractError(
                        "observation is missing required modalities: "
                        + ", ".join(missing)
                    )
                allowed_observation_schemas = self.manifest.observation_schema_ids
                if allowed_observation_schemas:
                    declared_modalities = set(self.manifest.required_modalities).union(
                        self.manifest.optional_modalities
                    )
                    schema_values = (
                        observation.modalities.values()
                        if not declared_modalities
                        else (
                            modality
                            for name, modality in observation.modalities.items()
                            if name in declared_modalities
                        )
                    )
                    unsupported = sorted(
                        {
                            modality.schema_id
                            for modality in schema_values
                            if modality.schema_id not in allowed_observation_schemas
                        }
                    )
                    if unsupported:
                        raise PredictionContractError(
                            "observation contains undeclared schema(s): "
                            + ", ".join(unsupported)
                        )

                if request.step_id < context._last_step_id:
                    raise PredictionContractError(
                        f"stale observation step {request.step_id}; last completed "
                        f"step is {context._last_step_id}"
                    )
                if request.step_id == context._last_step_id:
                    if context._last_prediction is None:
                        raise LifecycleError("completed step has no cached prediction")
                    # episode_id + step_id form the low-level idempotency key.
                    # This avoids consuming the next cached model action when
                    # an HTTP response is lost and the same simulator tick is
                    # retried.
                    return context._last_prediction

            backend_result = self._backend.infer(request)
            if isinstance(backend_result, ActionEnvelope):
                prediction = PolicyPrediction(action=backend_result)
            elif isinstance(backend_result, PolicyPrediction):
                prediction = backend_result
            else:
                raise PredictionContractError(
                    "backend infer() must return PolicyPrediction or ActionEnvelope"
                )

            action = prediction.action
            if action.episode_id != request.episode_id:
                raise PredictionContractError(
                    "backend action episode_id does not match its observation"
                )
            if action.step_id != request.step_id:
                raise PredictionContractError(
                    "backend action step_id does not match its observation"
                )
            allowed_schemas = self.manifest.action_schema_ids
            if allowed_schemas and action.schema_id not in allowed_schemas:
                raise PredictionContractError(
                    f"backend returned undeclared action schema {action.schema_id!r}"
                )
            context._last_step_id = request.step_id
            context._last_prediction = prediction
            return prediction

    def end_episode(
        self,
        result: EpisodeResult | None = None,
        *,
        episode_id: str | None = None,
    ) -> None:
        with self._lock:
            if result is not None:
                if episode_id is not None and episode_id != result.episode_id:
                    raise PredictionContractError(
                        "episode_id does not match the supplied EpisodeResult"
                    )
                episode_id = result.episode_id
            if episode_id is None:
                if len(self._contexts) > 1:
                    raise LifecycleError(
                        "episode_id is required when multiple episodes are active"
                    )
                if len(self._contexts) == 1:
                    episode_id = next(iter(self._contexts))
            context = self._contexts.get(episode_id or "")
            if context is None:
                raise LifecycleError("no episode is active")
        self._finish_context(context, result)

    def _finish_context(
        self,
        context: LowLevelEpisodeContext,
        result: EpisodeResult | None,
    ) -> None:
        with context._lock:
            with self._lock:
                episode_id = context.episode.episode_id
                if self._contexts.get(episode_id) is not context or not context._active:
                    raise LifecycleError("episode context is no longer active")
                if result is not None:
                    if not isinstance(result, EpisodeResult):
                        raise TypeError("result must be an EpisodeResult or None")
                    if result.episode_id != context.episode.episode_id:
                        raise PredictionContractError(
                            "episode result does not match the active episode"
                        )
                if context._close_result_frozen:
                    if result is not None and result != context._close_result:
                        raise PredictionContractError(
                            "close retry conflicts with the original episode result"
                        )
                    result = context._close_result
                else:
                    # Freeze result semantics before calling the backend.  A
                    # later retry may identify only the episode, but must not
                    # silently drop success metrics from the first attempt.
                    context._close_result = result
                    context._close_result_frozen = True
                context._closing = True
                context._resetting = False
                context._reset_phase = None
            try:
                self._backend.end_episode(context.episode, result)
            except BaseException:
                raise
            with self._lock:
                context._active = False
                context._closing = False
                context._resetting = False
                context._reset_phase = None
                context._close_result = None
                context._close_result_frozen = False
                self._contexts.pop(episode_id, None)

    def reset_context(self, episode_id: str) -> LowLevelEpisodeContext:
        """Reset backend state for an episode while retaining its context handle."""

        with self._lock:
            context = self._contexts.get(episode_id)
        if context is None:
            raise LifecycleError(f"episode {episode_id!r} is not active")
        with context._lock:
            with self._lock:
                if self._contexts.get(episode_id) is not context or not context._active:
                    raise LifecycleError(f"episode {episode_id!r} is not active")
                if context._closing:
                    raise LifecycleError(f"episode {episode_id!r} is closing")
                if not context._resetting:
                    context._resetting = True
                    context._reset_phase = "ending"

            if context._reset_phase == "ending":
                self._backend.end_episode(context.episode, None)
                # Record the completed phase before begin_episode.  If begin
                # fails, a retry resumes here without invoking end_episode on
                # an already-ended backend context.
                context._reset_phase = "beginning"

            if context._reset_phase == "beginning":
                self._backend.begin_episode(context.episode)
            else:  # Internal state should never expose an ambiguous reset.
                raise LifecycleError(
                    f"episode {episode_id!r} has an invalid reset phase"
                )

            context._last_step_id = -1
            context._last_prediction = None
            context._resetting = False
            context._reset_phase = None
            return context

    def close_context(
        self,
        episode_id: str,
        result: EpisodeResult | None = None,
    ) -> None:
        with self._lock:
            context = self._contexts.get(episode_id)
        if context is None:
            return
        try:
            self._finish_context(context, result)
        except LifecycleError:
            # A concurrent or retried close that already completed is safe.
            with self._lock:
                if episode_id not in self._contexts:
                    return
            raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            contexts = tuple(self._contexts.values())
            self._contexts.clear()
            self._closed = True
        pending_error: BaseException | None = None
        for context in contexts:
            with context._lock:
                episode_id = context.episode.episode_id
                try:
                    result = (
                        context._close_result
                        if context._close_result_frozen
                        else None
                    )
                    self._backend.end_episode(context.episode, result)
                except BaseException as exc:  # preserve cleanup while propagating failure
                    if pending_error is None:
                        pending_error = exc
                finally:
                    context._active = False
                    context._closing = False
                    context._resetting = False
                    context._reset_phase = None
                    context._close_result = None
                    context._close_result_frozen = False
        if self._opened:
            try:
                self._backend.close()
            except BaseException as exc:
                if pending_error is None:
                    pending_error = exc
        if pending_error is not None:
            raise pending_error

    def __enter__(self) -> LowLevelService:
        return self.open()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = ["LowLevelEpisodeContext", "LowLevelService"]
