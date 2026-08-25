"""A transport- and framework-neutral high/low inference coordinator."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Mapping

from highlow.contracts import (
    ActionEnvelope,
    EpisodeContext,
    EpisodeResult,
    ObservationEnvelope,
    RequestContext,
)

from .protocols import HighLevelClient, LowLevelClient


@dataclass(frozen=True, slots=True)
class BridgeTurn:
    """Canonical input shared by request factories for one control tick."""

    request_context: RequestContext
    episode_context: EpisodeContext
    observation: ObservationEnvelope
    instruction: str
    instruction_revision: int
    high_level_call_count: int = 0

    @property
    def episode_id(self) -> str:
        return self.episode_context.episode_id


RequestFactory = Callable[[BridgeTurn], Any]
LowRequestFactory = Callable[[BridgeTurn, Any], Any]
HighLevelTrigger = Callable[[BridgeTurn], bool]


@dataclass(slots=True)
class _EpisodeState:
    context: EpisodeContext
    instruction: str
    opening: bool = True
    open_phase: str | None = "high"
    resetting: bool = False
    reset_phase: str | None = None
    instruction_revision: int = 0
    high_level_call_count: int = 0
    last_step_id: int = -1
    last_action: ActionEnvelope | None = field(default=None, repr=False)
    pending_turn: BridgeTurn | None = field(default=None, repr=False)
    pending_high_response: Any = field(default=None, repr=False)
    pending_high_invoked: bool = False
    closing: bool = False
    pending_close_value: EpisodeContext | EpisodeResult | None = field(
        default=None,
        repr=False,
    )
    lock: Any = field(default_factory=RLock, repr=False)


def _initial_step_trigger(turn: BridgeTurn) -> bool:
    """Conservative default: call high level once, then let low level run."""

    return turn.high_level_call_count == 0


def _optional_call(target: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(target, method_name, None)
    if method is None:
        return None
    if not callable(method):
        raise TypeError(f"{type(target).__name__}.{method_name} must be callable")
    return method(*args, **kwargs)


def _lookup(value: Any, names: tuple[str, ...]) -> Any:
    """Read the first present key/attribute without depending on its class."""

    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _instruction_from_response(response: Any, current: str) -> str:
    if response is None:
        return current
    if isinstance(response, str):
        candidate = response
    else:
        output = _lookup(response, ("output", "result"))
        source = response if output is None else output
        decision = _lookup(source, ("decision",))
        if decision is not None:
            decision = getattr(decision, "value", decision)
            if str(decision).strip().lower() != "switch":
                return current
        candidate = _lookup(
            source,
            (
                "instruction",
                "low_level_instruction",
                "next_instruction",
                "next_subtask",
            ),
        )
        if candidate is None:
            subtasks = _lookup(source, ("subtasks",))
            if isinstance(subtasks, (tuple, list)) and subtasks:
                candidate = subtasks[0]
    if candidate is None:
        return current
    if not isinstance(candidate, str):
        raise TypeError("high-level response instruction must be a string")
    candidate = candidate.strip()
    return candidate or current


def _validate_high_response(
    response: Any,
    turn: BridgeTurn,
    high_request: Any = None,
) -> None:
    """Reject a correlated response routed to the wrong episode/request."""

    if response is None:
        return
    episode_id = _lookup(response, ("episode_id",))
    if episode_id is not None and str(episode_id) != turn.episode_id:
        raise ValueError("high-level response episode_id does not match the request")

    response_context = _lookup(response, ("request_context", "context"))
    response_request_id = _lookup(response_context, ("request_id",))
    if (
        response_request_id is not None
        and str(response_request_id) != turn.request_context.request_id
    ):
        raise ValueError("high-level response request_id does not match the request")

    request_operation = _lookup(high_request, ("operation",))
    response_operation = _lookup(response, ("operation",))
    if request_operation is not None and response_operation is not None:
        request_value = getattr(request_operation, "value", request_operation)
        response_value = getattr(response_operation, "value", response_operation)
        if str(request_value) != str(response_value):
            raise ValueError("high-level response operation does not match the request")


def _action_from_response(response: Any) -> ActionEnvelope:
    if isinstance(response, ActionEnvelope):
        return response
    candidate = _lookup(response, ("action", "action_envelope", "output"))
    if isinstance(candidate, ActionEnvelope):
        return candidate

    # Accept a structurally compatible response while still normalising the
    # value crossing the bridge to the canonical contract.
    value = candidate if candidate is not None else response
    episode_id = _lookup(value, ("episode_id",))
    step_id = _lookup(value, ("step_id",))
    schema_id = _lookup(value, ("schema_id",))
    data = _lookup(value, ("data",))
    if episode_id is None or step_id is None or schema_id is None or data is None:
        raise TypeError(
            "low-level infer() must return ActionEnvelope or an object containing one"
        )
    horizon = _lookup(value, ("horizon",))
    coordinate_frame = _lookup(value, ("coordinate_frame",))
    metadata = _lookup(value, ("metadata",))
    return ActionEnvelope(
        episode_id=str(episode_id),
        step_id=int(step_id),
        schema_id=str(schema_id),
        data=data,
        horizon=1 if horizon is None else int(horizon),
        coordinate_frame="" if coordinate_frame is None else str(coordinate_frame),
        metadata={} if metadata is None else metadata,
    )


class DualSystemBridge:
    """Coordinate high- and low-level services without driving a simulator.

    The official simulator loop supplies :class:`ObservationEnvelope` values,
    executes the returned :class:`ActionEnvelope`, and retains ownership of
    environment reset/step/action calls.  This class only manages inference
    context and instruction handoff.
    """

    def __init__(
        self,
        high_client: HighLevelClient,
        low_client: LowLevelClient,
        *,
        high_request_factory: RequestFactory | None = None,
        low_request_factory: LowRequestFactory | None = None,
        high_level_trigger: HighLevelTrigger | None = None,
    ) -> None:
        if not callable(getattr(high_client, "infer", None)):
            raise TypeError("high_client must provide infer(request)")
        if not callable(getattr(low_client, "infer", None)):
            raise TypeError("low_client must provide infer(request)")
        self.high_client = high_client
        self.low_client = low_client
        if high_request_factory is None or low_request_factory is None:
            # Lazy import avoids a module cycle and keeps bridge discovery free
            # of service/model dependencies. These factories produce the same
            # public contracts accepted by the bundled inference services.
            from .service_factories import low_level_request, planner_request

            high_request_factory = high_request_factory or planner_request
            low_request_factory = low_request_factory or low_level_request
        self.high_request_factory = high_request_factory
        self.low_request_factory = low_request_factory
        self.high_level_trigger = high_level_trigger or _initial_step_trigger
        self._episodes: dict[str, _EpisodeState] = {}
        self._lock = RLock()

    def open_context(self, context: EpisodeContext) -> None:
        """Open inference state for an episode already created by the loop."""

        if not isinstance(context, EpisodeContext):
            raise TypeError("context must be an EpisodeContext")

        with self._lock:
            existing = self._episodes.get(context.episode_id)
            if existing is None:
                state = _EpisodeState(
                    context=context,
                    instruction=context.instruction,
                )
                # Publish an opening tombstone before either remote call.  If
                # a response is lost, the caller retains the episode id and a
                # retry resumes the same open operation.
                self._episodes[context.episode_id] = state
            else:
                state = existing

        with state.lock:
            with self._lock:
                if self._episodes.get(context.episode_id) is not state:
                    raise RuntimeError(f"episode {context.episode_id!r} failed to open")
            if state.context != context:
                raise ValueError(f"episode {context.episode_id!r} is already open")
            if state.closing:
                raise RuntimeError(f"episode {context.episode_id!r} is closing")
            if state.resetting:
                raise RuntimeError(f"episode {context.episode_id!r} is resetting")
            if not state.opening:
                return

            if state.open_phase == "high":
                _optional_call(self.high_client, "open_context", context)
                state.open_phase = "low"
            if state.open_phase == "low":
                _optional_call(self.low_client, "open_context", context)
            else:
                raise RuntimeError(
                    f"episode {context.episode_id!r} has an invalid open phase"
                )
            state.opening = False
            state.open_phase = None

    def reset_context(self, context: EpisodeContext | str) -> None:
        """Reset inference state only; this never resets the simulator."""

        episode_id = context if isinstance(context, str) else context.episode_id
        with self._lock:
            try:
                state = self._episodes[episode_id]
            except KeyError as exc:
                raise KeyError(f"episode {episode_id!r} is not open") from exc

        with state.lock:
            with self._lock:
                if self._episodes.get(episode_id) is not state:
                    raise KeyError(f"episode {episode_id!r} is not open")
            if state.closing:
                raise RuntimeError(f"episode {episode_id!r} is closing")
            if state.opening:
                raise RuntimeError(f"episode {episode_id!r} has not finished opening")
            if not state.resetting:
                state.resetting = True
                state.reset_phase = "high"
            if state.reset_phase == "high":
                _optional_call(self.high_client, "reset_context", state.context)
                state.reset_phase = "low"
            if state.reset_phase == "low":
                _optional_call(self.low_client, "reset_context", state.context)
            else:
                raise RuntimeError(
                    f"episode {episode_id!r} has an invalid reset phase"
                )
            state.instruction = state.context.instruction
            state.instruction_revision = 0
            state.high_level_call_count = 0
            state.last_step_id = -1
            state.last_action = None
            state.pending_turn = None
            state.pending_high_response = None
            state.pending_high_invoked = False
            state.resetting = False
            state.reset_phase = None

    def close_context(
        self,
        context: EpisodeContext | EpisodeResult | str,
    ) -> None:
        """Close service context; idempotent and independent of env teardown."""

        episode_id = context if isinstance(context, str) else context.episode_id
        with self._lock:
            state = self._episodes.get(episode_id)
        if state is None:
            return
        with state.lock:
            with self._lock:
                if self._episodes.get(episode_id) is not state:
                    return
                state.closing = True
            requested_close_value = (
                context if isinstance(context, EpisodeResult) else state.context
            )
            if state.pending_close_value is None:
                # Freeze the first close's result semantics. A retry that only
                # carries an episode id must still deliver the original result.
                state.pending_close_value = requested_close_value
            elif (
                isinstance(context, EpisodeResult)
                and context != state.pending_close_value
            ):
                raise ValueError("close retry conflicts with the original episode result")
            close_value = state.pending_close_value
            first_error: BaseException | None = None
            for client in (self.low_client, self.high_client):
                try:
                    _optional_call(client, "close_context", close_value)
                except BaseException as exc:  # close both sides before surfacing it
                    if first_error is None:
                        first_error = exc
            if first_error is not None:
                # Retain a non-inferable local tombstone so close can be
                # retried after a transient transport failure. Remote close
                # endpoints are required to be idempotent.
                raise first_error
            with self._lock:
                if self._episodes.get(episode_id) is state:
                    del self._episodes[episode_id]

    def infer(
        self,
        observation: ObservationEnvelope,
        *,
        request_context: RequestContext | None = None,
    ) -> ActionEnvelope:
        """Run high-level handoff followed by one low-level inference call."""

        with self._lock:
            try:
                state = self._episodes[observation.episode_id]
            except KeyError as exc:
                raise KeyError(
                    f"episode {observation.episode_id!r} is not open"
                ) from exc
        # Only requests for the same episode serialize; service inference does
        # not hold the bridge-wide registry lock.
        with state.lock:
            with self._lock:
                if self._episodes.get(observation.episode_id) is not state:
                    raise KeyError(
                        f"episode {observation.episode_id!r} is not open"
                    )
            if state.closing:
                raise RuntimeError(
                    f"episode {observation.episode_id!r} is closing"
                )
            if state.opening:
                raise RuntimeError(
                    f"episode {observation.episode_id!r} has not finished opening"
                )
            if state.resetting:
                raise RuntimeError(
                    f"episode {observation.episode_id!r} has an incomplete reset"
                )
            if observation.step_id < state.last_step_id:
                raise ValueError(
                    f"stale observation step {observation.step_id}; "
                    f"last completed step is {state.last_step_id}"
                )
            if observation.step_id == state.last_step_id:
                if state.last_action is None:
                    raise RuntimeError("completed bridge step has no cached action")
                return state.last_action

            if state.pending_turn is not None:
                if state.pending_turn.observation.step_id != observation.step_id:
                    raise RuntimeError(
                        f"step {state.pending_turn.observation.step_id} has an unresolved "
                        "low-level request and must be retried before advancing"
                    )
                after = state.pending_turn
                high_response = state.pending_high_response
                high_invoked = state.pending_high_invoked
            else:
                request_context = request_context or RequestContext()
                before = BridgeTurn(
                    request_context=request_context,
                    episode_context=state.context,
                    observation=observation,
                    instruction=state.instruction,
                    instruction_revision=state.instruction_revision,
                    high_level_call_count=state.high_level_call_count,
                )
                high_response = None
                high_invoked = self.high_level_trigger(before)
                if high_invoked:
                    high_request = self.high_request_factory(before)
                    high_response = self.high_client.infer(high_request)
                    _validate_high_response(high_response, before, high_request)
                instruction = _instruction_from_response(
                    high_response, state.instruction
                )
                revision = state.instruction_revision + int(
                    instruction != state.instruction
                )
                after = BridgeTurn(
                    request_context=request_context,
                    episode_context=state.context,
                    observation=observation,
                    instruction=instruction,
                    instruction_revision=revision,
                    high_level_call_count=(
                        state.high_level_call_count + int(high_invoked)
                    ),
                )
                # Keep the semantic decision stable if a remote low-level call
                # executes but its response is lost. The retry reuses both the
                # high-level output and the original request id.
                state.pending_turn = after
                state.pending_high_response = high_response
                state.pending_high_invoked = high_invoked

            low_response = self.low_client.infer(
                self.low_request_factory(after, high_response)
            )
            action = _action_from_response(low_response)
            if action.episode_id != after.observation.episode_id:
                raise ValueError(
                    "low-level action episode_id does not match the observation"
                )
            if action.step_id != after.observation.step_id:
                raise ValueError("low-level action step_id does not match the observation")

            state.instruction = after.instruction
            state.instruction_revision = after.instruction_revision
            state.high_level_call_count = after.high_level_call_count
            state.last_step_id = after.observation.step_id
            state.last_action = action
            state.pending_turn = None
            state.pending_high_response = None
            state.pending_high_invoked = False
            return action

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "bridge": "dual_system",
            "high_level": _optional_call(self.high_client, "capabilities") or {},
            "low_level": _optional_call(self.low_client, "capabilities") or {},
        }

    def health(self) -> Mapping[str, Any]:
        return {
            "status": "ok",
            "high_level": _optional_call(self.high_client, "health") or {"status": "ok"},
            "low_level": _optional_call(self.low_client, "health") or {"status": "ok"},
        }


__all__ = [
    "BridgeTurn",
    "DualSystemBridge",
]
