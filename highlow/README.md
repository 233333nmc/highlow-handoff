# High/Low Policy Runtime

`highlow` is the new simulator-neutral runtime for the dual-policy system. It
is deliberately side-by-side with the legacy hierarchical implementation;
the official RoboTwin loop is not rewritten or silently redirected.

## Boundaries

```text
official simulator loop
        |
        | native observation / native action
        v
simulator adapter <-> bridge <-> inference clients
                               |              |
                               v              v
                    high-level service   low-level service
                    (stateless)          (episode stateful)
                         |                    |
                    SGLang backend       policy plugin
                                            |
                                          HyVLA
```

The source layout mirrors those boundaries:

```text
highlow/
  contracts/                       shared request/episode/observation/action values
  inference/common/                JSON, health, errors, dependency-free ASGI
  inference/high_level/            planner/verifier service + SGLang backend
  inference/low_level/             generic service, manifest, registry and API
    plugins/hyvla/                 one low-level plugin, not the low-level core
  clients/                         in-process and JSON/HTTP clients
  bridge/                          per-episode high/low handoff coordinator
  adapters/                        thin RoboTwin conversion adapters and hook
  compositions/                    explicit cross-boundary deployment wiring
robotwin_dual_system/              four-hook shim for RoboTwin's policy loader
```

The dependency rules are strict:

- shared contracts and bridge code import no model framework;
- low-level core imports no concrete policy or simulator;
- concrete policies live only below `inference/low_level/plugins`;
- the generic HyVLA plugin receives a policy factory by injection;
- the optional RoboTwin/HyVLA composition root lazily loads the unchanged
  official wrapper only when its backend starts;
- `RoboTwinAdapter` and `RoboTwinPolicyHook` never call simulator `reset`,
  `step`, `take_action`, task, physics, render, or success methods.

## One control tick

1. The official simulator loop produces a native observation.
2. `RoboTwinAdapter` wraps three RGB views and the dual-arm state in an
   `ObservationEnvelope`.
3. `DualSystemBridge` calls the high level on a configured boundary. The
   ready-made RoboTwin wiring plans on its first high-level call and verifies
   on later calls. Raw RGB arrays are JPEG-encoded only on those high-level
   ticks; low-level calls keep the original arrays.
4. A planner subtask, or a verifier `switch`, becomes the low-level language
   conditioning. A verifier `same` leaves it unchanged. Semantic validation
   prevents `completed/same` and `switch` without a next subtask.
5. The low-level service routes by `episode_id`, invokes its selected plugin,
   and returns an `ActionEnvelope`.
6. The adapter converts the action, and the official simulator loop executes
   it. Simulator lifecycle and success remain entirely outside this package.

The bridge and low-level service cache a completed `(episode_id, step_id)`.
If an HTTP response is lost, retrying the same tick does not call the model or
consume HyVLA's next cached action again. The bridge also retains the pending
high-level decision so a retry cannot commit a different, non-deterministic
subtask. Open, reset, and close keep non-inferable local tombstones while an
operation is incomplete, and retries resume the same episode and lifecycle
phase. Low-level plugin lifecycle methods must therefore be idempotent for an
identical `EpisodeContext`; the bundled HyVLA plugin follows that contract.
If a reset finishes remotely but its HTTP response is lost, a caller retry can
repeat the complete reset; a production backend should make reset semantically
idempotent (or add deployment-level operation-ID deduplication).

## Serving

The ASGI surface is dependency-free; use the process manager/ASGI server of
your deployment. For example, after installing `uvicorn`:

For high-only, generic low-only, bridge, and RoboTwin-shim images, install the
framework-light distribution without the root project's CUDA/training stack:

```bash
pip install './packaging/highlow-runtime[server]'
# Add robotwin-codec where JPEG conversion is needed:
pip install './packaging/highlow-runtime[server,robotwin-codec]'
```

### High level backed by SGLang

Run SGLang separately, then configure this stateless facade:

```bash
export HIGHLOW_SGLANG_BASE_URL=http://sglang:30000
export HIGHLOW_SGLANG_MODEL=your-served-model-name
export HIGHLOW_SGLANG_API_KEY=optional-key

uvicorn 'highlow.inference.high_level:create_sglang_app_from_env' \
  --factory --host 0.0.0.0 --port 8100
```

Routes:

- `GET /health`
- `GET /capabilities`
- `POST /v1/high-level/infer`

The SGLang client uses its OpenAI-compatible `/v1/chat/completions` endpoint,
multimodal `image_url` parts, and JSON-schema constrained output. Prompt and
response-schema versions are resolved explicitly by the service.

### Generic low level

A low-level deployment selects an explicit `module:factory`; adding another
policy does not require editing a central switch statement. The bundled
RoboTwin/HyVLA composition uses the factory below.

This concrete factory additionally requires the repository's HyVLA model and
`robotwin_eval` runtime in the low-level image; the framework-light wheel alone
is sufficient only for core services and plugins whose own dependencies are
installed separately:

```bash
export HIGHLOW_LOW_LEVEL_FACTORY=highlow.compositions.robotwin_hyvla:create_robotwin_hyvla_backend
export HIGHLOW_LOW_LEVEL_CONFIG='{
  "policy_config": {
    "ckpt_path": "/models/hyvla",
    "norm_path": "/models/hyvla/norm_stats.pkl"
  }
}'

uvicorn 'highlow.inference.low_level:create_policy_app_from_env' \
  --factory --host 0.0.0.0 --port 8200
```

Routes:

- `GET /health` and `GET /capabilities` (`/v1/...` aliases also exist)
- `POST /v1/episodes/open`
- `POST /v1/episodes/reset`
- `POST /v1/episodes/close`
- `POST /v1/infer`

ASGI startup opens the selected backend and shutdown closes all contexts and
the model. HyVLA advertises one concurrent episode because its current wrapper
and action cache are single-episode state. Other plugins may advertise a
larger limit; the service serializes within an episode and permits different
episodes to run concurrently.

High-level replicas are stateless and can sit behind ordinary load balancing.
Low-level episode context is intentionally worker-local: when running multiple
replicas, route every open/infer/reset/close for one `episode_id` to the same
replica (sticky routing or explicit episode sharding). The bridge always sends
that stable ID; a random per-request load balancer will correctly receive a
409 when inference lands on a worker where the episode was not opened.

## Connecting the official RoboTwin loop

The unchanged RoboTwin policy loader expects four module functions. Select the
new `robotwin_dual_system` policy package and its `deploy_policy.yml`; its thin
shim exposes `encode_obs`, `get_model`, `eval`, and `reset_model`. Configure the
two remote URLs there (or through the evaluator's normal overrides). The shim's
`eval` performs the one `TASK_ENV.take_action(...)` call required by RoboTwin's
existing policy protocol. It delegates all conversion and inference to the
adapter/hook below; neither the bridge nor an inference service owns the env.
The shim closes the prior context through `reset_model` and registers a final
process-exit close for the last rollout. Production low-level deployments
should also expire abandoned episode leases after hard client failures.

For a custom simulator integration layer, configure the same pieces directly:

```python
from highlow.adapters import RoboTwinAdapter, RoboTwinPolicyHook, create_robotwin_bridge
from highlow.clients import JsonHttpInferenceClient

high = JsonHttpInferenceClient(
    "http://high-level:8100",
    infer_endpoint="/v1/high-level/infer",
    health_endpoint="/health",
    capabilities_endpoint="/capabilities",
)
low = JsonHttpInferenceClient(
    "http://low-level:8200",
    infer_endpoint="/v1/infer",
    open_context_endpoint="/v1/episodes/open",
    reset_context_endpoint="/v1/episodes/reset",
    close_context_endpoint="/v1/episodes/close",
    health_endpoint="/health",
    capabilities_endpoint="/capabilities",
)

bridge = create_robotwin_bridge(
    high,
    low,
    verifier_interval_steps=100,  # replace with an event/oracle trigger if desired
)
policy_hook = RoboTwinPolicyHook(RoboTwinAdapter(), bridge)
```

Such a custom loop calls only `policy_hook.start_episode(...)`,
`policy_hook.act(native_observation, step_id, timestamp_s)`, optionally
`policy_hook.reset()`, and `policy_hook.finish(result)`. It remains responsible
for obtaining observations and executing the returned native action. The
bundled four-hook shim is the compatibility layer when the official loader
owns that calling convention.

For co-located testing, wrap `HighLevelService` and `LowLevelService` with
`InProcessHighLevelClient` and `InProcessLowLevelClient`; the bridge interface
does not change.

The default HTTP codec uses JSON, base64 for encoded images, and `tolist()` for
arrays. This is a portable control-plane baseline, not a high-throughput image
transport. Production deployments can inject a binary codec/transport into
`JsonHttpInferenceClient` without changing policy, bridge, or simulator APIs.
The baseline client deliberately exposes transport failures to its caller and
does not implement automatic retry/backoff; retry the same episode/step or
lifecycle operation so the idempotency guards can take effect.

This repository retains its original monolithic training dependencies in the
root `pyproject.toml` for legacy HyVLA training/evaluation. Do not install that
root distribution in high-only or bridge-only images; use the separate
`packaging/highlow-runtime` distribution above. A low-level HyVLA image uses
the root distribution (which already contains `highlow`) or an equivalent
model-specific package; do not install two wheels that both provide the same
`highlow` package into one environment.

## Extending the system

To add a low-level policy, create
`inference/low_level/plugins/<policy>/` with a `PolicyManifest` and a `Backend`
implementation. Keep model imports inside `open()` (or the injected factory),
and expose a composition factory for deployment. Do not add it to the generic
`plugins/__init__.py`.

To add a simulator, implement the three conversion methods in
`SimulatorAdapter`: create an episode context, encode an already-produced
observation, and decode an action. Do not wrap or duplicate the simulator
control loop.

## Verification

The focused suite is standard-library `unittest` and does not require a live
SGLang server or GPU:

```bash
PYTHONPATH=. python -m unittest -v \
  tests.test_highlow_contracts \
  tests.test_highlow_common \
  tests.test_highlow_high_level \
  tests.test_highlow_low_level \
  tests.test_highlow_bridge
```
