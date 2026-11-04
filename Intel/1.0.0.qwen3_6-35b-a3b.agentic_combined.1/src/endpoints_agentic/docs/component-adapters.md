# Component adapter contracts

The launcher separates model runtimes from routing stacks. A profile under
`platforms/<family>/profiles/` selects one adapter of each type and provides
image references, model identity, and ports. Profiles and adapters are
trusted Bash files included in the benchmark package. Platform-specific
implementations live with their platform files; reusable implementations
live under `components/`.

Benchmark-side image, served-model, and SWE-bench port settings live in
`configs/benchmark.conf`. `run_benchmark.sh` loads that file directly and
does not discover serving topology or load runtime and routing adapters. For
a managed local server, it records the server profile and manifest as
provenance rather than requiring a matching profile.

## Runtime adapter

A runtime adapter implements:

- `runtime_validate`: validate adapter configuration and reserve every
  worker port with `reserve_port`.
- `runtime_prepare_image`: pull or build the runtime image and run
  profile-specific capability checks without downloading model assets.
- `runtime_assert_image_prepared`: verify that the runtime image is available
  locally and run the same capability checks without pulling or building.
- `runtime_prepare_assets`: call `runtime_prepare_image`, then download the
  selected model into the shared Hugging Face cache.
- `runtime_assert_prepared`: fail without network access unless the runtime
  image, required capabilities, and downloaded model cache are locally
  available. For `MODEL_LOCAL_PATH`, loading validates the weights at
  startup.
- `runtime_start_workers`: start every entry in the worker plan and register
  each health-gated critical process with `register_ready_service`.

`collect_submission_runtime_metadata.sh` resolves the selected profile and
submission-point topology overrides from the environment. By default it
prepares only the runtime image. Set `SUBMISSION_RUNTIME_LOCAL_ONLY=1` to
require an already local image and run offline capability and software
probes. The wrapper accepts no positional arguments, writes one JSON object
to stdout, and sends diagnostics to stderr. It never downloads or mounts a
model or runs a benchmark.

The runtime metadata object reports the observed image ID and reference,
runtime and framework versions, container platform, and container OS. Its
software mapping includes installed framework and dependency versions.
`driver_version` appears only when the image probe successfully observes it.

Set `SUBMISSION_EXPECTED_RUNTIME_IMAGE_ID` when a later normal
`runtime_prepare_assets` call must reuse the image collected for submission.
The adapter checks that the configured image reference still resolves to
that ID, skips pulling or building, and starts workers by immutable image ID.

The neutral worker plan is exposed through parallel arrays:

- `WORKER_IDS`
- `WORKER_ROLES`
- `WORKER_PARALLEL_WIDTHS`
- `WORKER_PORTS`
- `BACKEND_URLS`

The selected profile owns its resource dimensions and runtime arguments.
Runtime adapters validate those values, build the worker plan, and expose
readiness endpoints. Profiles may also set `RUNTIME_EXTRA_ARGS=(...)`.
Preserve the shared Hugging Face cache contract and required capability
gates. Package and platform probes do not establish which kernels serve a
workload.

CPU profiles use the configured `VLLM_CPU_IMAGE` tag, reusing it locally or
pulling it when absent. Accelerator profiles pin `RUNTIME_IMAGE` by digest
and provide `RUNTIME_IMAGE_DISPLAY_TAG` as a local alias for image listings.
Workers launch by resolved image ID. The image reference, ID, and any display
alias are recorded in `resolved-stack.env`.

## Routing adapter

A routing adapter implements:

- `routing_validate`: validate configuration, reserve every routing-stack
  port, and set `PUBLIC_ENDPOINT_URL`.
- `routing_prepare_assets`: pull or build routing-stack images.
- `routing_assert_prepared`: verify that every routing image is local.
- `routing_configure`: render and locally validate server-session
  configuration without pulling images.
- `routing_start`: start and register all processes in the routing stack.

A routing adapter consumes `BACKEND_URLS`. It may register one process or
several processes, such as a gateway and a separate scheduler. The benchmark
orchestrator does not assume that routing is a single container.

The shared llm-d gateway allows up to two retries for connection failures,
upstream resets before response headers, and HTTP 502/503/504. Retries target
the same backend and cannot restart an SSE response already sent to the client.
Envoy's default backoff, 1 MiB retry buffer, and concurrent retry limits apply.
For P/D, retries repeat the outer decode-sidecar request.

## Lifecycle rules

Use component validators and propagate their diagnostics and failure status.
Keep launcher checks for immutable pins, offline assets, safe rendering,
resource assignments, and required compatibility between components.

Register a service before starting its container:

```bash
register_ready_service \
    "stable-service-id" \
    "${container_name}" \
    "http://127.0.0.1:${port}/health" \
    300
```

The server registry provides readiness polling, critical-service monitoring,
log capture, and reverse-order cleanup. Benchmark client and SWE-bench
services use a separate process-local registry, so ending a benchmark never
removes runtime or routing containers. Component adapters must not add their
own exit traps or remove containers owned by another component.

To add a component, add its adapter under the appropriate platform or shared
component path, add a profile selecting it, pin every image or source
revision, and extend the mocked adapter tests. Do not add component-specific
branches to `run_submission.sh`.

The optional routing integration test uses pinned HTTP test containers
instead of model workers to verify request distribution, session stickiness,
payload preservation, and fail-closed behavior. Mocked endpoint tests do not
exercise a model worker or establish live serving, accuracy, or performance.
