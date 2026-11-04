# Agentic endpoint benchmark

This directory contains the launcher and benchmark assets for the Qwen agentic
endpoint benchmark.

## Reproduce the submitted configuration

The submission uses INT8 weights, FP8 KV cache, three TP2 CPU replicas,
and llm-d routing, with MTP disabled.

Before running, configure the runtime image in `configs/vllm-cpu.conf`
and complete `configs/submission_disclosures.yaml` as described under
[Full submission campaign](#full-submission-campaign). Consult
`docs/specifications/vllm-runtime-builds.md` in the submission package
for the builds used at each measured point.

vLLM PR #61329 reports 5.4% lower attention-kernel latency; we therefore
expect no performance regression from adopting the merged fix. The full
benchmark has not been rerun after the merge.

From this directory, run:

```bash
STACK_PROFILE=qwen-vllm-llm-d WEIGHT_DTYPE=int8 KV_CACHE_DTYPE=fp8 \
  bash run_full_submission.sh
```

The default campaign runs concurrency points 8, 12, 16, 32, 64, 128,
and 256, evaluates accuracy, and packages the results after validation.
The command uses the configured runtime image; select the corresponding
images separately to reproduce the builds used at each measured point.
See the [CPU guide](platforms/intel-cpu/README.md) for calibration and
CPU placement details.

## General workflow

The workflow has preparation, serving, and benchmark phases:

```bash
bash preflight.sh
bash start_server.sh
```

`preflight.sh` prepares the selected runtime and routing assets, pulls the
digest-pinned upstream Endpoints image, builds the benchmark client image with
the official agentic dataset, pulls the fixed network-probe image, and
downloads the selected model into `HF_HOME`. The client image contains the
checksum-pinned 613-trajectory JSONL and its license files in
`/workspace/dataset_cache`. SWE-bench task images are prepared separately.

`start_server.sh` starts the model workers and routing stack, waits for routed
readiness, and remains attached as their supervisor. It does not pull images or
download models. Run the benchmark against that server from another terminal:

```bash
bash run_benchmark.sh
```

To benchmark another compatible OpenAI endpoint, provide its base URL:

```bash
ENDPOINT_URL=http://server.example:30000 bash run_benchmark.sh
```

The original one-command workflow remains available. It prepares assets,
starts and supervises the server, runs the benchmark, and then stops the
server:

```bash
bash run_submission.sh
```

`run_benchmark.sh` does not load a server profile. For a managed local server,
it records the active server's profile and manifest as provenance. Use
`STACK_PROFILE` to select a profile explicitly; profile definitions are
shipped under `platforms/<family>/profiles/`. The launcher selects an
available profile automatically when `STACK_PROFILE` is unset. Unknown
profiles and component adapters fail before Docker operations. Consult the
included [platform directories](platforms/): each
`platforms/<family>/README.md` documents its profiles, hardware requirements,
runtime options, and packaging selector. See
[component adapter contracts](docs/component-adapters.md) when adding a
profile or adapter.

For component setup and options, see the upstream documentation:
[vLLM](https://docs.vllm.ai/),
[Endpoints](https://github.com/mlcommons/endpoints), and
[llm-d](https://llm-d.ai/docs).

## Benchmark modes

Run only the accuracy phase:

```bash
MODE=Accuracy bash run_submission.sh
```

Run only the performance phase:

```bash
MODE=Performance bash run_submission.sh
```

Run a functional smoke diagnostic with 20 performance trajectories and 10
SWE-bench instances:

```bash
MODE=Smoke bash run_submission.sh
```

`MODE=Both` is the default and runs the full configured counts. Mode values are
case-insensitive. Other values are rejected before benchmark-owned services
are started. Performance mode makes the endpoint client skip the accuracy
phase. The one-command wrapper still performs normal preparation and server
startup first. Smoke output is functional diagnostic evidence, not publishable
performance evidence.

## Requirements

- Bash 5.1 or newer (`preflight.sh` uses `wait -n -p`).
- An x86_64 Linux host with Docker installed, a running Docker daemon, and
  access to `/var/run/docker.sock`. The launcher starts model workers with
  host networking and IPC.
- Network access during preparation for the selected runtime and routing
  images, the digest-pinned Endpoints image, model assets, benchmark image
  layers, and the fixed SWE-bench network-probe image. Docker reuses completed
  build layers. For `MODE=Both` (the default), `Accuracy`, and `Smoke`,
  preparation also pulls the selected SWE-bench Verified task images (200 by
  default, honoring `SWEBENCH_INSTANCE_COUNT`), with up to eight missing
  images pulled concurrently. A failure stops preparation before server
  startup, after already-started pulls finish. `MODE=Performance` skips this
  task-image pre-pull.
- When outbound access requires a proxy, export `HTTP_PROXY`, `HTTPS_PROXY`,
  and `NO_PROXY` (lowercase forms are also accepted). Include `localhost` and
  `127.0.0.1` in `NO_PROXY` so local readiness probes bypass the proxy. If
  `${DOCKER_CONFIG:-$HOME/.docker}/config.json` is absent, the launcher uses
  `jq` to create it with Docker's `proxies.default` settings, mode `0600`, and
  uppercase values taking precedence over lowercase values. An existing
  readable configuration is mounted unchanged; an existing unreadable or
  non-regular file is never replaced. The configuration is mounted read-only
  into the trusted SWE-bench service so nested task containers receive proxy
  settings. Proxy values are not logged, but Docker stores them as plaintext
  and the generated file persists after the run. Configure image-pull proxies
  separately using [Docker's daemon proxy guide](https://docs.docker.com/engine/daemon/proxy/).
- Enough local disk for the model, dataset, Docker layers, and SWE-bench task
  images. The launcher requires at least `MIN_FREE_GIB` free GiB in the
  checkout filesystem (default: 10); SWE-bench may require substantially
  more.
- A writable Hugging Face cache. Set `HF_HOME` to an absolute path to select
  it, and set `HF_TOKEN` when the model or other Hugging Face assets require
  authentication.
- Host networking with the selected runtime's worker ports and ports 30000
  and 18080 available.

Before pulling or building images or downloading models, preflight creates and
checks its output directory and the directories required by the selected
calibration mode. Each write check creates and removes a temporary directory as
the launching user. Failures report the path, UID/GID, and how to choose writable
storage using `OUTPUT_ROOT` or `HF_HOME`; missing private directories also get an
administrator provisioning command. The checks do not change existing ownership
or permissions. Standalone calibration uses the same calibration checks.
Existing model caches managed by containers do not need to be writable by the
host user when no host calibration writes are required. Local checkpoints are
not probed for write access.

### Using a local model directory

Set `MODEL_LOCAL_PATH` to an absolute checkpoint directory. Workers mount it
read-only, skipping repository download and model preparation. `MODEL_NAME`
remains campaign metadata; local checkpoints have no approved repository
revision. The client still fetches its tokenizer from huggingface.co.

## Configuration and outputs

The benchmark uses the performance and accuracy configuration in
`configs/qwen_agentic_benchmark.yaml`. Its scalar values are the tuning
baseline. The launcher supports these environment-variable overrides:

| Environment variable | YAML setting |
| --- | --- |
| `TARGET_CONCURRENCY` | `settings.load_pattern.target_concurrency` |
| `PERFORMANCE_TRAJECTORY_COUNT` | `datasets[].agentic_inference.num_trajectories_to_issue` |
| `SWEBENCH_INSTANCE_COUNT` | `datasets[].accuracy_config.extras.num_instances` |
| `TURN_TIMEOUT_S` | `datasets[].agentic_inference.turn_timeout_s` |
| `MAX_ISSUE_DURATION_MS` | `settings.runtime.max_issue_duration_ms` |
| `WARMUP_CONNECTIONS` | `settings.client.warmup_connections` |
| `MAX_IDLE_TIME` | `settings.client.max_idle_time` |
| `TEMPERATURE` | `model_params.temperature` |
| `TOP_K` | `model_params.top_k` |
| `TOP_P` | `model_params.top_p` |
| `REPETITION_PENALTY` | `model_params.repetition_penalty` |
| `PRESENCE_PENALTY` | `model_params.presence_penalty` |
| `MAX_NEW_TOKENS` | `model_params.max_new_tokens` |
| `PRESERVE_THINKING` | `model_params.chat_template_kwargs.preserve_thinking` |
| `ENABLE_SALT` | `datasets[].agentic_inference.enable_salt` |
| `INJECT_TOOL_DELAY` | `datasets[].agentic_inference.inject_tool_delay` |

For normal performance and combined runs, the launcher derives the trajectory
count from effective target concurrency when `PERFORMANCE_TRAJECTORY_COUNT`
is unset. It rounds five times the target concurrency up to a multiple of 613,
with a minimum of 613. For example, concurrency 240 issues 1226 trajectories.
`TARGET_CONCURRENCY` overrides the YAML value. A full submission campaign
requires `PERFORMANCE_TRAJECTORY_COUNT` to be a whole multiple of 613. The
trajectory count and `MAX_ISSUE_DURATION_MS` are separate workload limits.
Campaign accounting checks that configured trajectories were issued and
completed, including the post-issue drain. It has no local minimum-duration
or steady-state-window gate for agentic runs. When a report has no
`steady_state` window, the selected checker snapshot falls back to whole-run
metrics, warns about that basis, and warns when duration is below its generic
per-region minimum. The package retains those warnings; checker errors block
the archive.

Explicit environment overrides take precedence over the reduced trajectory
and instance counts selected by `MODE=Smoke`. Smoke keeps its
20-trajectory default, and accuracy-only runs do not adjust the trajectory
count. Other unset settings use their YAML values:

```bash
TARGET_CONCURRENCY=64 PERFORMANCE_TRAJECTORY_COUNT=40 \
  SWEBENCH_INSTANCE_COUNT=20 TEMPERATURE=0.7 bash run_submission.sh
```

Alternatively, edit YAML baseline settings directly. Arbitrary tuning may not
comply with submission requirements. The launcher does not gate, reject, or
label a run on that basis, so submission compliance must be assessed
separately.

The benchmark client writes each run below `OUTPUT_ROOT` (an absolute path;
default: `run_output/`) in a UTC timestamped directory. Server sessions use
`OUTPUT_ROOT/server_runs/<server-id>` and publish
`OUTPUT_ROOT/active-server.env` only after routed readiness. The active marker
is removed when its attached supervisor exits. Set `OUTPUT_ROOT` consistently
across terminals to choose another location.

The launcher scripts accept no positional arguments. Environment variables
are the supported configuration interface.

For a managed local server, the benchmark compares the served model name in
the server manifest with its own `SERVED_MODEL_NAME`. A mismatch emits a
warning and continues; it is provenance information, not a model-content
gate.

During preparation, the launcher uses an official SWE-bench task image to
verify that a nested task container can query PyPI over Docker's default
bridge. This fails early when Docker proxy configuration or bridge egress is
unavailable.

## Full submission campaign

`run_full_submission.sh` runs a dedicated concurrency sweep and builds a local
submission package after the official checker reports no errors and the
campaign's local acceptance gates pass. Checker warnings are retained in its
report. `run_submission.sh` remains available for an individual benchmark
run.

The default campaign is defined in `configs/submission.yaml`:

```yaml
points:
  - concurrency: 8
  - concurrency: 12
  - concurrency: 16
  - concurrency: 32
  - concurrency: 64
  - concurrency: 128
  - concurrency: 256
# Both paths are relative to this config.
metadata: submission_metadata.yaml
disclosures: submission_disclosures.yaml
```

Start the curve at concurrency 8 or 16 and cover all four computed regions.
Each point sets a concurrency and may override the selected profile or
topology; all points use the same physical system, model, and dataset.
Topology is checked before preparation.

Edit `configs/submission_disclosures.yaml` with verified human disclosures,
then validate and run the campaign:

```bash
SUBMISSION_VALIDATE_ONLY=1 bash run_full_submission.sh
bash run_full_submission.sh
```

Set `COLLECT_EMON=1` to collect EMON locally; default is `0`. Collection uses
1-second event-group intervals, starts 30 minutes into each run, and stops
after 30 minutes, at 1 GiB, or when the run ends. Buffered writes may exceed
the size threshold. Artifacts are saved under
`RUN_DIR/emon/`; collection failures warn and continue.
See [EMON setup](docs/emon/README.md).

The campaign preserves the disclosures file. Required `system` values are
`submitter_org_names`, `submitter_contact`, `submission_id`, `system_name`,
`shortened_system_name`, `publication_status`, `system_size`, and `division`.
The shortened name must be nonblank and at most 20 characters. `Available`
requires `availability_url`; `Preview` requires `target_availability_date` in
`YYYY-MM-DD` format. Other dates and notes are optional. Supply verified
values. Collector and runtime observations provide host and software facts;
use `system.node_types` only to fill missing host facts such as memory
configuration, NIC count, filesystem, or CPU core/vCPU counts. Conflicting
manual values fail validation. The campaign generates model, dataset,
topology, concurrency, framework, cohort, and seed fields.

Complete `system_power` from trusted inventory and specifications using the
checker's Appendix E model. Disclose actual cooling and scale-out, and provide
a sourced provisioned-power rating for the whole system or each node set.
Give each node set a unique ID; its type and provisioned count must match host
inventory. Include a public source or the applicable Appendix D reference; do
not estimate power. See
[`submission_disclosures.example.yaml`](configs/submission_disclosures.example.yaml)
for fields and examples.

Preflight records the selected checker revision, policy, and seed-registry
provenance. The campaign creates run-specific metadata snapshots from disclosures, host facts, and
runtime observations. The `metadata` path in `configs/submission.yaml` is the
optional output location for the preview helper:

```bash
bash prepare_submission_collector.sh
bash prepare_submission_collector.sh --assert-only
bash init_submission_metadata.sh
```

The host collector uses MLCflow **1.3.8** and the clean `mlperf-automations`
checkout pinned to
`00f82fe05cd63330a3b5ef97a5f18176aa5f7d55` in `configs/mlcflow.conf`. It has a
separate host virtual environment keyed by both pins; set `MLCFLOW_VENV_DIR`
to choose its location. `--assert-only` checks the pinned environment without
installing or downloading. Only the collector entrypoint runs with `sudo`:
it can prompt through `/dev/tty` when a controlling terminal is available,
and uses `sudo -n` otherwise. Optionally authenticate beforehand with `sudo -v`
in the launching terminal. Preparation and campaign execution are unprivileged,
and collection does not install host packages.

The unchanged native probes report the host userspace OS and raw host kernel
release; provenance retains both. Container software observations remain
separate. Successful collection saves the unchanged native JSON beside the
configured metadata destination with a provenance sidecar. Later preparation
and resume runs reuse populated JSON without collector preparation or sudo,
while revalidating disclosures and runtime software. Resume prefers its
saved metadata snapshot. The sidecar's JSON hash and collector pins are
checked; sidecar-less JSON is rejected for automatic cache reuse. Invalid or
mismatched cached data fails instead of silently recollecting. Choose a new
metadata destination to capture fresh host data.

To explicitly import an existing official collector output:

```bash
SUBMISSION_SYSTEM_DESCRIPTION=/absolute/path/system_desc.json \
  bash init_submission_metadata.sh
```

Imported output must identify the clean pinned Git checkout in
`mlc_scripts_version`; reported node revisions must agree. Explicit imports
bypass collector preparation and sudo and retain their declared provenance,
including when the source JSON has no cache sidecar. The helper creates a
preview; the campaign uses the manual disclosures file.
Raw JSON, logs, and provenance remain caller-owned in private
`run_output/submission-metadata-*` captures after failure or cancellation.
The helper refuses to overwrite an existing preview.

`SUBMISSION_DOCS_DIR` selects public evidence copied into
`docs/specifications/`. `SUBMISSION_VALIDATE_ONLY=1` validates inputs and
local runtime, topology, and storage without downloading benchmark assets or
models, pulling or building images, or launching benchmarks.
`prepare_submission.sh` prepares the official checker in a separate
environment. Set `SUBMISSION_VENV_DIR` to choose its path; the default is
under `~/.cache/endpoints-agentic/`.

| Environment variable | Campaign behavior |
| --- | --- |
| `SUBMISSION_CONFIG` | Optional absolute path to a custom campaign YAML; defaults to `configs/submission.yaml`. |
| `SUBMISSION_RESUME_DIR` | Optional existing absolute campaign directory to resume. |
| `SUBMISSION_VALIDATE_ONLY` | Set to `1` to validate inputs and local probes without image pulls/builds, model downloads, or benchmarks. |
| `MLCFLOW_VENV_DIR` | Optional absolute path for the isolated pinned host collector environment. |
| `SUBMISSION_VENV_DIR` | Optional absolute path for the isolated official checker environment. |
| `SUBMISSION_SYSTEM_DESCRIPTION` | Optional absolute official MLCflow JSON input; replaces automatic host collection and retains its declared provenance checks. |
| `SUBMISSION_DOCS_DIR` | Optional absolute public specification directory; copied into `docs/specifications/`. |
| `COLLECT_EMON` | Set to `1` to collect host-local EMON data during each run; defaults to `0`. |
| `MODEL_LOCAL_PATH` | Optional absolute checkpoint directory. Campaigns use it directly, skip repository download and model preparation, and omit repository revision metadata. |
| `MODEL_REVISION` | Pins repository checkpoints. Campaigns select an approved revision and reject a supplied mismatch; local checkpoints have no approved repository SHA. |
| `OUTPUT_ROOT` | Absolute base campaign output directory; campaign records are created below `submission_campaigns/`. Defaults to `run_output/`. |

The full campaign fixes Qwen sampling to `temperature=1.0`, `top_k=20`,
`top_p=0.95`, `repetition_penalty=1.0`, `presence_penalty=1.5`, and
`max_new_tokens=8192`; it also requires `preserve_thinking=true`, salt and
tool delay, and `stop_issuing_on_first_user_complete=false`. It checks
complete issue, completion, and drain accounting. Trajectory counts must be
whole multiples of 613, and every SWE-bench accuracy evaluation uses exactly
200 instances.

Campaign points automatically select the approved revision for repository
checkpoints; an explicit `MODEL_REVISION` mismatch is rejected. A local
checkpoint omits `MODEL_REVISION` and receives no approved repository SHA.

SWE-bench accuracy uses 200 instances at representatives 8, 12, 32, and 256,
with additional points as needed to cover each region and serving
configuration. For Qwen3.6-35B-A3B, targets are at least 55.86% inline
accuracy, a 69% four-region SWE-bench mean, and full-run output sequences of
344–422 tokens. The selected checker evaluates these targets for the final
bundle.
There is no local per-region minimum-duration or steady-state-window
gate for campaign acceptance. The selected checker may report duration or
missing-window warnings; those warnings remain in its report, while checker
errors prevent archive creation. Native inline accuracy retains
`dataset_name=performance` and `dataset_type=performance`. The package
preserves performance summary bytes and does not synthesize fields that the
upstream report omits, such as interactivity input totals.

Resume reuses a completed point when its performance summary is complete at
the target concurrency, inline accuracy and required SWE-bench evaluation
markers are complete, required package-owned inputs exist, and SHA-256 values
match for `config.yaml`, `performance/result_summary.json`, and
`accuracy/accuracy_results.json`. It requires a nonempty `events.jsonl`, but
does not parse or hash the full event log or compare its size with a previous
run. Packaging reads only the event-log prefix needed to measure warmup.
Older records use the hashes for these files directly; extra historical hashes
remain provenance only. Sample maps and final snapshots are not needed for
reuse. Failure counts, sample accounting, numerical validity, output lengths,
accuracy targets, and coverage are left to the official checker, which
determines final bundle validity. Source, checker revision, and image changes do not alone block resume;
host and checkpoint identity protections still apply. Completed points retain
their recorded settings and images; unfinished points use current settings. A resume may
add system disclosure fields required by a newer checker. Existing captured
values, raw host JSON, and the benchmark template remain intact. Additions are
recorded in `campaign.json` and must match on later resumes; they do not change
completed point settings.

If a serving-configuration group lacks SWE-bench results, the campaign
repeats its highest-concurrency point in full `Both` mode using saved
settings. Each attempt records requested settings and budgets before launch,
then its actual server manifest after launcher
completion. Interrupted or partial attempts retain their evidence and rerun
into a fresh attempt directory; earlier attempts remain intact. Source is
exported once for packaging after validation. A successful campaign creates
a local review package and does not submit or upload it.

## Submitted source package

This directory contains the packaged source. Run the launch and reproduction
commands above directly from it. The development-only archive generator is
not included and is not needed to reproduce the submitted runs.
