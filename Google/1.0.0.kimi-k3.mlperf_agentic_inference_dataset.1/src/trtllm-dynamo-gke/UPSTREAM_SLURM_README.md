> Site-specific names, paths and addresses are replaced with `<placeholders>`.

# Run Kimi K3

Kimi K3 is served with Dynamo in disaggregated mode: TensorRT-LLM prefill and
decode workers behind the Dynamo frontend and router, with DSpark speculative
decoding on the decode workers. etcd and NATS, started on the node of the first
prefill worker, connect the workers and the frontend. Each point YAML holds its
complete prefill, decode, router and client configuration. Point names are
concurrency settings, not measured capacity. No benchmark results are included in
this export.

| System | Approved points | First point |
| --- | --- | --- |
| GB300 NVL72, 32 or 48 GPUs | C12, C13, C16, C24, C32, C48, C64, C80, C192, C288, C384, C448, C512, C576, C640, C1024 | [point_12.yaml](configs/GB300-NVL72_GB300-288GB_aarch64x48_TRTLLM-Dynamo/point_12.yaml) |
| Vera Rubin NVL72, 32 or 48 GPUs | C15, C18, C40, C48, C56, C64, C96, C192, C193, C256, C512, C1024, C1536 | [point_15.yaml](configs/VR-NVL72_VR-288GB_aarch64x48_TRTLLM-Dynamo/point_15.yaml) |

Each point's `TOTAL_GPUS`, `TOTAL_NODES` and `SEGMENT_NODES` give its allocation in
whole four-GPU nodes ([Allocation](#allocation)). A dataset repeat is 613
trajectories; `NUM_TRAJECTORIES` issues whole repeats and the client completes
every one, including the tail. On GB300 the point's client configuration also
states the count as `num_trajectories_to_issue`; change both together. The client
runs in performance mode with inline accuracy scoring. GB300 C12, C16, C32 and
C1024 then also run SWE-bench Verified in the same client run (see
[SWE-bench accuracy](#swe-bench-accuracy)).

## Environment and images

- Slurm with Pyxis and Enroot on ARM64 nodes with four GPUs per node, and shared
  storage visible at the same absolute path on every node. Clone this repository
  on that shared storage: every node runs scripts from the checkout. The
  scheduler must support segment allocations (`--segment`); the launcher requests
  one and checks the allocation's `SegmentSize`.
- `python3` and `nvidia-smi` on every node's host: the launcher probes each node's
  NVLink fabric there, and the node of the first prefill worker runs etcd and NATS
  on its host. The SWE-bench points also need `git` on the client node's host.
- Outbound network access from the compute nodes: the control plane downloads etcd
  and NATS from GitHub, the SWE-bench service fetches its source from GitHub, `uv`
  and its Python packages from PyPI, a managed Python 3.12 if `uv` does not find one,
  and SWE-bench Verified from Hugging Face, and the serving containers download
  FlashInfer kernel metadata over HTTPS at start-up.
- Install the pinned SFlow into a virtual environment in this checkout. The
  launcher imports SFlow, and the batch job it submits reruns the launcher with the
  same interpreter on a compute node and runs it on every node for the fabric probe.
  Create the environment on an ARM64 compute node with the nodes' own `python3`, and
  keep it on the shared checkout:

```bash
srun -A your_account -p your_partition -N 1 --pty bash   # an ARM64 compute node
python3 -m venv .venv
.venv/bin/pip install 'sflow @ git+https://github.com/NVIDIA/nv-sflow.git@b8afb5b86b3f46d9fa771531990fc7afaefc69b9'
```

  The batch job calls `sflow` from `PATH`, which it inherits from the shell you
  submit from, so put `.venv/bin` on `PATH` before submitting (as the run commands
  below do).

Both systems use the same images, pinned in [slurm_env.yaml](slurm_env.yaml):

| Variable | Image |
| --- | --- |
| `SERVER_IMAGE` | `registry.gitlab.com#nvidia/mlperf-inference-partner/nv-mlpinf-partner/kimi-k3-dynamo@sha256:985260a01a2dfb66bdd0c68f0bd482e78f69c1b101120c7341782d101e2602a4` |
| `CLIENT_IMAGE` | `ghcr.io#mlcommons/endpoints@sha256:1523051b141437b25083924d6428082f435713516eee71c54ae207b393900b2d` (`d3c57f2`) |

The serving image is an ARM64 TensorRT-LLM build for GB300 and Vera Rubin with
Dynamo installed, the build NVIDIA ran these points with. It is in the partner
registry repository
`registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/kimi-k3-dynamo`
([registry page](https://gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/container_registry/12597129))
with tag `candidate-backlog-03bb2f90fc43-r1`. Pull it by digest, as the points do; a tag
can be moved. Its Dynamo frontend is built from
[dynamo/kimi-k3.patch](dynamo/kimi-k3.patch) in two variants under
`/opt/k3-dynamo/`: `frontend` applies the patch without its `tcp/server.rs`
change, and `frontend-backlog` applies all of it, including that TCP listener
backlog fix. Each point selects one and checks its hash against
`DYNAMO_FRONTEND_CORE_SHA256` before the frontend starts. The image also holds
the decode helpers [dynamo_entry_v4.py](dynamo/dynamo_entry_v4.py) and
[visible_eos_v4.py](dynamo/visible_eos_v4.py). Use the image unchanged.

> **Note:** this serving image will soon be copied to a public container
> repository. A later update of these instructions will give its public location;
> pulling it from there will need no credentials.

Until then, pulling it needs an Enroot `.credentials` entry for netrc machine
`registry.gitlab.com` with your login and a token that has `read_registry` access:

```text
machine registry.gitlab.com login USER password TOKEN
```

Every server task starts a container, so import the serving image once to
SquashFS and pass the `.sqsh` path with `--set SERVER_IMAGE=...`; do the same for
`CLIENT_IMAGE` to avoid pulling it at job start:

```bash
enroot import -o /shared/images/kimi-k3-dynamo.sqsh \
  'docker://registry.gitlab.com#nvidia/mlperf-inference-partner/nv-mlpinf-partner/kimi-k3-dynamo@sha256:985260a01a2dfb66bdd0c68f0bd482e78f69c1b101120c7341782d101e2602a4'
```

## Control plane

The workflow's `control_plane` task downloads the pinned ARM64 etcd and NATS server
releases listed in [scripts/control_plane.json](scripts/control_plane.json),
verifies the archive and binary hashes, and places the binaries in
`CONTROL_PLANE_DIR` (default: `client-results/control-plane/` in the launcher's
`--output` directory). It
runs them on the host of the first prefill worker's node, with their data in a
node-local temporary directory, and stops them when the benchmark finishes. Set
`CONTROL_PLANE_DIR` to a shared directory to reuse verified binaries across runs.

## Model and data

Place these on shared storage and pass their paths at launch:

| Variable | Artifact |
| --- | --- |
| `MODEL_PATH` | [nvidia/Kimi-K3-NVFP4 at b2428a0b](https://huggingface.co/nvidia/Kimi-K3-NVFP4/tree/b2428a0b83a8b712ff2e1a8448a103d4175341f1) |
| `SGL_DRAFT_PATH` | [RadixArk/Kimi-K3-DSpark at 3c5bac30](https://huggingface.co/RadixArk/Kimi-K3-DSpark/tree/3c5bac301d9cf392706189d82ed947feca6c2f0f) |
| `DATASET_PATH` | `mlperf_agentic_inference_dataset.jsonl`; see the [dataset instructions](../../../dataset/README.md#kimi-k3) |

```bash
hf download nvidia/Kimi-K3-NVFP4 --revision b2428a0b83a8b712ff2e1a8448a103d4175341f1 \
  --local-dir /shared/models/Kimi-K3-NVFP4
hf download RadixArk/Kimi-K3-DSpark --revision 3c5bac301d9cf392706189d82ed947feca6c2f0f \
  --local-dir /shared/models/Kimi-K3-DSpark
```

## Serving behavior

The frontend parses Kimi K3 reasoning and tool calls, runs with
`--migration-limit 0`, and routes each conversation's prefill to the same worker
and engine rank for up to four hours. Each point fixes its decode routing
(`DYNAMO_DECODE_ROUTER_MODE`) and its decode entry point (`DYNAMO_DECODE_ENTRY`):
the image's visible-EOS helper, which keeps stop tokens visible in the output while
applying the requested stop conditions, or the stock `dynamo.trtllm` module. Keep
each point's engine configurations, environment and these settings as they are;
they differ between points.

## Run

Both systems run every point through one launcher,
[scripts/run.py](scripts/run.py). It composes [slurm_env.yaml](slurm_env.yaml),
[setup.yaml](setup.yaml), [disagg.yaml](disagg.yaml), the point and your settings,
and runs the result with `sflow run`. Set these in a local environment file passed
with `--env`, or one at a time with `--set KEY=VALUE`:

| Setting | Value |
| --- | --- |
| `SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_QOS`, `WALL_TIME` | Your account, partition, QoS (default `normal`; an empty value omits `--qos`) and time limit (default `04:00:00`). Set the account: it is passed as given. |
| `MODEL_PATH`, `SGL_DRAFT_PATH`, `DATASET_PATH` | Shared paths to the artifacts in [Model and data](#model-and-data). |
| `SERVER_IMAGE`, `CLIENT_IMAGE` | The pinned images, or unchanged SquashFS imports of them. |
| `CONTAINER_SCRATCH` | Writable node-local disk for container extraction and the SWE-bench containers (default `/raid/scratch`). |
| `SLURM_EXTRA_ARGS` | Additional `sbatch` options as a YAML list, for example `['--gpus-per-node=4']` where jobs must request GPUs. Set it only in the `--env` file: a `--set` value arrives as a string, which the launcher splits into one character per option. |

The launcher sets `REPO_ROOT` from the checkout and `RESULTS_PATH` from `--output`.
A local environment file holds a `variables:` block in the format of
[slurm_env.yaml](slurm_env.yaml), and its values override the defaults:

```yaml
variables:
  SLURM_ACCOUNT:
    value: your_account
  SLURM_PARTITION:
    value: your_partition
  WALL_TIME:
    value: '05:00:00'
  MODEL_PATH:
    value: /shared/models/Kimi-K3-NVFP4
  SGL_DRAFT_PATH:
    value: /shared/models/Kimi-K3-DSpark
  DATASET_PATH:
    value: /shared/data/mlperf_agentic_inference_dataset.jsonl
  SERVER_IMAGE:
    value: /shared/images/kimi-k3-dynamo.sqsh
```

Submit a point from the repository root with `--submit`, keeping run output outside
this checkout:

```bash
export PATH="$PWD/.venv/bin:$PATH"
```

```bash
.venv/bin/python NVIDIA/src/kimi-k3/scripts/run.py \
  NVIDIA/src/kimi-k3/configs/GB300-NVL72_GB300-288GB_aarch64x48_TRTLLM-Dynamo/point_12.yaml \
  --env /shared/kimi-k3-env.yaml --submit --output /shared/runs/kimi-k3-c12-attempt-1
```

```bash
.venv/bin/python NVIDIA/src/kimi-k3/scripts/run.py \
  NVIDIA/src/kimi-k3/configs/VR-NVL72_VR-288GB_aarch64x48_TRTLLM-Dynamo/point_15.yaml \
  --env /shared/kimi-k3-env.yaml --submit --output /shared/runs/kimi-k3-vr-c15-attempt-1
```

With `--submit`, the launcher creates the output directory, writes `run.sbatch`
there and submits it with the point's node count and segment size, `--exclusive`,
your account, partition, QoS, wall time and `SLURM_EXTRA_ARGS`. The job reruns the
launcher inside the allocation. Without `--submit`, run the same command inside an
existing allocation; its node count, partition and segment size must match the
point. Use a new output directory for every attempt; the launcher does not reuse
one.

Inside the allocation the launcher probes each node's NVLink fabric, keeps every
communicating worker within one measured fabric domain (different workers may use
different domains), writes the placement plan and the composed workflow, and runs
it. Plan the wall time per point: loading the model and starting every worker takes
tens of minutes, and the client then runs whole dataset passes; the
lowest-concurrency points and the SWE-bench points run longest.

### Allocation

| System | Points | GPUs | Nodes | Segment |
| --- | --- | ---: | ---: | ---: |
| GB300 | C1024, C640, C576, C512, C448 | 32 | 8 | 8 |
| GB300 | C384, C288, C192 | 48 | 12 | 12 |
| GB300 | C80, C64, C48, C32, C24, C16, C13 | 32 | 8 | 2 |
| GB300 | C12 | 48 | 12 | 2 |
| Vera Rubin | C1536, C1024, C512, C256, C193 | 32 | 8 | 4 |
| Vera Rubin | C192 | 48 | 12 | 4 |
| Vera Rubin | C96, C64, C48, C18 | 32 | 8 | 2 |
| Vera Rubin | C56, C40, C15 | 48 | 12 | 2 |

On Vera Rubin the launcher also binds each worker to its GPUs' NUMA node and pins
GPU memory clocks to 4752 MHz for the run, monitors them, and sets them back to the
range 3186–4752 MHz on exit, so passwordless `sudo nvidia-smi` must work on the
allocated nodes. Before the first run, check that `nvidia-smi -q -d SUPPORTED_CLOCKS`
lists 4752 MHz memory clocks. The restore locks the range NVIDIA's nodes default to;
if your nodes' default differs, reset the clocks afterwards with
`sudo nvidia-smi -rmc`. If a run fails, keep its output and confirm the memory clocks
were restored before releasing the allocation.

### Network

Prefill and decode transfer the KV cache with NIXL over UCX.
GB300 C192 to C640 set `UCX_NET_DEVICES` in `PREFILL_ENV` and `DECODE_ENV` to the
measuring site's network interface. Change the value to an interface your nodes
reach each other on, in both environments of each of those points; this is the one
environment setting you may change. Keep the key, which the workflow passes on under
`set -u`. C1024 sets it to `all`,
and C12 to C80 leave it unset, so UCX chooses.
Vera Rubin points clear `UCX_NET_DEVICES` and `UCX_TLS`, so UCX chooses.
The client and the frontend share a node (`CLIENT_NODE` and `FRONTEND_NODE`); the
client connects to `http://127.0.0.1:8000`.

### SWE-bench accuracy

GB300 C12, C16, C32 and C1024 run performance and SWE-bench Verified accuracy back
to back in one client run (`--mode both`; 200 instances, 16 workers). A
`swe_service` task on the client node starts the
[SWE-bench service](https://github.com/mlcommons/endpoints/tree/d3c57f2eba127c819e5a8fd83101d9e5f4e15fce/src/inference_endpoint/evaluation/swebench_service)
with [scripts/swe_service.py](scripts/swe_service.py). On first use it prepares the
service under `SWE_ROOT` (default: `client-results/swe/` in the launcher's `--output`
directory): it
fetches the Endpoints source at the client's revision from GitHub, a
checksum-verified ARM64 `uv` from PyPI, and the service's locked Python
environment, and downloads SWE-bench Verified from Hugging Face. Set `SWE_ROOT` to
a shared directory to reuse the prepared service across runs.

Its agent and evaluation containers run as Pyxis steps in the same allocation, from
the ARM64 images in `SWE_IMAGE_REGISTRY`. These points set it to the partner
registry's
`registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/swebench-arm64`;
the `registry.gitlab.com` credential for the serving image also pulls these images.
To use your own mirror of the same images, pass `--set SWE_IMAGE_REGISTRY=...`. The
service keeps one ~2.4 GB container per instance until the run ends, so the client
node needs about 500 GB free under `CONTAINER_SCRATCH`. SWE-bench adds about 60 to
70 minutes after performance (about 100 minutes at C1024); allow about five hours of
wall time for these points. A failed SWE-bench run still exits 0: check that
`accuracy/accuracy_results.json` shows `swe_bench` with `"complete": true` and a
score.

## Collect your run output

Each workflow starts etcd and NATS, the Dynamo TensorRT-LLM prefill and decode
workers, the Dynamo frontend, and then the client, and stops the services when the
benchmark finishes. Under the `--output` directory:

| Path | Contents |
| --- | --- |
| `client-results/report/` | The client report: `config.yaml`, `performance/result_summary.json` and `accuracy/accuracy_results.json`. |
| `output/` | The workflow output, with the server logs and configurations. |
| `fabric/`, `placement.json`, `workflow.yaml` | The fabric probe of each node, the placement plan and the composed workflow. |
| `run.sbatch`, `slurm-<job>.log`, `exit_code` | The batch script, its log and the workflow's exit code. |
| `clocks/` | Vera Rubin only: the memory clock pin, monitor and restore logs. |

Keep the report and the workflow output. Confirm the client completed every issued
trajectory without failures before treating a run as successful.
[submission/README.md](submission/README.md) covers packaging runs for an MLCommons
submission.
