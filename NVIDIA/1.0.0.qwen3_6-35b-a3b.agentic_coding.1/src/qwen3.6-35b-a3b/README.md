# Qwen3.6-35B-A3B

For the Qwen serving image, see the [build instructions](BUILD.md).

Each point's `run.sh` runs **performance followed by SWE-bench** by default at the accuracy
points (C32, C36, C64, C192) and performance only at the others.
The customer recipes use four TP1 vLLM workers behind Dynamo on one four-GPU
ARM64 node. A separate CPU allocation runs the client and native ARM64 SWE
service. Performance issues 613 × `copies` trajectories (4–25 complete copies of the
613-trajectory dataset, per point, set in its `client.yaml`, so the performance window exceeds 40 minutes); SWE-bench evaluates 200 instances once.

## Source layout

```text
NVIDIA/src/qwen3.6-35b-a3b/
├── README.md
└── configs/
    ├── dependencies.json              # sole dependency, image and dataset pins
    ├── _shared/                       # common Dynamo/SWE helpers, CUDA-compat launch, autotune pins, FI K=64 tactics
    │   ├── dynamo-server.py
    │   ├── dynamo_cpu_affinity.py
    │   ├── swe-service.sh
    │   └── swe-service.py
    └── GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo/
        └── point_32/                  # one directory per concurrency
            ├── client.yaml            # full performance and SWE configuration
            ├── server.yaml            # full vLLM configuration for this point
            ├── workflow.yaml          # concurrency, Slurm resources and tasks
            ├── run.sh                 # direct sflow run command
            ├── client.sh              # client startup
            └── server.sh              # server startup
```

Each point contains its full configuration and its own run, client and server
launchers. Common Dynamo and SWE services live in [`configs/_shared/`](configs/_shared/README.md),
and version pins live in [`configs/dependencies.json`](configs/dependencies.json). Open the
point's `workflow.yaml` to see its concurrency and resources, `server.yaml` for
engine settings, and `client.yaml` for the performance and SWE workload.
The recipes assume the cluster and software environments are already prepared;
installation, registry authentication and storage are site-specific.
A copied point can set `REPO_ROOT` to the checkout containing this model's
`configs/dependencies.json` and `configs/_shared/`.

GB300 points: **32, 36, 64, 128, 144, 192, 256, 512, 768**.

> **Stacks.** GB300 runs s8g at C32-C256 and s7xq at C512/C768 (C768 with least-loaded Dynamo routing).
> Each point's `server.yaml` carries its full engine flags
> and feature environment (`SERVER_ENV`). The image ([`BUILD.md`](BUILD.md)) is built from YibaiMeng/vllm
> `mlperf-end-multiturn-v1.0-qwen-latest` at `VLLM_COMMIT`.

`configs/dependencies.json` is also a SFlow configuration. It pins the Endpoints
client image to `ghcr.io/mlcommons/endpoints:d3c57f2` at
`sha256:1523051b141437b25083924d6428082f435713516eee71c54ae207b393900b2d`.
The SWE service source is pinned to the matching revision
`d3c57f2eba127c819e5a8fd83101d9e5f4e15fce`, which executes agent tool commands
through a persistent worker in each instance container. `DATASET_SHA256` is the checksum of the input
**dataset** ([Dataset](#dataset)). The server image (`GB300_SERVER_IMAGE`) and the
SWE image registry (`SWE_IMAGE_REGISTRY`) are site-provided placeholders: build the server image with
[`BUILD.md`](BUILD.md) and prepare the SWE images as in [SWE-bench images](#swe-bench-images).
Dynamo's session TTL and CPU defaults live in `configs/_shared/dynamo-server.py`.

## Prerequisites

The cluster administrator prepares the following before running a point:

| Requirement | Expected setup |
| --- | --- |
| Cluster | Slurm, ARM64 GB300 GPUs, Enroot, Pyxis, and writable shared storage visible at the same paths on every node |
| Launcher | The pinned SFlow from `configs/dependencies.json` on the login node's PATH |
| Server | The pinned vLLM engine packages and `ai-dynamo==1.4.2` in the DP4 server environment; install Dynamo without replacing the image's vLLM, torch or transformers |
| SWE service | The native ARM64 service from the pinned Endpoints revision, installed with its locked dependencies in a Python 3.12 environment on the client nodes |
| Host Python | Native ARM64 `python3` on compute nodes |
| Images | The server image built per [`BUILD.md`](BUILD.md), the pinned client image and the ARM64 SWE images, with registry credentials and Enroot storage configured by the site |
| Assets | The pinned BF16 model and complete dataset on shared storage |

The point commands start the prepared services. They do not install packages,
patch Endpoints, or configure the site's Enroot storage. For SWE, `python3` must
import `swebench_service`, or pass `--set SWE_SERVICE_PYTHON=/path/to/service-venv/bin/python`.
The client needs sufficient writable `/dev/shm` for the pinned Endpoints runtime.
The client partition needs at least 69 CPUs per allocation. The default is the
server partition; use `--set CLIENT_SLURM_PARTITION=<cpu-partition>` when available.
The full run must fit the selected partitions' time limits.

Clone this repository beneath a writable shared root. If SFlow is not already
prepared, install it once on the login node:

```bash
export SHARED_ROOT=/path/to/shared/storage
cd "$SHARED_ROOT/mlperf-endpoints"
python3 -m venv .venv
.venv/bin/pip install \
  'sflow @ git+https://github.com/NVIDIA/nv-sflow.git@b8afb5b86b3f46d9fa771531990fc7afaefc69b9'
export PATH="$PWD/.venv/bin:$PATH"
```

The administrator can use the upstream
[SWE service setup instructions](https://github.com/mlcommons/endpoints/tree/d3c57f2eba127c819e5a8fd83101d9e5f4e15fce/src/inference_endpoint/evaluation/swebench_service)
when preparing the pinned service environment.

Download the **BF16 source model** at the revision in `configs/dependencies.json`.
Online MXFP8 quantization is configured by the recipe; prequantized weights
are not compatible with these settings.

```bash
.venv/bin/pip install huggingface_hub
.venv/bin/hf download Qwen/Qwen3.6-35B-A3B \
  --revision "$(python3 -c 'import json; print(json.load(open("NVIDIA/src/qwen3.6-35b-a3b/configs/dependencies.json"))["variables"]["MODEL_REVISION"]["value"])')" \
  --local-dir "$SHARED_ROOT/models/Qwen3.6-35B-A3B"
```

### Dataset

Download the dataset using the
[MLCommons Agentic Inference instructions](https://github.com/mlcommons/endpoints/tree/d3c57f2eba127c819e5a8fd83101d9e5f4e15fce/examples/10_Agentic_Inference#dataset).
Pass `--set DATASET_PATH=<path>` for the resulting JSONL, or place it at
`$SHARED_ROOT/datasets/mlperf_agentic_inference_dataset.jsonl`. Verify its content before running:

```bash
export DATASET_PATH=/path/to/shared/storage/datasets/mlperf_agentic_inference_dataset.jsonl
python3 - "$DATASET_PATH" <<'PY'
import hashlib, json, sys
pins = json.load(open('NVIDIA/src/qwen3.6-35b-a3b/configs/dependencies.json'))['variables']
with open(sys.argv[1], 'rb') as source:
    actual = hashlib.file_digest(source, 'sha256').hexdigest()
assert actual == pins['DATASET_SHA256']['value'], 'Dataset checksum mismatch'
PY
```

### SWE-bench images

Configure Enroot registry access using the site's standard credential location.
Set `SWE_IMAGE_REGISTRY` (in `configs/dependencies.json`, or with `--set`) to a namespace you can access that
contains the native `sweb.eval.arm64.<instance_id>:v4.1.0-arm64` images (`SWE_REGISTRY_CHECK_TAG`). For an authenticated Enroot
registry, include the username in the namespace and the matching credential
entry, for example `username@registry.example.com/team/swebench-arm64`.
The upstream guide also documents how to build the ARM64 SWE images.

## Run a point

From the repository root on a login node, fill in the account, partition and
shared root. SFlow allocates the server and client nodes:

```bash
SLURM_ACCOUNT=<account> \
SLURM_PARTITION=<gpu-partition> \
SHARED_ROOT=/path/to/shared/storage \
bash NVIDIA/src/qwen3.6-35b-a3b/configs/GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo/point_32/run.sh \
  --set CLIENT_SLURM_PARTITION=<cpu-partition> \
  --set DATASET_PATH=/path/to/shared/storage/datasets/mlperf_agentic_inference_dataset.jsonl
```

Set `GB300_SERVER_IMAGE` in `configs/dependencies.json` to the image you built
([`BUILD.md`](BUILD.md)), or pass `--set SERVER_IMAGE=<prepared-server-image>`.
Each script contains one `sflow run`
command that loads the dependency pins and that point's three YAML files.
Add `--dry-run` to validate and display SFlow's plan without allocating nodes.
All extra arguments pass directly to SFlow; see `sflow run --help`.

To submit a batch job, run `bash NVIDIA/src/qwen3.6-35b-a3b/configs/GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo/point_32/batch.sh` with the same environment and extra arguments as `run.sh`; it returns the job ID.

### Allocate the client node with the server

By default SFlow allocates the client separately from the server. On
clusters without a CPU partition, the server can then consume its time limit
while the client allocation waits in the queue. Append this overlay as the last
`-f` argument to a point's `run.sh` or `batch.sh` command to request both nodes
in one Slurm job:

```bash
  -f NVIDIA/src/qwen3.6-35b-a3b/configs/_shared/coallocate-client.yaml
```

Slurm starts both nodes together or neither. The server runs on the job's first
node; the performance client and SWE service run on the second node with the
point's normal settings, so every `MODE` is supported. `CLIENT_SLURM_PARTITION`
is unused; both nodes come from `SLURM_PARTITION`.

### Colocate the performance client

On clusters without CPU nodes, append this overlay as the last `-f` argument to
a point's `run.sh` or `batch.sh` command to use only the existing four-GPU server
allocation:

```bash
  -f NVIDIA/src/qwen3.6-35b-a3b/configs/_shared/colocate-client.yaml
```

The overlay selects `MODE=perf`; it rejects `acc` and `both`. SWE-bench creates
additional Slurm container steps and continues to use the default separate-client
setup. Existing commands without the overlay keep their current behavior.

The server already reserves **eight physical cores for the client**. The overlay
uses these otherwise-unused cores, reading their recorded mask from
`dp4-topology.json`. Its eight HTTP workers and LoadGen processes share that mask.
Endpoints' internal CPU affinity is disabled in the runtime client config so it
cannot replace the reserved mask with a host-wide allocation.

Eight client cores are a starting configuration, not a measured throughput
guarantee. Compare achieved concurrency and throughput with the separate client
before using colocated measurements as a baseline.

| SFlow override (`--set NAME=value`) | Default / purpose |
| --- | --- |
| `MODE` | `both` at C32, C36, C64 and C192, `perf` at the other points; select `perf`, `acc` or `both` explicitly |
| `SLURM_TIME` | `05:00:00`; must fit both partitions |
| `MODEL_PATH` | `$SHARED_ROOT/models/Qwen3.6-35B-A3B` |
| `DATASET_PATH` | `$SHARED_ROOT/datasets/mlperf_agentic_inference_dataset.jsonl` |
| `STORAGE_ROOT` | `$SHARED_ROOT`; a common container bind root for the checkout and assets |
| `SWE_IMAGE_REGISTRY` | Registry namespace with the ARM64 SWE images; required for SWE ([SWE-bench images](#swe-bench-images)) |
| `SWE_SERVICE_PYTHON` | `python3`; Python environment with the pinned SWE service installed |

For example, append `--set MODE=perf` or `--set SLURM_TIME=1-00:00:00`.
The script uses `sflow` on PATH.
`QWEN_RUN_DIR` selects the output directory, otherwise each run gets a unique
directory beneath `$SHARED_ROOT/runs/`. `LOCAL_SCRATCH_ROOT` defaults to
`/tmp/qwen-<uid>`; set it to the site's approved node-local path.

Additional `--set NAME=value` arguments reach SFlow. The recipes preserve the
quantization exclusions, MTP3, graph settings, unlimited performance run/drain
timeouts, inline accuracy, and disabled early stopping.

SFlow reports its output locations. Retain the resolved YAMLs, dependency pins,
logs and client reports.
A successful Slurm exit is only the execution result; verify full performance
coverage and a complete 200-instance SWE report before accepting a measurement.
