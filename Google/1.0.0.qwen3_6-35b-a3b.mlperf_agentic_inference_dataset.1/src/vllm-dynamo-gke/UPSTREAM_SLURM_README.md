> Site-specific names, paths and addresses are replaced with `<placeholders>`.

# Run Qwen3.6-35B-A3B

Every point serves the model with four TP1 vLLM workers behind a Dynamo frontend
on one four-GPU node; a separate CPU allocation runs the Endpoints client and the
ARM64 SWE-bench service. Point names are concurrency settings, not measured
capacity. No benchmark results are included in this export.

| System | Approved concurrency points |
| --- | --- |
| GB300 (`GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo`) | **C32, C36, C64, C128, C144, C192, C256, C512, C768** |
| VR-NVL72 (`VR-NVL72_VR-288GB_aarch64x4_vLLM-Dynamo`) | **C32, C36, C64, C128, C144, C192, C256, C320, C384, C512** |

Each point directory is complete: `workflow.yaml` sets concurrency, Slurm
resources and tasks; `server.yaml` holds the vLLM settings; `client.yaml` holds
the performance and SWE-bench workload; `run.sh` and `batch.sh` launch it. Version
pins live in [dependencies.json](configs/dependencies.json), and the shared Dynamo
and SWE launchers live in `configs/_shared/`.

> **Serving images.** Each system has its own ARM64 build with `ai-dynamo` 1.4.2 in the
> partner registry repository
> `registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/centml-gb300-rubin`
> ([registry page](https://gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/container_registry/12460102)), pinned in `dependencies.json` by digest.
> Each digest also carries a tag, listed on that page:
>
> | System | Pin | vLLM build | Tag | Digest |
> | --- | --- | --- | --- | --- |
> | GB300 | `GB300_SERVER_IMAGE` | CentML vLLM `VLLM_BRANCH` at `VLLM_COMMIT` | [`gb300-vllm-7cc4a74c`](https://gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/container_registry/12460102) | `sha256:cb0b77df93c0f9d89b8bca7f59a2520e582b9dbebdd606f7d9f36f57632a8b71` |
> | VR-NVL72 | `VR_NVL72_SERVER_IMAGE` | CentML vLLM `VR_NVL72_VLLM_BRANCH` at `VR_NVL72_VLLM_COMMIT` | [`vr-nvl72-vllm-d48687fd`](https://gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/container_registry/12460102) | `sha256:f56c2ac713a2228372afcf295dc102c29d07609462a2a9e734033a5f4223f7e8` |
>
> Pull by digest, as the points do; a tag can be moved. Pulling needs the partner-registry
> credential described under SWE-bench setup.
> Each point's `server.sh` refuses an image whose `VLLM_BUILD_COMMIT` is not its system's
> pinned commit, and any Dynamo version other than `DYNAMO_VERSION` (`1.4.2`). The point's
> `server.yaml` carries its complete vLLM settings and feature environment (VR-NVL72 points
> extend `configs/_shared/env/VR-NVL72.env`); nothing is overlaid at run time.
> `server.sh` starts the workers through `configs/_shared/cuda-compat.sh`, which uses the
> image's CUDA forward-compatibility driver only when the host driver is older, and
> writes its decision to `cuda-compat.txt` in the run output.

## Environment and images

| Requirement | Expected setup |
| --- | --- |
| Cluster | Slurm, Pyxis and Enroot on ARM64 nodes, with writable shared storage at the same path on every node |
| Client partition | At least 69 CPUs per allocation, sufficient writable `/dev/shm`, and in SWE-bench modes about 500 GB free where Enroot keeps containers (`ENROOT_DATA_PATH`): the service keeps one ~2.4 GB container per instance until the run ends |
| Tools | `git` and `uv` on the login node, and the Hugging Face `hf` CLI (`uv tool install huggingface_hub`) |
| Host Python | Native ARM64 `python3` on compute nodes |
| Network | In `acc` and `both` modes the client node downloads SWE-bench Verified from Hugging Face at every run, and pulls the SWE-bench images from the partner registry |
| SWE service | The native ARM64 service from the pinned Endpoints revision in a Python 3.12 environment on the client nodes (SWE-bench modes only) |
| VR-NVL72 only | The `enroot` CLI on compute nodes, PyYAML for the host `python3`, and both images pre-imported as `.sqsh` files |

Install the pinned runner:

```bash
uv tool install 'git+https://github.com/NVIDIA/nv-sflow.git@b8afb5b86b3f46d9fa771531990fc7afaefc69b9'
```

The client image is `ghcr.io/mlcommons/endpoints:d3c57f2` at
`sha256:1523051b141437b25083924d6428082f435713516eee71c54ae207b393900b2d`
([Endpoints d3c57f2eba127c819e5a8fd83101d9e5f4e15fce](https://github.com/mlcommons/endpoints/tree/d3c57f2eba127c819e5a8fd83101d9e5f4e15fce)).
Every client config sets `submission_ref` to ruleset
`mlperf-endpoints-v1.0-2026-10-C1-A`, which pins the published cohort's
scheduler, dataloader and warmup seeds. Keep that binding unchanged.

## Model and dataset

Download the **BF16 source model** at the pinned revision. The recipe applies
online MXFP8 quantization; prequantized weights are not compatible.

```bash
hf download Qwen/Qwen3.6-35B-A3B \
  --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --local-dir "$SHARED_ROOT/models/Qwen3.6-35B-A3B"
```

Follow the [dataset instructions](../../../dataset/README.md#qwen36-35b-a3b) for the
613-trajectory JSONL and verify its checksum against `DATASET_SHA256` in
`dependencies.json`. Performance issues a whole number of dataset passes, set per
point by `num_trajectories_to_issue` in its `client.yaml` (4 to 15 passes on GB300 and
4 to 20 on VR-NVL72, more at higher concurrency) so the performance window exceeds 40
minutes; SWE-bench evaluates 200 instances once. The four accuracy points (C32, C36,
C64 and C192) default to `MODE=both`; the other points default to `MODE=perf`, as NVIDIA
measured them.
Each point's `workflow.yaml` sets `SLURM_TIME` to `05:00:00` for both phases; raise it if
your run does not finish within that.

## SWE-bench setup (acc or both)

Install the service from
[Endpoints d3c57f2](https://github.com/mlcommons/endpoints/tree/d3c57f2eba127c819e5a8fd83101d9e5f4e15fce/src/inference_endpoint/evaluation/swebench_service)
with its locked dependencies into a venv on shared storage, from an ARM64 node. A
uv-managed Python installed on the same shared storage keeps the venv usable on every
client node:

```bash
git clone https://github.com/mlcommons/endpoints.git "$SHARED_ROOT/endpoints"
git -C "$SHARED_ROOT/endpoints" checkout d3c57f2eba127c819e5a8fd83101d9e5f4e15fce
UV_PROJECT_ENVIRONMENT="$SHARED_ROOT/venvs/swebench-service" \
UV_PYTHON_INSTALL_DIR="$SHARED_ROOT/venvs/python" \
  uv sync --frozen --no-dev --python 3.12 --python-preference only-managed \
  --project "$SHARED_ROOT/endpoints/src/inference_endpoint/evaluation/swebench_service"
```

Then pass `--set SWE_SERVICE_PYTHON=$SHARED_ROOT/venvs/swebench-service/bin/python`
unless `python3` already imports `swebench_service`.

The service runs native ARM64 `sweb.eval.arm64.<instance_id>:v4.1.0-arm64` images
from `SWE_IMAGE_REGISTRY`, which defaults to the partner registry's
`registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/swebench-arm64`.
To use your own mirror of the same images, pass `--set SWE_IMAGE_REGISTRY=...`. Pulling
from the partner registry needs an Enroot `.credentials` entry for netrc machine
`registry.gitlab.com` with your login and a token with `read_registry` access:

```text
machine registry.gitlab.com login USER password TOKEN
```

Keep tokens in that private credentials file; configs and launch arguments carry
only the registry path and login name.

## Run a point

Clone this repository with `git` beneath a writable shared root: the launchers find
the repository root with `git` (from an archive, set `REPO_ROOT` to it instead). The
checkout, model, dataset and run directories must all be within `STORAGE_ROOT`, which
defaults to `SHARED_ROOT`; on VR-NVL72, which mounts only `SHARED_ROOT`, they must be
within `SHARED_ROOT`. A `QWEN_RUN_DIR` must also lie beneath `SHARED_ROOT`. Run from the
repository root on a login node.

A `both` run takes about 2.5 to 2.75 hours at C192 on GB300. The server starts in 11 to
25 minutes; the first start on a new `SHARED_ROOT` spends about 15 minutes compiling
kernels with no log output, and later starts reuse that cache. Performance then takes about
70 minutes, SWE-bench about 60, and finishing 6 to 12. `SLURM_TIME` defaults to five
hours; lower it if your partitions allow less.

### GB300

Pyxis imports the pinned serving image at job start with your Enroot credential:

```bash
SLURM_ACCOUNT=your_account \
SLURM_PARTITION=your_gpu_partition \
SHARED_ROOT=/path/to/shared/storage \
bash NVIDIA/src/qwen3.6-35b-a3b/configs/GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo/point_32/run.sh \
  --set CLIENT_SLURM_PARTITION=your_cpu_partition \
  --set DATASET_PATH=/path/to/shared/storage/datasets/mlperf_agentic_inference_dataset.jsonl \
  --set MODE=perf --dry-run --skip-artifact-check
```

### VR-NVL72

VR points start both containers with Enroot from `.sqsh` files imported in
advance. Import the pinned serving and client images, then pass both paths:

```bash
enroot import -o /path/to/shared/storage/images/qwen-server-vr.sqsh \
  'docker://registry.gitlab.com#nvidia/mlperf-inference-partner/nv-mlpinf-partner/centml-gb300-rubin@sha256:f56c2ac713a2228372afcf295dc102c29d07609462a2a9e734033a5f4223f7e8'
enroot import -o /path/to/shared/storage/images/endpoints-d3c57f2.sqsh \
  'docker://ghcr.io#mlcommons/endpoints@sha256:1523051b141437b25083924d6428082f435713516eee71c54ae207b393900b2d'

SLURM_ACCOUNT=your_account \
SLURM_PARTITION=your_gpu_partition \
SHARED_ROOT=/path/to/shared/storage \
bash NVIDIA/src/qwen3.6-35b-a3b/configs/VR-NVL72_VR-288GB_aarch64x4_vLLM-Dynamo/point_32/run.sh \
  --set SERVER_SQSH=/path/to/shared/storage/images/qwen-server-vr.sqsh \
  --set CLIENT_SQSH=/path/to/shared/storage/images/endpoints-d3c57f2.sqsh \
  --set CLIENT_SLURM_PARTITION=your_cpu_partition \
  --set DATASET_PATH=/path/to/shared/storage/datasets/mlperf_agentic_inference_dataset.jsonl \
  --set MODE=perf --dry-run --skip-artifact-check
```

Each VR run records the checksums of both `.sqsh` files in `images.sha256`.
Leave `GPU_MEM_CLOCK_LOCK` unset: that optional memory-clock lock calls a helper
that is not part of this release. NVIDIA's Vera Rubin measurements ran with every
GPU's memory clock locked at 4752 MHz. To run the same way, lock the clocks on each
allocated node before the run (`sudo nvidia-smi -lmc 4752,4752`) and reset them
afterwards (`sudo nvidia-smi -rmc`).

Remove the dry-run options to execute once the serving image and assets are in
place. `batch.sh` accepts the same environment and arguments and submits the
workflow with `sflow batch`.

### Client placement

By default the Endpoints client and SWE service get their own allocation. Two optional
overlays change that; append one as the last `-f` argument to a point's `run.sh` or
`batch.sh` command:

| Overlay | Placement | Modes |
| --- | --- | --- |
| `configs/_shared/coallocate-client.yaml` | Server and client nodes in one Slurm job (the server on the first node). Use it where the client allocation would otherwise queue behind the server. `CLIENT_SLURM_PARTITION` is unused. | All |
| `configs/_shared/colocate-client.yaml` | The client runs on the server's node, on the eight physical cores the server reserves for it | `perf` only |

NVIDIA's GB300 performance-only measurements used `colocate-client.yaml`; its
performance-plus-SWE-bench and VR-NVL72 measurements used a separate client node.

SFlow does not add a GPU request to the Slurm allocations it makes. If your site
rejects GPU-partition jobs without one, do not pass `--gpus-per-node` through
`batch.sh`: it applies to every allocation, including the client's CPU-partition
one. Allocate the GPU node yourself and run `run.sh` from the login node against
it:

```bash
salloc --no-shell -A your_account -p your_gpu_partition -N1 --exclusive \
  --gpus-per-node=4 --mem=0 -t 05:00:00 -J qwen-point
export SLURM_JOB_ID=<that job id> SLURM_JOB_NODELIST=<its node> \
  SFLOW_SLURM_MULTI_BACKEND_SALLOC=1 SFLOW_SLURM_WRAPPER_BACKEND=cluster
# then the run.sh command above without the dry-run options. It stays in the
# foreground for the whole run, so detach it (setsid nohup ... &, or tmux), and
# release the allocation with `scancel` when it finishes
```

| SFlow override (`--set NAME=value`) | Default / purpose |
| --- | --- |
| `MODE` | `both` at C32, C36, C64 and C192 (performance, then SWE-bench), `perf` at the other points; `acc` runs SWE-bench alone |
| `SLURM_TIME` | `05:00:00`; must fit both partitions |
| `CLIENT_SLURM_PARTITION` | The GPU partition; use a CPU partition when available |
| `MODEL_PATH` | `$SHARED_ROOT/models/Qwen3.6-35B-A3B` |
| `DATASET_PATH` | `$SHARED_ROOT/datasets/mlperf_agentic_inference_dataset.jsonl` |
| `STORAGE_ROOT` | `$SHARED_ROOT`; common container bind root (GB300 only) |
| `SWE_IMAGE_REGISTRY` | Partner registry namespace above |
| `SWE_SERVICE_PYTHON` | `python3` |

## Collect and check output

GB300 runs write beneath `$SHARED_ROOT/runs/` (or `QWEN_RUN_DIR`); VR runs also
keep their Enroot run tree beneath `$SHARED_ROOT/runs/<system>/c<point>/<label>`.
Retain the resolved YAMLs, dependency pins, server/client logs and the client's
native reports (about 66 GB at C192, mostly `events.jsonl`). A successful Slurm exit
only means the workflow ran: confirm full performance coverage and a complete
200-instance SWE-bench report before accepting a run. A failed SWE-bench run still exits 0: before you accept a `both` run, check that
`accuracy/accuracy_results.json` in the client report shows `swe_bench` with
`"complete": true` and a score. Keep measured
output outside this source checkout.
