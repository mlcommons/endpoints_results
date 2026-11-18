# Run DeepSeek-V4.1-Flash

Every point serves `deepseek-ai/DeepSeek-V4.1-Flash` on one four-GPU node. Point
names are concurrency settings, not measured capacity. The measured results
are under `results/`.

| System | Points |
| --- | --- |
| Vera Rubin NVL72, 4 GPUs | **C16, C24, C32, C64, C128, C256, C512, C1024, C2048, C3072** |

One Slurm allocation
starts these workflow tasks:

1. A host task that drops the page cache before the model loads. It also
   pins the GPU memory clocks to 4752 MHz and checks them.
2. A Dynamo frontend that routes requests round-robin while keeping each agentic
   conversation on the same worker for 3,600 seconds.
3. Four `dynamo.vllm` data-parallel ranks with expert parallelism, one GPU each,
   behind the frontend. Each point's `VLLM_ARGS` holds its engine settings.
4. A native SWE-bench service that launches the ARM64 evaluation sandboxes.
5. The MLCommons Endpoints client: agentic performance with inline accuracy and,
   in `both` mode, 200 SWE-bench Verified tasks.

The MLCommons accuracy thresholds for this model, as checker 1.5.1.0 applies them:
inline accuracy at least 52.36 and mean output length 793 to 970 tokens at every point,
and a SWE-bench Verified mean of at least 96.4 across the line's accuracy points.

## Environment and images

- Slurm with Pyxis and Enroot on ARM64 nodes; one node with four GPUs per job.
- Shared storage visible at the same absolute path on login and compute nodes.
- Git and Python 3.12 or newer on the login node, and Hugging Face access for the
  model download with the `hf` CLI (`pip install -U huggingface_hub`). Clone this
  repository with `git`: `run_point.sh` finds the repository root with it, so an archive
  download does not work.
- On compute nodes: `python3` 3.10 or newer (each job builds its SFlow environment with
  it) and GitHub and PyPI access (each job installs SFlow, and the SWE-bench service
  setup installs its packages). In `both` mode the SWE-bench sandboxes also need about
  500 GB free where Enroot keeps containers on the GPU node.
- The batch script requests the whole node (`--exclusive`) and no GPUs by count. A site
  that requires a GPU request needs it added to the `sbatch` arguments in the recipe's
  `run_point.sh`.
- Compute-node access to the client image below and to your rebuilt serving image:

| Variable | Image |
| --- | --- |
| `VLLM_CONTAINER_IMAGE` | Required: your image built with [BUILD.md](BUILD.md), passed as `docker://<registry>/<repository>@sha256:<full-manifest-digest>` |
| `SWE_IMAGE_REGISTRY` | Required: your registry/repository prefix containing the ARM64 SWE-bench instance images built with [BUILD.md](BUILD.md#build-the-arm64-swe-bench-images) |
| `ENDPOINT_CONTAINER_IMAGE` | `ghcr.io/mlcommons/endpoints:4235a9c` (ARM64 manifest `sha256:d863e3b97a83dc73b18bd7d4824f2d40b9979a8b482b9a9defd80e5bfe96fd68`; a tag can be moved, so confirm the digest after pulling) |

Build the serving image and ARM64 SWE-bench images using [BUILD.md](BUILD.md).
Export both required settings in the shell that invokes `run.sh`:

```bash
export VLLM_CONTAINER_IMAGE="docker://registry.example.com/your-project/centml-rubin-vllm@sha256:<full-manifest-digest>"
export SWE_IMAGE_REGISTRY="registry.example.com/your-project/swebench-arm64"
```

Both defaults are empty; the launcher stops if either is missing. If your
chosen registry requires authentication, configure Enroot credentials for
that registry on the compute nodes. The SWE-bench value is a repository
prefix, without `docker://`, an instance name or a tag.

Adapt these site assumptions before running:

- Passwordless `sudo -n /sbin/sysctl` must work on the compute node for the page
  cache drop.
- On Vera Rubin, passwordless `sudo -n nvidia-smi` must also work for the memory
  clock gate, and the recipe's `slurm_env.yaml` and SWE-bench setup job pass
  `--segment=1` to Slurm. Remove it if your scheduler does not support `--segment`.
  Set `SLURM_CONSTRAINT` or `SLURM_EXCLUDE` to pass a node constraint or exclusion.
- The SWE-bench service runs on the host with the `PATH` from each point's
  `endpoint_env.yaml`, and it launches sandboxes with `srun`. Add your Slurm
  binary directory to that `PATH` if `srun` is not in a standard location.

Install the pinned runner into a virtual environment in this checkout:

```bash
python3 -m venv .venv
.venv/bin/pip install 'sflow @ git+https://github.com/NVIDIA/nv-sflow.git@b8afb5b86b3f46d9fa771531990fc7afaefc69b9'
```

## Model and data

`SCRATCH_DIR` is a shared storage root with this layout:

```text
${SCRATCH_DIR}/
├── models/DeepSeek-V4.1-Flash/
└── data/agentic_dataset/
    ├── perf/mlperf_agentic_inference_dataset.jsonl
    └── dataset_cache/
```

Download the checkpoint at the pinned revision:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277 \
  --local-dir "$SCRATCH_DIR/models/DeepSeek-V4.1-Flash"
```

Follow the [dataset instructions](../../docs/dataset.md#deepseek-v41-flash) for
the performance JSONL and the SWE-bench Verified cache.

## SWE-bench service

On Vera Rubin, install the service once. Submit the setup job from the recipe
directory so it runs on an ARM64 compute node and creates `swebench_service/.venv`
there:

```bash
cd NVIDIA/src/deepseek-v4.1-flash/configs/VR-NVL72_VR-288GB_aarch64x4_vLLM-Dynamo
sbatch --account=your_account --partition=your_partition setup_swebench_service.sbatch
```

The [setup script](configs/VR-NVL72_VR-288GB_aarch64x4_vLLM-Dynamo/setup_swebench_service.sbatch)
is safe to rerun.

Both setup scripts pin Endpoints commit `e9f427ce74fd90e7fafdbc77ee7746cc00245692`.
The service runs the native ARM64 SWE-bench images from `SWE_IMAGE_REGISTRY`,
built at the pinned service revision as described in [BUILD.md](BUILD.md#build-the-arm64-swe-bench-images).
The workflow and client share a fixed bearer token for it, and it only listens on
`127.0.0.1`. Run one SWE-bench campaign at a time; the service allows one
concurrent run.

## Run a point

Each point's `run.sh` hands the point to its recipe's shared `run_point.sh`, which
freezes the point's inputs into a new directory under the recipe's `run_logs/`
(set `RUN_LOG_ROOT` to keep output elsewhere), renders a batch script with a
per-job SFlow bootstrap, and submits it. SFlow defaults to this checkout's
`.venv/bin/sflow`; set `SFLOW` to use another install, such as the
`~/.local/bin/sflow` that `uv tool install` creates. `run_point.sh` resolves `SFLOW`
with `readlink -f` and runs the `python` beside the real executable, which must
import PyYAML (an SFlow dependency); it stops with an error if it can't.

```bash
point=NVIDIA/src/deepseek-v4.1-flash/configs/VR-NVL72_VR-288GB_aarch64x4_vLLM-Dynamo/point_16
SLURM_ACCOUNT=your_account \
SLURM_PARTITION=your_partition \
SCRATCH_DIR=/path/to/shared/storage \
"$point/run.sh" --dry-run
```

`--dry-run` renders the batch script without submitting it; replace it with
`--submit` to queue the job. `SLURM_TIME` comes from the recipe's `slurm_env.yaml`
(five hours) unless you set it; it must fit your partition's
time limit. A `both` point
takes about 2 hours 20 minutes, for example C24: workers ready in about 7 minutes
with warm compile caches, performance about 58 minutes, SWE-bench about 75 minutes. Each point's `MODE` and `DATASET_REPEATS` (whole passes of the
613-trajectory dataset) are part of its configuration:

| System | Points | `MODE` |
| --- | --- | --- |
| Vera Rubin NVL72 | C16, C24, C128, C3072 | `both`: performance with inline accuracy, then 200 SWE-bench tasks |
| Vera Rubin NVL72 | C32, C64, C256, C512, C1024, C2048 | `perf`: performance with inline accuracy |

## Collect and check output

Retain each run directory under `run_logs/`: the frozen inputs, the Slurm logs, the
endpoint report, `result_summary.json`, the SWE-bench status and the server logs.
A `COMPLETED` scheduler state alone does not qualify a run: confirm the endpoint
report is complete, no performance samples failed, and, in `both` mode, all 200
SWE-bench tasks were evaluated without service errors. A failed SWE-bench run still exits 0: before you accept a `both` run, check that
`accuracy/accuracy_results.json` in the client report shows `swe_bench` with
`"complete": true` and a score. Keep
measured output outside this source checkout.
