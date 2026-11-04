> Site-specific names, paths and addresses are replaced with `<placeholders>`.

# Run GPT-OSS-120B on GB300

Approved concurrency points: **C17, C35, C70, C140, C272, C544, C1024, C2048, C3840, C7168, C22528**. Every point uses 72 GB300 GPUs on
eighteen four-GPU nodes. Each point directory defines the prefill/decode replicas,
GPUs per replica, server options and client workload. Point names are concurrency
settings, not measured capacity. No benchmark results are included in this export.

## Environment and images

Use ARM64 GB300 nodes, Slurm/Pyxis/Enroot and shared storage. Keep the allocation
in one NVLink domain. Adapt the default `--segment=18` and `--exclusive` placement
in [slurm_env.yaml](configs/_shared/slurm_env.yaml) to the scheduler's supported
options while preserving each worker's topology.

Install the pinned runner:

```bash
uv tool install 'git+https://github.com/NVIDIA/nv-sflow.git@b8afb5b86b3f46d9fa771531990fc7afaefc69b9'
```

| Variable | Image |
| --- | --- |
| `CONTAINER_IMAGE` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14` |
| `ENDPOINT_CONTAINER_IMAGE` | `ghcr.io/mlcommons/endpoints:e9f427c` (ARM64 manifest `sha256:f52feabcf1e2107c213c1f5699d8101119193b46a25d1af51b63efb7ac541fac`; a tag can be moved, so confirm the digest after pulling) |
| `LCB_CONTAINER_IMAGE` | `ghcr.io/mlcommons/endpoints:9d749906-livecodebench` |

Pre-import on an ARM64 host if compute nodes cannot reach the registries:

```bash
mkdir -p /shared/images
enroot import -o /shared/images/trtllm.sqsh \
  'docker://nvcr.io#nvidia/tensorrt-llm/release:1.3.0rc14'
enroot import -o /shared/images/endpoints.sqsh \
  'docker://ghcr.io#mlcommons/endpoints:e9f427c'
enroot import -o /shared/images/lcb.sqsh \
  'docker://ghcr.io#mlcommons/endpoints:9d749906-livecodebench'
```

The TRT-LLM ARM64 manifest is
`sha256:e71eb8056ac1c717e6b727f9d66ebba9442e95f7ebc3514007b541046e84b088`.
Record resolved image digests. The client corresponds to
[Endpoints e9f427ce74fd90e7fafdbc77ee7746cc00245692](https://github.com/mlcommons/endpoints/tree/e9f427ce74fd90e7fafdbc77ee7746cc00245692).
The workflow starts the LiveCodeBench service in every mode, including `perf`;
all three images are required by the current template.

## Model, datasets and caches

Every point serves
[openai/gpt-oss-120b at b5c939de](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a),
which includes its tokenizer. Download the model at that revision to the point
configs' default `MODEL_PATH`, `$SCRATCH_DIR/models/gpt-oss-120b`. The serving
image needs only the top-level model files, so skip the repository's `original/`
and `metal/` weight copies:

```bash
scratch_dir=/shared/benchmark-assets
hf download openai/gpt-oss-120b \
  --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --exclude 'original/*' --exclude 'metal/*' \
  --local-dir "$scratch_dir/models/gpt-oss-120b"
```

Do not substitute another precision or model variant. The default `DATA_PATH`
is `$SCRATCH_DIR/data/gpt-oss`. If you stage the model or data elsewhere, pass
`--set MODEL_PATH=<dir>` or `--set DATA_PATH=<dir>`.

The `DATA_PATH` directory must contain:

```text
v4/perf/perf_eval_ref.parquet
endpoint_accuracy/          # Populated accuracy dataset cache
harmony_cache_gptoss/       # Populated Harmony tokenizer cache
```

Use the [dataset instructions](../../../dataset/README.md#gpt-oss-120b) for download
and staging. The site template mounts `endpoint_accuracy/` read-only at the client's
`<node-mount>/inference-endpoint/dataset_cache`, and the client loads AIME25, GPQA and
LiveCodeBench from it in every mode, `perf` included. Populate it once before the first
run, as the dataset instructions show; an empty cache fails at client setup. GPT-OSS's accuracy datasets are loaded by their named
client definitions. Provision required dataset access, including gated GPQA,
and populate caches before running on nodes without network egress. Never put
access tokens in exported configs or launch logs.

Populate the Harmony cache from a network-enabled host using the shipped helper:

```bash
CACHE_DIR=/shared/benchmark-assets/data/gpt-oss/harmony_cache_gptoss \
  bash NVIDIA/src/sflow/tools/setup_harmony_cache.sh
```

## Site network overlay (RoCE)

The prefill and decode servers transfer the KV cache over UCX. Each point's
`server-env.yaml` is the shipped baseline. NVIDIA measured these configurations
only on RoCE, with overlays like the ones below; on InfiniBand, start from the shipped
settings and confirm the KV cache does not fall back to TCP. On a RoCE fabric, give the
servers a site overlay: a flat YAML of environment variables merged over the
point's `server-env.yaml` for every prefill and decode server, later keys winning.
Without one, UCX can move the KV cache over TCP and decode stalls. Replace the device
names and GID index below with your NICs':

```yaml
UCX_TLS: '^posix,tcp'                        # keep KV transfer off TCP; skip POSIX shared memory across Pyxis steps
UCX_IB_GID_INDEX: '3'                        # your NICs' RoCE v2 GID index
UCX_NET_DEVICES: 'mlx5_${SLURM_LOCALID}:1'   # one RoCE device per local rank
```

The servers read the overlay inside their containers, where only this checkout
and `SCRATCH_DIR` are mounted, so save it under one of them and add
`--set SITE_ENV_YAML=<path>` to the run. `${SLURM_LOCALID}` expands per rank, and
the run output keeps a copy as `site-env.yaml`.

C70 and C140 place their two prefill workers and one decode worker on the same
node. With the overlay above, that same-node KV transfer did not complete on
NVIDIA's RoCE system, and neither point has been validated. On RoCE, confirm
same-node transfer works with your overlay before running them.

## Run a point

From the repository root, customize the following site paths and allocation:

```bash
point_dir="$PWD/NVIDIA/src/gpt-oss-120b/configs/GB300-NVL72_GB300-288GB_aarch64x72_TRTLLM/point_17"
scratch_dir=/shared/benchmark-assets
run_dir=/shared/runs/gptoss-c17
ls "$scratch_dir"/data/gpt-oss/endpoint_accuracy/{aime25/aime25.parquet,gpqa/diamond/gpqa_diamond.parquet,livecodebench/release_v6/livecodebench_release_v6.parquet}
mkdir -p "$run_dir/workspace" "$run_dir/output"
sflow run \
  -f "$point_dir/gptoss_config.yaml" \
  -f NVIDIA/src/gpt-oss-120b/configs/_shared/slurm_env.yaml \
  -f NVIDIA/src/gpt-oss-120b/configs/_shared/templates/trtllm_disagg_endpoints_gptoss_perf_acc.yaml \
  --set WORK_DIR="$PWD" --set SCRATCH_DIR="$scratch_dir" \
  --set SLURM_ACCOUNT=your_account --set SLURM_PARTITION=your_partition \
  --set CONTAINER_IMAGE=/shared/images/trtllm.sqsh \
  --set ENDPOINT_CONTAINER_IMAGE=/shared/images/endpoints.sqsh \
  --set LCB_CONTAINER_IMAGE=/shared/images/lcb.sqsh \
  --set MODE=perf --workspace-dir "$run_dir/workspace" --output-dir "$run_dir/output" \
  --dry-run --skip-artifact-check
```

Remove the final dry-run options to execute after verifying actual assets. Keep
model/data paths beneath `SCRATCH_DIR` or add their container mounts explicitly.
Use fresh output/workspace directories each time. Set `MODE=both` for performance
followed by accuracy, which uses AIME25, GPQA and LiveCodeBench with the repetitions
and token limits in the selected client YAML. With client `e9f427c`, `MODE=perf` still
runs those accuracy datasets after performance, and `MODE=acc` runs them alone at
concurrency 1 over one connection, which does not finish within the five-hour job; use
`both` for accuracy. C17 to C140 allow the client 16,200 s (`run_timeout_s`) for
performance and accuracy together.

For `sflow batch --submit`, use the same files/overrides and explicitly supply `-N 18`,
`-p your_partition`, `-A your_account` and `--time 05:00:00`. YAML allocation
variables alone do not set the batch command-line allocation options.

## Collect and check output

Retain the full SFlow output: `endpoint_benchmark/`, native reports/event logs,
resolved client config, accuracy outputs, server logs and `snapshot_configs/`.
The client report directory is `${SFLOW_TASK_OUTPUT_DIR}`. Check completion,
failure counts and actual evaluator outputs before calling a run successful.
Dry-runs and serving startup alone do not validate benchmark execution. Keep
measured output outside this source checkout.
