> Site-specific names, paths and addresses are replaced with `<placeholders>`.

# Run DeepSeek-R1 on GB300

Approved concurrency points: **C32, C48, C64, C128, C256, C512, C1024, C2048, C4096, C8192, C16384, C28080**. Every point uses 72 GB300 GPUs on
eighteen four-GPU nodes. The point directory defines its prefill/decode replica
counts, GPUs per replica, server options and client workload. These are launch
configurations; no benchmark results are included or certified by this export.

## Prepare the environment

Use Slurm with Pyxis/Enroot, ARM64 nodes and shared storage. Keep the allocation
within an NVLink domain. The default placement requests `--segment=18` and
`--exclusive`; adapt supported placement arguments in
[slurm_env.yaml](configs/_shared/slurm_env.yaml) to your scheduler and fabric.
Multi-node server workers must retain their configured topology.

Install the pinned workflow runner on the launch host:

```bash
uv tool install 'git+https://github.com/NVIDIA/nv-sflow.git@b8afb5b86b3f46d9fa771531990fc7afaefc69b9'
```

The defaults use these images; pre-import them on an ARM64 host and pass shared
`.sqsh` paths if compute nodes cannot access the registries:

| Variable | Image |
| --- | --- |
| `CONTAINER_IMAGE` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14` |
| `ENDPOINT_CONTAINER_IMAGE` | `ghcr.io/mlcommons/endpoints:e9f427c` (ARM64 manifest `sha256:f52feabcf1e2107c213c1f5699d8101119193b46a25d1af51b63efb7ac541fac`; a tag can be moved, so confirm the digest after pulling) |
| `LCB_CONTAINER_IMAGE` | `ghcr.io/mlcommons/endpoints:9d749906-livecodebench` |

```bash
mkdir -p /shared/images
enroot import -o /shared/images/trtllm.sqsh \
  'docker://nvcr.io#nvidia/tensorrt-llm/release:1.3.0rc14'
enroot import -o /shared/images/endpoints.sqsh \
  'docker://ghcr.io#mlcommons/endpoints:e9f427c'
enroot import -o /shared/images/lcb.sqsh \
  'docker://ghcr.io#mlcommons/endpoints:9d749906-livecodebench'
```

The TRT-LLM tag resolves to ARM64 manifest
`sha256:e71eb8056ac1c717e6b727f9d66ebba9442e95f7ebc3514007b541046e84b088`.
Record resolved image digests for your run. The client corresponds to
[Endpoints e9f427ce74fd90e7fafdbc77ee7746cc00245692](https://github.com/mlcommons/endpoints/tree/e9f427ce74fd90e7fafdbc77ee7746cc00245692).
The workflow launches the LCB service in every mode, including `perf`, so provide
all three images. Accuracy uses the client's `legacy_mlperf_deepseek_r1` evaluator.

## Model and data

Every point serves the ModelOpt NVFP4 checkpoint
[centml/DeepSeek-R1-NVFP4-v2-mlpinf at 93947a0d](https://huggingface.co/centml/DeepSeek-R1-NVFP4-v2-mlpinf/tree/93947a0d7bd04f73ff98636f6b18ff5839e7aaf9),
which includes its tokenizer. Download it at that revision to the point configs'
default `MODEL_PATH`, `$SCRATCH_DIR/models/DeepSeek-R1-NVFP4-v2`:

```bash
scratch_dir=/shared/benchmark-assets
hf download centml/DeepSeek-R1-NVFP4-v2-mlpinf \
  --revision 93947a0d7bd04f73ff98636f6b18ff5839e7aaf9 \
  --local-dir "$scratch_dir/models/DeepSeek-R1-NVFP4-v2"
```

Other DeepSeek-R1 NVFP4 builds, including `nvidia/DeepSeek-R1-NVFP4`, are
different checkpoints.

Points that set `speculative_config` in `server-gen.yaml` decode with the
checkpoint's native MTP head and three draft tokens; C28080 sets none and runs without
speculative decoding. Checker `<ip>`'s approved list includes this head
(`centml/DeepSeek-R1-NVFP4-v2-mlpinf` at `93947a0d`).

The [dataset instructions](../../../dataset/README.md#deepseek-r1) download the
public prepared parquet and give its checksum. Place it at the default
`DATA_PATH`, `$SCRATCH_DIR/data/deepseek_r1_eval.parquet`. If you stage the
model or dataset elsewhere, pass `--set MODEL_PATH=<dir>` or
`--set DATA_PATH=<file>`. Accuracy may also need evaluator dependencies and
cached Hugging Face data; follow the pinned
[Endpoints example](https://github.com/mlcommons/endpoints/tree/e9f427ce74fd90e7fafdbc77ee7746cc00245692/examples/07_DeepSeekR1_Example).

Create a writable `hf_cache` directory beneath `SCRATCH_DIR`. The site template
mounts it at `/hf_cache`, and a job fails to start without it. The server
workflow separately creates a rank-local `HF_MODULES_CACHE` to avoid shared locks.
Keep model/data paths beneath `SCRATCH_DIR`, or add their mounts explicitly.

## Site network overlay (RoCE)

The prefill and decode servers transfer the KV cache over UCX. Each point's
`server-env.yaml` already sets `UCX_TLS: '^posix'`. NVIDIA measured these
configurations only on RoCE, with overlays like the ones below; on InfiniBand, start
from the shipped settings and confirm the KV cache does not fall back to TCP. On a
RoCE fabric, give the servers a site overlay: a flat YAML of
environment variables merged over the point's `server-env.yaml` for every prefill
and decode server, later keys winning. Replace the device names and GID index below
with your NICs':

```yaml
UCX_TLS: '^posix,tcp'                        # keep KV transfer off TCP; skip POSIX shared memory across Pyxis steps
UCX_IB_GID_INDEX: '3'                        # your NICs' RoCE v2 GID index
UCX_NET_DEVICES: 'mlx5_${SLURM_LOCALID}:1'   # one RoCE device per local rank
```

Points whose decode server spans 64 GPUs (`GPUS_PER_GEN_RANK: 64` in
`dsr1_config.yaml`) start a 64-rank MPI job across sixteen nodes. On NVIDIA's RoCE
system that start-up failed with UCX's default DC transport, so those runs used
the RC transport instead:

```yaml
UCX_TLS: 'rc,cuda,self,sysv'                 # RC instead of DC for the 64-rank decode server
UCX_IB_GID_INDEX: '3'                        # your NICs' RoCE v2 GID index
UCX_NET_DEVICES: 'mlx5_${SLURM_LOCALID}:1'   # one RoCE device per local rank
```

The servers read the overlay inside their containers, where only this checkout
and `SCRATCH_DIR` are mounted, so save it under one of them and add
`--set SITE_ENV_YAML=<path>` to the run. `${SLURM_LOCALID}` expands per rank, and
the run output keeps a copy as `site-env.yaml`.

## Run a point

From the exported repository root, customize the paths/account/partition below:

```bash
point_dir="$PWD/NVIDIA/src/deepseek-r1/configs/GB300-NVL72_GB300-288GB_aarch64x72_TRTLLM/point_32"
scratch_dir=/shared/benchmark-assets
run_dir=/shared/runs/dsr1-c32
mkdir -p "$scratch_dir/hf_cache" "$run_dir/workspace" "$run_dir/output"
sflow run \
  -f "$point_dir/dsr1_config.yaml" \
  -f NVIDIA/src/deepseek-r1/configs/_shared/slurm_env.yaml \
  -f NVIDIA/src/deepseek-r1/configs/_shared/templates/trtllm_disagg_endpoints_dsr1_perf_acc.yaml \
  --set WORK_DIR="$PWD" --set SCRATCH_DIR="$scratch_dir" \
  --set SLURM_ACCOUNT=your_account --set SLURM_PARTITION=your_partition \
  --set CONTAINER_IMAGE=/shared/images/trtllm.sqsh \
  --set ENDPOINT_CONTAINER_IMAGE=/shared/images/endpoints.sqsh \
  --set LCB_CONTAINER_IMAGE=/shared/images/lcb.sqsh \
  --set MODE=perf --workspace-dir "$run_dir/workspace" --output-dir "$run_dir/output" \
  --dry-run --skip-artifact-check
```

Remove the final dry-run options to execute after verifying the actual assets.
Use a fresh run directory each time. Set `MODE=both` for performance followed by
accuracy. With client `e9f427c`, `MODE=perf` still runs the accuracy dataset after
performance, and `MODE=acc` runs it alone at concurrency 1 over one connection, which
does not finish within the five-hour job; use `both` for accuracy. Change the point directory to select another
approved concurrency; retain its full server/client settings.

For `sflow batch --submit`, use the same files/overrides and explicitly pass `-N 18`,
`-p your_partition`, `-A your_account`, and `--time 05:00:00`. The YAML allocation
variables alone do not set batch command-line allocation options.

The generic server environment is included. Supply site network settings with
the overlay above rather than editing `server-env.yaml`; the export does not
establish compatibility with every site's transport settings.

## Collect and check output

Keep the complete SFlow output, especially `endpoint_benchmark/`, its resolved
client YAML, native reports/event logs and evaluator output, together with all
server logs and `snapshot_configs/`. The client uses `${SFLOW_TASK_OUTPUT_DIR}`
as its report directory. Inspect completion and failure counts, and require
actual evaluator results for accuracy; successful serving or a dry-run is not
a completed benchmark. Store all measured output outside this source checkout.
