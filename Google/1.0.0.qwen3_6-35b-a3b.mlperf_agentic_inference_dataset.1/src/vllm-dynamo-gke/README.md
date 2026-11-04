> Site-specific names, paths and addresses are replaced with `<placeholders>`.

# Qwen3.6-35B-A3B on GKE (MLPerf Endpoints v1.0.3)

This directory is the GKE sflow port of NVIDIA's v1.0.3 Slurm recipe for
Qwen3.6-35B-A3B (`GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo`), following
[the guide](../../SLURM_TO_GKE.md). Every recipe passes `--check` and
`sflow run --dry-run`.

Source drop: `code-drop-nvidia/nv-mlpinf-partner-partner-endpoints-v1.0.3/NVIDIA/src/qwen3.6-35b-a3b/`
and `configs/GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo/`.

| File | What it is |
|---|---|
| [generate_point_recipes.py](generate_point_recipes.py) | Emits all 9 GB300 points (`32, 36, 64, 128, 144, 192, 256, 512, 768`) directly from the v1.0.3 drop and [recipe_template.yaml](recipe_template.yaml); `--check` verifies all 522 assertions |
| [recipe_template.yaml](recipe_template.yaml) | GKE layer (backend, operators, fabric preflight, benchmark pattern). Hand-maintained single source |
| `1nodes/point_<C>/` | Generated recipe `qwen36_35b_a3b_gke_dynamo_point<C>.yaml` + `launch_test.sh` (REQUIRED_NODES=1) |
| [stage-assets.yaml](stage-assets.yaml) | DaemonSet: model (`hf://` pinned) and perf dataset (`sha256` pinned) on **every** node |
| [mk-report-dir.yaml](mk-report-dir.yaml) | DaemonSet: `qwen3.6-35b-a3b-reports/` (durable reports, server logs, and compiled kernel caches) |
| [prepull-images.yaml](prepull-images.yaml) | DaemonSet: pre-pull server + client images |

## 1. Topology: NVIDIA's native 1-node layout (9 official GB300 points covering all 4 concurrency regions)

Every point uses one GB300 node (4 GPUs) running 4 `dynamo.vllm` TP1 workers (`DP=4`)
and 1 `dynamo.frontend` (`PORT=30003`), plus the MLCommons client (`MODE=perf` by default).

With `c_min = 32` and `c_max = 768`, MLPerf Endpoints v1.0 `compute_regions(768, 32)` defines:
- Ultra Low (`low_latency`): `[1, 32]` -> `C32` (accuracy point)
- Low (`low_concurrency`): `[33, 41]` -> `C36` (accuracy point)
- Medium (`med_concurrency`): `[42, 114]` -> `C64` (accuracy point)
- High (`high_concurrency`): `[115, 768]` -> `C128, C144, C192 (accuracy point), C256, C512, C768`

| C | workers × TP | DP | GPUs | nodes | trajectories (passes) | max-num-batched-tokens | max-num-seqs/worker | router-mode | speculative-config |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 32 | 4 × TP1 | 4 | 4 | 1 | 2452 (4×) | 32768 | 256 | `round-robin` | `qwen3_next_mtp`, 3 tokens |
| 36 | 4 × TP1 | 4 | 4 | 1 | 2452 (4×) | 32768 | 256 | `round-robin` | `qwen3_next_mtp`, 3 tokens |
| 64 | 4 × TP1 | 4 | 4 | 1 | 3678 (6×) | 32768 | 256 | `round-robin` | `qwen3_next_mtp`, 3 tokens |
| 128 | 4 × TP1 | 4 | 4 | 1 | 6130 (10×) | 32768 | 256 | `round-robin` | `qwen3_next_mtp`, 3 tokens |
| 144 | 4 × TP1 | 4 | 4 | 1 | 6130 (10×) | 32768 | 256 | `round-robin` | `qwen3_next_mtp`, 3 tokens |
| 192 | 4 × TP1 | 4 | 4 | 1 | 6743 (11×) | 32768 | 256 | `round-robin` | `qwen3_next_mtp`, 3 tokens |
| 256 | 4 × TP1 | 4 | 4 | 1 | 6743 (11×) | 32768 | 256 | `round-robin` | `qwen3_next_mtp`, 3 tokens |
| 512 | 4 × TP1 | 4 | 4 | 1 | 8582 (14×) | 32768 | 256 | `round-robin` | `qwen3_next_mtp`, 3 tokens |
| 768 | 4 × TP1 | 4 | 4 | 1 | 14712 (24×) | 32768 | 256 | `least-loaded` | `qwen3_next_mtp`, 3 tokens |

Settings shared by every point, all verbatim from the drop:
- Model precision: `mxfp8` weights (with `bfloat16` embed/head/norms/gates/MTP projections) and `fp8` KV cache (`mamba-ssm-cache-dtype: float32`).
- Speculative decoding: `qwen3_next_mtp`, `num_speculative_tokens: 5` on `C32`–`C256` and `3` on `C512`–`C768`.
- `--language-model-only`, `--tool-call-parser qwen3_xml`, `--reasoning-parser qwen3`, `--prefix-match-unit 32`, `gdn_prefill_backend: flashinfer`.
- Client settings: `max_new_tokens 8192`, `temperature 1.0`, `top_p 0.95`, `top_k 20`, `presence_penalty 1.5`,
  `submission_ref.ruleset: mlperf-endpoints-v1.0-2026-10-C1-A`, `metrics_tokenizer_workers 32`.

## 2. Provenance (record with every result)

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd98c81ed38be0c5b274b04031597b0`: 26 safetensors, 71,926,865,825 B |
| Server image | `registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/centml-gb300-rubin@sha256:cb0b77df93c0f9d89b8bca7f59a2520e582b9dbebdd606f7d9f36f57632a8b71` (tag `gb300-vllm-7cc4a74c`, CentML vLLM `mlperf-end-multiturn-v1.0-qwen-latest` @ `7cc4a74c34270ca783e535f7ec07134f92159482`, `ai-dynamo 1.4.2`) |
| Client image | `ghcr.io/mlcommons/endpoints@sha256:1523051b141437b25083924d6428082f435713516eee71c54ae207b393900b2d` (tag `d3c57f2`, commit `d3c57f2eba127c819e5a8fd83101d9e5f4e15fce`) |
| Perf dataset | `mlperf_agentic_inference_dataset.jsonl` (613 trajectories, 4 to 24 passes = 2452 to 14712 trajectories), sha256 `1beb24c882122df96571cf11b390acbea388944038bc55c78b891475459014ae` |
| Ruleset / Seed Set A | `mlperf-endpoints-v1.0-2026-10-C1-A` (`scheduler_rng_seed: 10_487_924_139_932_647_040`, `sample_index_rng_seed: 586_478_644_936_801_402`, `model_seed: 9_315_206_023_656_308_754` declared in `point.yaml`, server `seed: 42`) |

## 3. GKE deviations from the NVIDIA v1.0.3 Slurm drop

- **`client.yaml` rewrites (2 total, verified by `--check`):**
  1. `report_dir: logs/qwen_agentic` -> `report_dir: ${REPORT_DIR}` (durable node-local SSD hostPath).
  2. `metrics_tokenizer_workers: 2` -> `metrics_tokenizer_workers: 32` (prevents post-run tokenization bottleneck on 144-vCPU Grace nodes).
- **`SERVER_ENV`, `SERVER_CONFIG`, and `_shared` helpers (`dynamo-server.py`, `dynamo_cpu_affinity.py`, `cuda-compat.sh`, `autotune/GB300-NVL72.json`):** Embedded 100% byte-identically as ConfigMap artifacts and staged to `QWEN_SHARED_DIR=/tmp/qwen-shared`, with `SHARED_ROOT` pointing at `REPORTS_ROOT` on the node SSD so compiled GDN/FlashInfer/EWS kernels (`${SHARED_ROOT}/caches/qwen-gb300/*`) persist across runs on a node.
- **Single-node NCCL environment (`NCCL_NET=Socket NCCL_ENV_PLUGIN=none NCCL_NET_PLUGIN=none`):** The container's `/etc/profile` exports `NCCL_ENV_PLUGIN=gcp` and `NCCL_NET_PLUGIN=gcp` when `sflow` starts the container with `bash -l`, overriding pod-level `rdma: disable` environment variables. Disabling the GCP plugin and selecting `Socket` allows each single-node `TP=1` worker's `init_worker_distributed_environment` to initialize cleanly without multi-node IB/RoCE plugins.
- **Default `MODE=perf`:** Runs performance + inline accuracy (`agentic_inference_inline`) without the Slurm/Pyxis SWE-bench service.
