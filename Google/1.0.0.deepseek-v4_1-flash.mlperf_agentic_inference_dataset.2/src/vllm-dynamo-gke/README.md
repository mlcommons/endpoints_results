> Site-specific names, paths and addresses are replaced with `<placeholders>`.

# DeepSeek-V4.1-Flash on GKE (MLPerf Endpoints v1.0.3)

This directory is the GKE sflow port of NVIDIA's v1.0.3 Slurm recipe for
DeepSeek-V4.1-Flash (`GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo`), following
[the guide](../../SLURM_TO_GKE.md). Every recipe passes `--check` and
`sflow run --dry-run`.

Source drop: `code-drop-nvidia/nv-mlpinf-partner-partner-endpoints-v1.0.3/NVIDIA/src/deepseek-v4.1-flash/`
and `configs/GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo/`.

| File | What it is |
|---|---|
| [generate_point_recipes.py](generate_point_recipes.py) | Emits all 9 points from the v1.0.3 drop and [recipe_base.yaml](recipe_base.yaml); `--check` verifies them, `--table` prints the per-point table |
| [recipe_base.yaml](recipe_base.yaml) | GKE layer (backend, operators, fabric preflight, benchmark pattern). Hand-maintained single source |
| `1nodes/point_<C>/` | Generated recipe `deepseek_v41_flash_gke_vllm_point<C>.yaml` + `launch_test.sh` (`REQUIRED_NODES=1`) |
| [stage-assets.yaml](stage-assets.yaml) | DaemonSet: model (`hf://` pinned), perf dataset (`sha256`), SWE-bench cache + parquet, on **every** node |
| [mk-report-dir.yaml](mk-report-dir.yaml) | DaemonSet: `dsv41-flash-reports/` (durable reports) + `dsv41-flash-runtime/` (vLLM caches + `host_cpus.txt`) |
| [prepull-images.yaml](prepull-images.yaml) | DaemonSet: pre-pull server (`nv-gitlab-pull`) + client images |

## 1. Topology: NVIDIA's native 1-node DEP4 layout (9 points)

`slurm_env.yaml`: `SLURM_NODES: 1`, `GPUS_PER_NODE: 4`. Every point is one GB300 node
(4 GPUs) running one Dynamo frontend, 4 × `dynamo.vllm` workers in `DEP4`
(`--tensor-parallel-size 1 --data-parallel-size 4 --data-parallel-rank $IDX --enable-expert-parallel`),
and the MLCommons client.

On GKE (`k8s_cluster`), `vllm_worker` declares `replicas.count: 4` and `resources.gpus.count: 1`;
`sflow`'s default `merge_colocated_gpu_pods` merges all 4 colocated replicas into a single 4-GPU pod
where `server-env.yaml` sets `CUDA_VISIBLE_DEVICES: "0,1,2,3"` and
`VLLM_DP_RANK_LOCAL_OVERRIDE: "${SFLOW_REPLICA_INDEX}"`.

| C | workers × TP (DP/EP) | EP | GPUs | nodes | drop MODE | DATASET_REPEATS | trajectories | max-num-seqs/worker | max-num-batched-tokens / max-cudagraph-capture-size |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 4 × TP1 (DP4) | on | 4 | 1 | both | 1 | 613 | 512 | 10240 / 10230 |
| 24 | 4 × TP1 (DP4) | on | 4 | 1 | both | 1 | 613 | 512 | 10240 / 10230 |
| 32 | 4 × TP1 (DP4) | on | 4 | 1 | perf | 1 | 613 | 512 | 10240 / 10230 |
| 64 | 4 × TP1 (DP4) | on | 4 | 1 | perf | 2 | 1226 | 512 | 10240 / 10230 |
| 128 | 4 × TP1 (DP4) | on | 4 | 1 | both | 2 | 1226 | 512 | 10240 / 10230 |
| 256 | 4 × TP1 (DP4) | on | 4 | 1 | perf | 4 | 2452 | 512 | 10240 / 10230 |
| 512 | 4 × TP1 (DP4) | on | 4 | 1 | perf | 8 | 4904 | 512 | 3072 / 3072 |
| 1024 | 4 × TP1 (DP4) | on | 4 | 1 | perf | 8 | 4904 | 512 | 3072 / 3072 |
| 2048 | 4 × TP1 (DP4) | on | 4 | 1 | both | 12 | 7356 | 512 | 3600 / 3600 |

Settings shared by every point, all verbatim from the drop:
- Speculative decoding `mtp`, 5 tokens (`num_speculative_tokens: 5`, `parallel_drafting: true`, `speculative_token_tree`).
- `--attention-backend FLASHINFER_MLA_SPARSE_DSV41`, `--kv-cache-dtype fp8_ds_mla`, `--engram-Config '{"cpu_offload": true}'`.
- `--language-model-only`, `--tokenizer-mode deepseek_v41`, and the `deepseek_v41` tool/reasoning parsers.
- NUMA and CPU pinning via `tools/cpuset.py` (12 physical cores + SMT per worker on its GPU's NUMA node via `numactl --physcpubind --membind`, remaining cores for `dynamo_frontend` and `endpoint_client` via `taskset -c`).
- Client settings: `max_new_tokens 8192`, `temperature 1.0`, `top_p 0.95`,
  `thinking: true`, `reasoning_effort: max`, `metrics_drain_timeout_s 1800`, `metrics_tokenizer_workers 16`.

## 2. Provenance (record with every result)

| Item | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40aa45a94ad051997016db3960a90277`: 48 safetensors, 510,296,708,312 B; not gated |
| Server image (`VLLM_IMAGE`) | `registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/centml-rubin-vllm@sha256:141b04dc8fb0461f9875bd116cf831aa5876b655ff210465e4bca8e21f004dcb` (tag `dsv41-rubin-7cf5542c25-r1`; `vLLM 0.20.2rc1.dev6106+g7cf5542c2.d20261003 behind Dynamo`), pulled with secret `nv-gitlab-pull` |
| Client image (`CLIENT_IMAGE`) | `ghcr.io/mlcommons/endpoints:4235a9c`: multi-arch index `sha256:6cf516d0932d6790590308e7d04e5a866fffbab3a9001d65222b639ee8350240` (linux/arm64 `sha256:d863e3b97a83dc73b18bd7d4824f2d40b9979a8b482b9a9defd80e5bfe96fd68`) |
| Perf dataset | `perf/mlperf_agentic_inference_dataset.jsonl` (613 trajectories), sha256 `1beb24c882122df96571cf11b390acbea388944038bc55c78b891475459014ae` |
| SWE-bench Verified | HF `princeton-nlp/SWE-bench_Verified` @ `c104f840cc67f8b6eec6f759ebc8b2693d585d4a` (`dataset_cache/swe_bench_verified_test.parquet`) |
