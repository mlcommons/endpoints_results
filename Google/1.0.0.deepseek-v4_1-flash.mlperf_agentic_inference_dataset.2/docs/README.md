# deepseek-v4_1-flash v1.0.3 on GKE: implementation notes

## 1. Software stack and checkpoint

- Checkpoint: `deepseek-ai/DeepSeek-V4.1-Flash` revision `dba1be0a40aa45a94ad051997016db3960a90277`
- Serving framework: vLLM 0.20.2rc1.dev6106+g7cf5542c2.d20261003 behind Dynamo (DEP4: 4 x TP1, DP4, EP4)
- Server container: `registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/centml-rubin-vllm@sha256:141b04dc8fb0461f9875bd116cf831aa5876b655ff210465e4bca8e21f004dcb`
- Other software: MLCommons endpoints client ghcr.io/mlcommons/endpoints@sha256:d863e3b97a83dc73b18bd7d4824f2d40b9979a8b482b9a9defd80e5bfe96fd68 (tag 4235a9c); containerd 2.2.7
- Speculative decoding: `dspark` (5 speculative tokens, `deepseek-ai/DeepSeek-V4.1-Flash` @ `git-sha1:dba1be0a40aa45a94ad051997016db3960a90277`)
- Ruleset & Seed Set: `mlperf-endpoints-v1.0-2026-10-C1-A` (Seed Set A, `model_seed=9_315_206_023_656_308_754`).

## 2. GKE adaptations from the NVIDIA v1.0.3 Slurm drop

- Co-located 4-GPU `merged_vllm_worker` pod running the 4 `TP1` ranks with `DP4`/`EP4` across all 4 GPUs on the node.
- Durable node-local `report_dir` and shared `/dev/shm/dynamo-discovery` SSD hostPath between `dynamo_frontend` and `merged_vllm_worker`.
- `RUN_TAG = cksum % 2700` to keep `DYN_SYSTEM_PORT` within signed 16-bit integer range (`<= 32767`).

`src/vllm-dynamo-gke/README.md` and these excerpts needed no placeholders.

## 3. Measured points

| C | Run ID | Inline Accuracy (%) | Full-Run OSL (tokens) | System TPS | Interactivity (tok/s) | TTFT P90 (ms) | TPOT P90 (ms) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | `dsv41_flash_vllm_endpoint_gke-20261014-171009-a48436` | 53.41 | 907.9 | 2928.80 | 0.00 | 119.2 | 4.27 |
| 24 | `dsv41_flash_vllm_endpoint_gke-20261014-171011-820695` | 53.05 | 894.8 | 4002.34 | 0.00 | 124.1 | 4.84 |
| 32 | `dsv41_flash_vllm_endpoint_gke-20261014-171013-a74e8a` | 53.28 | 910.0 | 4964.06 | 0.00 | 127.1 | 5.33 |
| 64 | `dsv41_flash_vllm_endpoint_gke-20261014-171015-2d7ee6` | 53.24 | 917.6 | 8544.92 | 0.00 | 143.0 | 6.62 |
| 128 | `dsv41_flash_vllm_endpoint_gke-20261014-171017-0cdacc` | 53.07 | 913.7 | 13677.67 | 0.00 | 165.1 | 8.64 |
| 256 | `dsv41_flash_vllm_endpoint_gke-20261014-171019-4d3faa` | 53.04 | 911.9 | 21067.71 | 0.00 | 212.4 | 11.94 |
| 512 | `dsv41_flash_vllm_endpoint_gke-20261014-171021-8aebd6` | 52.97 | 903.4 | 32947.79 | 0.00 | 339.2 | 15.76 |
| 1024 | `dsv41_flash_vllm_endpoint_gke-20261016-132100-f42406` | 53.13 | 908.4 | 49342.94 | 0.00 | 6152.7 | 18.72 |
| 2048 | `dsv41_flash_vllm_endpoint_gke-20261014-171025-4748eb` | 53.11 | 908.9 | 54206.00 | 0.00 | 20722.4 | 20.34 |

## 4. SWE-bench Verified accuracy (4 mandatory bands)

| Band | C | Resolved / 200 | Accuracy (%) | Results File |
|---|---:|---:|---:|---|
| `ultra_low_concurrency` | 16 | 195 / 200 | 97.50 | `docs/swe_bench/c16_swe_bench_results.json` |
| `low_concurrency` | 24 | 195 / 200 | 97.50 | `docs/swe_bench/c24_swe_bench_results.json` |
| `med_concurrency` | 128 | 197 / 200 | 98.50 | `docs/swe_bench/c128_swe_bench_results.json` |
| `high_concurrency` | 256 | 197 / 200 | 98.50 | `docs/swe_bench/c256_swe_bench_results.json` |

- **4-band arithmetic mean:** **98.00%**
