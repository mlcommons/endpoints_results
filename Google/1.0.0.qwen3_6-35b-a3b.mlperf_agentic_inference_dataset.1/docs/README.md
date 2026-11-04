# qwen3_6-35b-a3b v1.0.3 on GKE: implementation notes

## 1. Software stack and checkpoint

- Checkpoint: `Qwen/Qwen3.6-35B-A3B` revision `995ad96eacd98c81ed38be0c5b274b04031597b0`
- Serving framework: vLLM Qwen-latest 7cc4a74c3427 behind Dynamo 1.4.2 (4 x TP1 aggregated workers)
- Server container: `registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/centml-gb300-rubin@sha256:cb0b77df93c0f9d89b8bca7f59a2520e582b9dbebdd606f7d9f36f57632a8b71`
- Other software: MLCommons endpoints client ghcr.io/mlcommons/endpoints@sha256:1523051b141437b25083924d6428082f435713516eee71c54ae207b393900b2d (tag d3c57f2); Dynamo 1.4.2; containerd 2.2.7
- Speculative decoding: `qwen3_next_mtp` (5 speculative tokens on C32–C256, 3 on C512–C768; `Qwen/Qwen3.6-35B-A3B` @ `git-sha1:995ad96eacd98c81ed38be0c5b274b04031597b0`)
- Ruleset & Seed Set: `mlperf-endpoints-v1.0-2026-10-C1-A` (Seed Set A, `model_seed=9_315_206_023_656_308_754`).

## 2. GKE adaptations from the NVIDIA v1.0.3 Slurm drop

- Durable node-local `report_dir` (`${REPORT_DIR}`) and `metrics_tokenizer_workers: 32` in `client.yaml`.
- Single-node NCCL environment (`NCCL_NET=Socket NCCL_ENV_PLUGIN=none NCCL_NET_PLUGIN=none`) in `serve_endpoint` so `bash -l`'s `/etc/profile` GCP plugin settings do not override single-node `rdma: disable`.
- At `C768`, `num_trajectories_to_issue` in `client.yaml` (`point_768`) is set to `14712` (24 passes of 613 trajectories, compared to 15 passes / 9,195 trajectories in the upstream Slurm drop) so the high-concurrency measurement window runs for `4,784.10 s`.

`src/vllm-dynamo-gke/README.md` and these excerpts needed no placeholders.

## 3. Measured points

| C | Run Tag | Inline Accuracy (%) | Full-Run OSL (tokens) | System TPS | Interactivity (tok/s) | TTFT P90 (ms) | TPOT P90 (ms) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | `c32-20261014-171027-0624d0` | 56.58 | 375.5 | 5463.26 | 0.00 | 64.6 | 2.14 |
| 36 | `c36-20261014-171029-aa367b` | 56.64 | 375.0 | 6054.65 | 0.00 | 66.2 | 2.26 |
| 64 | `c64-20261014-171031-45d87c` | 56.51 | 372.3 | 9784.02 | 0.00 | 70.7 | 3.02 |
| 128 | `c128-20261014-171033-92303f` | 56.61 | 374.8 | 16084.47 | 0.00 | 83.8 | 4.82 |
| 144 | `c144-20261014-171036-e47991` | 56.68 | 374.4 | 17276.30 | 0.00 | 87.1 | 5.27 |
| 192 | `c192-20261014-171037-ec8d16` | 56.60 | 373.7 | 20478.90 | 0.00 | 96.7 | 6.46 |
| 256 | `c256-20261014-171039-92d779` | 56.57 | 374.1 | 23941.63 | 0.00 | 111.2 | 8.06 |
| 512 | `c512-20261014-171042-ead62e` | 56.61 | 374.9 | 31911.65 | 0.00 | 234.5 | 14.46 |
| 768 | `c768-20261016-101049-5357bb` | 56.58 | 374.1 | 35238.49 | 0.00 | 387.3 | 21.58 |

## 4. SWE-bench Verified accuracy (4 mandatory bands)

| Band | C | Resolved / 200 | Accuracy (%) | Results File |
|---|---:|---:|---:|---|
| `ultra_low_concurrency` | 32 | 142 / 200 | 71.00 | `docs/swe_bench/c32_swe_bench_results.json` |
| `low_concurrency` | 36 | 145 / 200 | 72.50 | `docs/swe_bench/c36_swe_bench_results.json` |
| `med_concurrency` | 64 | 138 / 200 | 69.00 | `docs/swe_bench/c64_swe_bench_results.json` |
| `high_concurrency` | 192 | 145 / 200 | 72.50 | `docs/swe_bench/c192_swe_bench_results.json` |

- **4-band arithmetic mean:** **71.25%**
