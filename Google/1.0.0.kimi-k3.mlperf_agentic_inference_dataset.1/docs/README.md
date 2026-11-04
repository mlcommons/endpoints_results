> Site-specific names, paths and addresses are replaced with `<placeholders>`.

# kimi-k3 v1.0.3 on GKE: implementation notes

## 1. Software stack and checkpoint

- Checkpoint: `nvidia/Kimi-K3-NVFP4` revision `b2428a0b83a8b712ff2e1a8448a103d4175341f1`
- Draft model: `RadixArk/Kimi-K3-DSpark` revision `3c5bac301d9cf392706189d82ed947feca6c2f0f` (DSpark speculative decoding)
- Serving framework: TensorRT-LLM 1.3.0rc25.post1 + Dynamo 1.6.0.dev20260918
- Server container: `kimi-k3-dynamo@sha256:985260a01a2dfb66bdd0c68f0bd482e78f69c1b101120c7341782d101e2602a4 (built from public sources via `src/trtllm-dynamo-gke/container/build.sh`)`
- Other software: MLCommons endpoints client ghcr.io/mlcommons/endpoints@sha256:ba85b060b9c1de0d3b44d52f6fce8369e81f88ac31c460807cdc1e61441249bf (tag 89904fb); containerd 2.2.7
- Ruleset & Seed Set: `mlperf-endpoints-v1.0-2026-10-C1-A` (Seed Set A, `model_seed=9_315_206_023_656_308_754`).

## 2. GKE adaptations from the NVIDIA v1.0.3 Slurm drop

- `k8s_mpi` per prefill/decode server replica with `ComputeDomain` IMEX channel allocation.
- Dedicated `control_plane` pod hosting NATS (`:4222`) and etcd (`:2379`) on node 0 for Dynamo service discovery.
- `flock`-serialized per-node `G_FABRIC_PROBE` on co-located 1-GPU prefill workers (`C192..C1024`).

In `src/trtllm-dynamo-gke/README.md` and in these excerpts, site-specific names, paths and
addresses are replaced with these placeholders:

- `<bucket>`: a Cloud Storage bucket
- `<node-local-ssd>`: a directory on the node's local SSD
- `<workdir>`: a directory on the operator's workstation
- `<ip>`: an IPv4 address

## 3. Measured points

| C | Run ID | Inline Accuracy (%) | Full-Run OSL (tokens) | System TPS | Interactivity (tok/s) | Draft Len | TTFT P90 (ms) | TPOT P90 (ms) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 13 | `kimi_k3_trtllm_v1_0_3_gke_c13-20261015-042358-df77de` | 58.60 | 474.3 | 912.21 | 92.60 | 5 | 1287.6 | 12.23 |
| 16 | `kimi_k3_trtllm_v1_0_3_gke_c16-20261014-214006-316cfb` | 59.41 | 467.0 | 1033.14 | 83.55 | 3 | 1350.4 | 14.00 |
| 24 | `kimi_k3_trtllm_v1_0_3_gke_c24-20261014-193820-dff3e2` | 59.20 | 461.5 | 1366.92 | 72.56 | 3 | 1600.1 | 15.25 |
| 32 | `kimi_k3_trtllm_v1_0_3_gke_c32-20261014-193826-74ffff` | 58.77 | 468.0 | 1650.87 | 64.98 | 3 | 1808.3 | 17.12 |
| 48 | `kimi_k3_trtllm_v1_0_3_gke_c48-20261014-221426-9ca45c` | 59.01 | 468.4 | 2161.43 | 54.42 | 3 | 2362.8 | 20.12 |
| 64 | `kimi_k3_trtllm_v1_0_3_gke_c64-20261015-101329-a18abe` | 58.90 | 468.6 | 2605.04 | 47.95 | 3 | 3202.6 | 22.46 |
| 80 | `kimi_k3_trtllm_v1_0_3_gke_c80-20261015-010605-5d4c22` | 58.62 | 469.6 | 2976.12 | 43.60 | 3 | 4496.4 | 22.76 |
| 448 | `kimi_k3_trtllm_v1_0_3_gke_c448-20261015-035715-ad4eba` | 58.74 | 459.9 | 9351.84 | 24.17 | 3 | 5943.8 | 40.82 |
| 640 | `kimi_k3_trtllm_v1_0_3_gke_c640-20261016-103805-ad5c07` | 58.76 | 460.4 | 11807.07 | 20.72 | 3 | 7473.2 | 48.55 |
| 1024 | `kimi_k3_trtllm_v1_0_3_gke_c1024-20261015-003709-613b63` | 58.77 | 461.2 | 13560.52 | 16.29 | 3 | 15995.5 | 53.15 |

## 4. SWE-bench Verified accuracy (4 mandatory bands)

| Band | C | Resolved / 200 | Accuracy (%) | Results File |
|---|---:|---:|---:|---|
| `ultra_low_concurrency` | 13 | 187 / 200 | 93.50 | `docs/swe_bench/c13_swe_bench_results.json` |
| `low_concurrency` | 16 | 190 / 200 | 95.00 | `docs/swe_bench/c16_swe_bench_results.json` |
| `med_concurrency` | 48 | 189 / 200 | 94.50 | `docs/swe_bench/c48_swe_bench_results.json` |
| `high_concurrency` | 448 | 189 / 200 | 94.50 | `docs/swe_bench/c448_swe_bench_results.json` |

- **4-band arithmetic mean:** **94.38%**

## Public Server Container Build (`src/trtllm-dynamo-gke/container/build.sh`)

The `kimi-k3-dynamo` server image (`TensorRT-LLM v1.3.0rc25.post1` + `Dynamo v1.6.0.dev20260918`, digest `sha256:985260a01a2dfb66bdd0c68f0bd482e78f69c1b101120c7341782d101e2602a4`) is built entirely from public sources using `src/trtllm-dynamo-gke/container/build.sh`:
- **TensorRT-LLM:** public branch [`feat/k3_26.08`](https://github.com/NVIDIA/TensorRT-LLM/tree/feat/k3_26.08) at commit [`45c5ae4150e51df678b2435457ff39679e8fd8d6`](https://github.com/NVIDIA/TensorRT-LLM/commit/45c5ae4150e51df678b2435457ff39679e8fd8d6)
- **Dynamo wheels:** `ai-dynamo==1.6.0.dev20260918` and `ai-dynamo-runtime==1.6.0.dev20260918` from PyPI
- **Dynamo frontend:** built from [`ai-dynamo/dynamo` commit `a9792dbe2402d379232f0aa88eb647cddb78dcab`](https://github.com/ai-dynamo/dynamo/tree/a9792dbe2402d379232f0aa88eb647cddb78dcab) and `dynamo-renderer` crate `v5.2.0` from crates.io, patched with `src/trtllm-dynamo-gke/dynamo/kimi-k3.patch` (`kimi-k3-build.patch`).
