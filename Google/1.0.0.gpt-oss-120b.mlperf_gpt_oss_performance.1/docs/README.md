# gpt-oss-120b v1.0.3 on GKE: implementation notes

## 1. Software stack and checkpoint

- Reference drop: `nv-mlpinf-partner-partner-endpoints-v1.0.3` (`NVIDIA/src/gpt-oss-120b`)
- Checkpoint: `openai/gpt-oss-120b` revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`
- Serving framework: TensorRT-LLM 1.3.0rc14 (disaggregated prefill/decode)
- Server container: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14`
- Other software: MLCommons endpoints client ghcr.io/mlcommons/endpoints:e9f427c; LiveCodeBench scorer ghcr.io/mlcommons/endpoints:9d749906-livecodebench; containerd 2.2.7
- Speculative decoding: None
- Model seed: Seed Set A (`9_315_206_023_656_308_754`) passed via `client.yaml` `model_params.seed`.

## 2. GKE deviations from the NVIDIA v1.0.3 Slurm drop

- Disjoint per-pod/per-replica `/proc/sys/kernel/ns_last_pid` offset in `gptoss-rank-env-{prefill,decode}.sh` before launching MPI worker ranks: prevents cross-pod `(system_uuid, pid)` peer-cache collisions in `ucp_cuda_ipc` MNNVL when multiple pods or replicas share a node or NVLink domain in isolated Kubernetes PID namespaces.
- `model_params.seed` set to Seed Set A (`9_315_206_023_656_308_754`) in `client.yaml`.
- `metrics_tokenizer_workers: 32`, durable node-local `report_dir`, and `run_timeout_s >= 14400` floor in `client.yaml` (in v1.0.3, upstream `client.yaml` already sets `run_timeout_s: 16200.0` on C17/C35/C70/C140).
- `UCX_CUDA_IPC_ENABLE_MNNVL=y`, `UCX_TLS=cuda_copy,cuda_ipc,sm,self,tcp`, and `NCCL_NET_PLUGIN=none` for GKE MNNVL fabric.
- Atomic startup patch extending `TRTLLM_HANG_DETECTOR_TIMEOUT` to 3600 s and frontend `--request_timeout 14400`.

In `src/trtllm-gke/README.md` and in these excerpts, site-specific names, paths and
addresses are replaced with these placeholders:

- `<bucket>`: a Cloud Storage bucket

## 3. Measured points

| C | Run ID | TPS (tok/s) | Weighted Accuracy (%) | AIME25 (%) | GPQA (%) | LiveCodeBench (%) |
|---:|---|---:|---:|---:|---:|---:|
| 17 | `trtllm_disagg_gptoss_gke-20261012-071338-208420` | 6700.9 | 83.27 | 80.00 | 74.75 | 84.96 |
| 35 | `trtllm_disagg_gptoss_gke-20261015-221313-79b5b5` | 12762.3 | 82.75 | 79.58 | 73.33 | 84.61 |
| 70 | `trtllm_disagg_gptoss_gke-20261013-001315-49d518` | 23201.7 | 82.80 | 79.17 | 73.84 | 84.58 |
| 140 | `trtllm_disagg_gptoss_gke-20261012-015231-ddbb68` | 41220.6 | 83.26 | 80.42 | 74.65 | 84.96 |
| 272 | `trtllm_disagg_gptoss_gke-20261011-212415-a6f094` | 71891.3 | 82.63 | 77.92 | 73.64 | 84.45 |
| 1024 | `trtllm_disagg_gptoss_gke-20261013-164526-91e72c` | 192483.8 | 82.89 | 80.83 | 75.05 | 84.42 |
| 3840 | `trtllm_disagg_gptoss_gke-20261012-230936-529758` | 445733.1 | 83.10 | 80.00 | 74.85 | 84.74 |
| 22528 | `trtllm_disagg_gptoss_gke-20261016-072142-12fa0e` | 948368.8 | 82.65 | 78.33 | 73.33 | 84.52 |
