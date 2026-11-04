# deepseek-r1 v1.0.3 on GKE: implementation notes

## 1. Software stack and checkpoint

- Reference drop: `nv-mlpinf-partner-partner-endpoints-v1.0.3` (`NVIDIA/src/deepseek-r1`)
- Checkpoint: `centml/DeepSeek-R1-NVFP4-v2-mlpinf` revision `93947a0d7bd04f73ff98636f6b18ff5839e7aaf9`
- Serving framework: TensorRT-LLM 1.3.0rc14 (disaggregated prefill/decode)
- Server container: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14`
- Other software: MLCommons endpoints client ghcr.io/mlcommons/endpoints:e9f427c; containerd 2.2.7
- Speculative decoding: MTP (`num_nextn_predict_layers: 3`, native head in `centml/DeepSeek-R1-NVFP4-v2-mlpinf@93947a0d7bd04f73ff98636f6b18ff5839e7aaf9`)
- Model seed: Seed Set A (`9_315_206_023_656_308_754`) passed via `client.yaml` `model_params.seed`.

## 2. GKE deviations from the NVIDIA v1.0.3 Slurm drop

- Disjoint per-pod `/proc/sys/kernel/ns_last_pid` offset in `dsr1-rank-env.sh` before launching MPI worker ranks: prevents cross-pod `(system_uuid, pid)` peer-cache collisions in `ucp_cuda_ipc` MNNVL when multiple pods share a node or NVLink domain in isolated Kubernetes PID namespaces.
- `model_params.seed` set to Seed Set A (`9_315_206_023_656_308_754`) in `client.yaml`.
- `metrics_tokenizer_workers: 32` and durable node-local `report_dir` in `client.yaml`.
- `UCX_CUDA_IPC_ENABLE_MNNVL=y` and `UCX_TLS=cuda_copy,cuda_ipc,sm,self,tcp` for GKE MNNVL fabric.
- Atomic startup patch extending `TRTLLM_HANG_DETECTOR_TIMEOUT` to 3600 s and frontend `--request_timeout 14400`.

In `src/trtllm-gke/README.md` and in these excerpts, site-specific names, paths and
addresses are replaced with these placeholders:

- `<node-local-ssd>`: a directory on the node's local SSD

## 3. Measured points

| C | Run ID | TPS (tok/s) | Accuracy (%) | Avg OSL |
|---:|---|---:|---:|---:|
| 32 | `dsr1_endpoints_v1_0_2_disagg_gke-20261011-100521-c9a809` | 6794.5 | 81.1531 | 3823.5 |
| 48 | `dsr1_endpoints_v1_0_2_disagg_gke-20261014-080518-f2c936` | 10334.8 | 81.2215 | 3865.1 |
| 64 | `dsr1_endpoints_v1_0_2_disagg_gke-20261016-003115-407d89` | 13605.5 | 80.8797 | 3813.6 |
| 128 | `dsr1_endpoints_v1_0_2_disagg_gke-20261016-050730-8540bd` | 27256.0 | 80.7657 | 3851.7 |
| 256 | `dsr1_endpoints_v1_0_2_disagg_gke-20261011-155619-495ee8` | 45933.1 | 81.1759 | 3825.4 |
| 512 | `dsr1_endpoints_v1_0_2_disagg_gke-20261011-165041-4db70e` | 82685.8 | 80.8341 | 3859.0 |
| 1024 | `dsr1_endpoints_v1_0_2_disagg_gke-20261011-072048-e2fb2b` | 131440.4 | 81.0848 | 3822.8 |
| 2048 | `dsr1_endpoints_v1_0_2_disagg_gke-20261013-050827-baded3` | 198873.0 | 80.5378 | 3783.8 |
| 4096 | `dsr1_endpoints_v1_0_2_disagg_gke-20261011-140244-7ed915` | 308595.0 | 80.9708 | 3757.6 |
| 8192 | `dsr1_endpoints_v1_0_2_disagg_gke-20261013-065251-e52753` | 395546.7 | 81.0392 | 3795.4 |
