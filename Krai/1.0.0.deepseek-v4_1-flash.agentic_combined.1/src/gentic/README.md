# gentic: DeepSeek-V4.1-Flash agentic SUT on single-node HPE Cray XD670 (8x H200 SXM), vLLM

Benchmark harness used for KRAI's MLPerf Endpoints v1.0 DeepSeek-V4.1-Flash agentic submission (Standardized
division, Client on Prem). For each point it starts the vLLM server, an nginx router, the SWE-bench scoring service
and the unmodified MLPerf Endpoints reference client on one node, then collects the results. vLLM and the
reference client are built unmodified from public commits by the Dockerfiles in `src/docker/`.

## Setup

Requirements:
- one node with 8x NVIDIA H200 141 GB (SXM);
- Docker (usable without `sudo`), the NVIDIA Container Toolkit and `uv`;
- about 1 TB of storage;
- ports 8000 (router), 9000–9003 (vLLM) and 18080 (SWE-bench service) free. The node is used exclusively.

```bash
# 1. Install the harness at $HOME/gentic (the scripts resolve paths from this location)
cp -r <submission>/src/gentic $HOME/gentic && cd $HOME/gentic && uv sync

# 2. Data directory for weights, dataset, image caches and results
export GENTIC_DATA_DIR=/path/to/large/storage
mkdir -p $GENTIC_DATA_DIR/hf_cache $GENTIC_DATA_DIR/dataset_cache

# 3. Model weights at the approved revision (stored under $GENTIC_DATA_DIR/hf_cache/hub/)
HF_HOME=$GENTIC_DATA_DIR/hf_cache uvx --from huggingface_hub hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277

# 4. Dataset, via the MLCommons downloader (creates mlperf_agentic_inference_dataset/ in the current directory)
cd $GENTIC_DATA_DIR/dataset_cache
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://endpoints.mlcommons-storage.org/metadata/mlperf_agentic_inference_dataset.uri
sha256sum mlperf_agentic_inference_dataset/mlperf_agentic_inference_dataset.jsonl
# expected: 1beb24c882122df96571cf11b390acbea388944038bc55c78b891475459014ae
cd $HOME/gentic

# 5. Container images (optional: if skipped, the first run pulls the published tags)
docker login   # SWE-bench pulls about 200 task images from Docker Hub
docker build -f src/docker/vllm_server.dockerfile        -t wongcannan/vllm-v0.30.0-nsys-withsrc:v1.0 src/docker
docker build -f src/docker/inference_endpoint.dockerfile -t wongcannan/mlperf-endpoints:v1.1 src/docker
```

Resulting layout of `$GENTIC_DATA_DIR`:

```text
$GENTIC_DATA_DIR/
├── hf_cache/hub/            model weights (step 3); SWE-bench Verified is added by the client
├── dataset_cache/           dataset (step 4)
├── docker_cache/            image tarballs and SWE-bench task images   (created by the first run)
├── root_cache/              vLLM compile caches                        (created by the first run)
└── results/mlperf/          one directory per run, 6.5–52 GB each      (created by the first run)
```

Notes:
- A local image with a matching tag takes precedence over the cache and the registry, so images built in step 5
  are the ones used.
- `$HOME/gentic` must hold this directory: if another checkout is there, the run uses its tokenizer, router
  configuration and pull script.
- The client container requests 128 GB of shared memory, and vLLM keeps the Engram tables in host memory. The
  submitted runs used nodes with 2 TiB of host memory.
- `HF_TOKEN` is optional. It is passed to the client for the SWE-bench Verified download.

## Reproduce a point

Each point's `results/<system>/deepseek-v4_1-flash/r<N>/server_configs/` contains the two configurations it was
measured with:

```bash
cd $HOME/gentic
P=<submission>/results/<system>/deepseek-v4_1-flash/r<N>/server_configs
bash src/e2e_benchmarking/entry_local.sh --config "$P/server_config_resolved.yaml" "$P/bench_config_resolved.yaml"
```

- `prepare_runs.py` derives the run name, report directory, endpoint and concurrency from these files. It
  regenerates the server and client commands of the submitted points exactly, apart from host paths.
- Results are written to `$GENTIC_DATA_DIR/results/mlperf/repro_v1.0.0__.../`: `report.txt`,
  `performance/result_summary.json`, `accuracy/accuracy_results.json`, `events.jsonl` and the provenance files.
  The job log is under `$HOME/gentic/engines/`.
- A point takes 1.5–5.5 h, increasing with concurrency and trajectory count. Mandatory points add about 1 h of
  SWE-bench.
- The submitted runs were dispatched with `sbatch --exclusive --gpus-per-node=8` around the same
  `execute_runs.sh`. Slurm is not required.

## Contents

```text
gentic/
├── pyproject.toml, uv.lock, .python-version  harness Python environment (uv)
├── LICENSE                                    MIT
└── src/
    ├── docker/
    │   ├── vllm_server.dockerfile             vllm/vllm-openai:v0.30.0 (ced6857a) + Nsight Systems CLI (unused)
    │   └── inference_endpoint.dockerfile      MLPerf Endpoints client d3c57f2, unmodified, + SWE-bench service
    ├── docker_pull_w_cache.sh                 local image, else cached tarball, else pull and cache
    └── e2e_benchmarking/
        ├── entry_local.sh                     entry point: --config <server yaml> <bench yaml>
        ├── prepare_runs.py                    configurations -> run manifest + resolved YAMLs
        ├── scripts/
        │   ├── execute_runs.sh                runs each manifest line
        │   ├── param_search_multireplica_mlperf.sh
        │   │                                  one point: vLLM, nginx, SWE-bench service, client, evidence files
        │   └── build_swebench_image_manifest.py
        │                                      SWE-bench task images to pre-pull
        ├── configs/
        │   ├── nginx_dp_router.conf           router template: pins each X-Session-ID to one DP rank
        │   ├── server_configs/mlperf_dsv41f_long_endpoints_nvidia_eval.yaml
        │   │                                  server template (working copy)
        │   └── bench_configs/mlperf_dsv41f_long_endpoints.yaml
        │                                      client template (working copy)
        └── tokenizers/deepseek-v4.1-flash/    client tokenizer and chat template from the model repository at
                                               2cba9e42 (MIT); SHA256SUMS is verified on every run
```

The authoritative configuration of each point is in that point's `server_configs/`, not in the templates.

## License

MIT License, Copyright (c) 2026 Krai Ltd; see `LICENSE`. This directory is submitted under the MLCommons CLA.
The tokenizer files in `src/e2e_benchmarking/tokenizers/deepseek-v4.1-flash/` are DeepSeek's and carry their own
MIT license.
