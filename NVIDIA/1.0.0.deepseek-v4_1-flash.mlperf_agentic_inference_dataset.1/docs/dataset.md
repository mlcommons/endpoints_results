# Benchmark input data

Datasets are external dependencies and are not bundled in this repository. Each
model's runbook specifies its download source, revision or checksum, and the
site variables needed to locate the data on shared storage. Add a model-specific
section here when introducing another model.

## DeepSeek-V4.1-Flash

Download the unchanged 613-trajectory Agentic Inference JSONL from
[MLCommons storage](https://endpoints.mlcommons-storage.org/index.html#mlperf-agentic-inference),
following the [Agentic Inference README](https://github.com/mlcommons/endpoints/tree/e9f427ce74fd90e7fafdbc77ee7746cc00245692/examples/10_Agentic_Inference#dataset).
Its SHA-256 must be `1beb24c882122df96571cf11b390acbea388944038bc55c78b891475459014ae`.
Save it as `${SCRATCH_DIR}/data/agentic_dataset/perf/mlperf_agentic_inference_dataset.jsonl`:

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d "$SCRATCH_DIR/data/agentic_dataset/perf" \
  https://endpoints.mlcommons-storage.org/metadata/mlperf_agentic_inference_dataset.uri
sha256sum "$SCRATCH_DIR/data/agentic_dataset/perf/mlperf_agentic_inference_dataset.jsonl"
```

The client and SWE-bench service run with Hugging Face offline, so seed the
`princeton-nlp/SWE-bench_Verified` cache once from an ARM64 compute node after
installing the SWE-bench service (see the runbook). `SERVICE_VENV` is the service
venv the setup job created:
`NVIDIA/src/deepseek-v4.1-flash/swebench_service/.venv` for GB300.

```bash
DATA_PATH="$SCRATCH_DIR/data/agentic_dataset"
mkdir -p "$DATA_PATH/dataset_cache"
HF_HOME="$DATA_PATH/dataset_cache" HF_HUB_OFFLINE=0 \
  "$SERVICE_VENV/bin/python" -c \
  'from datasets import load_dataset; load_dataset("princeton-nlp/SWE-bench_Verified", split="test")'
```

The client also loads SWE-bench from its own copy in the cache, which each point
mounts read-only. Generate that copy once with the pinned client image, from an
ARM64 compute node, after seeding the cache above:

```bash
srun --account=<account> --partition=<ARM64 partition> --nodes=1 \
  --container-image=docker://ghcr.io/mlcommons/endpoints:4235a9c \
  --container-mounts=$SCRATCH_DIR:$SCRATCH_DIR,$DATA_PATH/dataset_cache:/mnt/inference-endpoint/dataset_cache \
  --container-workdir=/mnt/inference-endpoint \
  --export=ALL,HF_HOME=$DATA_PATH/dataset_cache,HF_HUB_OFFLINE=1 \
  /opt/venv/bin/python3 -c 'from pathlib import Path; from inference_endpoint.dataset_manager.predefined.swe_bench import SWEBench; SWEBench.generate(datasets_dir=Path("dataset_cache"), subset="verified", split="test")'
ls "$DATA_PATH/dataset_cache/swe_bench/verified/test/swe_bench_verified_test.parquet"
```

Without this file every point stops during client setup with
`Read-only file system: 'dataset_cache/swe_bench'`.

See the [DeepSeek-V4.1-Flash runbook](../src/deepseek-v4.1-flash/README.md) for
the full layout.

