# Benchmark input data

Datasets are external dependencies and are not bundled in this repository. Each
model's runbook specifies its download source, revision or checksum, and the
site variables needed to locate the data on shared storage. Add a model-specific
section here when introducing another model.

## Qwen3.6-35B-A3B

Download the unchanged 613-trajectory `mlperf_agentic_inference_dataset.jsonl` from
[MLCommons storage](https://endpoints.mlcommons-storage.org/index.html#mlperf-agentic-inference),
following the [Agentic Inference README](https://github.com/mlcommons/endpoints/tree/d3c57f2eba127c819e5a8fd83101d9e5f4e15fce/examples/10_Agentic_Inference#dataset).
Its SHA-256 must equal `DATASET_SHA256` in the model's `configs/dependencies.json`
(`1beb24c882122df96571cf11b390acbea388944038bc55c78b891475459014ae`). Set
`DATASET_PATH` to its shared-storage path; the point configurations default to
`$SHARED_ROOT/datasets/mlperf_agentic_inference_dataset.jsonl`:

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d "$SHARED_ROOT/datasets" \
  https://endpoints.mlcommons-storage.org/metadata/mlperf_agentic_inference_dataset.uri
sha256sum "$SHARED_ROOT/datasets/mlperf_agentic_inference_dataset.jsonl"
```

For SWE-bench modes, also supply the service and images described in the
[Qwen3.6-35B-A3B runbook](../src/qwen3.6-35b-a3b/README.md).

