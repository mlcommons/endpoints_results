# Benchmark input data

Datasets are external dependencies and are not bundled in this repository. Each
model's runbook specifies its download source, revision or checksum, and the
site variables needed to locate the data on shared storage. Add a model-specific
section here when introducing another model.

## Kimi K3

Download the unchanged `mlperf_agentic_inference_dataset.jsonl` (613 trajectories) from
[MLCommons storage](https://endpoints.mlcommons-storage.org/index.html#mlperf-agentic-inference)
and set `DATASET_PATH` to its shared-storage path:

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d /shared/data \
  https://endpoints.mlcommons-storage.org/metadata/mlperf_agentic_inference_dataset.uri
sha256sum /shared/data/mlperf_agentic_inference_dataset.jsonl
```
Its SHA-256 must be
`1beb24c882122df96571cf11b390acbea388944038bc55c78b891475459014ae`. The point configurations set
how many trajectories to issue, in whole passes of the dataset. See the
[Kimi K3 runbook](../src/kimi-k3/README.md#model-and-data) for the model
and draft checkpoints.

