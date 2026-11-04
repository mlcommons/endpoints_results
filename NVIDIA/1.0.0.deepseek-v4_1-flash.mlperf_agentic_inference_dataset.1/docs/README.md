# DeepSeek-V4.1-Flash on GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo

NVIDIA's submission for one line: DeepSeek-V4.1-Flash on `GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo`, at C16, C24, C32, C64, C128, C256, C512, C1024, C2048.

| Directory | What it holds |
| --- | --- |
| `src/deepseek-v4.1-flash/` | The recipes that reproduce these points, and [how to run them](../src/deepseek-v4.1-flash/README.md). |
| `src/sflow/` | Helpers the recipes call. |
| `results/` | One directory per point, and the system's power description. |

The [software disclosure](software_disclosure.md) lists the software that ran, where it comes from,
and how the client reached the server. The [dataset notes](dataset.md)
say where the benchmark data comes from.

## Running the recipes

The recipes are written for a git checkout of NVIDIA's repository: the launchers find the repository
root with `git rev-parse --show-toplevel`, and the workflows refer to `NVIDIA/src/deepseek-v4.1-flash/...` and
`NVIDIA/src/sflow/...`. To run them from this submission, place `src/` as `NVIDIA/src/` in a git checkout
(for example, `git init` an empty folder, copy `src/` to `NVIDIA/src/` and commit), then follow
[the runbook](../src/deepseek-v4.1-flash/README.md). A submission archive may not keep executable bits, so start the
launchers with `bash`, for example `bash NVIDIA/src/deepseek-v4.1-flash/configs/GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo/point_<C>/run.sh`.

## Recorded values

Each point's `config.yaml` is the benchmark client's record of the run. Where the client recorded a value
specific to NVIDIA's site (the tokenizer, dataset and report paths, and the server URL), this submission
shows the recipe's expression instead: `${MODEL_PATH}`, `${DATA_PATH}`, `${SFLOW_TASK_OUTPUT_DIR}` and the
workflow's `FRONTEND_URL`. Each site sets these to its own storage and node. Every other value is as the
client recorded it.
