# Qwen3.6-35B-A3B on GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo

NVIDIA's submission for one line: Qwen3.6-35B-A3B on `GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo`, at C32, C36, C64, C128, C144, C192, C256, C512, C768.

| Directory | What it holds |
| --- | --- |
| `src/qwen3.6-35b-a3b/` | The recipes that reproduce these points, and [how to run them](../src/qwen3.6-35b-a3b/README.md). |
| `src/sflow/` | Helpers the recipes call. |
| `results/` | One directory per point, and the system's power description. |

The [software disclosure](software_disclosure.md) lists the software that ran, where it comes from,
and how the client reached the server. The [dataset notes](dataset.md)
say where the benchmark data comes from.

The [calibration notes](calibration.adoc) say how the weights, the drafter and the KV cache
are quantized; no calibration data is used.

## Running the recipes

The recipes are written for a git checkout of NVIDIA's repository: each point's `run.sh` finds the
repository root with `git rev-parse --show-toplevel` unless `REPO_ROOT` names it, and the workflows refer
to `NVIDIA/src/qwen3.6-35b-a3b/...`. To run them from this submission, place `src/` as `NVIDIA/src/` in a git
checkout (for example, `git init` an empty folder, copy `src/` to `NVIDIA/src/` and commit) or set `REPO_ROOT`
to the folder that holds `NVIDIA/src/`, then follow [the runbook](../src/qwen3.6-35b-a3b/README.md). A submission
archive may not keep executable bits, so start the launchers with `bash`, for example
`bash NVIDIA/src/qwen3.6-35b-a3b/configs/GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo/point_32/run.sh`.

## Recorded values

Each point's `config.yaml` is the benchmark client's record of the run. Where the client recorded a value
specific to NVIDIA's site (the tokenizer, dataset and report paths, and the server URL), this submission
shows the recipe's expression instead: `${MODEL_PATH}`, `${DATA_PATH}`, `${SFLOW_TASK_OUTPUT_DIR}` and the
workflow's `FRONTEND_URL`. Each site sets these to its own storage and node. Every other value is as the
client recorded it.

Each point's `system_desc.json` names the serving image by name and digest, without the path of the
NVIDIA-internal registry it is kept in.
