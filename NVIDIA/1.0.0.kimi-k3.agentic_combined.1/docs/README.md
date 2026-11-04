# Kimi K3 on GB300-NVL72_GB300-288GB_aarch64x48_TRTLLM-Dynamo

NVIDIA's submission for one line: Kimi K3 on `GB300-NVL72_GB300-288GB_aarch64x48_TRTLLM-Dynamo`, at C12, C13, C16, C24, C32, C48, C64, C80, C192, C288, C384, C448, C512, C576, C640, C1024.

| Directory | What it holds |
| --- | --- |
| `src/kimi-k3/` | The recipes that reproduce these points, and [how to run them](../src/kimi-k3/README.md). |
| `src/sflow/` | Helpers the recipes call. |
| `results/` | One directory per point, and the system's power description. |

The [software disclosure](software_disclosure.md) lists the software that ran, where it comes from,
and how the client reached the server. The [dataset notes](dataset.md)
say where the benchmark data comes from.

## Running the recipes

The recipes are written for a checkout of NVIDIA's repository: the launcher,
`NVIDIA/src/kimi-k3/scripts/run.py`, takes the repository root from its own location, and the
runbook's commands use `NVIDIA/src/kimi-k3/...` paths from that root. To run them from this submission,
place `src/` as `NVIDIA/src/` in an empty folder and follow [the runbook](../src/kimi-k3/README.md) from
that folder. A submission archive may not keep executable bits, so start scripts with `bash` or
`python3`, for example `bash NVIDIA/src/kimi-k3/container/build.sh`.

## Recorded values

Each point's `config.yaml` is the benchmark client's record of the run, as it recorded it: the recipes
give the client container paths and a loopback endpoint, so it holds no value specific to NVIDIA's site.

Each point's `system_desc.json` names the serving image by name and digest, without the path of the
NVIDIA-internal registry it is kept in.
