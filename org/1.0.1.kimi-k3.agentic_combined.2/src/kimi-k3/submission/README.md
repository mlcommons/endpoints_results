# Submitting Kimi K3 results

[SUBMITTING.md](../../../SUBMITTING.md) describes the submission folder; this page
adds what is specific to Kimi K3.

- **Model name:** `kimi-k3`.
- **Client report directory:** `client-results/report/` under the launcher's `--output`
  directory (`config.yaml`, `performance/result_summary.json` and
  `accuracy/accuracy_results.json`).
- **`point.yaml`:** start from [point.yaml.template](point.yaml.template). Copy
  `max_draft_len` from the `speculative_config` in the point's YAML; it is 3 or 5 by
  point. Keep the drafter's `model_id` and `weight_checksum`: the checker identifies
  the approved `RadixArk/Kimi-K3-DSpark` head by that checksum (§2.9.4).
- **32 and 48 GPUs on one line:** each point runs on 8 or 12 four-GPU nodes. Describe
  the line's system once: every point's `system_desc.json` is the same, for the 12
  nodes you provision. Each point's `point.yaml` states the nodes it engaged in
  `nodes_used` (8 or 12). Only the parallelism fields, `disaggregated`, `batch`,
  `tps_utilization`, `config_summary` and `config_summary_notes` may differ between
  the points' system descriptions; the checker rejects a line whose descriptions
  differ in anything else.
- **System description and power:** NVIDIA's files for each line. Set
  `max_supported_concurrency`, `endpoint_url` and each point's `config_summary`, and
  replace what differs on your system.
  - GB300: [system_desc.json](GB300-NVL72_GB300-288GB_aarch64x48_TRTLLM-Dynamo/system_desc.json) and
    [system_power.json](GB300-NVL72_GB300-288GB_aarch64x48_TRTLLM-Dynamo/system_power.json),
    which provisions the 12 nodes.
  - Vera Rubin: performance-only in this release. NVIDIA has not run SWE-bench on this
    line yet, which a submission needs at one point per band, so no files ship for it.
- **Offline:** none; Kimi K3 is an agentic benchmark.

## Accuracy points

The client scores inline accuracy at every point, and the SWE-bench points also run
SWE-bench Verified (200 instances) after performance (`--mode both`). Keep
`accuracy_results.json` only at one point per mandatory band:

GB300 (C12 to C1024):

| Band | Concurrency on this line | Point with accuracy |
| --- | --- | --- |
| Ultra Low Concurrency | 1-32 | C12 |
| Low Concurrency | 13-22 | C16 |
| Medium Concurrency | 23-113 | C32 |
| High Concurrency | 114-1024 | C1024 |

C16 and C32 also lie in Ultra Low, but each fills its own band. Checker 1.5.0.0
averages the SWE-bench results at 32 or below (C12, C16 and C32) into one band for
the line's SWE-bench mean and warns that it covers fewer than four bands; that is
not an error.

Leave `accuracy_results.json` out at every other point. With a different set of
points the bands move: compute them for your line
([SUBMITTING.md](../../../SUBMITTING.md#bands)) and run SWE-bench at one point per band.
