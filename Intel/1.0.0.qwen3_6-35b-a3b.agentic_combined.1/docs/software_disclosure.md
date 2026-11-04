# Software disclosure

Versions below are recorded in the per-point `system_desc.json`. Exact
source revisions, image IDs, and registry references are in
[vLLM runtime build provenance](specifications/vllm-runtime-builds.md)
and each point's `serving_config.json`.

| Points | vLLM | Python | PyTorch | Transformers | Container OS |
| --- | --- | --- | --- | --- | --- |
| 8, 12, 32, 128 | `0.30.1rc1.dev57+g67ddac3ea` | 3.12.14 | 2.13.0+cpu | 5.17.0 | Ubuntu 25.04 |
| 64 | `0.31.1rc1.dev366+g7a0d68f29` | 3.12.15 | 2.13.0+cpu | 5.19.0 | Ubuntu 22.04.5 LTS |
| 16, 256 | `0.32.0rc1.dev8+g7a0d68f29.d20261018` | 3.12.15 | 2.13.0+cpu | 5.19.0 | Ubuntu 22.04.5 LTS |

## Host

Host OS: Ubuntu 26.04. Inference runs on CPU; accelerator drivers are
inapplicable. Hardware details are in [systemdescription.json](systemdescription.json).

## Routing and benchmark client

All points record the same llm-d and Envoy routing images:

| Component | Recorded image reference |
| --- | --- |
| llm-d endpoint picker `v0.10.0` | `ghcr.io/llm-d/llm-d-router-endpoint-picker@sha256:9862ae38658824b2560c547237363fe21f269e649b873a4ee84fe3137b554ba9` |
| Envoy `distroless-v1.33.2` | `docker.io/envoyproxy/envoy@sha256:0d8c8b3fc3250b5f664dd93ac305738f9c4e6ff38fcb8499088cca1b827f787e` |

The benchmark client image is recorded as the locally built tag
`endpoints-agentic-client:4235a9c`, based on
`ghcr.io/mlcommons/endpoints@sha256:6cf516d0932d6790590308e7d04e5a866fffbab3a9001d65222b639ee8350240`.
The per-point serving manifests retain the tag and base image digest, but not
the final client image digest. The submitted client image source declares
MLCflow `1.3.8` and mlperf-automations revision
`00f82fe05cd63330a3b5ef97a5f18176aa5f7d55`; these are build-time pins, not
runtime-probed versions.

## Calibration environment

The serving manifests record `endpoints-agentic-qwen36-calibration:v8`
and recipe version `8`. The
[calibration Dockerfile](../src/endpoints_agentic/platforms/intel-cpu/docker/Dockerfile.calibration)
pins a Python 3.12 slim-trixie base by digest and declares PyTorch
`2.13.0+cpu`, llmcompressor `0.13.0`, compressed-tensors `0.18.0`,
Transformers `5.10.1`, datasets `5.0.1`, and safetensors `0.8.0`.
These are build-time pins; the original serving manifests lack the built
calibration image digest and runtime package inventory.

Additional client build pins are in the
[endpoint Dockerfile](../src/endpoints_agentic/docker/Dockerfile.endpoints)
and [MLCflow configuration](../src/endpoints_agentic/configs/mlcflow.conf).

## Submission metadata

[cli_metadata.json](../cli_metadata.json) records checker
`1.5.3.1.dev2+gfda43ef88`. Preparation resolves and installs the official
checker's then-current `main` commit.

In published `serving_config.json` files, `${HF_HOME}` replaces the personal
model-cache path and `${SERVER_RUN_DIR}` replaces the original server-output
path. These are disclosure placeholders; actual launch paths are configured
on the reproduction system.
