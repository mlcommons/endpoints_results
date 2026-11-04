# Software disclosure: DeepSeek-V4.1-Flash on GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo

## Serving stack

| Component | Version | Source |
| --- | --- | --- |
| vLLM | `0.20.2rc1.dev6106+g7cf5542c2.d20261003.cu134` | [YibaiMeng/vllm `dsv41-flash-rubin-opt`](https://github.com/YibaiMeng/vllm/commits/dsv41-flash-rubin-opt), commit `a562490e7a6ffe57dcba72a6bb1ef7b563233cd8` |
| Torch | `2.15.0.dev20260908+cu134` | PyTorch nightly index (`https://download.pytorch.org/whl/nightly/cu134`) |
| Transformers | `5.18.0` | PyPI |
| ai-dynamo | `1.6.0.dev20260924` | PyPI |
| CUDA | 13.4.1 | `nvcr.io/nvidia/cuda-dl-base:26.08-cuda13.4-devel-ubuntu24.04` |

The image was built from vLLM commit `7cf5542c25a83d229abfe6779b99d42aa0b5c9bc`. Commit `a562490e` on the
public branch has the same files (root tree `ecb70496`), squashed onto upstream vLLM, so a rebuild differs
only in the version string's commit ID. `.d20261003` is the build date: vLLM's Dockerfile rewrites its
requirements files after its repository check, and the version tool then marks the source tree as
modified. The Rubin pre-release components the build installs (PyTorch nightly, FlashInfer, cutlass-dsl
and the others in `requirements/rubin-prerelease.txt` at that commit) come from public package indexes.

## Images

| Role | What ran | What `src/` pins |
| --- | --- | --- |
| Serving | `centml-rubin-vllm:dsv41-rubin-7cf5542c25` (`sha256:f141ad7e…`) | `centml-rubin-vllm@sha256:141b04dc…` (`-r1`): the same layers; only image metadata that named a private registry path is rewritten |
| Client, C16–C1024 | `ghcr.io/mlcommons/endpoints:4235a9c` | `4235a9c`, the recipe default |
| Client, C2048 | `ghcr.io/mlcommons/endpoints:d3c57f2`, launched with `ENDPOINT_CONTAINER_IMAGE=docker://ghcr.io/mlcommons/endpoints:d3c57f2` | `4235a9c`, the recipe default; set `ENDPOINT_CONTAINER_IMAGE` as shown to reproduce C2048 |

## Client and server (rules §2.1.1)

Each point is one single-node job. The benchmark client and the Dynamo frontend run on the same GB300
node, and the client reaches the frontend at the node's own address (the workflow's `FRONTEND_URL`)
through the host network stack, with no switch hops. Baseline network latency: not measured.
