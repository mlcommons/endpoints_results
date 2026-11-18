# Software disclosure: DeepSeek-V4.1-Flash on VR-NVL72_VR-288GB_aarch64x4_vLLM-Dynamo

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
in its version string's commit ID. A rebuild can also differ in image metadata
and resolved dependencies; its manifest digest must be recorded independently. `.d20261003` is the build date: vLLM's Dockerfile rewrites its
requirements files after its repository check, and the version tool then marks the source tree as
modified. The Rubin pre-release components the build installs (PyTorch nightly, FlashInfer, cutlass-dsl
and the others in `requirements/rubin-prerelease.txt` at that commit) come from public package indexes.

## Images

| Role | What ran | Reproduction |
| --- | --- | --- |
| Serving | `centml-rubin-vllm:dsv41-rubin-7cf5542c25@sha256:f141ad7ef6fb3c823c62bc87a988bdf2ac7f33f143743c6313cb4f08c0c1c53e` | Build from public sources with [BUILD.md](../src/deepseek-v4.1-flash/BUILD.md), then set `VLLM_CONTAINER_IMAGE` to your published image |
| Client, C16, C24, C32, C512, C1024, C2048 and C3072 | `ghcr.io/mlcommons/endpoints:4235a9c` | `4235a9c`, the recipe default |
| Client, C64, C128 and C256 | `ghcr.io/mlcommons/endpoints:d3c57f2`, launched with `ENDPOINT_CONTAINER_IMAGE=docker://ghcr.io/mlcommons/endpoints:d3c57f2` | `4235a9c`, the recipe default; set `ENDPOINT_CONTAINER_IMAGE` as shown to reproduce these points |
| SWE-bench validation | ARM64 SWE-bench instance images, tag `v4.1.0-arm64`; service source `e9f427ce74fd90e7fafdbc77ee7746cc00245692` | Build and push to your registry with [BUILD.md](../src/deepseek-v4.1-flash/BUILD.md#build-the-arm64-swe-bench-images), then set `SWE_IMAGE_REGISTRY` |

## Client and server (rules §2.1.1)

Each point is one single-node job. The benchmark client and the Dynamo frontend run on the same Vera Rubin
node, and the client reaches the frontend at the node's own address (the workflow's `FRONTEND_URL`)
through the host network stack, with no switch hops. Baseline network latency: 0.76 ms over loopback (median HTTP time to
first byte, measured on a Vera Rubin node of the same cluster).

## System settings

Each vLLM worker runs on 12 dedicated physical cores of its GPU's CPU socket with its memory bound there
(`numactl --physcpubind/--membind`); the Dynamo frontend and the client run on the remaining CPUs.

## Model weights and precision

Original mixed-precision checkpoint: MXFP4 MoE expert weights and FP8 weights with 32x32 UE8M0 block scales (quant_method=fp8, expert_dtype=fp4); unquantized tensors retain BF16. No submitter weight quantization or calibration. Runtime uses fp8_ds_mla KV cache, MXFP4 indexer KV, sparse indexer logits and dynamic MXFP8 activation quantization; see docs/calibration.adoc.

See [calibration.adoc](calibration.adoc) for the checkpoint scheme and the
KV-cache, indexer sparsity and activation-quantization mechanisms required
by §§2.9.3, 2.9.5, 2.9.6.4 and 8.1. These runtime paths are enabled during
both performance and accuracy.

## GPU clocks

All four GPUs have HBM clocks locked to **4752 MHz** with
`nvidia-smi -lmc 4752,4752`; the workflow verifies the lock before launching
the workload. This is recorded in each point's `system_desc.json` `hw_notes`.
