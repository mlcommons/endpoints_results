# Software disclosure: Qwen3.6-35B-A3B on GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo

## Serving stack

| Component | Version | Source |
| --- | --- | --- |
| vLLM | `0.1.dev1+g7cc4a74c3.d20261007.cu134` | [YibaiMeng/vllm `mlperf-end-multiturn-v1.0-qwen-latest`](https://github.com/YibaiMeng/vllm/tree/mlperf-end-multiturn-v1.0-qwen-latest) at commit `7cc4a74c34270ca783e535f7ec07134f92159482`; the recipes pin the public commit `78386fe6cda9bd9ef688c0ad56e542039e429789`, which has the same files ([BUILD.md](../src/qwen3.6-35b-a3b/BUILD.md)) |
| Dynamo | `ai-dynamo==1.4.2` | PyPI |
| PyTorch | `2.15.0.dev20260908+cu134` | PyTorch nightly index, installed by the build ([BUILD.md](../src/qwen3.6-35b-a3b/BUILD.md)) |
| Transformers | `5.19.0` | PyPI |
| CUDA | 13.4.1 | `nvcr.io/nvidia/cuda-dl-base:26.09-cuda13.4` build and runtime bases ([BUILD.md](../src/qwen3.6-35b-a3b/BUILD.md)) |
| Model | `Qwen/Qwen3.6-35B-A3B` at `995ad96e`, BF16 weights quantized online to MXFP8 | [Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/995ad96eacd98c81ed38be0c5b274b04031597b0) |
| Speculative decoding | the checkpoint's native MTP head, three tokens | the same checkpoint |

## Images

| Role | What ran | What `src/` and `system_desc.json` name |
| --- | --- | --- |
| Serving | `qwen3.6-vllm-dynamo@sha256:cb0b77df93c0f9d89b8bca7f59a2520e582b9dbebdd606f7d9f36f57632a8b71`, the same image at every point | `system_desc.json` names that image; `configs/dependencies.json` leaves `GB300_SERVER_IMAGE` to the site, and [BUILD.md](../src/qwen3.6-35b-a3b/BUILD.md) builds it from public sources |
| Client | `ghcr.io/mlcommons/endpoints:d3c57f2` (`sha256:1523051b141437b25083924d6428082f435713516eee71c54ae207b393900b2d`) | the same |
| SWE-bench agent and evaluation, at C32, C36, C64 and C192 | NVIDIA's native ARM64 `sweb.eval.arm64.<instance_id>:v4.1.0-arm64` images | `SWE_IMAGE_REGISTRY` names the site's copy of the same images |

## Client and server (rules §2.1.1)

Each point serves the model with four TP1 vLLM workers behind one Dynamo frontend on one four-GPU GB300 node.
At the accuracy points (C32, C36, C64, C192) the benchmark client and the SWE-bench service ran on a second GB300
node in the same NVL72 rack and reached the frontend at the server node's address over the cluster network. At the
other points (C128, C144, C256, C512, C768) the client ran on the server node, on eight CPU cores the server
reserves for it, and reached the frontend through the host network stack with no switch hops
([`configs/_shared/colocate-client.yaml`](../src/qwen3.6-35b-a3b/configs/_shared/colocate-client.yaml)).
Baseline network latency: not measured.
