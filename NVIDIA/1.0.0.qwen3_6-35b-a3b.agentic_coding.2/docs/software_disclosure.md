# Software disclosure: Qwen3.6-35B-A3B on VR-NVL72_VR-288GB_aarch64x4_vLLM-Dynamo

## Serving stack

| Component | Version | Source |
| --- | --- | --- |
| vLLM | `0.1.dev1+gd48687fde.d20261008.cu134` | [YibaiMeng/vllm `mlperf-end-multiturn-v1.0-qwen-v2`](https://github.com/YibaiMeng/vllm/tree/mlperf-end-multiturn-v1.0-qwen-v2) at commit `77d1e9a77d3a7f2adb96e13478207c4ed8dfad2c`, which the recipes pin and `system_desc.json` names ([BUILD.md](../src/qwen3.6-35b-a3b/BUILD.md)). The version string records the build commit `d48687fd`; `77d1e9a7` is that branch squashed into one public commit with the same source tree |
| Dynamo | `ai-dynamo==1.4.2` | PyPI |
| PyTorch | `2.15.0.dev20260908+cu134` | PyTorch nightly index, installed by the build ([BUILD.md](../src/qwen3.6-35b-a3b/BUILD.md)) |
| Transformers | `5.19.0` | PyPI |
| CUDA | 13.4.1 | `nvcr.io/nvidia/cuda-dl-base:26.09-cuda13.4` build and runtime bases ([BUILD.md](../src/qwen3.6-35b-a3b/BUILD.md)) |
| Model | `Qwen/Qwen3.6-35B-A3B` at `995ad96e`, BF16 weights quantized online to MXFP8 | [Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/995ad96eacd98c81ed38be0c5b274b04031597b0) |
| Speculative decoding | the checkpoint's native MTP head: five draft tokens at C32–C320, four draft tokens at C384–C1024 | the same checkpoint |

## Images

| Role | What ran | What `src/` and `system_desc.json` name |
| --- | --- | --- |
| Serving | `qwen3.6-vllm-dynamo@sha256:f56c2ac713a2228372afcf295dc102c29d07609462a2a9e734033a5f4223f7e8`, the same image at every point, kept in an NVIDIA-internal registry that reviewers cannot pull from | `system_desc.json` sets no `container_link` (an optional field); `configs/dependencies.json` leaves `VR_NVL72_SERVER_IMAGE` to the site, and [BUILD.md](../src/qwen3.6-35b-a3b/BUILD.md) builds the image from public sources |
| Client | `ghcr.io/mlcommons/endpoints:d3c57f2` (`sha256:1523051b141437b25083924d6428082f435713516eee71c54ae207b393900b2d`) | the same |
| SWE-bench agent and evaluation, at C32, C36, C64 and C192 | NVIDIA's native ARM64 `sweb.eval.arm64.<instance_id>:v4.1.0-arm64` images | `SWE_IMAGE_REGISTRY` names the site's copy of the same images |

## Client and server (rules §2.1.1)

Each point serves the model with four TP1 vLLM workers behind one Dynamo frontend on one four-GPU Vera Rubin
node. From C32 to C512 the benchmark client (and, at C32, C36, C64 and C192, the SWE-bench service) ran on a
second node of the same Slurm job and reached the frontend at the server node's address over the cluster network
([`configs/_shared/coallocate-client.yaml`](../src/qwen3.6-35b-a3b/configs/_shared/coallocate-client.yaml)). At
C1024 the client ran on the server node, on eight CPU cores the server reserves for it, and reached the frontend
through the host network stack with no switch hops
([`configs/_shared/colocate-client.yaml`](../src/qwen3.6-35b-a3b/configs/_shared/colocate-client.yaml)).

The benchmark client is the Endpoints `d3c57f2` reference client at every point; only where it runs differs.
§2.1.1 allows the client to share the data center or a local network with the server, and the client-to-server
latency is inside every timing either way.

| Points | Client host | Path to the Dynamo frontend |
| --- | --- | --- |
| C32, C36, C64, C128, C144, C192, C256, C320, C384, C512 | A second node allocated with the server; its GPUs are idle. The SWE-bench service and task containers of C32, C36, C64 and C192 run on the same node | HTTP to the server node's address on the 400 Gb/s Ethernet port of the BlueField-4 integrated network controller; one routed hop |
| C1024 | Eight physical cores reserved on the server node, outside the CPU masks of the frontend and the four vLLM workers | HTTP to the server node's own address, which the kernel delivers over `lo`; no hops |

Baseline network latency of the two paths, measured between two nodes of the same system (100 ICMP echoes at
0.2 s intervals; 200 HTTP GETs with `curl` against `python3 -m http.server` on the server node's Ethernet address):

| Path | ICMP round trip min / avg / max (ms) | TCP connect p50 / p90 (ms) | HTTP first byte p50 / p90 (ms) |
| --- | --- | --- | --- |
| Client node → server node (C32 to C512) | 0.172 / 0.393 / 0.509 | 0.389 / 0.406 | 1.245 / 1.352 |
| Server node → its own address (C1024) | 0.004 / 0.036 / 0.064 | 0.045 / 0.064 | 0.757 / 0.883 |

HTTP first byte includes the time `python3 -m http.server` takes to answer; the TCP connect time is the network part.

## System settings

Each point locks the memory (HBM4) clock of every GPU in the job to 4752 MHz with `nvidia-smi -lmc 4752,4752`
before the server starts, records the clocks every 30 s, and resets them to the default range with
`nvidia-smi -lmc 3186,4752` when the job ends
([`configs/_shared/gpu-mem-clock.sh`](../src/qwen3.6-35b-a3b/configs/_shared/gpu-mem-clock.sh)). 4752 MHz is the
memory clock `nvidia-smi -q -d CLOCK` reports under Max Clocks and the top of the default range, so the lock holds
HBM4 at its rated maximum; it is not an overclock. SM clocks and GPU power limits are left at their defaults.
