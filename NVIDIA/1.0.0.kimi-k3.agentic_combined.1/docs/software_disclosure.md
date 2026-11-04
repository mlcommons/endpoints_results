# Software disclosure: Kimi K3 on GB300-NVL72_GB300-288GB_aarch64x48_TRTLLM-Dynamo

## Serving stack

| Component | Version | Source |
| --- | --- | --- |
| TensorRT-LLM | `1.3.0rc25.post1` | Built from the public branch [`feat/k3_26.08`](https://github.com/NVIDIA/TensorRT-LLM/tree/feat/k3_26.08) at commit [`45c5ae41`](https://github.com/NVIDIA/TensorRT-LLM/commit/45c5ae4150e51df678b2435457ff39679e8fd8d6); [container/build.sh](../src/kimi-k3/container/build.sh) builds that commit |
| Dynamo | `1.6.0.dev20260918` | `ai-dynamo` and `ai-dynamo-runtime` wheels from PyPI; the frontend is rebuilt from [ai-dynamo/dynamo `a9792dbe`](https://github.com/ai-dynamo/dynamo/tree/a9792dbe2402d379232f0aa88eb647cddb78dcab) with [dynamo/kimi-k3.patch](../src/kimi-k3/dynamo/kimi-k3.patch) |
| Model | `nvidia/Kimi-K3-NVFP4` at `b2428a0b` | [Hugging Face](https://huggingface.co/nvidia/Kimi-K3-NVFP4/tree/b2428a0b83a8b712ff2e1a8448a103d4175341f1) |
| Drafter | `RadixArk/Kimi-K3-DSpark` at `3c5bac30` | [Hugging Face](https://huggingface.co/RadixArk/Kimi-K3-DSpark/tree/3c5bac301d9cf392706189d82ed947feca6c2f0f); DSpark speculative decoding on the decode workers |

## Images

| Role | What ran | What `src/` and `system_desc.json` name |
| --- | --- | --- |
| Serving | NVIDIA's `kimi-k3-dynamo` build, imported to SquashFS; the same image at every point | `kimi-k3-dynamo@sha256:796bbbb499a61221fc04930179763f8be3c9af412782843bc9ec763cf2e4c2c8` (tag `20261015`), a rebuild of the same TensorRT-LLM and Dynamo stack that adds `libjemalloc2`, which no GB300 recipe preloads; [container/build.sh](../src/kimi-k3/container/build.sh) builds the image from public sources |
| Client | `ghcr.io/mlcommons/endpoints@sha256:1523051b141437b25083924d6428082f435713516eee71c54ae207b393900b2d` (`d3c57f2`) | the same |
| SWE-bench agent and evaluation, at C12, C16, C32 and C1024 | NVIDIA's ARM64 builds of the SWE-bench Verified images | the same images in `registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/swebench-arm64` |

## Client and server (rules §2.1.1)

Each point is one Slurm job on the nodes its `point.yaml` lists in `nodes_used`. The benchmark client and
the Dynamo frontend run on the same node, and the client reaches the frontend at `http://127.0.0.1:8000`
over loopback, with no network hops. Prefill and decode workers exchange the KV cache over NVLink and
InfiniBand. At C12, C16, C32 and C1024 the same client run then scores SWE-bench Verified with the
Endpoints SWE-bench service at `d3c57f2`, on the client node. Baseline network latency: not measured.
