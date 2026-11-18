# Software disclosure: Kimi K3 on GB300-NVL72_GB300-288GB_aarch64x48_TRTLLM-Dynamo

## Serving stack

| Component | Version / revision | Source |
|---|---|---|
| TensorRT-LLM | `1.3.0rc25.post1` | [System description](../results/Azure_ND_GB300-v6/kimi-k3/r1024/system_desc.json) |
| Dynamo | `1.6.0.dev20260918` | [Recipe](../src/kimi-k3/README.md); patched frontend variants with startup hash verification |
| Model | `nvidia/Kimi-K3-NVFP4` | [Revision b2428a0b](https://huggingface.co/nvidia/Kimi-K3-NVFP4/tree/b2428a0b83a8b712ff2e1a8448a103d4175341f1) |
| Drafter | `RadixArk/Kimi-K3-DSpark` | [Revision 3c5bac30](https://huggingface.co/RadixArk/Kimi-K3-DSpark/tree/3c5bac301d9cf392706189d82ed947feca6c2f0f); speculative decoding on decode workers |

Submitter-supplied build provenance identifies TensorRT-LLM branch
`feat/k3_26.08` at `45c5ae41`, `ai-dynamo` and `ai-dynamo-runtime` wheels
from PyPI, and a frontend rebuilt from `ai-dynamo/dynamo` at `a9792dbe` with
[dynamo/kimi-k3.patch](../src/kimi-k3/dynamo/kimi-k3.patch).
Image equivalence to these build revisions is unverified; `container/build.sh`
is not included in this checkout.

## Images

| Role | Image / implementation |
|---|---|
| Serving | `registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/kimi-k3-dynamo@sha256:985260a01a2dfb66bdd0c68f0bd482e78f69c1b101120c7341782d101e2602a4` |
| Client | `ghcr.io/mlcommons/endpoints@sha256:1523051b141437b25083924d6428082f435713516eee71c54ae207b393900b2d` (revision `d3c57f2`) |
| SWE-bench service | [Launcher](../src/kimi-k3/scripts/swe_service.py) pins Endpoints `d3c57f2eba127c819e5a8fd83101d9e5f4e15fce`; task images come from `SWE_IMAGE_REGISTRY` |

Serving/client images are pinned in [slurm_env.yaml](../src/kimi-k3/slurm_env.yaml);
the serving digest also matches all seven submitted system descriptions.
SWE-bench task image digests are not recorded here.

### Build SWE-bench ARM64 images without NVIDIA registry access

For C12/C16/C32/C1024, the public
[build_and_push.py](https://github.com/mlcommons/endpoints/blob/d3c57f2eba127c819e5a8fd83101d9e5f4e15fce/examples/10_Agentic_Inference/build_and_push.py)
builds and pushes 200 ARM64 agent/evaluation task images:
`sweb.eval.arm64.<instance_id>:v4.1.0-arm64`. It uses SWE-bench 4.1.0 with
included ARM compatibility adjustments, pins Verified dataset revision
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`, and checks the instance-ID checksum.

Requires native ARM64, Docker daemon access, `uv`, internet and registry push
access. Replace the example registry with your own empty namespace; no NVIDIA
partner-registry credentials are needed.

```bash
git clone https://github.com/mlcommons/endpoints.git endpoints-swe-build
cd endpoints-swe-build
git checkout --detach d3c57f2eba127c819e5a8fd83101d9e5f4e15fce
export SWE_IMAGE_REGISTRY=registry.example.com/your-project/swebench-arm64
docker login registry.example.com
REGISTRY="$SWE_IMAGE_REGISTRY" WORKERS=4 \
  uv run --frozen --no-dev --python 3.12 \
  --project src/inference_endpoint/evaluation/swebench_service \
  python "$PWD/examples/10_Agentic_Inference/build_and_push.py"
```

The script verifies ARM64 architecture and fails on build/push errors.
Existing tags may be reused, so start with an empty namespace. Add
`--set SWE_IMAGE_REGISTRY="$SWE_IMAGE_REGISTRY"` to the existing Kimi K3 launcher
command; configure compute-node Enroot credentials for private registries per the
[public service instructions](https://github.com/mlcommons/endpoints/blob/d3c57f2eba127c819e5a8fd83101d9e5f4e15fce/src/inference_endpoint/evaluation/swebench_service/README.md#pyxis-runtime).
Rebuilds are not asserted to be byte-identical to the private images.

## Client and server (Rules Section 2.1.1)

Each point's `point.yaml` declares its nodes. Client and Dynamo frontend share
a node and communicate over HTTP/TCP at `http://127.0.0.1:8000` through IPv4
loopback, with **zero physical network hops**. Distributed serving is a separate
path. C12/C16/C32/C1024 also score SWE-bench Verified using the client-node service.

**Baseline network latency: 0.137 ms over loopback** (median HTTP time to first
byte, pooled across 6,000 fresh-connection requests after warmup in a supplemental
test against a static local HTTP endpoint on a GB300 node).
