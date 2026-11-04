# Qwen server image

Each system pins its own YibaiMeng/vllm source in [`configs/dependencies.json`](configs/dependencies.json). The Dockerfile
has no default commit: every build must pass `--build-arg VLLM_BUILD_COMMIT=<40-hex commit>` and use a YibaiMeng/vllm
checkout at that commit as the build context; a compile without it fails at its first step. The commit is stamped into
the image, and every point's `server.sh` refuses an image built from a commit other than its system's pin.
No vLLM overlay is applied at build time or at run time.

| system | branch / commit pin | image pin |
| --- | --- | --- |
| GB300 | `mlperf-end-multiturn-v1.0-qwen-latest` (`VLLM_BRANCH`) at `VLLM_COMMIT` (`78386fe6cda9bd9ef688c0ad56e542039e429789`, `mlperf-end-multiturn-v1.0-qwen-latest~1`) | `GB300_SERVER_IMAGE` |

The image stamps and builds the commit passed as `VLLM_BUILD_COMMIT`. The points refuse an image whose stamped commit
is not their system's pin in `dependencies.json`.

| | |
| --- | --- |
| `BUILD_BASE_IMAGE` | `nvcr.io/nvidia/cuda-dl-base:26.09-cuda13.4-devel-ubuntu24.04@sha256:5a78f43c0ced5796f7cfa1e52277d035f96610d0513c42e9c1eaac6a2a3f4af2`. The build stages install packages with apt on Ubuntu bases (dnf on manylinux bases) and take CPython 3.12 from uv. |
| `FINAL_BASE_IMAGE` | `nvcr.io/nvidia/cuda-dl-base:26.09-cuda13.4-runtime-ubuntu24.04@sha256:9452535d44ed0c9abebcf81a1dd8f761e66dd0d6a2d275d153e65fed6a735b37` |
| `CUDA_VERSION` | `13.4.1` |
| `torch_cuda_arch_list` | `10.0 10.3 10.7f`. vLLM's CMake keeps 10.0 and 10.7f for CUDA 13.4; its 10.0 / 10.0f kernels run on sm_103. Runtime `torch.utils.cpp_extension` JIT also uses 10.3. |
| `INSTALL_RUBIN_PRERELEASE` | `true`. This installs the CUDA 13.4 torch nightly, FlashInfer 0.6.18.post1 with the cu134 jit-cache, and runs `tools/install_flashinfer_sm107.py` (the sm_107 GEMM sources). The Triton source pin is in the Dockerfile. |
| FlashInfer K=64 tactics / autotune pin | vLLM features of the branch: `VLLM_FLASHINFER_MXFP8_K64` (default on, sm_107 only) and `VLLM_FLASHINFER_AUTOTUNE_FILE` (set per point). |

```bash
# Docker on native ARM64: full vLLM + Dynamo build of the GB300 image: VLLM_COMMIT on VLLM_BRANCH (table above).
SYSTEM=GB300
QWEN_CONFIGS="$(git rev-parse --show-toplevel)/NVIDIA/src/qwen3.6-35b-a3b/configs"
pin() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["variables"][sys.argv[2]]["value"])' "$QWEN_CONFIGS/dependencies.json" "$1"; }
OUTPUT_REGISTRY_IMAGE=<registry>/<namespace>/qwen-server:$SYSTEM
DYNAMO_VERSION=$(pin DYNAMO_VERSION)
VLLM_BRANCH=$(pin VLLM_BRANCH)
VLLM_BUILD_COMMIT=$(pin VLLM_COMMIT)
docker login <registry>
git clone --branch "$VLLM_BRANCH" https://github.com/YibaiMeng/vllm.git vllm
git -C vllm checkout "$VLLM_BUILD_COMMIT"
docker buildx build --platform linux/arm64 \
  --file "$QWEN_CONFIGS/Dockerfile" --target qwen-unified \
  --build-arg "BUILD_BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.09-cuda13.4-devel-ubuntu24.04@sha256:5a78f43c0ced5796f7cfa1e52277d035f96610d0513c42e9c1eaac6a2a3f4af2" \
  --build-arg "FINAL_BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.09-cuda13.4-runtime-ubuntu24.04@sha256:9452535d44ed0c9abebcf81a1dd8f761e66dd0d6a2d275d153e65fed6a735b37" \
  --build-arg "CUDA_VERSION=13.4.1" \
  --build-arg "VLLM_BUILD_COMMIT=$VLLM_BUILD_COMMIT" \
  --build-arg "DYNAMO_VERSION=$DYNAMO_VERSION" \
  --tag "$OUTPUT_REGISTRY_IMAGE" --push vllm
```

Set the pushed digest as `GB300_SERVER_IMAGE` in `configs/dependencies.json`, which ships it as the
placeholder `<gb300-server-image>`.
