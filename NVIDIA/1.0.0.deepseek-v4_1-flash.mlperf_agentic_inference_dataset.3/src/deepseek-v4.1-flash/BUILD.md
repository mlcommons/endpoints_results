# Build the DeepSeek-V4.1-Flash serving image

## Rebuild the shared vLLM-Dynamo container

Build on a native ARM64 machine with Docker and Buildx. These instructions
rebuild the shared GB300 and Vera Rubin vLLM-Dynamo serving stack from public
sources. No NVIDIA partner-registry serving image is needed for this build.

The measured image was tagged `dsv41-rubin-7cf5542c25`. The source commit and
package versions below describe that stack; a rebuild has its own image digest.

Choose a registry and repository accessible to your compute nodes, and replace
`registry.example.com/your-project` in the commands below. After publishing,
set `VLLM_CONTAINER_IMAGE` to your image when launching a point. The serving-image
and SWE-bench registry defaults are empty and must both be supplied.

### Build the vLLM image

Use the [`dsv41-flash-rubin-opt`](https://github.com/YibaiMeng/vllm/commits/dsv41-flash-rubin-opt)
branch of YibaiMeng/vllm and the official vLLM `docker/Dockerfile`. The published image records source
commit `7cf5542c25a83d229abfe6779b99d42aa0b5c9bc` (vLLM version `…+g7cf5542c2…`). The branch publishes that source
as one commit, `a562490e7a6ffe57dcba72a6bb1ef7b563233cd8`, squashed onto upstream vLLM `ea84051819`: it has exactly
the files of `7cf5542c` but a different commit ID, so a rebuild's vLLM version string differs. Source equivalence does not imply
a byte-identical rebuilt image; record the digest of each rebuild.

```bash
git clone --branch dsv41-flash-rubin-opt https://github.com/YibaiMeng/vllm.git
cd vllm
git checkout a562490e7a6ffe57dcba72a6bb1ef7b563233cd8  # same files as 7cf5542c
# The fork has no tags; GIT_REPO_CHECK=1 (tools/check_repo.sh) and the vLLM version need upstream release tags.
git fetch --no-tags https://github.com/vllm-project/vllm.git 'refs/tags/v*:refs/tags/v*'
git describe --tags
git rev-parse HEAD

# docker-build-metadata-args.sh records this as the image's VLLM_IMAGE_TAG
# variable and labels; name the tag you will publish, not a private registry.
export VLLM_IMAGE_TAG=registry.example.com/your-project/centml-rubin-vllm:dsv41-rebuilt

docker buildx inspect endpoints-buildx >/dev/null 2>&1 || \
  docker buildx create --name endpoints-buildx --driver docker-container
docker buildx inspect endpoints-buildx --bootstrap

docker buildx build \
  --builder endpoints-buildx \
  --load \
  $(bash .buildkite/scripts/docker-build-metadata-args.sh cu134) \
  --platform linux/arm64 \
  --build-arg max_jobs=96 \
  --build-arg nvcc_threads=32 \
  --build-arg GIT_REPO_CHECK=1 \
  --build-arg CUDA_VERSION=13.4.1 \
  --build-arg torch_cuda_arch_list="10.0" \
  --build-arg INSTALL_KV_CONNECTORS=true \
  --build-arg INSTALL_RUBIN_PRERELEASE=true \
  --build-arg TRITON_INSTALL_FROM_SOURCE_REPO=https://github.com/triton-lang/triton.git \
  --build-arg TRITON_INSTALL_FROM_SOURCE_REVISION=3f6e41132b5edf639bfb872ad73d4688765e08b8 \
  --build-arg RUN_WHEEL_CHECK=false \
  --build-arg BUILD_BASE_IMAGE=pytorch/manylinuxaarch64-builder:cuda13.4@sha256:487d5f86222afdfda9426da96332ca2603ff0c5024c56bd1b363241240ef7a8e \
  --build-arg FINAL_BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.08-cuda13.4-devel-ubuntu24.04@sha256:843a5e81ed49ad2a2787aa10e48e926dc1f9440522a122ec2a637c27039fc77d \
  --target vllm-openai \
  --progress plain \
  -t vllm-rubin:base \
  -f docker/Dockerfile .
```

### Add Dynamo directly to the image

Install Dynamo into the container's default Python environment; no additional
venv is created. Capture and protect the rebuilt base image's vLLM, Torch, and
Transformers versions as shown below. The measured image contains:

| Package | Version |
| --- | --- |
| vLLM | `0.20.2rc1.dev6106+g7cf5542c2.d20261003.cu134` |
| Torch | `2.15.0.dev20260908+cu134` |
| Transformers | `5.18.0` |
| Protobuf | `7.36.2` |
| ai-dynamo | `1.6.0.dev20260924` |

The measured image's build history uses `uv pip install --system` with
constraints for vLLM, Torch, and Transformers; it lets Dynamo resolve Protobuf.
Follow that approach below, capturing the three protected versions from the
rebuilt base image (including the build date in vLLM's version string).

```bash
export DYNAMO_BUILD_DIR=$(mktemp -d)
docker run --rm --entrypoint python3 vllm-rubin:base \
  -c 'from importlib.metadata import version; print("\n".join(f"{name}=={version(name)}" for name in ("vllm", "torch", "transformers")))' \
  > "$DYNAMO_BUILD_DIR/dynamo-constraint.txt"
cat "$DYNAMO_BUILD_DIR/dynamo-constraint.txt"

cat > "$DYNAMO_BUILD_DIR/Dockerfile" <<'EOF'
FROM vllm-rubin:base
COPY dynamo-constraint.txt /opt/dynamo-constraint.txt
RUN uv pip install --system \
      --constraint /opt/dynamo-constraint.txt 'ai-dynamo==1.6.0.dev20260924'
RUN python3 -c 'from importlib.metadata import version; from pathlib import Path; expected = dict(line.strip().split("==", 1) for line in Path("/opt/dynamo-constraint.txt").read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")); actual = {name: version(name) for name in expected}; assert actual == expected, (expected, actual); assert version("ai-dynamo") == "1.6.0.dev20260924"'
EOF

docker build --platform linux/arm64 \
  -t centml-rubin-vllm:dsv41-rubin-7cf5542c25 "$DYNAMO_BUILD_DIR"
docker run --rm --gpus all --entrypoint python3 \
  centml-rubin-vllm:dsv41-rubin-7cf5542c25 \
  -c 'import importlib; importlib.import_module("dynamo.frontend"); importlib.import_module("dynamo.vllm")'
```

Review the captured versions before adding Dynamo. Installation and the
post-install checks preserve those exact versions. Use a new tag for a changed
build. The published prerelease image currently reports existing `pip check`
conflicts (including Torch/Triton and Protobuf bounds); the validation above
checks the protected versions and imports both Dynamo modules.

Publish the combined Docker image after validation, using credentials for your
chosen registry:

```bash
docker login registry.example.com
export PUBLISHED_IMAGE=registry.example.com/your-project/centml-rubin-vllm:dsv41-rebuilt
docker tag centml-rubin-vllm:dsv41-rubin-7cf5542c25 "$PUBLISHED_IMAGE"
docker push "$PUBLISHED_IMAGE"
docker buildx imagetools inspect "$PUBLISHED_IMAGE"
```

Build the SWE-bench images below before running. Use the published images for
Slurm/Pyxis runs (replace `point_<C>` with a point):

```bash
export VLLM_CONTAINER_IMAGE="docker://$PUBLISHED_IMAGE"
export SWE_IMAGE_REGISTRY="registry.example.com/your-project/swebench-arm64"
point=/path/to/checkout/NVIDIA/src/deepseek-v4.1-flash/configs/YOUR_SYSTEM/point_16
bash "$point/run.sh" --submit
```

For repeatable runs, inspect the published manifest and use its full digest in
`VLLM_CONTAINER_IMAGE` instead of a mutable tag.

## Build the ARM64 SWE-bench images

Use a native ARM64 host with Docker and push access to your registry. The
[MLCommons builder](https://github.com/mlcommons/endpoints/blob/e9f427ce74fd90e7fafdbc77ee7746cc00245692/examples/10_Agentic_Inference/build_and_push.py)
at the pinned SWE-bench service revision builds the first
200 pinned SWE-bench Verified instances. It verifies the ordered task-ID hash
and publishes instance images with tag `v4.1.0-arm64`.

```bash
git clone https://github.com/mlcommons/endpoints.git endpoints-swe-images
cd endpoints-swe-images
git checkout --detach e9f427ce74fd90e7fafdbc77ee7746cc00245692
python3 -m venv .venv-swe-images
.venv-swe-images/bin/pip install "swebench==4.1.0"
docker login registry.example.com
export SWE_IMAGE_REGISTRY=registry.example.com/your-project/swebench-arm64
REGISTRY="$SWE_IMAGE_REGISTRY" WORKERS=16 \
  .venv-swe-images/bin/python examples/10_Agentic_Inference/build_and_push.py
```

The upstream builder applies the ARM64 compatibility adjustments documented in
its [README](https://github.com/mlcommons/endpoints/tree/e9f427ce74fd90e7fafdbc77ee7746cc00245692/examples/10_Agentic_Inference#arm64-compatibility-and-x86-comparability).
Keep that script unchanged. Export `SWE_IMAGE_REGISTRY` in the shell that invokes
`point_<C>/run.sh`. Supply the registry/repository prefix only, without
`docker://`, an instance name, or a tag. The service appends each instance's
image name and `v4.1.0-arm64` tag. The workflow passes this
value to `--image-registry`.
