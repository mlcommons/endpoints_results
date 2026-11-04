#!/usr/bin/env bash
# Build the Kimi K3 ARM64 server image from public sources.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: $0 [image-tag] (default: kimi-k3-dynamo:local; optional BUILD_DIR, TRTLLM_COMMIT)"
  exit 0
fi
if (( $# > 1 )); then echo "Expected at most one image tag" >&2; exit 2; fi
if [[ "$(uname -m)" != aarch64 && "$(uname -m)" != arm64 ]]; then
  echo "Run this build on an ARM64 host" >&2; exit 1
fi
for command in git make docker; do command -v "$command" >/dev/null; done
git lfs version >/dev/null
image="${1:-kimi-k3-dynamo:local}"
# The TensorRT-LLM commit of the published kimi-k3-dynamo:20261015 image, on feat/k3_26.08.
trtllm_commit="${TRTLLM_COMMIT:-45c5ae4150e51df678b2435457ff39679e8fd8d6}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$(cd -- "$script_dir/../dynamo" && pwd)"
build_dir="${BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/kimi-k3-build.XXXXXX")}"
mkdir -p "$build_dir"
trtllm="$build_dir/TensorRT-LLM"
if [[ -e "$trtllm" ]]; then echo "Use a fresh BUILD_DIR: $trtllm already exists" >&2; exit 1; fi
container=""
cleanup() {
  if [[ -n "$container" ]]; then docker rm -f "$container" >/dev/null || true; fi
}
trap cleanup EXIT
printf 'Build sources: %s (TensorRT-LLM feat/k3_26.08 at %s)\n' "$build_dir" "$trtllm_commit"
git clone --branch feat/k3_26.08 --recurse-submodules \
  https://github.com/NVIDIA/TensorRT-LLM.git "$trtllm"
git -C "$trtllm" checkout --detach "$trtllm_commit"
git -C "$trtllm" submodule update --init --recursive
git -C "$trtllm" lfs install --local
git -C "$trtllm" lfs pull
git -C "$trtllm" submodule foreach --recursive 'git lfs install --local && git lfs pull'
base_image="kimi-k3-trtllm:build-$$"
make -C "$trtllm/docker" release_build \
  CUDA_ARCHS="'100-real;103-real;107-real'" IMAGE_WITH_TAG="$base_image"
base_entrypoint="$(docker inspect --format '{{json .Config.Entrypoint}}' "$base_image")"
container="$(docker create \
  --entrypoint /bin/bash \
  --mount "type=bind,src=$runtime_dir,dst=/runtime,readonly" \
  "$base_image" -c 'sleep infinity')"
docker start "$container" >/dev/null
docker exec -i "$container" /bin/bash <<'CONTAINER_BUILD'
set -euo pipefail
extract_wheel() {
  python3 - "$1" "$2" <<'PYWHEEL'
import sys, zipfile
from pathlib import Path
wheels = list(Path(sys.argv[1]).glob("ai_dynamo_runtime-*.whl"))
assert len(wheels) == 1, f"Expected one runtime wheel, found {wheels}"
with zipfile.ZipFile(wheels[0]) as archive:
    archive.extractall(sys.argv[2])
PYWHEEL
}
python3 -m pip install --index-url https://pypi.org/simple --no-deps \
  msgspec==0.21.1 httptools==0.8.0 \
  fla-core==0.5.2 flash-linear-attention==0.5.2
python3 -m pip install --index-url https://pypi.org/simple --no-deps --only-binary=:all: \
  ai-dynamo==1.6.0.dev20260918 ai-dynamo-runtime==1.6.0.dev20260918
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl git build-essential protobuf-compiler pkg-config clang libssl-dev libjemalloc2
curl --fail --location https://static.rust-lang.org/rustup/dist/aarch64-unknown-linux-gnu/rustup-init -o /tmp/rustup-init
chmod +x /tmp/rustup-init
/tmp/rustup-init -y --default-toolchain 1.96.1 --profile minimal
export PATH="/root/.cargo/bin:$PATH"
python3 -m pip install --index-url https://pypi.org/simple maturin
export CUDARC_CUDA_VERSION=13000 CARGO_BUILD_JOBS=8

mkdir -p /build/k3-dynamo
cd /build/k3-dynamo
git clone https://github.com/ai-dynamo/dynamo.git source
git -C source checkout a9792dbe2402d379232f0aa88eb647cddb78dcab
curl --fail --location https://static.crates.io/crates/dynamo-renderer/dynamo-renderer-5.2.0.crate -o renderer.crate
tar -xzf renderer.crate
git apply --exclude='source/lib/runtime/src/pipeline/network/tcp/server.rs' /runtime/kimi-k3.patch

sed -i 's/version = "1.6.0"/version = "1.6.0-dev20260918"/g' source/Cargo.toml
sed -i 's/^\(version = "[^"]*\)"$/\1-dev20260918"/' \
  source/lib/bindings/{python,python/codegen,kvbm}/Cargo.toml \
  source/lib/kvbm-{common,engine,kernels,logical,physical}/Cargo.toml
sed -i -e 's/^\(version = "[^"]*\)"$/\1.dev20260918"/' \
  -e 's/ai-dynamo-runtime==1.6.0"/ai-dynamo-runtime==1.6.0.dev20260918"/' \
  source/pyproject.toml source/lib/{bindings/python,bindings/kvbm,gpu_memory_service}/pyproject.toml
cat >> source/lib/bindings/python/Cargo.toml <<'EOF'

[patch.crates-io]
dynamo-renderer = { path = "/build/k3-dynamo/dynamo-renderer-5.2.0" }
EOF
sed -i '/^name = "\(dynamo-[^"]*\|kvbm-[^"]*\)"$/,/^\[\[package\]\]/{s/^version = "1.6.0"$/version = "1.6.0-dev20260918"/;}' \
  source/lib/bindings/python/Cargo.lock
sed -i '/^name = "dynamo-renderer"$/,/^\[\[package\]\]/{/^source =/d; /^checksum =/d;}' \
  source/lib/bindings/python/Cargo.lock

cd /build/k3-dynamo/source/lib/bindings/python
maturin build --locked --release \
  --features kv-indexer,slot-tracker,select-service,mm-routing,aic-forward-pass,request-trace-s3 \
  --out /build/k3-dynamo/wheels
extract_wheel /build/k3-dynamo/wheels /opt/k3-dynamo/frontend

cd /build/k3-dynamo
git apply --include='source/lib/runtime/src/pipeline/network/tcp/server.rs' /runtime/kimi-k3.patch
cd source/lib/bindings/python
maturin build --locked --release \
  --features kv-indexer,slot-tracker,select-service,mm-routing,aic-forward-pass,request-trace-s3 \
  --out /build/k3-dynamo/wheels-backlog
extract_wheel /build/k3-dynamo/wheels-backlog /opt/k3-dynamo/frontend-backlog

mkdir -p /opt/k3-dynamo/helpers
cp /runtime/{dynamo_entry_v4.py,visible_eos_v4.py} /opt/k3-dynamo/helpers/
sha256sum /opt/k3-dynamo/{frontend,frontend-backlog}/dynamo/_core.abi3.so

FLASHINFER_METADATA_DIR=/root/.cache/flashinfer/cubins/2d6a5a029eefcc388ec0ceb87efb55d8bcce5c3c/fmha/trtllm-gen
FLASHINFER_METADATA_URL=https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/2d6a5a029eefcc388ec0ceb87efb55d8bcce5c3c/fmha/trtllm-gen
mkdir -p "$FLASHINFER_METADATA_DIR/include"
curl --fail --location "$FLASHINFER_METADATA_URL/checksums.txt" -o "$FLASHINFER_METADATA_DIR/checksums.txt"
curl --fail --location "$FLASHINFER_METADATA_URL/include/flashInferMetaInfo.h" -o "$FLASHINFER_METADATA_DIR/include/flashInferMetaInfo.h"
(cd "$FLASHINFER_METADATA_DIR" && sha256sum --check <<'HASHES'
d79b5c51fc8597fac57dae0da4afa114fb2014575e4ec3df099ad856d97cabc3  checksums.txt
892b5a755dde041ec97eacd1ba2156643977cc8e34d882a108a7e45096c0fae4  include/flashInferMetaInfo.h
HASHES
)
CONTAINER_BUILD
docker commit --change "ENTRYPOINT ${base_entrypoint:-[]}" \
  --change 'ENV PYTHONNOUSERSITE=1' --change 'CMD ["/bin/bash"]' "$container" "$image"
printf 'Built %s. Frontend hashes are printed above.\n' "$image"
