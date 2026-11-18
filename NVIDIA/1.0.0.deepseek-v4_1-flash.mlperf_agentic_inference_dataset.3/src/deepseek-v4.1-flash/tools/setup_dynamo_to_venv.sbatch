#!/usr/bin/env bash
#SBATCH --job-name=dsv41-dynamo-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=setup_dynamo_to_venv_%j.out
#SBATCH --error=setup_dynamo_to_venv_%j.err

set -euo pipefail
umask 022

MODEL_DIR=$(cd "${MODEL_DIR:-${SLURM_SUBMIT_DIR:?Submit this job from the deepseek-v4.1-flash directory}}" && pwd)
RECIPE_DIR=${RECIPE_DIR:-"$MODEL_DIR/configs/GB300-NVL72_GB300-288GB_aarch64x4_vLLM"}
DYNAMO_VENV=${DYNAMO_VENV:-"$MODEL_DIR/dynamo_gb300_venv"}
DYNAMO_CONSTRAINT=${DYNAMO_CONSTRAINT:-"$RECIPE_DIR/dynamo-constraint.txt"}
DYNAMO_VERSION=${DYNAMO_VERSION:-1.6.0.dev20260924}
VLLM_CONTAINER_IMAGE=${VLLM_CONTAINER_IMAGE:-"docker://vllm/vllm-openai@sha256:865784ba59b46e3e1370df8158c1bbff8769c3ab88d63551d3f25eabc1a61c45"}

if [[ $(uname -m) != aarch64 ]]; then
  echo "ERROR: this setup must run on an aarch64 compute node; got $(uname -m)" >&2
  exit 1
fi

test -r "$DYNAMO_CONSTRAINT"
mkdir -p "$DYNAMO_VENV"

srun \
  --nodes=1 \
  --ntasks=1 \
  --container-image="$VLLM_CONTAINER_IMAGE" \
  --container-mounts="$MODEL_DIR:$MODEL_DIR" \
  --container-workdir="$MODEL_DIR" \
  --container-remap-root \
  env \
    DYNAMO_VENV="$DYNAMO_VENV" \
    DYNAMO_CONSTRAINT="$DYNAMO_CONSTRAINT" \
    DYNAMO_VERSION="$DYNAMO_VERSION" \
  bash -lc '
    set -euo pipefail

    # The benchmark uses this venv inside the same pinned vLLM image. Keeping
    # system site packages preserves the image-provided vLLM, Torch, and
    # Transformers builds constrained below.
    verify_dynamo() {
      "$DYNAMO_VENV/bin/python" - <<"PY"
import importlib
from importlib import metadata
from pathlib import Path
import os

expected = {}
for line in Path(os.environ["DYNAMO_CONSTRAINT"]).read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        name, version = line.split("==", 1)
        expected[name] = version

installed = {name: metadata.version(name) for name in expected}
if installed != expected:
    raise RuntimeError(f"Protected package mismatch: {expected} -> {installed}")

dynamo_version = metadata.version("ai-dynamo")
expected_dynamo_version = os.environ["DYNAMO_VERSION"]
if dynamo_version != expected_dynamo_version:
    raise RuntimeError(
        f"Unexpected ai-dynamo version: {dynamo_version} != {expected_dynamo_version}"
    )

for module in ("dynamo.frontend", "dynamo.vllm"):
    importlib.import_module(module)

print(f"DYNAMO_SETUP_READY ai-dynamo={dynamo_version} protected={installed}")
PY
    }

    if [[ -x "$DYNAMO_VENV/bin/python" ]] && verify_dynamo; then
      echo "Dynamo is already installed and verified"
      exit 0
    fi

    python3 -m venv --system-site-packages "$DYNAMO_VENV"
    "$DYNAMO_VENV/bin/python" -m pip install \
      --disable-pip-version-check \
      --constraint "$DYNAMO_CONSTRAINT" \
      "ai-dynamo==$DYNAMO_VERSION"
    verify_dynamo
  '

echo "Dynamo setup complete"
echo "dynamo_version=$DYNAMO_VERSION"
echo "dynamo_venv=$DYNAMO_VENV"
echo "container_image=$VLLM_CONTAINER_IMAGE"
