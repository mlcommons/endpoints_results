#!/usr/bin/env bash
#SBATCH --job-name=dsv41-swebench-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --segment=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=setup_swebench_service_%j.out
#SBATCH --error=setup_swebench_service_%j.err

set -euo pipefail
umask 022

RECIPE_DIR=${RECIPE_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}}
RECIPE_DIR=$(cd "$RECIPE_DIR" && pwd)
SERVICE_ROOT="$RECIPE_DIR/swebench_service"
CHECKOUT="$SERVICE_ROOT/endpoints"
PROJECT="$CHECKOUT/src/inference_endpoint/evaluation/swebench_service"
VENV="$SERVICE_ROOT/.venv"
UV_BOOTSTRAP="$SERVICE_ROOT/.uv-bootstrap"
ENDPOINTS_REPO=https://github.com/mlcommons/endpoints.git
ENDPOINTS_COMMIT=e9f427ce74fd90e7fafdbc77ee7746cc00245692

if [[ $(uname -m) != aarch64 ]]; then
  echo "ERROR: this setup must run on an aarch64 compute node; got $(uname -m)" >&2
  exit 1
fi

if [[ -x "$VENV/bin/python" ]] && \
   MODULE_PATH=$("$VENV/bin/python" -c 'import swebench_service; print(swebench_service.__file__)' 2>/dev/null) && \
   [[ $MODULE_PATH == "$VENV"/* ]]; then
  echo "SWE-bench service is already installed non-editably: $MODULE_PATH"
  exit 0
fi

mkdir -p "$SERVICE_ROOT"

if [[ -e "$CHECKOUT" && ! -d "$CHECKOUT/.git" ]]; then
  echo "ERROR: $CHECKOUT exists but is not a Git checkout." >&2
  exit 1
fi

if [[ ! -d "$CHECKOUT/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone "$ENDPOINTS_REPO" "$CHECKOUT"
fi
git -C "$CHECKOUT" fetch --depth=1 origin "$ENDPOINTS_COMMIT"
git -C "$CHECKOUT" checkout --detach "$ENDPOINTS_COMMIT"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$ENDPOINTS_COMMIT"

if [[ ! -x "$UV_BOOTSTRAP/bin/uv" ]]; then
  python3 -m venv "$UV_BOOTSTRAP"
  "$UV_BOOTSTRAP/bin/python" -m pip install --upgrade "uv==0.12.6"
fi

export UV_CACHE_DIR="$SERVICE_ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$SERVICE_ROOT/python"
export UV_PROJECT_ENVIRONMENT="$VENV"

"$UV_BOOTSTRAP/bin/uv" sync \
  --frozen \
  --no-dev \
  --no-editable \
  --python 3.12 \
  --project "$PROJECT"

MODULE_PATH=$("$VENV/bin/python" -c 'import swebench_service; print(swebench_service.__file__)')
if [[ $MODULE_PATH != "$VENV"/* ]]; then
  echo "ERROR: expected a non-editable installation under $VENV, got $MODULE_PATH" >&2
  exit 1
fi

"$VENV/bin/python" -m swebench_service --help >/dev/null

echo "SWE-bench service setup complete"
echo "endpoint_commit=$ENDPOINTS_COMMIT"
echo "service_venv=$VENV"
echo "module=$MODULE_PATH"
