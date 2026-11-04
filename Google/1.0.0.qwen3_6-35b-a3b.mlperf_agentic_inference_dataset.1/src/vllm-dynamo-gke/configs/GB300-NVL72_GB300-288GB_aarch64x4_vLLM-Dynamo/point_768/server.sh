#!/usr/bin/env bash
set -euo pipefail
export CPUSET="${CPUSET:-$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')}"
# Container-local temp must not inherit the host's private import path.
export TMPDIR="$(mktemp -d /tmp/qw.XXXXXXXX)"
export TMP="$TMPDIR" TEMP="$TMPDIR"
umask 077
case "$MODE" in perf|acc|both) ;; *) echo "MODE must be perf, acc, or both" >&2; exit 2;; esac
for value in "$STORAGE_ROOT" "$SHARED_ROOT" "$REPO_ROOT" "$MODEL_PATH" "$DATASET_PATH" "$SYSTEM_ID"; do
  case "$value" in ''|*'<'*|*'>'*) echo 'Supply the placeholder variables with sflow --set NAME=value' >&2; exit 1;; esac
done
for path in "$REPO_ROOT" "$SHARED_ROOT" "$MODEL_PATH" "$DATASET_PATH"; do
  if [[ "$STORAGE_ROOT" != / && "$path" != "$STORAGE_ROOT" && "$path" != "$STORAGE_ROOT/"* ]]; then
    echo 'Code, run root, model and dataset must be within STORAGE_ROOT' >&2; exit 1
  fi
done
for path in "$SFLOW_WORKSPACE_DIR" "$SFLOW_WORKFLOW_OUTPUT_DIR"; do
  case "$path" in "$SHARED_ROOT"/*) ;; *) echo 'Set --workspace-dir and --output-dir below SHARED_ROOT' >&2; exit 1;; esac
done
# Use the assigned GPU if one was passed; otherwise self-select the first visible GPU.
case "$GPU_UUID" in GPU-*) ;; *) GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader)"; GPU_UUID="${GPU_UUID%%$'\n'*}" ;; esac
[[ "$GPU_UUID" == GPU-* ]]
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME="$HF_CACHE_DIR" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600 OMP_NUM_THREADS=8
cache_root="$SHARED_ROOT/caches/$SYSTEM_ID/$GPU_UUID"
export VLLM_CACHE_ROOT="$cache_root/vllm" TRITON_CACHE_DIR="$cache_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor" FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
taskset -pc "$CPUSET" $$
printf '%s\n' "$SERVER_IMAGE" > "$SFLOW_TASK_OUTPUT_DIR/server-image.txt"
cp "$SERVER_CONFIG" "$SFLOW_TASK_OUTPUT_DIR/server.yaml"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv > "$SFLOW_TASK_OUTPUT_DIR/gpu.csv"
if [[ "${NSYS:-0}" == 1 ]]; then
  test -x "$NSYS_DIR/nsys"
  hardware="${SYSTEM_ID%%-*}"
  export NSYS_NAME="vllm-delay${NSYS_DELAY:-1200}-dur${NSYS_DURATION:-60}-dp${DP}-point${CONCURRENCY}-${hardware,,}-worker0"
  export NSYS_START_FILE="$SFLOW_WORKFLOW_OUTPUT_DIR/nsys-client-start"
fi
# The image is built from CentML/vllm VLLM_BRANCH at VLLM_COMMIT
# (dependencies.json); no vLLM overlay is applied. Refuse any other image.
[[ "$VLLM_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "VLLM_COMMIT is not pinned in dependencies.json: $VLLM_COMMIT" >&2; exit 2; }
printf '%s\n' "${VLLM_BUILD_COMMIT:-}" > "$SFLOW_TASK_OUTPUT_DIR/vllm-build-commit.txt"
[[ "${VLLM_BUILD_COMMIT:-}" == "$VLLM_COMMIT" ]] || { echo "Image vLLM ${VLLM_BUILD_COMMIT:-unknown} is not $VLLM_BRANCH $VLLM_COMMIT" >&2; exit 2; }
# Server environment: the SERVER_ENV artifact of server.yaml, one KEY=VALUE per line (# comments).
server_env=()
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  [[ "$line" =~ ^[A-Z][A-Z0-9_]*= ]] || { echo "Bad line in $SERVER_ENV: $line" >&2; exit 2; }
  server_env+=("$line")
done < "$SERVER_ENV"
# The FlashInfer autotune cache dir is fresh per run: VLLM_FLASHINFER_AUTOTUNE_FILE (SERVER_ENV) seeds it only
# when no cache file exists yet.
[[ ! -e "$SFLOW_TASK_OUTPUT_DIR/flashinfer-autotune" ]] || { echo "FlashInfer autotune cache dir is not fresh: $SFLOW_TASK_OUTPUT_DIR/flashinfer-autotune" >&2; exit 2; }
cp "$SERVER_ENV" "$SFLOW_TASK_OUTPUT_DIR/server.env"
# Optional space-separated NAME=value pairs for the Dynamo frontend and vLLM workers.
read -r -a server_extra_env <<< "${SERVER_EXTRA_ENV:-}"
for assignment in "${server_extra_env[@]}"; do
  [[ "$assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || { echo "Invalid SERVER_EXTRA_ENV entry: $assignment" >&2; exit 2; }
done
if (( ${#server_extra_env[@]} )); then printf '%s\n' "${server_extra_env[@]}" > "$SFLOW_TASK_OUTPUT_DIR/server-extra-env.txt"; fi
# CUDA forward compat is chosen per host driver (_shared/cuda-compat.sh), never from the image ENV.
exec env "${server_env[@]}" VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$SFLOW_TASK_OUTPUT_DIR/flashinfer-autotune" "${server_extra_env[@]}" \
  PYTHONPATH="$QWEN_SHARED_DIR" QWEN_CUDA_COMPAT_LOG="$SFLOW_TASK_OUTPUT_DIR/cuda-compat.txt" \
  bash "$QWEN_SHARED_DIR/cuda-compat.sh" python3 "$DYNAMO_LAUNCHER" --config "$SFLOW_TASK_OUTPUT_DIR/server.yaml" --output "$SFLOW_TASK_OUTPUT_DIR" --version "$DYNAMO_VERSION" --router-mode least-loaded > >(tee "$SFLOW_TASK_OUTPUT_DIR/vllm.log") 2>&1
