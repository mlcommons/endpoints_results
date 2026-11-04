#!/usr/bin/env bash
set -euo pipefail
umask 077
export CPUSET="${CPUSET:-$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')}"
export LOCAL_SCRATCH="$LOCAL_SCRATCH_ROOT/${SFLOW_BACKEND_JOB_ID:-$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH/tmp"
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
cp "$SERVER_CONFIG" "$RUN_DIR/server.yaml"
cp "$CLIENT_CONFIG" "$RUN_DIR/client.yaml"
# The image is built from CentML/vllm VR_NVL72_VLLM_BRANCH at
# VR_NVL72_VLLM_COMMIT (dependencies.json). The image commit is checked below.
[[ "$SERVER_IMAGE" == *@sha256:* ]] || { echo "Set SERVER_IMAGE to an image digest (see the runbook)" >&2; exit 2; }
[[ "$VR_NVL72_VLLM_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "VR_NVL72_VLLM_COMMIT is not pinned in dependencies.json: $VR_NVL72_VLLM_COMMIT" >&2; exit 2; }
# Server environment: SERVER_ENV_COMMON (_shared/env/VR-NVL72.env) followed by this point's SERVER_ENV artifact of
# server.yaml, one KEY=VALUE per line (# comments), each key once. The concatenation is the run's server.env record.
cat "$SERVER_ENV_COMMON" "$SERVER_ENV" > "$RUN_DIR/server.env"
SERVER_ENV="$RUN_DIR/server.env"
server_env=()
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  [[ "$line" =~ ^[A-Z][A-Z0-9_]*= ]] || { echo "Bad line in $SERVER_ENV: $line" >&2; exit 2; }
  server_env+=("$line")
done < "$SERVER_ENV"
duplicates=$(printf '%s\n' "${server_env[@]%%=*}" | sort | uniq -d)
[[ -z "$duplicates" ]] || { echo "Keys set twice in $SERVER_ENV: $duplicates" >&2; exit 2; }
# The FlashInfer autotune cache dir is fresh per run: VLLM_FLASHINFER_AUTOTUNE_FILE (SERVER_ENV) seeds it only
# when no cache file exists yet.
[[ ! -e "$RUN_DIR/flashinfer-autotune" ]] || { echo "FlashInfer autotune cache dir is not fresh: $RUN_DIR/flashinfer-autotune" >&2; exit 2; }
# Images are prepared by the cluster administrator before running this point.
test -s "$SERVER_SQSH"
test -s "$CLIENT_SQSH"
sha256sum "$SERVER_SQSH" "$CLIENT_SQSH" > "$RUN_DIR/images.sha256"
client_source="${CLIENT_IMAGE%:*}@$CLIENT_IMAGE_DIGEST"
printf '%s\n' "$SERVER_IMAGE" > "$RUN_DIR/server-image.txt"
printf '%s\n' "$CLIENT_IMAGE" "$client_source" > "$RUN_DIR/client-image.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv > "$RUN_DIR/gpus.csv"
name="qwen-$SLURM_JOB_ID-c$CONCURRENCY-$RUN_LABEL-server"
enroot create --name "$name" "$SERVER_SQSH" > "$RUN_DIR/enroot-server.log" 2>&1
cache="$LOCAL_SCRATCH/caches/g$GPU_INDEX"
mkdir -p "$cache/vllm" "$cache/triton" "$cache/inductor" "$cache/flashinfer" "$cache/hf"
# Override the image's `vllm serve` entrypoint; the command below invokes it.
printf '#!/bin/sh\nexec "$@"\n' > "$cache/server.rc"
# Refuse any image that is not the pinned build of the merged branch. (CUDA forward compat is chosen per host driver
# by _shared/cuda-compat.sh at the exec below, never from the image ENV.)
image_commit=$(enroot start --rc "$cache/server.rc" "$name" sh -c 'printf %s "${VLLM_BUILD_COMMIT:-}"')
printf '%s\n' "$image_commit" > "$RUN_DIR/vllm-build-commit.txt"
[[ "$image_commit" == "$VR_NVL72_VLLM_COMMIT" ]] || { echo "Image vLLM $image_commit is not $VR_NVL72_VLLM_BRANCH $VR_NVL72_VLLM_COMMIT" >&2; exit 2; }
nsys_args=()
if [[ "${NSYS:-0}" == 1 ]]; then
  test -x "$NSYS_DIR/nsys"
  hardware="${SYSTEM_ID%%-*}"
  nsys_name="vllm-delay${NSYS_DELAY:-1200}-dur${NSYS_DURATION:-60}-dp${DP}-point${CONCURRENCY}-${hardware,,}-worker0"
  nsys_args=(--mount "$NSYS_DIR:$NSYS_DIR" --env NSYS=1 --env "NSYS_DIR=$NSYS_DIR"
    --env "NSYS_ARGS=${NSYS_ARGS:-}"
    --env "NSYS_DELAY=${NSYS_DELAY:-1200}" --env "NSYS_DURATION=${NSYS_DURATION:-60}"
    --env "NSYS_NAME=$nsys_name" --env "NSYS_START_FILE=$SFLOW_WORKFLOW_OUTPUT_DIR/nsys-client-start")
fi
# Optional space-separated NAME=value pairs for the Dynamo frontend and vLLM workers.
read -r -a server_extra_env <<< "${SERVER_EXTRA_ENV:-}"
for assignment in "${server_extra_env[@]}"; do
  [[ "$assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || { echo "Invalid SERVER_EXTRA_ENV entry: $assignment" >&2; exit 2; }
done
if (( ${#server_extra_env[@]} )); then printf '%s\n' "${server_extra_env[@]}" > "$RUN_DIR/server-extra-env.txt"; fi
# Optional DRAM clock lock (ci/qwen/gpu-mem-clock.sh), held by a separate job step until the job ends.
if [[ -n "${GPU_MEM_CLOCK_LOCK:-}" ]]; then bash "$REPO_ROOT/ci/qwen/gpu-mem-clock.sh" start; fi
exec enroot start --rc "$cache/server.rc" --root --rw --mount "$SHARED_ROOT:$SHARED_ROOT" --mount "$LOCAL_SCRATCH:$LOCAL_SCRATCH" "${nsys_args[@]}" "$name" \
  env CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HOME="$cache/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  VLLM_CACHE_ROOT="$cache/vllm" TRITON_CACHE_DIR="$cache/triton" TORCHINDUCTOR_CACHE_DIR="$cache/inductor" \
  FLASHINFER_WORKSPACE_BASE="$cache/flashinfer" VLLM_ENGINE_READY_TIMEOUT_S=3600 OMP_NUM_THREADS=8 \
  XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LOCAL_SCRATCH/xdg-cache}" XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$LOCAL_SCRATCH/xdg-config}" \
  XDG_DATA_HOME="${XDG_DATA_HOME:-$LOCAL_SCRATCH/xdg-data}" XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$LOCAL_SCRATCH/xdg-runtime}" \
  TMPDIR="$LOCAL_SCRATCH/tmp" TMP="$LOCAL_SCRATCH/tmp" TEMP="$LOCAL_SCRATCH/tmp" "${server_env[@]}" \
  VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$RUN_DIR/flashinfer-autotune" "${server_extra_env[@]}" \
  env PYTHONPATH="$QWEN_SHARED_DIR" QWEN_CUDA_COMPAT_LOG="$RUN_DIR/cuda-compat.txt" \
  bash "$QWEN_SHARED_DIR/cuda-compat.sh" python3 "$DYNAMO_LAUNCHER" --config "$RUN_DIR/server.yaml" --output "$RUN_DIR" --version "$DYNAMO_VERSION" \
  > >(tee "$RUN_DIR/vllm.log") 2>&1
