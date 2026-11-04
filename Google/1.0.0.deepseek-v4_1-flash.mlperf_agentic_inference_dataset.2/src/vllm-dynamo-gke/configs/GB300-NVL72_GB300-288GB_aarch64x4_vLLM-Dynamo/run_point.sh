#!/usr/bin/env bash
set -euo pipefail
usage() { echo "Usage: point_<C>/run.sh --dry-run|--submit" >&2; }
[[ $# -eq 2 ]] || { usage; exit 2; }
POINT_DIR=$(cd "$1" && pwd)
ACTION=$2
RECIPE_DIR=$(cd "$POINT_DIR/.." && pwd)
REPO_ROOT=$(git -C "$RECIPE_DIR" rev-parse --show-toplevel)
CONCURRENCY=${POINT_DIR##*/point_}
case "$ACTION" in --dry-run|--submit) ;; *) usage; exit 2 ;; esac
if [[ -z ${SFLOW:-} ]]; then
  SFLOW=$REPO_ROOT/.venv/bin/sflow
fi
RUN_LOG_ROOT=${RUN_LOG_ROOT:-$RECIPE_DIR/run_logs}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-your_account}
SLURM_PARTITION=${SLURM_PARTITION:-your_partition}
SFLOW_VERSION=${SFLOW_VERSION:-b8afb5b86b3f46d9fa771531990fc7afaefc69b9}
[[ -x "$SFLOW" ]] || { echo "ERROR: sflow executable missing: $SFLOW" >&2; exit 1; }
# SFlow's Python sits beside the real executable; resolve symlinks such as a uv tool shim.
SFLOW_PYTHON=$(dirname "$(readlink -f "$SFLOW")")/python
"$SFLOW_PYTHON" -c "import yaml" 2>/dev/null || { echo "ERROR: $SFLOW_PYTHON cannot import yaml; set SFLOW to an sflow whose Python has PyYAML" >&2; exit 1; }
SLURM_TIME=${SLURM_TIME:-$("$SFLOW_PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["variables"]["SLURM_TIME"]["value"])' "$RECIPE_DIR/slurm_env.yaml")}
[[ -f "$RECIPE_DIR/vllm_acc_perf_acc.yaml" ]] || { echo "ERROR: workflow template missing" >&2; exit 1; }
mkdir -p "$RUN_LOG_ROOT"
RUN_DIR=$(mktemp -d "$RUN_LOG_ROOT/full_c${CONCURRENCY}_gb300_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
mkdir -p "$RUN_DIR/workspace" "$RUN_DIR/output" "$RUN_DIR/inputs"
cp "$POINT_DIR/client.yaml" "$POINT_DIR/endpoint_env.yaml" "$POINT_DIR/server-env.yaml" "$POINT_DIR/dynamo_env.yaml" "$RUN_DIR/inputs/"
echo "run_dir=$RUN_DIR"
ARGS=(batch -f "$RECIPE_DIR/slurm_env.yaml" -f "$RECIPE_DIR/vllm_acc_perf_acc.yaml" -f "$POINT_DIR/dsv4.1_flash_config.yaml"
  --set "WORK_DIR=$REPO_ROOT" --set "RECIPE_DIR=$RECIPE_DIR"
  --set "ENDPOINT_CONFIG=$RUN_DIR/inputs/client.yaml" --set "ENDPOINT_ENV_YAML=$RUN_DIR/inputs/endpoint_env.yaml"
  --set "SERVER_ENV_YAML=$RUN_DIR/inputs/server-env.yaml" --set "DYNAMO_ENV_YAML=$RUN_DIR/inputs/dynamo_env.yaml"
  --set "SLURM_ACCOUNT=$SLURM_ACCOUNT" --set "SLURM_PARTITION=$SLURM_PARTITION" --set "SLURM_TIME=$SLURM_TIME"
  --workspace-dir "$RUN_DIR/workspace" --output-dir "$RUN_DIR/output"
  --job-name "dsv41-vllm-gb300-c${CONCURRENCY}" --partition "$SLURM_PARTITION"
  --account "$SLURM_ACCOUNT" --time "$SLURM_TIME" --nodes 1 --sflow-version "$SFLOW_VERSION"
  --sbatch-extra-args="--exclusive" --sbatch-extra-args="--signal=B:TERM@60"
  --sbatch-output "$RUN_DIR/slurm_%j.out" --sbatch-error "$RUN_DIR/slurm_%j.err" --sbatch-path "$RUN_DIR/slurm_job.sh")
for KEY in SCRATCH_DIR VLLM_CONTAINER_IMAGE ENDPOINT_CONTAINER_IMAGE; do
  if printenv "$KEY" >/dev/null; then ARGS+=(--set "$KEY=${!KEY}"); fi
done
"$SFLOW" "${ARGS[@]}" > "$RUN_DIR/sflow_batch.log" 2>&1 || { cat "$RUN_DIR/sflow_batch.log"; exit 1; }
"$SFLOW_PYTHON" "$REPO_ROOT/NVIDIA/src/deepseek-v4.1-flash/tools/prepare_batch.py" "$RUN_DIR" "$SFLOW_VERSION" "$POINT_DIR" "$RECIPE_DIR/vllm_acc_perf_acc.yaml"
bash -n "$RUN_DIR/slurm_job.sh"
if [[ $ACTION == --submit ]]; then
  EXTRA=()
  if [[ -n ${SLURM_DEPENDENCY:-} ]]; then EXTRA+=("--dependency=$SLURM_DEPENDENCY"); fi
  sbatch --parsable "${EXTRA[@]}" "$RUN_DIR/slurm_job.sh" > "$RUN_DIR/submission_job_id.txt"
  cat "$RUN_DIR/submission_job_id.txt"
  touch "$RUN_DIR/SUBMITTED"
else
  touch "$RUN_DIR/DRY_RUN_COMPLETE"
fi
