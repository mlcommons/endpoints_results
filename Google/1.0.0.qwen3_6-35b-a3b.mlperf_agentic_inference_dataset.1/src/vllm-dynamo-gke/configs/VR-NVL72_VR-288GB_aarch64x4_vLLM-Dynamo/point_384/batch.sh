#!/usr/bin/env bash
set -euo pipefail
umask 077

POINT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "$POINT_DIR" rev-parse --show-toplevel)}"
RUN_LABEL="${RUN_LABEL:-manual-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
OUTPUT_DIR="${QWEN_RUN_DIR:-${SHARED_ROOT:?}/runs/qwen-$RUN_LABEL}"
POINT_NAME="$(basename "$POINT_DIR")"

mkdir -p "$OUTPUT_DIR/workspace" "$OUTPUT_DIR/sflow_output"

exec sflow batch \
  -f "$REPO_ROOT/NVIDIA/src/qwen3.6-35b-a3b/configs/dependencies.json" \
  -f "$POINT_DIR/client.yaml" \
  -f "$POINT_DIR/server.yaml" \
  -f "$POINT_DIR/workflow.yaml" \
  --set "REPO_ROOT=$REPO_ROOT" --set "POINT_DIR=$POINT_DIR" \
  --set "SHARED_ROOT=${SHARED_ROOT:?Set SHARED_ROOT}" \
  --set "SLURM_ACCOUNT=${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}" \
  --set "SLURM_PARTITION=${SLURM_PARTITION:?Set SLURM_PARTITION}" \
  --set "LOCAL_SCRATCH_ROOT=${LOCAL_SCRATCH_ROOT:-/tmp/qwen-$UID}" \
  --set "RUN_LABEL=$RUN_LABEL" \
  --account "$SLURM_ACCOUNT" \
  --partition "$SLURM_PARTITION" \
  --workspace-dir "$OUTPUT_DIR/workspace" \
  --output-dir "$OUTPUT_DIR/sflow_output" \
  --job-name "qwen-$POINT_NAME-$RUN_LABEL" \
  --sbatch-output "$OUTPUT_DIR/sflow_output/%j-sflow-submit.out" \
  --sbatch-error "$OUTPUT_DIR/sflow_output/%j-sflow-submit.err" \
  --sbatch-extra-args '--time=${{ variables.SLURM_TIME }}' \
  --sbatch-path "$OUTPUT_DIR/sflow-batch.sh" \
  --submit \
  "$@"
