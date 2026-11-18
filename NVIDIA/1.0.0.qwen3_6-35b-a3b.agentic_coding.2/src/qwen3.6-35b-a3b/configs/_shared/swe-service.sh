#!/usr/bin/env bash
set -euo pipefail
export SLURM_JOB_ID="${SFLOW_BACKEND_JOB_ID:-$SLURM_JOB_ID}"
export SLURMD_NODENAME="$(hostname -s)"
exec env PYTHONPATH="$QWEN_SHARED_DIR" python3 "$SWE_LAUNCHER"
