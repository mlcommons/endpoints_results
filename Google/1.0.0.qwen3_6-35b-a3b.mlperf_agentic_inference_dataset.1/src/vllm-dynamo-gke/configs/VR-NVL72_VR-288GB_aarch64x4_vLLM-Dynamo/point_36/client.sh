#!/usr/bin/env bash
set -euo pipefail
umask 077
export SLURM_JOB_ID="$SFLOW_BACKEND_JOB_ID"
export SLURMD_NODENAME="$(hostname -s)"
export CPUSET="$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))) )')"
export LOCAL_SCRATCH="$LOCAL_SCRATCH_ROOT/$SLURM_JOB_ID"
mkdir -p "$LOCAL_SCRATCH/tmp"
test -s "$RUN_DIR/client.yaml"
python3 - "$RUN_DIR/client.yaml" "$SWE_CONNECTION" "$MODE" "$RUN_DIR/client.yaml" <<'PY_CLIENT'
import json, sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
if sys.argv[3] != 'perf':
    connection = json.load(open(sys.argv[2]))
    extras = config['datasets'][1]['accuracy_config']['extras']
    extras['swebench_service_url'] = connection['url']
    extras['swebench_service_auth_token'] = connection['token']
with open(sys.argv[4], 'w') as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY_CLIENT
# Reuse a prepared container when supplied.
name="${CLIENT_CONTAINER:-qwen-$SLURM_JOB_ID-c$CONCURRENCY-$RUN_LABEL-client}"
if [[ -z "${CLIENT_CONTAINER:-}" ]]; then
  enroot create --name "$name" "$CLIENT_SQSH" > "$RUN_DIR/enroot-client.log" 2>&1
fi
hf_offline=1
if [[ "$MODE" != perf ]]; then hf_offline=0; fi
if [[ "${NSYS:-0}" == 1 ]]; then touch "$SFLOW_WORKFLOW_OUTPUT_DIR/nsys-client-start"; fi
exec enroot start --root --rw --mount "$SHARED_ROOT:$SHARED_ROOT" --mount "$LOCAL_SCRATCH:$LOCAL_SCRATCH" "$name" \
  env CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void HF_HUB_OFFLINE="$hf_offline" HF_DATASETS_OFFLINE="$hf_offline" TRANSFORMERS_OFFLINE=1 \
  HF_HOME="$LOCAL_SCRATCH/caches/hf-client" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONPYCACHEPREFIX="$LOCAL_SCRATCH/client-pycache" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LOCAL_SCRATCH/xdg-cache}" XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$LOCAL_SCRATCH/xdg-config}" \
  XDG_DATA_HOME="${XDG_DATA_HOME:-$LOCAL_SCRATCH/xdg-data}" XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$LOCAL_SCRATCH/xdg-runtime}" \
  TMPDIR="$LOCAL_SCRATCH/tmp" TMP="$LOCAL_SCRATCH/tmp" TEMP="$LOCAL_SCRATCH/tmp" \
  bash -c 'cd "$1"; exec taskset -c "$2" /opt/venv/bin/inference-endpoint benchmark from-config -c client.yaml --report-dir "$1/report" --mode "$3"' \
  bash "$RUN_DIR" "$CPUSET" "$MODE" > >(tee "$RUN_DIR/client.log") 2>&1
