#!/usr/bin/env bash
set -euo pipefail
export CPUSET="${CPUSET:-$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')}"
export SLURM_JOB_ID="$SFLOW_BACKEND_JOB_ID"
export SLURMD_NODENAME="$(hostname -s)"
export CPUSET="$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))) )')"
# Container-local temp must not inherit the host's private import path.
export TMPDIR="$(mktemp -d /tmp/qw.XXXXXXXX)"
export TMP="$TMPDIR" TEMP="$TMPDIR"
umask 077
case "$MODE" in perf|acc|both) ;; *) echo "MODE must be perf, acc, or both" >&2; exit 2;; esac
export CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void
export HF_HOME="$HF_CACHE_DIR" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ "$MODE" != perf ]]; then
  export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
fi
# Tokenizers: disable internal Rayon threads. OpenMP: one thread per client process.
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
taskset -pc "$CPUSET" $$
/opt/venv/bin/python - "$CLIENT_CONFIG" "$SWE_CONNECTION" "$MODE" "$SFLOW_TASK_OUTPUT_DIR/client.yaml" <<'PY_CLIENT'
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
printf '%s\n' "$CLIENT_IMAGE" "$CLIENT_IMAGE_DIGEST" > "$SFLOW_TASK_OUTPUT_DIR/client-image.txt"
if [[ "${NSYS:-0}" == 1 ]]; then touch "$SFLOW_WORKFLOW_OUTPUT_DIR/nsys-client-start"; fi
exec /opt/venv/bin/inference-endpoint benchmark from-config \
  -c "$SFLOW_TASK_OUTPUT_DIR/client.yaml" --report-dir "$SFLOW_TASK_OUTPUT_DIR/report" --mode "$MODE"
