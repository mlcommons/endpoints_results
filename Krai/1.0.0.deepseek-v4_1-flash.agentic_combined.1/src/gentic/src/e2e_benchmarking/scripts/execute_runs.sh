#!/usr/bin/env bash

set -uo pipefail
umask 002

if (( $# != 1 )); then
  echo "Usage: $0 MANIFEST" >&2
  exit 2
fi

MANIFEST=$(realpath "$1")

if [[ ! -s "$MANIFEST" ]]; then
  echo "Runs file is missing or empty: $MANIFEST" >&2
  exit 2
fi

uv sync

TOTAL_RUNS=$(grep -cve '^[[:space:]]*$' "$MANIFEST")
CURRENT_RUN=0
FAILED_RUNS=0

# echo "Slurm job: ${SLURM_JOB_ID}"
echo "Runs manifest: $MANIFEST"
echo "Total runs: $TOTAL_RUNS"

while IFS= read -r LINE || [[ -n "$LINE" ]]; do
  [[ -z "$LINE" ]] && continue

  if [[ "$LINE" != *$'\t'* ]]; then
    echo "Malformed runs.txt line: missing tab separator" >&2
    FAILED_RUNS=$((FAILED_RUNS + 1))
    continue
  fi

  RUN_ID=${LINE%%$'\t'*}
  COMMAND=${LINE#*$'\t'}

  CURRENT_RUN=$((CURRENT_RUN + 1))

  echo "========================================"
  echo "[$CURRENT_RUN/$TOTAL_RUNS] Starting $RUN_ID"
  echo "Command: $COMMAND"
  echo "Started: $(date --iso-8601=seconds)"

  bash -c "$COMMAND"
  # echo "$COMMAND"
  STATUS=$?

  echo "Finished: $(date --iso-8601=seconds)"
  echo "Exit code: $STATUS"

  if (( STATUS != 0 )); then
    echo "Run failed: $RUN_ID" >&2
    FAILED_RUNS=$((FAILED_RUNS + 1))
  fi
done < "$MANIFEST"

echo "========================================"
echo "Completed $TOTAL_RUNS runs"
echo "Failures: $FAILED_RUNS"

if (( FAILED_RUNS != 0 )); then
  exit 1
fi