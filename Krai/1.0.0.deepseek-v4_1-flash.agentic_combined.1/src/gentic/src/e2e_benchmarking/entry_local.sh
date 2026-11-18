#!/bin/bash

set -eo pipefail
umask 002
REPO_ROOT="$HOME/gentic"

VERSION="v1.0.0"
UBUNTU_IMG="ubuntu:24.04"

DATETIME=$(date +%Y-%m-%d_%H-%M-%S)

echo "==============="
echo "Running Version: ${VERSION}"

if [[ "$1" == "--config" ]]; then
    if (( $# != 3 )); then
      echo "Usage: $0 --config SERVER_YAML BENCH_YAML" >&2
      exit 2
    fi
    CONFIG_MODE=True
    SERVER_YAML=$(realpath "$2")
    BENCH_YAML=$(realpath "$3")
elif [[ " $* " == *" --custom "* ]]; then
    CUSTOM=True
    MODEL=$2
  else
    if (( $# != 6 )); then
      echo "Usage: $0 MODEL BENCH_SIZE SCENARIO PLATFORM MODE" >&2
      exit 2
    fi
    MODEL=$2
    BENCH_SIZE=$3
    SCENARIO=$4
    PLATFORM=$5
    MODE=$6
fi


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

echo "1"
{
  echo "========================================"
  echo "Search initiating: $(date --iso-8601=seconds)"

  echo "Preparing runs..."
  if [[ "$CONFIG_MODE" == "True" ]]; then
    echo "Reproducing explicit configs:"
    KERNEL_RESULTS_DIR=$(uv run python3 -u "$SCRIPT_DIR/prepare_runs.py" \
    --version "$VERSION" \
    --server_config "$SERVER_YAML" \
    --bench_config "$BENCH_YAML" \
    3>&1 1>&2)

  elif [[ "$CUSTOM" == "True" ]]; then
    echo "Running custom:"
    KERNEL_RESULTS_DIR=$(uv run python3 -u "$SCRIPT_DIR/prepare_runs.py" \
    --model "$MODEL" \
    --custom_run \
    3>&1 1>&2)

  else
    echo "Running default with params:"
    KERNEL_RESULTS_DIR=$(uv run python3 -u "$SCRIPT_DIR/prepare_runs.py" \
    --version "$VERSION" \
    --model "$MODEL" \
    --bench_size "$BENCH_SIZE" \
    --scenario "$SCENARIO" \
    --platform "$PLATFORM" \
    --mode "$MODE" \
    3>&1 1>&2)
  fi

  # UPDATED EXEC_LOG
  EXEC_LOG="$KERNEL_RESULTS_DIR/log/exec.log"

  mkdir -p "$KERNEL_RESULTS_DIR/log"


  PREPARE_STATUS=$?
  if (( PREPARE_STATUS != 0 )); then
    echo "Preparation failed with exit code $PREPARE_STATUS"
    exit "$PREPARE_STATUS"
  fi

  echo "Executing runs, check script logs at $EXEC_LOG..."

  echo "Running $KERNEL_RESULTS_DIR/manifest.txt" >> "$EXEC_LOG"

  bash \
    "$KERNEL_RESULTS_DIR/execute_runs.sh" \
    "$KERNEL_RESULTS_DIR/manifest.txt" >> "$EXEC_LOG" 2>&1
  

} # >> "$EXEC_LOG" 2>&1
