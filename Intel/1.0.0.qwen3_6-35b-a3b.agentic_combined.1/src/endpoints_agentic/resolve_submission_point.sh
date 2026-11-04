#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=lib/config.sh
source "${SCRIPT_DIR}/lib/config.sh"
# shellcheck source=lib/topology.sh
source "${SCRIPT_DIR}/lib/topology.sh"

log() {
    # This resolver returns machine-readable configuration, not startup logs.
    case "$*" in
        ERROR:*|WARNING:*)
            printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
            ;;
    esac
}

write_topology_json() {
    python3 - "$@" <<'PY'
import json
import sys

integer_fields = {
    "TP_SIZE",
    "REPLICA_COUNT",
    "PREFILL_TP_SIZE",
    "PREFILL_REPLICA_COUNT",
    "DECODE_TP_SIZE",
    "DECODE_REPLICA_COUNT",
}
fields = {}
for argument in sys.argv[1:]:
    key, value = argument.split("=", 1)
    fields[key] = int(value) if key in integer_fields else value
json.dump(fields, sys.stdout, separators=(",", ":"), sort_keys=True)
sys.stdout.write("\n")
PY
}

main() {
    (( $# == 0 )) \
        || fail "resolve_submission_point.sh does not accept positional arguments."

    load_stack_profile
    MODEL_CACHE_DIR="${HF_HOME:-}"
    configure_model_cache
    configure_calibration_settings
    discover_topology

    local topology_mode="${TOPOLOGY_MODE:-replica}"
    local -a topology_fields=(
        "STACK_PROFILE=${STACK_PROFILE}"
        "STACK_FAMILY=${STACK_FAMILY}"
        "RUNTIME_DEVICE=${RUNTIME_DEVICE}"
        "TOPOLOGY_MODE=${topology_mode}"
        "MODEL_NAME=${MODEL_NAME}"
        "MODEL_REVISION=${MODEL_REVISION:-}"
        "MODEL_LOCAL_PATH=${MODEL_LOCAL_PATH:-}"
        "RUNTIME_IMAGE=${RUNTIME_IMAGE}"
        "WEIGHT_DTYPE=${WEIGHT_DTYPE}"
        "RUNTIME_DEFAULT_WEIGHT_PRECISION=${RUNTIME_DEFAULT_WEIGHT_PRECISION}"
        "KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
    )

    if [[ "${topology_mode}" == "pd" ]]; then
        topology_fields+=(
            "PREFILL_TP_SIZE=${PREFILL_TP_SIZE}"
            "PREFILL_REPLICA_COUNT=${PREFILL_REPLICA_COUNT}"
            "DECODE_TP_SIZE=${DECODE_TP_SIZE}"
            "DECODE_REPLICA_COUNT=${DECODE_REPLICA_COUNT}"
        )
    elif [[ "${TOPOLOGY_MODE:-replica}" == "gpu_replica" ]]; then
        topology_fields+=(
            "TP_SIZE=${ACCEL_TP_SIZE}"
            "REPLICA_COUNT=${ACCEL_REPLICA_COUNT}"
        )
    else
        topology_fields+=(
            "TP_SIZE=${TP_SIZE}"
            "REPLICA_COUNT=${REPLICA_COUNT}"
        )
    fi

    write_topology_json "${topology_fields[@]}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
