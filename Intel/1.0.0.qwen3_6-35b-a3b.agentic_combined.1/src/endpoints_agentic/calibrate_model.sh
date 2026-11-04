#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=lib/config.sh
source "${SCRIPT_DIR}/lib/config.sh"
# shellcheck source=lib/topology.sh
source "${SCRIPT_DIR}/lib/topology.sh"
# shellcheck source=lib/benchmark.sh
source "${SCRIPT_DIR}/lib/benchmark.sh"
# shellcheck source=lib/services.sh
source "${SCRIPT_DIR}/lib/services.sh"
# shellcheck source=lib/orchestration.sh
source "${SCRIPT_DIR}/lib/orchestration.sh"

main() {
    (( $# == 0 )) \
        || fail "calibrate_model.sh does not accept positional arguments; configure it with environment variables."
    initialize_selected_stack preflight
    require_common_launcher_environment
    require_server_topology_environment
    [[ "${CALIBRATION_REQUIRED}" == "true" ]] \
        || {
            log "Calibration is not required for WEIGHT_DTYPE=${WEIGHT_DTYPE}, KV_CACHE_DTYPE=${KV_CACHE_DTYPE}."
            return 0
        }
    calibration_prepare_model
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
