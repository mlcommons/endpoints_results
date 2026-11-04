#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=lib/config.sh
source "${SCRIPT_DIR}/lib/config.sh"
# shellcheck source=lib/topology.sh
source "${SCRIPT_DIR}/lib/topology.sh"
# shellcheck source=lib/services.sh
source "${SCRIPT_DIR}/lib/services.sh"
# shellcheck source=lib/swebench.sh
source "${SCRIPT_DIR}/lib/swebench.sh"
# shellcheck source=lib/benchmark.sh
source "${SCRIPT_DIR}/lib/benchmark.sh"
# shellcheck source=lib/orchestration.sh
source "${SCRIPT_DIR}/lib/orchestration.sh"

main() {
    (( $# == 0 )) \
        || fail "preflight.sh does not accept positional arguments; configure it with environment variables."
    initialize_selected_stack preflight
    require_common_launcher_environment
    require_server_topology_environment
    require_benchmark_environment
    configure_benchmark_mode
    configure_swebench_docker_client
    validate_selected_stack
    reserve_port "${SWEBENCH_PORT}" "SWE-bench service"
    check_reserved_ports
    validate_free_space
    prepare_asset_directories
    calibration_prepare_directories

    runtime_prepare_assets
    benchmark_prepare_assets
    calibration_prepare_model
    routing_prepare_assets
    check_swebench_task_network
    prepull_swebench_task_images

    log "Preflight completed for stack profile ${STACK_PROFILE}."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
