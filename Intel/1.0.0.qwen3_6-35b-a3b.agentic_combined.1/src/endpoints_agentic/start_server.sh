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
# shellcheck source=lib/orchestration.sh
source "${SCRIPT_DIR}/lib/orchestration.sh"

main() {
    (( $# == 0 )) \
        || fail "start_server.sh does not accept positional arguments; configure it with environment variables."
    initialize_selected_stack server
    require_common_launcher_environment
    require_server_topology_environment
    validate_selected_stack
    check_reserved_ports
    prepare_server_directories
    install_server_cleanup_traps

    calibration_assert_prepared
    runtime_assert_prepared
    routing_assert_prepared
    routing_configure

    log "Server run directory: ${SERVER_RUN_DIR}"
    runtime_start_workers
    routing_start
    write_server_manifest
    publish_server_ready
    supervise_server
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
