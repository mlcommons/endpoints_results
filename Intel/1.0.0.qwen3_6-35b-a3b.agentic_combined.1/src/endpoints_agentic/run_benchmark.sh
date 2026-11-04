#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=lib/config.sh
source "${SCRIPT_DIR}/lib/config.sh"
# shellcheck source=lib/services.sh
source "${SCRIPT_DIR}/lib/services.sh"
# shellcheck source=lib/swebench.sh
source "${SCRIPT_DIR}/lib/swebench.sh"
# shellcheck source=lib/benchmark.sh
source "${SCRIPT_DIR}/lib/benchmark.sh"
# shellcheck source=lib/emon.sh
source "${SCRIPT_DIR}/lib/emon.sh"
# shellcheck source=lib/orchestration.sh
source "${SCRIPT_DIR}/lib/orchestration.sh"

main() {
    (( $# == 0 )) \
        || fail "run_benchmark.sh does not accept positional arguments; configure it with environment variables."
    initialize_selected_stack benchmark
    configure_emon
    require_common_launcher_environment
    require_benchmark_environment
    configure_benchmark_mode
    configure_swebench_docker_client
    assert_port_available "${SWEBENCH_PORT}" "SWE-bench service"
    validate_free_space
    benchmark_assert_prepared
    select_benchmark_endpoint
    prepare_benchmark_directories
    install_benchmark_cleanup_traps
    prepare_emon

    log "Benchmark run directory: ${RUN_DIR}"
    local service_token
    service_token=$(openssl rand -hex 32)
    render_runtime_config "${service_token}"
    validate_runtime_config
    write_benchmark_manifest
    start_swebench_service "${service_token}"
    run_endpoint_client
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
