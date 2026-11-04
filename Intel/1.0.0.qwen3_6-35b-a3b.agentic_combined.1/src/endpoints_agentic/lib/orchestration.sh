#!/usr/bin/env bash
# Cross-phase orchestration and persisted state.
# shellcheck disable=SC2034

initialize_selected_stack() {
    local phase=$1
    if [[ "${phase}" == "benchmark" ]]; then
        initialize_benchmark_context
        return
    fi

    load_stack_profile
    case "${phase}" in
        preflight)
            initialize_base_configuration
            ;;
        server)
            initialize_server_configuration
            ;;
        *)
            fail "Unknown launcher phase: ${phase}"
            ;;
    esac
    configure_model_cache
    configure_calibration_settings
    discover_topology
    if [[ "${RUNTIME_DEVICE}" == "cpu" ]]; then
        resolve_effective_target_concurrency
        resolve_cpu_kv_cache_allocation
    fi
    load_component_adapters
}

require_common_launcher_environment() {
    local command
    local -a required_commands=(
        awk curl date df docker grep mkdir sed sleep ss uname
    )
    for command in "${required_commands[@]}"; do
        require_command "${command}"
    done

    [[ "$(uname -m)" == "x86_64" ]] \
        || fail "This initial setup requires x86_64."
    [[ "${OUTPUT_ROOT}" == /* ]] \
        || fail "OUTPUT_ROOT must be an absolute path."
    [[ -S /var/run/docker.sock ]] \
        || fail "Docker socket /var/run/docker.sock is unavailable."
    docker info >/dev/null \
        || fail "Docker daemon is unavailable."

    [[ "${MIN_FREE_GIB}" =~ ^[0-9]+$ ]] \
        || fail "MIN_FREE_GIB must be a non-negative integer."
    MIN_FREE_GIB=$((10#${MIN_FREE_GIB}))
}

require_server_topology_environment() {
    local command
    for command in lscpu sort wc; do
        require_command "${command}"
    done
}

require_benchmark_environment() {
    local command
    for command in id openssl tee; do
        require_command "${command}"
    done
}

validate_free_space() {
    local free_kib required_kib
    free_kib=$(df -Pk "${SCRIPT_DIR}" | awk 'NR == 2 {print $4}')
    required_kib=$((MIN_FREE_GIB * 1024 * 1024))
    (( free_kib >= required_kib )) \
        || fail "Only $((free_kib / 1024 / 1024)) GiB is free; MIN_FREE_GIB=${MIN_FREE_GIB}. SWE-bench task images may require substantially more."
}

validate_selected_stack() {
    runtime_validate
    routing_validate
}

assert_port_available() {
    local port=$1
    local owner=$2
    if ss -H -ltn "sport = :${port}" | grep -q .; then
        fail "TCP port ${port} for ${owner} is already in use."
    fi
}

prepare_asset_directories() {
    if ! mkdir -p -- "${MODEL_CACHE_DIR}"; then
        fail "Could not create model cache directory '${MODEL_CACHE_DIR}' for UID $(id -u)/GID $(id -g). Set HF_HOME to a writable absolute path, or ask an administrator to create this directory for the launching user."
    fi
    prepare_writable_directory "${OUTPUT_ROOT}" "preflight output" OUTPUT_ROOT
}

prepare_benchmark_directories() {
    mkdir -p \
        "${LOG_DIR}" \
        "${SWEBENCH_ARTIFACT_ROOT}"
}

prepare_server_directories() {
    mkdir -p "${LOG_DIR}"
}

write_stack_manifest_entries() {
    printf 'STACK_PROFILE=%q\n' "${STACK_PROFILE}"
    printf 'RUNTIME_ADAPTER=%q\n' "${RUNTIME_ADAPTER}"
    printf 'ROUTING_ADAPTER=%q\n' "${ROUTING_ADAPTER}"
    printf 'RUNTIME_DEVICE=%q\n' "${RUNTIME_DEVICE}"
    printf 'MODEL_NAME=%q\n' "${MODEL_NAME}"
    printf 'MODEL_REVISION=%q\n' "${MODEL_REVISION:-}"
    printf 'EFFECTIVE_MODEL_NAME=%q\n' "${EFFECTIVE_MODEL_NAME:-${MODEL_NAME}}"
    printf 'SERVED_MODEL_NAME=%q\n' "${SERVED_MODEL_NAME}"
    printf 'MODEL_CACHE_DIR=%q\n' "${MODEL_CACHE_DIR:-}"
    printf 'MODEL_LOCAL_PATH=%q\n' "${MODEL_LOCAL_PATH:-}"
    printf 'CALIBRATED_MODEL_HOST_PATH=%q\n' "${CALIBRATED_MODEL_HOST_PATH:-}"
    printf 'MTP_ENABLED=%q\n' "${MTP_ENABLED}"
    printf 'WEIGHT_DTYPE=%q\n' "${WEIGHT_DTYPE}"
    printf 'KV_CACHE_DTYPE=%q\n' "${KV_CACHE_DTYPE}"
    printf 'CALIBRATION_REQUIRED=%q\n' "${CALIBRATION_REQUIRED}"
    if [[ "${CALIBRATION_REQUIRED}" == "true" ]]; then
        printf 'CALIBRATION_RECIPE_VERSION=%q\n' \
            "${CALIBRATION_RECIPE_VERSION}"
        printf 'CALIBRATION_IMAGE=%q\n' "${CALIBRATION_IMAGE}"
        [[ -z "${CALIBRATION_IMAGE_ID:-}" ]] \
            || printf 'CALIBRATION_IMAGE_ID=%q\n' "${CALIBRATION_IMAGE_ID}"
        printf 'CALIBRATION_SOURCE_MODEL=%q\n' "${CALIBRATION_SOURCE_MODEL}"
        printf 'CALIBRATION_SOURCE_REVISION=%q\n' \
            "${CALIBRATION_SOURCE_REVISION}"
        printf 'CALIBRATION_MANIFEST=%q\n' "${CALIBRATION_MANIFEST}"
    fi
    printf 'RUNTIME_IMAGE=%q\n' "${RUNTIME_IMAGE}"
    [[ -z "${RUNTIME_IMAGE_DISPLAY_TAG:-}" ]] \
        || printf 'RUNTIME_IMAGE_DISPLAY_TAG=%q\n' "${RUNTIME_IMAGE_DISPLAY_TAG}"
    [[ -z "${RUNTIME_IMAGE_ID:-}" ]] \
        || printf 'RUNTIME_IMAGE_ID=%q\n' "${RUNTIME_IMAGE_ID}"
    [[ -z "${RUNTIME_BASE_IMAGE:-}" ]] \
        || printf 'RUNTIME_BASE_IMAGE=%q\n' "${RUNTIME_BASE_IMAGE}"
    printf 'ROUTING_IMAGE=%q\n' "${ROUTING_IMAGE}"
    [[ -z "${ROUTING_IMAGE_DISPLAY_TAG:-}" ]] \
        || printf 'ROUTING_IMAGE_DISPLAY_TAG=%q\n' "${ROUTING_IMAGE_DISPLAY_TAG}"
    [[ -z "${ROUTING_IMAGE_ID:-}" ]] \
        || printf 'ROUTING_IMAGE_ID=%q\n' "${ROUTING_IMAGE_ID}"
    [[ -z "${ROUTING_PROXY_IMAGE:-}" ]] \
        || printf 'ROUTING_PROXY_IMAGE=%q\n' "${ROUTING_PROXY_IMAGE}"
    [[ -z "${ROUTING_PROXY_IMAGE_DISPLAY_TAG:-}" ]] \
        || printf 'ROUTING_PROXY_IMAGE_DISPLAY_TAG=%q\n' "${ROUTING_PROXY_IMAGE_DISPLAY_TAG}"
    [[ -z "${ROUTING_PROXY_IMAGE_ID:-}" ]] \
        || printf 'ROUTING_PROXY_IMAGE_ID=%q\n' "${ROUTING_PROXY_IMAGE_ID}"
    [[ -z "${LLMD_SIDECAR_IMAGE:-}" ]] \
        || printf 'LLMD_SIDECAR_IMAGE=%q\n' "${LLMD_SIDECAR_IMAGE}"
    [[ -z "${LLMD_SIDECAR_IMAGE_DISPLAY_TAG:-}" ]] \
        || printf 'LLMD_SIDECAR_IMAGE_DISPLAY_TAG=%q\n' "${LLMD_SIDECAR_IMAGE_DISPLAY_TAG}"
    [[ -z "${LLMD_SIDECAR_IMAGE_ID:-}" ]] \
        || printf 'LLMD_SIDECAR_IMAGE_ID=%q\n' "${LLMD_SIDECAR_IMAGE_ID}"
    [[ -z "${LLMD_SIDECAR_PORT_OFFSET:-}" ]] \
        || printf 'LLMD_SIDECAR_PORT_OFFSET=%q\n' \
            "${LLMD_SIDECAR_PORT_OFFSET}"
    printf 'ENDPOINT_BASE_IMAGE=%q\n' "${ENDPOINT_BASE_IMAGE}"
    printf 'ENDPOINT_IMAGE=%q\n' "${ENDPOINT_IMAGE}"
    printf 'STACK_FAMILY=%q\n' "${STACK_FAMILY}"
    printf 'RUNTIME_DEFAULT_WEIGHT_PRECISION=%q\n' \
        "${RUNTIME_DEFAULT_WEIGHT_PRECISION}"
    printf 'HOST_CPU_PROFILE=%q\n' "${HOST_CPU_PROFILE:-generic}"
    printf 'HOST_CPU_VENDOR=%q\n' "${HOST_CPU_VENDOR:-}"
    printf 'HOST_CPU_MODEL=%q\n' "${HOST_CPU_MODEL:-}"
    printf 'HOST_GPU_COUNT=%q\n' "${HOST_GPU_COUNT:-0}"
    printf 'HOST_GPU_NAME=%q\n' "${HOST_GPU_NAME:-}"
    printf 'NUMA_NODES=%q\n' "${NUMA_NODES[*]}"
    printf 'TOPOLOGY_MODE=%q\n' "${TOPOLOGY_MODE:-replica}"
    if [[ "${RUNTIME_DEVICE}" == "cpu" ]]; then
        printf 'TOPOLOGY_TOTAL_RANKS=%q\n' "${TOPOLOGY_TOTAL_RANKS}"
        printf 'TOPOLOGY_CAPACITY_MODE=%q\n' "${TOPOLOGY_CAPACITY_MODE}"
        printf 'TOPOLOGY_USED_NUMA_NODES=%q\n' \
            "${TOPOLOGY_USED_NUMA_NODES[*]}"
        printf 'TOPOLOGY_UNUSED_NUMA_NODES=%q\n' \
            "${TOPOLOGY_UNUSED_NUMA_NODES[*]}"
        if [[ "${TOPOLOGY_MODE:-replica}" == "pd" ]]; then
            printf 'PREFILL_TP_SIZE=%q\n' "${PREFILL_TP_SIZE}"
            printf 'PREFILL_REPLICA_COUNT=%q\n' \
                "${PREFILL_REPLICA_COUNT}"
            printf 'DECODE_TP_SIZE=%q\n' "${DECODE_TP_SIZE}"
            printf 'DECODE_REPLICA_COUNT=%q\n' \
                "${DECODE_REPLICA_COUNT}"
        else
            printf 'TP_SIZE=%q\n' "${TP_SIZE}"
            printf 'REPLICA_COUNT=%q\n' "${REPLICA_COUNT}"
        fi
    fi
    printf 'WORKER_IDS=%q\n' "${WORKER_IDS[*]}"
    printf 'WORKER_ROLES=%q\n' "${WORKER_ROLES[*]}"
    printf 'WORKER_NUMA_GROUPS=%q\n' "${WORKER_NUMA_GROUPS[*]}"
    printf 'WORKER_PARALLEL_WIDTHS=%q\n' "${WORKER_PARALLEL_WIDTHS[*]}"
    printf 'WORKER_OMP_THREADS_BINDS=%q\n' "${WORKER_OMP_THREADS_BINDS[*]}"
    printf 'WORKER_GPU_IDS=%q\n' "${WORKER_GPU_IDS[*]:-}"
    local -a worker_reserved_cpu_counts=()
    local index parallel_width
    if [[ "${RUNTIME_DEVICE}" == "cpu" ]]; then
        for index in "${!WORKER_PARALLEL_WIDTHS[@]}"; do
            parallel_width="${WORKER_PARALLEL_WIDTHS[$index]}"
            # Explicit OpenMP masks
            # leave two CPUs per rank outside the OpenMP thread lists.
            if [[ "${TOPOLOGY_MODE:-replica}" == "pd" \
                || "${WORKER_OMP_THREADS_BINDS[$index]:-auto}" != "auto" ]]; then
                worker_reserved_cpu_counts+=("$((2 * parallel_width))")
            else
                worker_reserved_cpu_counts+=("${parallel_width}")
            fi
        done
    fi
    printf 'WORKER_RESERVED_CPU_COUNTS=%q\n' "${worker_reserved_cpu_counts[*]}"
    if [[ "${RUNTIME_DEVICE}" == "cpu" ]]; then
        printf 'CPU_KV_CACHE_ALLOCATION_MODE=%q\n' \
            "${CPU_KV_CACHE_ALLOCATION_MODE}"
        printf 'CPU_KV_CACHE_ALLOCATION_SOURCE=%q\n' \
            "${CPU_KV_CACHE_ALLOCATION_SOURCE}"
        printf 'CPU_KV_CACHE_TARGET_CONCURRENCY=%q\n' \
            "${EFFECTIVE_TARGET_CONCURRENCY}"
        printf 'CPU_KV_CACHE_BASELINE_TOTAL_GIB=%q\n' \
            "${CPU_KV_CACHE_BASELINE_TOTAL_GIB}"
        printf 'CPU_KV_CACHE_BASELINE_CONCURRENCY=%q\n' \
            "${CPU_KV_CACHE_BASELINE_CONCURRENCY}"
        printf 'CPU_KV_CACHE_REQUESTED_TOTAL_GIB=%q\n' \
            "${CPU_KV_CACHE_REQUESTED_TOTAL_GIB}"
        printf 'CPU_KV_CACHE_EFFECTIVE_TOTAL_GIB=%q\n' \
            "${CPU_KV_CACHE_EFFECTIVE_TOTAL_GIB}"
        printf 'PREFILL_KV_CACHE_EFFECTIVE_TOTAL_GIB=%q\n' \
            "${PREFILL_KV_CACHE_EFFECTIVE_TOTAL_GIB}"
        printf 'DECODE_KV_CACHE_EFFECTIVE_TOTAL_GIB=%q\n' \
            "${DECODE_KV_CACHE_EFFECTIVE_TOTAL_GIB}"
        printf 'VLLM_CPU_KVCACHE_SPACE_GIB=%q\n' \
            "${VLLM_CPU_KVCACHE_SPACE_GIB:-}"
        printf 'WORKER_KV_CACHE_AGGREGATE_GIB=%q\n' \
            "${WORKER_KV_CACHE_AGGREGATE_GIB[*]}"
        printf 'WORKER_KV_CACHE_SPACE_GIB=%q\n' \
            "${WORKER_KV_CACHE_RANK_GIB[*]}"
    fi
    if declare -F platform_write_manifest_entries >/dev/null; then
        platform_write_manifest_entries
    fi
    printf 'WORKER_PORTS=%q\n' "${WORKER_PORTS[*]}"
    printf 'BACKEND_URLS=%q\n' "${BACKEND_URLS[*]}"
    if (( ${#WORKER_NIXL_SIDE_CHANNEL_PORTS[@]} > 0 )); then
        printf 'WORKER_NIXL_SIDE_CHANNEL_PORTS=%q\n' "${WORKER_NIXL_SIDE_CHANNEL_PORTS[*]}"
        printf 'PREFILL_BACKEND_URLS=%q\n' "${PREFILL_BACKEND_URLS[*]}"
        printf 'DECODE_BACKEND_URLS=%q\n' "${DECODE_BACKEND_URLS[*]}"
        printf 'VLLM_PD_NIXL_VERSION=%q\n' "${VLLM_PD_NIXL_VERSION}"
        printf 'VLLM_PD_NIXL_COMMIT=%q\n' "${VLLM_PD_NIXL_COMMIT}"
        printf 'VLLM_PD_UCX_COMMIT=%q\n' "${VLLM_PD_UCX_COMMIT}"
        printf 'VLLM_PD_KV_LEASE_DURATION_SECONDS=%q\n' \
            "${VLLM_PD_KV_LEASE_DURATION_SECONDS}"
    fi
    printf 'PUBLIC_ENDPOINT_URL=%q\n' "${PUBLIC_ENDPOINT_URL}"
    if [[ -n "${LLMD_EPP_CONFIG:-}" ]]; then
        printf 'LLMD_EPP_CONFIG=%q\n' "${LLMD_EPP_CONFIG}"
        printf 'LLMD_ENDPOINTS_CONFIG=%q\n' "${LLMD_ENDPOINTS_CONFIG}"
        printf 'LLMD_ENVOY_CONFIG=%q\n' "${LLMD_ENVOY_CONFIG}"
        printf 'LLMD_EPP_GRPC_PORT=%q\n' "${LLMD_EPP_GRPC_PORT}"
        printf 'LLMD_EPP_HEALTH_PORT=%q\n' "${LLMD_EPP_HEALTH_PORT}"
        printf 'LLMD_EPP_METRICS_PORT=%q\n' "${LLMD_EPP_METRICS_PORT}"
        printf 'LLMD_ENVOY_ADMIN_PORT=%q\n' "${LLMD_ENVOY_ADMIN_PORT}"
        printf 'LLMD_ENVOY_CONCURRENCY=%q\n' "${LLMD_ENVOY_CONCURRENCY}"
        printf 'LLMD_SESSION_TTL_SECONDS=%q\n' "${LLMD_SESSION_TTL_SECONDS}"
        printf 'LLMD_ROUTED_HEALTH_PATH=%q\n' "${LLMD_ROUTED_HEALTH_PATH}"
        if [[ "${TOPOLOGY_MODE:-replica}" == "pd" \
            && ${#DECODE_BACKEND_URLS[@]} -gt 0 ]]; then
            local decode_backend_url decode_backend_port routed_decode_port
            local -a routed_decode_endpoints=()
            for decode_backend_url in "${DECODE_BACKEND_URLS[@]}"; do
                decode_backend_port="${decode_backend_url##*:}"
                routed_decode_port=$(
                    pd_routed_backend_port decode "${decode_backend_port}"
                )
                routed_decode_endpoints+=(
                    "http://127.0.0.1:${routed_decode_port}"
                )
            done
            printf 'LLMD_ROUTED_DECODE_ENDPOINTS=%q\n' \
                "${routed_decode_endpoints[*]}"
            if (( ${#routed_decode_endpoints[@]} == 1 )); then
                printf 'LLMD_ROUTED_DECODE_ENDPOINT=%q\n' \
                    "${routed_decode_endpoints[0]}"
            fi
        fi
    fi
}

write_server_manifest() {
    local manifest_path="${SERVER_RUN_DIR}/resolved-stack.env"
    {
        printf 'SERVER_RUN_ID=%q\n' "${SERVER_RUN_ID}"
        printf 'SERVER_PID=%q\n' "$$"
        printf 'SERVER_RUN_DIR=%q\n' "${SERVER_RUN_DIR}"
        printf 'CONTAINER_PREFIX=%q\n' "${CONTAINER_PREFIX}"
        printf 'SERVICE_IDS=%q\n' "${SERVICE_IDS[*]}"
        printf 'SERVICE_CONTAINERS=%q\n' "${SERVICE_CONTAINERS[*]}"
        write_stack_manifest_entries
    } >"${manifest_path}"
    log "Resolved server manifest: ${manifest_path}"
}

write_benchmark_manifest() {
    local manifest_path="${RUN_DIR}/resolved-stack.env"
    {
        printf 'RUN_ID=%q\n' "${RUN_ID}"
        printf 'ENDPOINT_SOURCE=%q\n' "${ENDPOINT_SOURCE}"
        printf 'ENDPOINT_URL=%q\n' "${PUBLIC_ENDPOINT_URL}"
        [[ -z "${ACTIVE_SERVER_RUN_ID:-}" ]] \
            || printf 'SERVER_RUN_ID=%q\n' "${ACTIVE_SERVER_RUN_ID}"
        [[ -z "${ACTIVE_SERVER_MANIFEST:-}" ]] \
            || printf 'SERVER_MANIFEST=%q\n' "${ACTIVE_SERVER_MANIFEST}"
        [[ -z "${ACTIVE_SERVER_STACK_PROFILE:-}" ]] \
            || printf 'SERVER_STACK_PROFILE=%q\n' "${ACTIVE_SERVER_STACK_PROFILE}"
        printf 'ENDPOINT_BASE_IMAGE=%q\n' "${ENDPOINT_BASE_IMAGE}"
        printf 'ENDPOINT_IMAGE=%q\n' "${ENDPOINT_IMAGE}"
        printf 'ENDPOINT_IMAGE_ID=%q\n' \
            "$(docker image inspect --format '{{.Id}}' "${ENDPOINT_IMAGE}")"
        printf 'AGENTIC_DATASET_SHA256=%q\n' "${AGENTIC_DATASET_SHA256}"
        printf 'SWEBENCH_NETWORK_PROBE_IMAGE=%q\n' "${SWEBENCH_NETWORK_PROBE_IMAGE}"
        printf 'SERVED_MODEL_NAME=%q\n' "${SERVED_MODEL_NAME}"
        printf 'SWEBENCH_PORT=%q\n' "${SWEBENCH_PORT}"
        write_benchmark_manifest_entries
        write_emon_manifest_entries
    } >"${manifest_path}"
    log "Resolved benchmark manifest: ${manifest_path}"
}

publish_server_ready() {
    local temporary_file="${ACTIVE_SERVER_FILE}.tmp.$$"
    {
        printf 'SERVER_RUN_ID=%q\n' "${SERVER_RUN_ID}"
        printf 'SERVER_PID=%q\n' "$$"
        printf 'SERVER_RUN_DIR=%q\n' "${SERVER_RUN_DIR}"
        printf 'STACK_PROFILE=%q\n' "${STACK_PROFILE}"
        printf 'PUBLIC_ENDPOINT_URL=%q\n' "${PUBLIC_ENDPOINT_URL}"
        printf 'SERVER_MANIFEST=%q\n' "${SERVER_RUN_DIR}/resolved-stack.env"
    } >"${temporary_file}"
    mv -f -- "${temporary_file}" "${ACTIVE_SERVER_FILE}"

    if [[ -n "${SERVER_READY_FILE:-}" ]]; then
        temporary_file="${SERVER_READY_FILE}.tmp.$$"
        cp -- "${ACTIVE_SERVER_FILE}" "${temporary_file}"
        mv -f -- "${temporary_file}" "${SERVER_READY_FILE}"
    fi
    log "Server is ready at ${PUBLIC_ENDPOINT_URL}."
    log "Active server state: ${ACTIVE_SERVER_FILE}"
}

remove_owned_active_server_file() {
    [[ -f "${ACTIVE_SERVER_FILE:-}" ]] || return 0
    if grep -Fxq "SERVER_RUN_ID=${SERVER_RUN_ID}" "${ACTIVE_SERVER_FILE}"; then
        rm -f -- "${ACTIVE_SERVER_FILE}"
    fi
}

cleanup_server() {
    local status=$?
    trap - EXIT INT TERM
    set +e
    if [[ -d "${SERVER_RUN_DIR:-}" ]]; then
        write_server_manifest || true
    fi
    cleanup_registered_services
    remove_owned_active_server_file
    if [[ ${status} -eq 0 || ${status} -eq 130 || ${status} -eq 143 ]]; then
        log "Server stopped. Logs and manifest: ${SERVER_RUN_DIR}"
    else
        log "Server failed with status ${status}. Logs and manifest: ${SERVER_RUN_DIR}" >&2
    fi
    exit "${status}"
}

install_server_cleanup_traps() {
    trap cleanup_server EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
}

supervise_server() {
    log "Server supervisor is attached. Use another terminal for requests; press Ctrl-C to stop."
    while true; do
        local failed_service=""
        failed_service=$(first_failed_critical_service || true)
        [[ -z "${failed_service}" ]] \
            || fail "Critical server service ${failed_service} exited."
        sleep 5
    done
}

select_benchmark_endpoint() {
    if [[ -n "${ENDPOINT_URL:-}" ]]; then
        PUBLIC_ENDPOINT_URL="${ENDPOINT_URL%/}"
        ENDPOINT_SOURCE="override"
        ACTIVE_SERVER_RUN_ID=""
        ACTIVE_SERVER_MANIFEST=""
        ACTIVE_SERVER_STACK_PROFILE=""
    else
        local active_file="${OUTPUT_ROOT}/active-server.env"
        [[ -f "${active_file}" ]] \
            || fail "No active local server was found at ${active_file}. Run bash start_server.sh or set ENDPOINT_URL."

        local requested_stack_profile="${STACK_PROFILE:-}"
        local active_stack_profile active_endpoint active_server_run_id
        local active_server_pid active_server_manifest
        # The state file is generated by start_server.sh and contains shell-escaped scalars.
        # shellcheck disable=SC1090
        source "${active_file}"
        active_stack_profile="${STACK_PROFILE}"
        active_endpoint="${PUBLIC_ENDPOINT_URL}"
        active_server_run_id="${SERVER_RUN_ID}"
        active_server_pid="${SERVER_PID}"
        active_server_manifest="${SERVER_MANIFEST}"

        if [[ -n "${requested_stack_profile}" ]]; then
            STACK_PROFILE="${requested_stack_profile}"
        else
            unset STACK_PROFILE
        fi
        if [[ ! "${active_server_pid}" =~ ^[1-9][0-9]*$ ]] \
            || ! kill -0 "${active_server_pid}" 2>/dev/null; then
            fail "Active server state is stale: supervisor PID ${active_server_pid} is not running."
        fi
        [[ -f "${active_server_manifest}" ]] \
            || fail "Active server manifest is missing: ${active_server_manifest}"
        local benchmark_served_model_name="${SERVED_MODEL_NAME}"
        local active_served_model_name
        active_served_model_name=$(
            (
                unset SERVED_MODEL_NAME
                # The manifest is generated by start_server.sh and contains
                # shell-escaped scalar and array summaries.
                # shellcheck disable=SC1090
                source "${active_server_manifest}"
                printf '%s' "${SERVED_MODEL_NAME:-}"
            )
        )
        if [[ -n "${active_served_model_name}" \
            && "${active_served_model_name}" != "${benchmark_served_model_name}" ]]; then
            printf 'WARNING: managed server serves model %q, but benchmark configuration requests %q; continuing.\n' \
                "${active_served_model_name}" \
                "${benchmark_served_model_name}" >&2
        fi
        PUBLIC_ENDPOINT_URL="${active_endpoint%/}"
        ACTIVE_SERVER_RUN_ID="${active_server_run_id}"
        ACTIVE_SERVER_MANIFEST="${active_server_manifest}"
        ACTIVE_SERVER_STACK_PROFILE="${active_stack_profile}"
        ENDPOINT_SOURCE="local"
    fi

    [[ "${PUBLIC_ENDPOINT_URL}" =~ ^https?://[^[:space:]]+$ ]] \
        || fail "ENDPOINT_URL must be an absolute http:// or https:// URL."
    curl --connect-timeout 2 --max-time 10 -fsS \
        "${PUBLIC_ENDPOINT_URL}/v1/models" >/dev/null \
        || fail "Endpoint is not ready: ${PUBLIC_ENDPOINT_URL}/v1/models"
    log "Benchmark endpoint (${ENDPOINT_SOURCE}): ${PUBLIC_ENDPOINT_URL}"
}
