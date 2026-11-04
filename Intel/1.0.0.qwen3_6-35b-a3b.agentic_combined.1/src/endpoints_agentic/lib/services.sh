#!/usr/bin/env bash

initialize_service_registry() {
    SERVICE_IDS=()
    SERVICE_CONTAINERS=()
    SERVICE_READINESS_URLS=()
    SERVICE_TIMEOUTS=()
    SERVICE_CRITICAL=()
    RESERVED_PORTS=()
    RESERVED_PORT_OWNERS=()
}

reserve_port() {
    local port=$1
    local owner=$2
    if [[ ! "${port}" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
        fail "Invalid TCP port ${port} requested by ${owner}."
    fi

    local index
    for index in "${!RESERVED_PORTS[@]}"; do
        [[ "${RESERVED_PORTS[$index]}" != "${port}" ]] \
            || fail "TCP port ${port} is requested by both ${RESERVED_PORT_OWNERS[$index]} and ${owner}."
    done
    RESERVED_PORTS+=("${port}")
    RESERVED_PORT_OWNERS+=("${owner}")
}

check_reserved_ports() {
    local index port
    for index in "${!RESERVED_PORTS[@]}"; do
        port="${RESERVED_PORTS[$index]}"
        if ss -H -ltn "sport = :${port}" | grep -q .; then
            fail "TCP port ${port} for ${RESERVED_PORT_OWNERS[$index]} is already in use."
        fi
    done
}

register_service() {
    local service_id=$1
    local container_name=$2
    local critical=$3

    [[ "${critical}" == "true" || "${critical}" == "false" ]] \
        || fail "Service ${service_id} has invalid critical flag: ${critical}"
    local existing_id
    for existing_id in "${SERVICE_IDS[@]}"; do
        [[ "${existing_id}" != "${service_id}" ]] \
            || fail "Service ${service_id} was registered more than once."
    done

    SERVICE_IDS+=("${service_id}")
    SERVICE_CONTAINERS+=("${container_name}")
    SERVICE_READINESS_URLS+=("")
    SERVICE_TIMEOUTS+=(0)
    SERVICE_CRITICAL+=("${critical}")
}

register_ready_service() {
    local service_id=$1
    local container_name=$2
    local readiness_url=$3
    local timeout_seconds=$4

    [[ -n "${readiness_url}" ]] \
        || fail "Ready service ${service_id} requires a readiness URL."
    [[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]] \
        || fail "Ready service ${service_id} requires a positive timeout."

    register_service "${service_id}" "${container_name}" true
    local index=$((${#SERVICE_IDS[@]} - 1))
    SERVICE_READINESS_URLS[index]="${readiness_url}"
    SERVICE_TIMEOUTS[index]="${timeout_seconds}"
}

service_index() {
    local requested_id=$1
    local index
    for index in "${!SERVICE_IDS[@]}"; do
        if [[ "${SERVICE_IDS[$index]}" == "${requested_id}" ]]; then
            printf '%s\n' "${index}"
            return 0
        fi
    done
    return 1
}

wait_for_service() {
    local service_id=$1
    local description=$2
    local index
    index=$(service_index "${service_id}") \
        || fail "Cannot wait for unregistered service ${service_id}."

    local container_name="${SERVICE_CONTAINERS[$index]}"
    local url="${SERVICE_READINESS_URLS[$index]}"
    local timeout_seconds="${SERVICE_TIMEOUTS[$index]}"
    local start_seconds
    start_seconds=$(date +%s)

    while true; do
        if curl --connect-timeout 2 --max-time 5 -fsS "${url}" >/dev/null 2>&1; then
            log "${description} is ready at ${url}."
            return 0
        fi
        if [[ "$(docker inspect -f '{{.State.Running}}' "${container_name}" 2>/dev/null || true)" != "true" ]]; then
            docker logs "${container_name}" >&2 || true
            fail "${description} container exited before becoming ready."
        fi
        if (( $(date +%s) - start_seconds >= timeout_seconds )); then
            docker logs --tail 200 "${container_name}" >&2 || true
            fail "Timed out after ${timeout_seconds}s waiting for ${description}."
        fi
        sleep 5
    done
}

first_failed_critical_service() {
    local index
    for index in "${!SERVICE_IDS[@]}"; do
        [[ "${SERVICE_CRITICAL[$index]}" == "true" ]] || continue
        if [[ "$(docker inspect -f '{{.State.Running}}' "${SERVICE_CONTAINERS[$index]}" 2>/dev/null || true)" != "true" ]]; then
            printf '%s\n' "${SERVICE_IDS[$index]}"
            return 0
        fi
    done
    return 1
}

capture_service_logs() {
    local index container_name
    for index in "${!SERVICE_IDS[@]}"; do
        container_name="${SERVICE_CONTAINERS[$index]}"
        if docker container inspect "${container_name}" >/dev/null 2>&1; then
            docker logs "${container_name}" \
                >"${LOG_DIR}/${container_name}.log" 2>&1 || true
        fi
    done
}

stop_client_log_follower() {
    if [[ -n "${CLIENT_LOG_PID:-}" ]] && kill -0 "${CLIENT_LOG_PID}" 2>/dev/null; then
        kill "${CLIENT_LOG_PID}" 2>/dev/null || true
        wait "${CLIENT_LOG_PID}" 2>/dev/null || true
    fi
    CLIENT_LOG_PID=""
}

remove_registered_services() {
    local index
    for ((index=${#SERVICE_CONTAINERS[@]} - 1; index >= 0; index--)); do
        docker rm -f "${SERVICE_CONTAINERS[$index]}" >/dev/null 2>&1 || true
    done
}

cleanup_registered_services() {
    set +e
    stop_client_log_follower
    if [[ -d "${LOG_DIR:-}" ]]; then
        capture_service_logs
    fi
    remove_registered_services
}

cleanup_benchmark() {
    local status=$?
    trap - EXIT INT TERM
    set +e
    if declare -F finish_emon >/dev/null; then
        local emon_stop_reason=benchmark_failed
        case "${status}" in
            0) emon_stop_reason=benchmark_completed ;;
            130|143) emon_stop_reason=cancelled ;;
        esac
        finish_emon "${emon_stop_reason}"
    fi
    cleanup_registered_services
    if declare -F cleanup_swebench_task_containers >/dev/null; then
        cleanup_swebench_task_containers
    fi

    if [[ ${status} -eq 0 ]]; then
        log "Run completed. Artifacts: ${RUN_DIR}"
    elif [[ -d "${RUN_DIR:-}" ]]; then
        log "Run failed with status ${status}. Preserved artifacts: ${RUN_DIR}" >&2
    else
        log "Run failed with status ${status}. No run artifacts were created." >&2
    fi
    exit "${status}"
}

install_benchmark_cleanup_traps() {
    trap cleanup_benchmark EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
}
