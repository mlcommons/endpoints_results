#!/usr/bin/env bash
# PUBLIC_ENDPOINT_URL and generated config paths are consumed by other modules.
# shellcheck disable=SC2034

LLMD_EPP_GRPC_PORT="${LLMD_EPP_GRPC_PORT:-9002}"
LLMD_EPP_HEALTH_PORT="${LLMD_EPP_HEALTH_PORT:-9003}"
LLMD_EPP_METRICS_PORT="${LLMD_EPP_METRICS_PORT:-9090}"
LLMD_ENVOY_ADMIN_PORT="${LLMD_ENVOY_ADMIN_PORT:-19000}"
LLMD_ENVOY_CONCURRENCY="${LLMD_ENVOY_CONCURRENCY:-4}"
LLMD_SESSION_TTL_SECONDS="${LLMD_SESSION_TTL_SECONDS:-36000}"
LLMD_ROUTED_HEALTH_PATH="${LLMD_ROUTED_HEALTH_PATH:-/health}"

llm_d_require_positive_integer() {
    local variable_name=$1
    local value="${!variable_name:-}"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
        || fail "${variable_name} must be a positive integer."
}

llm_d_backend_port() {
    local index=$1
    local backend_url="${BACKEND_URLS[$index]}"
    local backend_port="${backend_url##*:}"
    pd_routed_backend_port "${WORKER_ROLES[$index]:-}" "${backend_port}"
}

llm_d_render_configs() {
    local config_dir="${RUN_DIR}/llm-d"
    LLMD_EPP_CONFIG="${config_dir}/epp.yaml"
    LLMD_ENDPOINTS_CONFIG="${config_dir}/endpoints.yaml"
    LLMD_ENVOY_CONFIG="${config_dir}/envoy.yaml"
    mkdir -p "${config_dir}"

    local epp_template="${SCRIPT_DIR}/configs/llm_d/epp.yaml.in"
    if [[ "${TOPOLOGY_MODE:-replica}" == "pd" ]]; then
        epp_template="${SCRIPT_DIR}/configs/llm_d/epp_pd.yaml.in"
    fi
    sed \
        -e "s/__SESSION_TTL_SECONDS__/${LLMD_SESSION_TTL_SECONDS}/g" \
        "${epp_template}" \
        >"${LLMD_EPP_CONFIG}"

    sed \
        -e "s/__ENVOY_ADMIN_PORT__/${LLMD_ENVOY_ADMIN_PORT}/g" \
        -e "s/__PUBLIC_ENDPOINT_PORT__/${PUBLIC_ENDPOINT_PORT}/g" \
        -e "s/__EPP_GRPC_PORT__/${LLMD_EPP_GRPC_PORT}/g" \
        "${SCRIPT_DIR}/configs/llm_d/envoy.yaml.in" \
        >"${LLMD_ENVOY_CONFIG}"

    {
        printf 'endpoints:\n'
        local index backend_url backend_port role
        for index in "${!BACKEND_URLS[@]}"; do
            backend_url="${BACKEND_URLS[$index]}"
            backend_port=$(llm_d_backend_port "${index}")
            printf '  - name: %s\n' "${WORKER_IDS[$index]:-worker${index}}"
            printf '    address: 127.0.0.1\n'
            printf '    port: "%s"\n' "${backend_port}"
            if [[ "${TOPOLOGY_MODE:-replica}" == "pd" ]]; then
                role="${WORKER_ROLES[$index]}"
                printf '    labels:\n'
                printf '      llm-d.ai/role: %s\n' "${role}"
            fi
        done
    } >"${LLMD_ENDPOINTS_CONFIG}"

    if grep -q '__[A-Z0-9_]*__' \
        "${LLMD_EPP_CONFIG}" "${LLMD_ENVOY_CONFIG}" "${LLMD_ENDPOINTS_CONFIG}"; then
        fail "Generated llm-d configuration contains unresolved placeholders."
    fi
}

routing_validate() {
    validate_digest_image ROUTING_IMAGE "llm-d routing"
    validate_digest_image ROUTING_PROXY_IMAGE "llm-d routing"
    [[ -n "${LLMD_ROUTED_HEALTH_PATH:-}" && "${LLMD_ROUTED_HEALTH_PATH}" == /* ]] \
        || fail "LLMD_ROUTED_HEALTH_PATH must be an absolute HTTP path."

    local variable_name
    for variable_name in \
        PUBLIC_ENDPOINT_PORT LLMD_EPP_GRPC_PORT LLMD_EPP_HEALTH_PORT \
        LLMD_EPP_METRICS_PORT LLMD_ENVOY_ADMIN_PORT \
        LLMD_SESSION_TTL_SECONDS; do
        llm_d_require_positive_integer "${variable_name}"
    done
    (( ${#BACKEND_URLS[@]} > 0 )) \
        || fail "llm-d routing requires at least one backend URL."
    if [[ "${TOPOLOGY_MODE:-replica}" == "pd" ]]; then
        validate_digest_image LLMD_SIDECAR_IMAGE "llm-d P/D routing"
        llm_d_require_positive_integer LLMD_SIDECAR_PORT_OFFSET
        (( ${#PREFILL_BACKEND_URLS[@]} > 0 )) \
            || fail "llm-d P/D routing requires at least one prefill backend."
        (( ${#DECODE_BACKEND_URLS[@]} > 0 )) \
            || fail "llm-d P/D routing requires at least one decode backend."
    fi
    local index backend_url backend_port routed_port
    for index in "${!BACKEND_URLS[@]}"; do
        backend_url="${BACKEND_URLS[$index]}"
        [[ "${backend_url}" =~ ^http://127\.0\.0\.1:([0-9]+)$ ]] \
            || fail "llm-d file discovery requires a literal http://127.0.0.1:<port> backend: ${backend_url}"
        backend_port="${BASH_REMATCH[1]}"
        (( backend_port >= 1 && backend_port <= 65535 )) \
            || fail "Invalid llm-d backend port in ${backend_url}."
        routed_port=$(llm_d_backend_port "${index}")
        if [[ "${TOPOLOGY_MODE:-replica}" == "pd" \
            && "${WORKER_ROLES[$index]}" == "decode" ]]; then
            (( routed_port >= 1 && routed_port <= 65535 )) \
                || fail "Invalid llm-d decode sidecar port derived from ${backend_url}."
        fi
    done

    reserve_port "${PUBLIC_ENDPOINT_PORT}" "llm-d Envoy public listener"
    reserve_port "${LLMD_EPP_GRPC_PORT}" "llm-d EPP ext-proc"
    reserve_port "${LLMD_EPP_HEALTH_PORT}" "llm-d EPP health"
    reserve_port "${LLMD_EPP_METRICS_PORT}" "llm-d EPP metrics"
    reserve_port "${LLMD_ENVOY_ADMIN_PORT}" "llm-d Envoy admin"
    PUBLIC_ENDPOINT_URL="http://localhost:${PUBLIC_ENDPOINT_PORT}"
}

routing_prepare_assets() {
    log "Pulling pinned llm-d EPP image ${ROUTING_IMAGE}."
    docker pull "${ROUTING_IMAGE}"
    apply_image_display_tag "${ROUTING_IMAGE}" "${ROUTING_IMAGE_DISPLAY_TAG}"
    log "Pulling pinned llm-d Envoy image ${ROUTING_PROXY_IMAGE}."
    docker pull "${ROUTING_PROXY_IMAGE}"
    apply_image_display_tag \
        "${ROUTING_PROXY_IMAGE}" \
        "${ROUTING_PROXY_IMAGE_DISPLAY_TAG}"
    if [[ "${TOPOLOGY_MODE:-replica}" == "pd" ]]; then
        log "Pulling pinned llm-d decode sidecar image ${LLMD_SIDECAR_IMAGE}."
        docker pull "${LLMD_SIDECAR_IMAGE}"
        apply_image_display_tag \
            "${LLMD_SIDECAR_IMAGE}" \
            "${LLMD_SIDECAR_IMAGE_DISPLAY_TAG}"
    fi
}

routing_assert_prepared() {
    ROUTING_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "${ROUTING_IMAGE}") \
        || fail "Required image is not available locally: ${ROUTING_IMAGE}. Run bash preflight.sh first."
    ROUTING_PROXY_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "${ROUTING_PROXY_IMAGE}") \
        || fail "Required image is not available locally: ${ROUTING_PROXY_IMAGE}. Run bash preflight.sh first."
    if [[ "${TOPOLOGY_MODE:-replica}" == "pd" ]]; then
        # Consumed by the persisted stack manifest.
        # shellcheck disable=SC2034
        LLMD_SIDECAR_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "${LLMD_SIDECAR_IMAGE}") \
            || fail "Required image is not available locally: ${LLMD_SIDECAR_IMAGE}. Run bash preflight.sh first."
    fi
}

routing_configure() {
    llm_d_render_configs
    log "Validating generated llm-d Envoy configuration."
    docker run --rm \
        --network none \
        -v "${LLMD_ENVOY_CONFIG}:/etc/envoy/envoy.yaml:ro" \
        "${ROUTING_PROXY_IMAGE}" \
        --mode validate \
        -c /etc/envoy/envoy.yaml \
        >/dev/null
}

llm_d_service_is_running() {
    local service_id=$1
    local index
    index=$(service_index "${service_id}") || return 1
    [[ "$(docker inspect -f '{{.State.Running}}' \
        "${SERVICE_CONTAINERS[$index]}" 2>/dev/null || true)" == "true" ]]
}

wait_for_llm_d_stack() {
    wait_for_service "routing-llm-d-envoy" "llm-d Envoy"

    local start_seconds
    start_seconds=$(date +%s)
    while true; do
        if curl --connect-timeout 2 --max-time 5 -fsS \
            "http://127.0.0.1:${LLMD_ENVOY_ADMIN_PORT}/stats?filter=^cluster\\.ext-proc\\.membership_healthy$" \
            | grep -Eq '^cluster\.ext-proc\.membership_healthy: [1-9][0-9]*$'; then
            break
        fi
        if ! llm_d_service_is_running "routing-llm-d-epp"; then
            docker logs "${CONTAINER_PREFIX}-llm-d-epp" >&2 || true
            fail "llm-d EPP exited before becoming ready."
        fi
        if ! llm_d_service_is_running "routing-llm-d-envoy"; then
            docker logs "${CONTAINER_PREFIX}-llm-d-envoy" >&2 || true
            fail "llm-d Envoy exited before EPP became ready."
        fi
        if (( $(date +%s) - start_seconds >= 120 )); then
            docker logs "${CONTAINER_PREFIX}-llm-d-epp" >&2 || true
            docker logs "${CONTAINER_PREFIX}-llm-d-envoy" >&2 || true
            fail "Timed out waiting for Envoy to report a healthy llm-d ext-proc endpoint."
        fi
        sleep 2
    done

    curl --connect-timeout 2 --max-time 10 -fsS \
        "${PUBLIC_ENDPOINT_URL}${LLMD_ROUTED_HEALTH_PATH}" >/dev/null \
        || fail "llm-d routing stack could not reach a backend health endpoint."
    log "llm-d routing stack is ready at ${PUBLIC_ENDPOINT_URL}."
}

routing_start() {
    local epp_container="${CONTAINER_PREFIX}-llm-d-epp"
    local envoy_container="${CONTAINER_PREFIX}-llm-d-envoy"
    local -a epp_args=(
        --config-file=/etc/epp/config.yaml
        --pool-name=endpoints-agentic
        --health-checking
        --secure-serving=false
        --tracing=false
        --enable-pprof=false
        --allow-experimental-plugins
    )
    # These are the pinned llm-d v0.10.0 defaults. Only pass overrides so the
    # normal profile does not restate values already supplied by the image.
    [[ "${LLMD_EPP_GRPC_PORT}" == 9002 ]] \
        || epp_args+=(--grpc-port="${LLMD_EPP_GRPC_PORT}")
    [[ "${LLMD_EPP_HEALTH_PORT}" == 9003 ]] \
        || epp_args+=(--grpc-health-port="${LLMD_EPP_HEALTH_PORT}")
    [[ "${LLMD_EPP_METRICS_PORT}" == 9090 ]] \
        || epp_args+=(--metrics-port="${LLMD_EPP_METRICS_PORT}")

    register_service \
        "routing-llm-d-epp" \
        "${epp_container}" \
        true
    log "Starting llm-d EPP on ext-proc port ${LLMD_EPP_GRPC_PORT}."
    docker run -d \
        --name "${epp_container}" \
        --network host \
        -v "${LLMD_EPP_CONFIG}:/etc/epp/config.yaml:ro" \
        -v "${LLMD_ENDPOINTS_CONFIG}:/etc/epp/endpoints.yaml:ro" \
        "${ROUTING_IMAGE}" \
        "${epp_args[@]}" \
        >/dev/null

    register_ready_service \
        "routing-llm-d-envoy" \
        "${envoy_container}" \
        "http://127.0.0.1:${LLMD_ENVOY_ADMIN_PORT}/ready" \
        120
    log "Starting llm-d Envoy gateway on port ${PUBLIC_ENDPOINT_PORT}."
    docker run -d \
        --name "${envoy_container}" \
        --network host \
        -v "${LLMD_ENVOY_CONFIG}:/etc/envoy/envoy.yaml:ro" \
        "${ROUTING_PROXY_IMAGE}" \
        --disable-hot-restart \
        --concurrency "${LLMD_ENVOY_CONCURRENCY}" \
        -c /etc/envoy/envoy.yaml \
        >/dev/null

    wait_for_llm_d_stack
}
