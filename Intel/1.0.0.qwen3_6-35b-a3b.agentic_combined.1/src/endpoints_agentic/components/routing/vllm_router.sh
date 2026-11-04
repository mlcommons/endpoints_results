#!/usr/bin/env bash
# PUBLIC_ENDPOINT_URL is consumed by the benchmark module.
# shellcheck disable=SC2034

routing_validate() {
    reserve_port "${PUBLIC_ENDPOINT_PORT}" "routing adapter ${ROUTING_ADAPTER}"
    PUBLIC_ENDPOINT_URL="http://localhost:${PUBLIC_ENDPOINT_PORT}"
}

routing_prepare_assets() {
    log "Building pinned router image ${ROUTING_IMAGE}."
    docker build \
        --pull \
        --file "${SCRIPT_DIR}/docker/Dockerfile.router" \
        --tag "${ROUTING_IMAGE}" \
        "${SCRIPT_DIR}"
}

routing_assert_prepared() {
    ROUTING_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "${ROUTING_IMAGE}") \
        || fail "Required image is not available locally: ${ROUTING_IMAGE}. Run bash preflight.sh first."
}

routing_configure() {
    :
}

routing_start() {
    local container_name="${CONTAINER_PREFIX}-router"

    register_ready_service \
        "routing-vllm-router" \
        "${container_name}" \
        "http://127.0.0.1:${PUBLIC_ENDPOINT_PORT}/health" \
        300

    log "Starting consistent-hash router on port ${PUBLIC_ENDPOINT_PORT}."
    docker run -d \
        --name "${container_name}" \
        --network host \
        "${ROUTING_IMAGE}" \
        --port "${PUBLIC_ENDPOINT_PORT}" \
        --worker-urls \
        "${BACKEND_URLS[@]}" \
        --policy consistent_hash \
        --worker-startup-timeout-secs 300 \
        >/dev/null

    wait_for_service "routing-vllm-router" "vLLM router"
}
