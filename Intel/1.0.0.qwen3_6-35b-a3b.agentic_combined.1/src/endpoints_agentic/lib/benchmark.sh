#!/usr/bin/env bash

declare -Ag RESOLVED_BENCHMARK_VALUES=()

BENCHMARK_TUNABLE_SPECS=(
    "TARGET_CONCURRENCY|target_concurrency|positive_integer"
    "PERFORMANCE_TRAJECTORY_COUNT|num_trajectories_to_issue|positive_integer"
    "SWEBENCH_INSTANCE_COUNT|num_instances|positive_integer"
    "TURN_TIMEOUT_S|turn_timeout_s|positive_number"
    "MAX_ISSUE_DURATION_MS|max_issue_duration_ms|duration_or_null"
    "WARMUP_CONNECTIONS|warmup_connections|warmup_connections"
    "MAX_IDLE_TIME|max_idle_time|number"
    "TEMPERATURE|temperature|optional_number"
    "TOP_K|top_k|optional_integer"
    "TOP_P|top_p|optional_number"
    "REPETITION_PENALTY|repetition_penalty|optional_number"
    "PRESENCE_PENALTY|presence_penalty|optional_number"
    "MAX_NEW_TOKENS|max_new_tokens|positive_integer"
    "PRESERVE_THINKING|preserve_thinking|boolean"
    "ENABLE_SALT|enable_salt|boolean"
    "INJECT_TOOL_DELAY|inject_tool_delay|boolean"
)

benchmark_prepare_image() {
    validate_digest_image ENDPOINT_BASE_IMAGE "benchmark client"
    log "Pulling pinned endpoint base image ${ENDPOINT_BASE_IMAGE}."
    docker pull "${ENDPOINT_BASE_IMAGE}"
    log "Building endpoint client and SWE-bench service image ${ENDPOINT_IMAGE}."
    docker build \
        --build-arg "ENDPOINT_BASE_IMAGE=${ENDPOINT_BASE_IMAGE}" \
        --build-arg "AGENTIC_DATASET_SHA256=${AGENTIC_DATASET_SHA256}" \
        --build-arg "MLCFLOW_VERSION=${MLCFLOW_VERSION}" \
        --build-arg "MLC_AUTOMATIONS_REF=${MLC_AUTOMATIONS_REF}" \
        --file "${SCRIPT_DIR}/docker/Dockerfile.endpoints" \
        --tag "${ENDPOINT_IMAGE}" \
        "${SCRIPT_DIR}"
}

benchmark_prepare_assets() {
    benchmark_prepare_image
    log "Pulling fixed SWE-bench network-probe image ${SWEBENCH_NETWORK_PROBE_IMAGE}."
    docker pull "${SWEBENCH_NETWORK_PROBE_IMAGE}"
}

benchmark_assert_prepared() {
    require_local_image "${ENDPOINT_IMAGE}"
    local actual_mlcflow_version actual_automations_ref
    actual_mlcflow_version=$(docker image inspect \
        --format '{{ index .Config.Labels "org.mlcommons.mlcflow.version" }}' \
        "${ENDPOINT_IMAGE}" 2>/dev/null) \
        || fail "Could not inspect MLCflow version label on ${ENDPOINT_IMAGE}. Run bash preflight.sh to rebuild the image."
    [[ "${actual_mlcflow_version}" == "${MLCFLOW_VERSION}" ]] \
        || fail "Endpoint image ${ENDPOINT_IMAGE} has MLCflow version label '${actual_mlcflow_version:-missing}', expected '${MLCFLOW_VERSION}'. Run bash preflight.sh to rebuild the image."

    actual_automations_ref=$(docker image inspect \
        --format '{{ index .Config.Labels "org.mlcommons.mlperf-automations.revision" }}' \
        "${ENDPOINT_IMAGE}" 2>/dev/null) \
        || fail "Could not inspect MLC automations revision label on ${ENDPOINT_IMAGE}. Run bash preflight.sh to rebuild the image."
    [[ "${actual_automations_ref}" == "${MLC_AUTOMATIONS_REF}" ]] \
        || fail "Endpoint image ${ENDPOINT_IMAGE} has MLC automations revision label '${actual_automations_ref:-missing}', expected '${MLC_AUTOMATIONS_REF}'. Run bash preflight.sh to rebuild the image."
    require_local_image "${SWEBENCH_NETWORK_PROBE_IMAGE}"
}

require_single_yaml_key() {
    local file=$1
    local key=$2
    local count
    count=$(grep -Ec "^[[:space:]]+${key}:[[:space:]]*" "${file}" || true)
    [[ "${count}" -eq 1 ]] \
        || fail "Benchmark config must contain exactly one ${key} entry (found ${count})."
}

require_safe_benchmark_scalar() {
    local variable_name=$1
    local value=$2

    [[ -n "${value}" && "${value}" =~ ^[-+A-Za-z0-9_.]+$ ]] \
        || fail "${variable_name} must be a nonempty scalar token safe for config rendering."
}

read_yaml_scalar() {
    local file=$1
    local key=$2
    awk -v key="${key}" '
        $0 ~ "^[[:space:]]+" key ":[[:space:]]*" {
            value = $0
            sub("^[[:space:]]+" key ":[[:space:]]*", "", value)
            sub("[[:space:]]*#.*$", "", value)
            sub("[[:space:]]+$", "", value)
            print value
        }
    ' "${file}"
}

render_runtime_config() {
    local token=$1
    local config_template="${BENCHMARK_CONFIG_TEMPLATE:-${SCRIPT_DIR}/configs/qwen_agentic_benchmark.yaml}"
    local placeholder spec variable_name yaml_key value
    local default_performance_trajectory_count=""
    local -a sed_args=(
        -e "s#__SWEBENCH_SERVICE_AUTH_TOKEN__#${token}#g"
        -e "s#__SWEBENCH_PORT__#${SWEBENCH_PORT}#g"
        -e "s#__ENDPOINT_URL__#${PUBLIC_ENDPOINT_URL}#g"
    )

    resolve_effective_target_concurrency
    if [[ ! -v PERFORMANCE_TRAJECTORY_COUNT \
        && ( "${BENCHMARK_MODE}" == "perf" || "${BENCHMARK_MODE}" == "both" ) ]]; then
        default_performance_trajectory_count=$((613 \
            * ((5 * EFFECTIVE_TARGET_CONCURRENCY + 612) / 613)))
    fi
    [[ -n "${PUBLIC_ENDPOINT_URL}" ]] \
        || fail "Routing adapter did not set PUBLIC_ENDPOINT_URL."
    for placeholder in \
        __SWEBENCH_SERVICE_AUTH_TOKEN__ \
        __SWEBENCH_PORT__ \
        __ENDPOINT_URL__; do
        require_single_literal "${config_template}" "${placeholder}"
    done

    for spec in "${BENCHMARK_TUNABLE_SPECS[@]}"; do
        IFS='|' read -r variable_name yaml_key _ <<<"${spec}"
        require_single_yaml_key "${config_template}" "${yaml_key}"
        if [[ "${variable_name}" == "PERFORMANCE_TRAJECTORY_COUNT" \
            && -n "${default_performance_trajectory_count}" ]]; then
            value="${default_performance_trajectory_count}"
        elif [[ -v "${variable_name}" ]]; then
            value="${!variable_name}"
        else
            continue
        fi
        require_safe_benchmark_scalar "${variable_name}" "${value}"
        sed_args+=(
            -e "s|^([[:space:]]+${yaml_key}:[[:space:]]*)[^#[:space:]]+([[:space:]]*(#.*)?)\$|\\1${value}\\2|"
        )
    done

    sed -E "${sed_args[@]}" "${config_template}" >"${RUNTIME_CONFIG}"
    for placeholder in \
        __SWEBENCH_SERVICE_AUTH_TOKEN__ \
        __SWEBENCH_PORT__ \
        __ENDPOINT_URL__; do
        if grep -Fq "${placeholder}" "${RUNTIME_CONFIG}"; then
            fail "Failed to render benchmark config placeholder ${placeholder}."
        fi
    done

    RESOLVED_BENCHMARK_VALUES=()
    for spec in "${BENCHMARK_TUNABLE_SPECS[@]}"; do
        IFS='|' read -r variable_name yaml_key _ <<<"${spec}"
        value=$(read_yaml_scalar "${RUNTIME_CONFIG}" "${yaml_key}")
        RESOLVED_BENCHMARK_VALUES["${variable_name}"]="${value}"
    done
    [[ "${RESOLVED_BENCHMARK_VALUES[TARGET_CONCURRENCY]}" \
        == "${EFFECTIVE_TARGET_CONCURRENCY}" ]] \
        || fail "Rendered target concurrency does not match the resolved target concurrency."
    log "Rendered benchmark config for ${BENCHMARK_MODE}: target concurrency ${RESOLVED_BENCHMARK_VALUES[TARGET_CONCURRENCY]}, ${RESOLVED_BENCHMARK_VALUES[PERFORMANCE_TRAJECTORY_COUNT]} performance trajectories, and ${RESOLVED_BENCHMARK_VALUES[SWEBENCH_INSTANCE_COUNT]} SWE-bench instances."
}

validate_runtime_config() {
    docker run --rm \
        --network none \
        --user "$(id -u):$(id -g)" \
        --env HOME=/tmp \
        -v "${RUNTIME_CONFIG}:/workspace/qwen_agentic_benchmark.yaml:ro" \
        --entrypoint inference-endpoint \
        "${ENDPOINT_IMAGE}" \
        validate-yaml \
        --config /workspace/qwen_agentic_benchmark.yaml
}

write_benchmark_manifest_entries() {
    local spec variable_name yaml_key
    for spec in "${BENCHMARK_TUNABLE_SPECS[@]}"; do
        IFS='|' read -r variable_name yaml_key _ <<<"${spec}"
        printf '%s=%q\n' \
            "${variable_name}" \
            "${RESOLVED_BENCHMARK_VALUES[$variable_name]}"
    done
}

run_endpoint_client() {
    local container_name="${CONTAINER_PREFIX}-client"
    register_service \
        "benchmark-client" \
        "${container_name}" \
        false

    log "Running the endpoint benchmark with target concurrency ${RESOLVED_BENCHMARK_VALUES[TARGET_CONCURRENCY]}, --mode ${BENCHMARK_MODE}, ${RESOLVED_BENCHMARK_VALUES[PERFORMANCE_TRAJECTORY_COUNT]} performance trajectories, and ${RESOLVED_BENCHMARK_VALUES[SWEBENCH_INSTANCE_COUNT]} SWE-bench instances."
    docker run -d \
        --name "${container_name}" \
        --network host \
        --ipc host \
        --user "$(id -u):$(id -g)" \
        --env HOME=/tmp \
        --workdir /workspace \
        -v "${RUNTIME_CONFIG}:/workspace/qwen_agentic_benchmark.yaml:ro" \
        -v "${RUN_DIR}:/output" \
        --entrypoint inference-endpoint \
        "${ENDPOINT_IMAGE}" \
        benchmark from-config \
        --config /workspace/qwen_agentic_benchmark.yaml \
        --mode "${BENCHMARK_MODE}" \
        --report-dir /output/reports \
        >/dev/null

    begin_emon_run
    docker logs --follow "${container_name}" 2>&1 \
        | tee "${LOG_DIR}/inference_endpoint_client.log" &
    CLIENT_LOG_PID=$!

    local failed_service=""
    while [[ "$(docker inspect -f '{{.State.Running}}' "${container_name}" 2>/dev/null || true)" == "true" ]]; do
        failed_service=$(first_failed_critical_service || true)
        if [[ -n "${failed_service}" ]]; then
            docker stop --time 10 "${container_name}" >/dev/null 2>&1 || true
            break
        fi
        tick_emon
        if [[ "${EMON_STATE}" == collecting ]]; then
            sleep 1
        else
            sleep 5
        fi
    done

    local client_exit_code
    client_exit_code=$(docker inspect -f '{{.State.ExitCode}}' "${container_name}" 2>/dev/null || printf '1')
    if [[ -n "${failed_service}" || "${client_exit_code}" != 0 ]]; then
        finish_emon benchmark_failed
    else
        finish_emon benchmark_completed
    fi
    wait "${CLIENT_LOG_PID}" || true
    CLIENT_LOG_PID=""

    if [[ -n "${failed_service}" ]]; then
        local failed_index failed_container failed_state
        failed_index=$(service_index "${failed_service}")
        failed_container="${SERVICE_CONTAINERS[$failed_index]}"
        failed_state=$(docker inspect "${failed_container}" \
            --format 'exit={{.State.ExitCode}}, oom={{.State.OOMKilled}}, error={{json .State.Error}}' \
            2>/dev/null || printf 'state unavailable')
        fail "Critical service ${failed_service} exited during the benchmark (${failed_state})."
    fi
    [[ "${client_exit_code}" -eq 0 ]] \
        || fail "Endpoint benchmark client exited with status ${client_exit_code}."
}
