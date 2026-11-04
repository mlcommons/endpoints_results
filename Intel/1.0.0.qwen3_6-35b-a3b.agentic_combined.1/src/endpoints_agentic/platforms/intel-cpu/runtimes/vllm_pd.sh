#!/usr/bin/env bash

# shellcheck source=components/runtimes/vllm_common.sh
source "${SCRIPT_DIR}/components/runtimes/vllm_common.sh"
VLLM_PYTHON_ENTRYPOINT="python"
VLLM_PD_KV_LEASE_DURATION_SECONDS="${VLLM_PD_KV_LEASE_DURATION_SECONDS:-300}"

runtime_validate() {
    vllm_configure_local_model
    runtime_validate_cpu
    local metadata_variable
    for metadata_variable in \
        VLLM_PD_NIXL_VERSION VLLM_PD_NIXL_COMMIT VLLM_PD_UCX_COMMIT; do
        [[ -n "${!metadata_variable:-}" ]] \
            || fail "${metadata_variable} must be non-empty for the P/D runtime image."
    done
    [[ "${VLLM_PD_NIXL_VERSION}" =~ ^[0-9]+(\.[0-9]+)+$ ]] \
        || fail "VLLM_PD_NIXL_VERSION must be a dotted numeric version."
    [[ "${VLLM_PD_NIXL_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
        || fail "VLLM_PD_NIXL_COMMIT must be a 40-character lowercase hexadecimal commit."
    [[ "${VLLM_PD_UCX_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
        || fail "VLLM_PD_UCX_COMMIT must be a 40-character lowercase hexadecimal commit."
    vllm_validate_worker_kv_cache_allocations
    local lease_duration="${VLLM_PD_KV_LEASE_DURATION_SECONDS}"
    local max_lease_duration=2147483647
    [[ "${lease_duration}" =~ ^[1-9][0-9]*$ ]] \
        || fail "VLLM_PD_KV_LEASE_DURATION_SECONDS must be a positive integer."
    if (( ${#lease_duration} > ${#max_lease_duration} )) \
        || { (( ${#lease_duration} == ${#max_lease_duration} )) \
            && (( lease_duration > max_lease_duration )); }; then
        fail "VLLM_PD_KV_LEASE_DURATION_SECONDS must not exceed ${max_lease_duration} seconds."
    fi
    VLLM_PD_KV_LEASE_DURATION_SECONDS=$(
        printf '%d' "${lease_duration}"
    )

    (( ${#WORKER_ROLES[@]} == ${#WORKER_IDS[@]} )) \
        || fail "P/D runtime requires one role per worker."
    local prefill_count=0 decode_count=0 role
    for role in "${WORKER_ROLES[@]}"; do
        case "${role}" in
            prefill)
                prefill_count=$((prefill_count + 1))
                ;;
            decode)
                decode_count=$((decode_count + 1))
                ;;
            *)
                fail "P/D runtime worker role must be prefill or decode: ${role}."
                ;;
        esac
    done
    (( prefill_count > 0 )) \
        || fail "P/D runtime requires at least one prefill worker."
    (( decode_count > 0 )) \
        || fail "P/D runtime requires at least one decode worker."
    (( ${#WORKER_NIXL_SIDE_CHANNEL_PORTS[@]} == ${#WORKER_IDS[@]} )) \
        || fail "P/D runtime requires one NIXL side-channel port per worker."

    local index
    for index in "${!WORKER_IDS[@]}"; do
        reserve_port "${WORKER_PORTS[$index]}" "runtime ${WORKER_IDS[$index]}"
        reserve_port \
            "${WORKER_NIXL_SIDE_CHANNEL_PORTS[$index]}" \
            "NIXL side channel ${WORKER_IDS[$index]}"
        if vllm_pd_uses_llm_d_sidecar \
            && [[ "${WORKER_ROLES[$index]}" == "decode" ]]; then
            local sidecar_port
            sidecar_port=$(vllm_pd_sidecar_port \
                "${WORKER_ROLES[$index]}" "${WORKER_PORTS[$index]}")
            reserve_port "${sidecar_port}" \
                "llm-d decode sidecar ${WORKER_IDS[$index]}"
        fi
    done
}

runtime_validate_cpu() {
    if [[ -n "${HOST_CPU_VENDOR:-}" ]]; then
        [[ "${HOST_CPU_VENDOR}" == "GenuineIntel" ]] \
            || fail "The Intel CPU P/D runtime requires an Intel host CPU."
    fi
    case "${MODEL_CPU_REQUIREMENT:-generic}" in
        generic|amx_fp8)
            ;;
        *)
            fail "Unsupported MODEL_CPU_REQUIREMENT: ${MODEL_CPU_REQUIREMENT}"
            ;;
    esac
}

vllm_pd_uses_llm_d_sidecar() {
    [[ "${ROUTING_ADAPTER:-}" == "llm_d" \
        && "${TOPOLOGY_MODE:-replica}" == "pd" ]]
}

vllm_pd_sidecar_port() {
    local role=$1
    local model_server_port=$2
    pd_routed_backend_port "${role}" "${model_server_port}"
}

vllm_pd_validate_runtime_capability() {
    docker run --rm \
        --network none \
        --entrypoint "${VLLM_PYTHON_ENTRYPOINT}" \
        "${RUNTIME_IMAGE_ID:-${RUNTIME_IMAGE}}" \
        -c 'import inspect, vllm; from vllm.platforms.cpu import CpuPlatform; source = inspect.getsource(CpuPlatform.check_and_update_config); assert "VLLM_SSM_CONV_STATE_LAYOUT\"] = \"SD\"" not in source, f"vLLM CPU {vllm.__version__} lacks the upstream DS state-layout support required for Qwen P/D"' \
        >/dev/null
}

vllm_validate_runtime_capability() {
    vllm_pd_validate_runtime_capability
}

runtime_prepare_image() {
    if [[ -n "${SUBMISSION_EXPECTED_RUNTIME_IMAGE_ID:-}" ]]; then
        log "Reusing prepared P/D vLLM CPU image ${RUNTIME_IMAGE} with ID ${SUBMISSION_EXPECTED_RUNTIME_IMAGE_ID}."
        vllm_assert_runtime_image_id
    else
        log "Building P/D vLLM CPU image ${RUNTIME_IMAGE} from ${RUNTIME_BASE_IMAGE}."
        vllm_build_derived_runtime \
            --build-arg "NIXL_VERSION=${VLLM_PD_NIXL_VERSION}" \
            --build-arg "NIXL_COMMIT=${VLLM_PD_NIXL_COMMIT}" \
            --build-arg "UCX_COMMIT=${VLLM_PD_UCX_COMMIT}"
        vllm_assert_runtime_image_id
    fi

    vllm_validate_runtime_capability
}

runtime_assert_image_prepared() {
    vllm_assert_runtime_image_id
    vllm_validate_runtime_capability
}

runtime_prepare_assets() {
    runtime_prepare_image
    vllm_download_model
}

runtime_assert_prepared() {
    runtime_assert_image_prepared
    vllm_assert_cached_model
}

runtime_start_workers() {
    local index worker_id role port numa_group parallel_width container_name
    local runtime_worker_image
    local sidecar_port nixl_port worker_kv_cache_space_gib
    local worker_kv_cache_aggregate_gib
    local kv_role kv_transfer_config sidecar_container
    local omp_threads_bind
    local effective_model_name="${EFFECTIVE_MODEL_NAME:-${MODEL_NAME}}"
    local -a server_args=()
    runtime_worker_image=$(vllm_runtime_worker_image_ref)
    for index in "${!WORKER_IDS[@]}"; do
        worker_id="${WORKER_IDS[$index]}"
        role="${WORKER_ROLES[$index]}"
        port="${WORKER_PORTS[$index]}"
        numa_group="${WORKER_NUMA_GROUPS[$index]}"
        parallel_width="${WORKER_PARALLEL_WIDTHS[$index]}"
        omp_threads_bind="${WORKER_OMP_THREADS_BINDS[$index]:-auto}"
        nixl_port="${WORKER_NIXL_SIDE_CHANNEL_PORTS[$index]}"
        container_name="${CONTAINER_PREFIX}-${worker_id}"
        worker_kv_cache_aggregate_gib="${WORKER_KV_CACHE_AGGREGATE_GIB[$index]}"
        worker_kv_cache_space_gib="${WORKER_KV_CACHE_RANK_GIB[$index]}"

        if [[ "${role}" == "prefill" ]]; then
            kv_role="kv_producer"
        else
            kv_role="kv_consumer"
        fi
        kv_transfer_config=$(printf \
            '{"kv_connector":"NixlConnector","kv_role":"%s","kv_buffer_device":"cpu","kv_load_failure_policy":"fail","kv_connector_extra_config":{"kv_lease_duration":%d}}' \
            "${kv_role}" "${VLLM_PD_KV_LEASE_DURATION_SECONDS}")
        vllm_build_server_args \
            server_args "${effective_model_name}" "${port}" "${parallel_width}"
        server_args+=(--kv-transfer-config "${kv_transfer_config}")

        register_ready_service \
            "runtime-${worker_id}" \
            "${container_name}" \
            "http://127.0.0.1:${port}/health" \
            1800

        log "Starting vLLM ${worker_id}: role=${role}, NUMA=${numa_group}, TP=${parallel_width}, KV cache=${worker_kv_cache_space_gib} GiB/rank (${worker_kv_cache_aggregate_gib} GiB aggregate), port=${port}, NIXL port=${nixl_port}."
        docker run -d \
            --name "${container_name}" \
            --network host \
            --ipc host \
            -e "CPU_VISIBLE_MEMORY_NODES=${numa_group}" \
            -e "VLLM_CPU_KVCACHE_SPACE=${worker_kv_cache_space_gib}" \
            -e "VLLM_CPU_OMP_THREADS_BIND=${omp_threads_bind}" \
            -e "VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1" \
            -e "VLLM_NIXL_SIDE_CHANNEL_PORT=${nixl_port}" \
            -e HF_HUB_OFFLINE=1 \
            -e TRANSFORMERS_OFFLINE=1 \
            -v "${MODEL_CACHE_DIR}:/root/.cache/huggingface" \
            "${VLLM_MODEL_MOUNT_ARGS[@]}" \
            --entrypoint vllm \
            "${runtime_worker_image}" \
            serve \
            "${server_args[@]}" \
            >/dev/null

        if vllm_pd_uses_llm_d_sidecar && [[ "${role}" == "decode" ]]; then
            sidecar_port=$(vllm_pd_sidecar_port "${role}" "${port}")
            sidecar_container="${CONTAINER_PREFIX}-${worker_id}-sidecar"
            register_ready_service \
                "runtime-${worker_id}-sidecar" \
                "${sidecar_container}" \
                "http://127.0.0.1:${sidecar_port}/health" \
                120
        fi
    done

    for index in "${!WORKER_IDS[@]}"; do
        worker_id="${WORKER_IDS[$index]}"
        wait_for_service "runtime-${worker_id}" "vLLM ${worker_id}"
    done

    for index in "${!WORKER_IDS[@]}"; do
        worker_id="${WORKER_IDS[$index]}"
        role="${WORKER_ROLES[$index]}"
        port="${WORKER_PORTS[$index]}"
        if vllm_pd_uses_llm_d_sidecar && [[ "${role}" == "decode" ]]; then
            sidecar_port=$(vllm_pd_sidecar_port "${role}" "${port}")
            sidecar_container="${CONTAINER_PREFIX}-${worker_id}-sidecar"
            log "Starting llm-d decode sidecar for ${worker_id}: sidecar port=${sidecar_port}, model server port=${port}."
            docker run -d \
                --name "${sidecar_container}" \
                --network host \
                "${LLMD_SIDECAR_IMAGE}" \
                --port="${sidecar_port}" \
                --model-server-port="${port}" \
                --secure-proxy=false \
                >/dev/null
            wait_for_service \
                "runtime-${worker_id}-sidecar" \
                "llm-d decode sidecar ${worker_id}"
        fi
    done
}
