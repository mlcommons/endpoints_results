#!/usr/bin/env bash

# shellcheck source=components/runtimes/vllm_common.sh
source "${SCRIPT_DIR}/components/runtimes/vllm_common.sh"
VLLM_PYTHON_ENTRYPOINT="python"

runtime_validate() {
    vllm_configure_local_model
    runtime_validate_cpu
    vllm_validate_worker_kv_cache_allocations

    local index
    for index in "${!WORKER_IDS[@]}"; do
        reserve_port "${WORKER_PORTS[$index]}" "runtime ${WORKER_IDS[$index]}"
    done
}

runtime_cpu_build_args() {
    :
}

runtime_prepare_image() {
    if [[ -n "${SUBMISSION_EXPECTED_RUNTIME_IMAGE_ID:-}" ]]; then
        log "Reusing prepared vLLM runtime image ${RUNTIME_IMAGE} with ID ${SUBMISSION_EXPECTED_RUNTIME_IMAGE_ID}."
        vllm_assert_runtime_image_id
    else
        if [[ -n "${RUNTIME_DOCKERFILE:-}" ]]; then
            log "Building thin vLLM runtime image ${RUNTIME_IMAGE} from ${RUNTIME_BASE_IMAGE}."
            local -a build_args=()
            runtime_cpu_build_args build_args
            vllm_build_derived_runtime "${build_args[@]}"
        else
            if docker image inspect "${RUNTIME_IMAGE}" >/dev/null 2>&1; then
                log "Reusing local vLLM CPU image ${RUNTIME_IMAGE}."
            else
                log "Pulling upstream vLLM CPU image ${RUNTIME_IMAGE}."
                docker pull "${RUNTIME_IMAGE}"
            fi
        fi
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
    local index worker_id port numa_group parallel_width container_name
    local runtime_worker_image
    local worker_kv_cache_space_gib worker_kv_cache_aggregate_gib
    local omp_threads_bind
    local effective_model_name="${EFFECTIVE_MODEL_NAME:-${MODEL_NAME}}"
    local -a server_args=()
    runtime_worker_image=$(vllm_runtime_worker_image_ref)
    for index in "${!WORKER_IDS[@]}"; do
        worker_id="${WORKER_IDS[$index]}"
        port="${WORKER_PORTS[$index]}"
        numa_group="${WORKER_NUMA_GROUPS[$index]}"
        parallel_width="${WORKER_PARALLEL_WIDTHS[$index]}"
        omp_threads_bind="${WORKER_OMP_THREADS_BINDS[$index]:-auto}"
        container_name="${CONTAINER_PREFIX}-${worker_id}"
        worker_kv_cache_aggregate_gib="${WORKER_KV_CACHE_AGGREGATE_GIB[$index]}"
        worker_kv_cache_space_gib="${WORKER_KV_CACHE_RANK_GIB[$index]}"
        vllm_build_server_args \
            server_args "${effective_model_name}" "${port}" "${parallel_width}"
        server_args+=(--mamba-ssm-cache-dtype bfloat16)

        register_ready_service \
            "runtime-${worker_id}" \
            "${container_name}" \
            "http://127.0.0.1:${port}/health" \
            1800

        log "Starting vLLM ${worker_id}: NUMA=${numa_group}, TP=${parallel_width}, OMP binding=${omp_threads_bind}, KV cache=${worker_kv_cache_space_gib} GiB/rank (${worker_kv_cache_aggregate_gib} GiB aggregate), port=${port}."
        docker run -d \
            --name "${container_name}" \
            --network host \
            --ipc host \
            -e "CPU_VISIBLE_MEMORY_NODES=${numa_group}" \
            -e "VLLM_CPU_KVCACHE_SPACE=${worker_kv_cache_space_gib}" \
            -e "VLLM_CPU_OMP_THREADS_BIND=${omp_threads_bind}" \
            -e HF_HUB_OFFLINE=1 \
            -e TRANSFORMERS_OFFLINE=1 \
            -v "${MODEL_CACHE_DIR}:/root/.cache/huggingface" \
            "${VLLM_MODEL_MOUNT_ARGS[@]}" \
            --entrypoint vllm \
            "${runtime_worker_image}" \
            serve \
            "${server_args[@]}" \
            >/dev/null

    done

    for index in "${!WORKER_IDS[@]}"; do
        worker_id="${WORKER_IDS[$index]}"
        wait_for_service "runtime-${worker_id}" "vLLM ${worker_id}"
    done
}
