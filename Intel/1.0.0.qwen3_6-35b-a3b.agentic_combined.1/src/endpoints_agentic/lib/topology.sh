#!/usr/bin/env bash
# Vendor-neutral worker planning; hardware probes live in the selected family.
# Shared arrays and profile variables are consumed by sourced modules.
# shellcheck disable=SC2034,SC2153

initialize_worker_topology() {
    WORKER_IDS=()
    WORKER_ROLES=()
    WORKER_NUMA_GROUPS=()
    WORKER_PARALLEL_WIDTHS=()
    WORKER_OMP_THREADS_BINDS=()
    WORKER_GPU_IDS=()
    WORKER_PORTS=()
    WORKER_NIXL_SIDE_CHANNEL_PORTS=()
    WORKER_KV_CACHE_AGGREGATE_GIB=()
    WORKER_KV_CACHE_RANK_GIB=()
    BACKEND_URLS=()
    PREFILL_BACKEND_URLS=()
    DECODE_BACKEND_URLS=()
    TOPOLOGY_TOTAL_RANKS=0
    TOPOLOGY_CAPACITY_MODE=""
    TOPOLOGY_USED_NUMA_NODES=()
    TOPOLOGY_UNUSED_NUMA_NODES=()
}

topology_validate_positive_integer() {
    local variable_name=$1
    local value="${!variable_name:-}"
    local max_value=2147483647
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
        || fail "${variable_name} must be a positive decimal integer."
    # Equal-length decimal strings are compared lexically to avoid overflow.
    # shellcheck disable=SC2071
    if (( ${#value} > ${#max_value} )) \
        || { (( ${#value} == ${#max_value} )) \
            && [[ "${value}" > "${max_value}" ]]; }; then
        fail "${variable_name} must not exceed ${max_value}."
    fi
    printf -v "${variable_name}" '%d' "$((10#${value}))"
}

topology_validate_cpu_configuration() {
    case "${TOPOLOGY_MODE:-replica}" in
        replica)
            topology_validate_positive_integer TP_SIZE
            topology_validate_positive_integer REPLICA_COUNT
            ;;
        pd)
            topology_validate_positive_integer PREFILL_TP_SIZE
            topology_validate_positive_integer PREFILL_REPLICA_COUNT
            topology_validate_positive_integer DECODE_TP_SIZE
            topology_validate_positive_integer DECODE_REPLICA_COUNT
            ;;
        *)
            fail "Unsupported topology mode: ${TOPOLOGY_MODE:-replica}"
            ;;
    esac
}

configure_topology() {
    NUMA_NODES=("$@")
    local node_count=${#NUMA_NODES[@]}
    (( node_count > 0 )) \
        || fail "No online CPU-bearing NUMA nodes were discovered."
    initialize_worker_topology

    local node
    local -A seen_nodes=()
    for node in "${NUMA_NODES[@]}"; do
        [[ "${node}" =~ ^[0-9]+$ ]] \
            || fail "NUMA node IDs must be non-negative integers: ${node}"
        [[ -z "${seen_nodes[${node}]:-}" ]] \
            || fail "NUMA node ${node} was provided more than once."
        seen_nodes["${node}"]=1
    done

    local topology_mode="${TOPOLOGY_MODE:-replica}"
    topology_validate_cpu_configuration

    local parallel_width
    local replica_index rank group_start numa_group
    local replica_count role
    local prefill_ranks decode_ranks total_ranks
    if [[ "${topology_mode}" == "pd" ]]; then
        [[ "${NIXL_SIDE_CHANNEL_BASE_PORT:-}" =~ ^[0-9]+$ ]] \
            || fail "NIXL_SIDE_CHANNEL_BASE_PORT must be set to a valid TCP port for P/D topology."

        if (( PREFILL_TP_SIZE > node_count )); then
            fail "P/D prefill TP=${PREFILL_TP_SIZE} exceeds discovered NUMA nodes (${node_count})."
        fi
        if (( PREFILL_REPLICA_COUNT > node_count / PREFILL_TP_SIZE )); then
            fail "P/D prefill replica count ${PREFILL_REPLICA_COUNT} exceeds NUMA capacity for TP=${PREFILL_TP_SIZE} (${node_count} nodes)."
        fi
        prefill_ranks=$((PREFILL_TP_SIZE * PREFILL_REPLICA_COUNT))
        if (( DECODE_TP_SIZE > node_count )); then
            fail "P/D decode TP=${DECODE_TP_SIZE} exceeds discovered NUMA nodes (${node_count})."
        fi
        if (( DECODE_REPLICA_COUNT > node_count / DECODE_TP_SIZE )); then
            fail "P/D decode replica count ${DECODE_REPLICA_COUNT} exceeds NUMA capacity for TP=${DECODE_TP_SIZE} (${node_count} nodes)."
        fi
        decode_ranks=$((DECODE_TP_SIZE * DECODE_REPLICA_COUNT))
        (( prefill_ranks <= node_count - decode_ranks )) \
            || fail "P/D total ranks ($((prefill_ranks + decode_ranks))) exceed discovered NUMA nodes (${node_count})."
        total_ranks=$((prefill_ranks + decode_ranks))
        TOPOLOGY_TOTAL_RANKS="${total_ranks}"
        TOPOLOGY_CAPACITY_MODE="numa_rank"
        TOPOLOGY_USED_NUMA_NODES=("${NUMA_NODES[@]:0:total_ranks}")
        TOPOLOGY_UNUSED_NUMA_NODES=("${NUMA_NODES[@]:total_ranks}")

        for role in prefill decode; do
            if [[ "${role}" == "prefill" ]]; then
                parallel_width="${PREFILL_TP_SIZE}"
                replica_count="${PREFILL_REPLICA_COUNT}"
            else
                parallel_width="${DECODE_TP_SIZE}"
                replica_count="${DECODE_REPLICA_COUNT}"
            fi
            for ((replica_index = 0; replica_index < replica_count; replica_index++)); do
                if [[ "${role}" == "prefill" ]]; then
                    group_start=$((replica_index * PREFILL_TP_SIZE))
                else
                    group_start=$((prefill_ranks + replica_index * DECODE_TP_SIZE))
                fi
                numa_group=""
                for ((rank = 0; rank < parallel_width; rank++)); do
                    [[ -z "${numa_group}" ]] \
                        || numa_group+=","
                    numa_group+="${NUMA_NODES[$((group_start + rank))]}"
                done
                WORKER_IDS+=("${role}${replica_index}")
                WORKER_ROLES+=("${role}")
                WORKER_NUMA_GROUPS+=("${numa_group}")
                WORKER_PARALLEL_WIDTHS+=("${parallel_width}")
                WORKER_OMP_THREADS_BINDS+=("auto")
            done
        done
    elif [[ "${topology_mode}" == "replica" ]]; then
        if declare -F platform_configure_cpu_workers >/dev/null; then
            platform_configure_cpu_workers
        else
            parallel_width="${TP_SIZE}"
            replica_count="${REPLICA_COUNT}"
            if (( parallel_width > node_count )); then
                fail "Regular TP=${parallel_width} exceeds discovered NUMA nodes (${node_count})."
            fi
            if (( replica_count > node_count / parallel_width )); then
                fail "Regular replica count ${replica_count} exceeds NUMA capacity for TP=${parallel_width} (${node_count} nodes)."
            fi
            total_ranks=$((parallel_width * replica_count))
            TOPOLOGY_TOTAL_RANKS="${total_ranks}"
            TOPOLOGY_CAPACITY_MODE="numa_rank"
            TOPOLOGY_USED_NUMA_NODES=("${NUMA_NODES[@]:0:total_ranks}")
            TOPOLOGY_UNUSED_NUMA_NODES=("${NUMA_NODES[@]:total_ranks}")
            for ((replica_index = 0; replica_index < replica_count; replica_index++)); do
                group_start=$((replica_index * parallel_width))
                numa_group=""
                for ((rank = 0; rank < parallel_width; rank++)); do
                    [[ -z "${numa_group}" ]] \
                        || numa_group+=","
                    numa_group+="${NUMA_NODES[$((group_start + rank))]}"
                done
                WORKER_NUMA_GROUPS+=("${numa_group}")
                WORKER_PARALLEL_WIDTHS+=("${parallel_width}")
                WORKER_OMP_THREADS_BINDS+=("auto")
            done
        fi
    else
        fail "Unsupported topology mode: ${topology_mode}"
    fi

    local worker_index port backend_url nixl_port
    for worker_index in "${!WORKER_NUMA_GROUPS[@]}"; do
        port=$((WORKER_BASE_PORT + worker_index))
        (( port <= 65535 )) \
            || fail "Generated worker port ${port} exceeds the TCP port range."
        backend_url="http://127.0.0.1:${port}"

        if [[ "${topology_mode}" == "replica" ]]; then
            WORKER_IDS+=("worker${worker_index}")
            WORKER_ROLES+=("replica")
        fi
        WORKER_PORTS+=("${port}")
        BACKEND_URLS+=("${backend_url}")

        if [[ "${topology_mode}" == "pd" ]]; then
            nixl_port=$((NIXL_SIDE_CHANNEL_BASE_PORT + worker_index))
            (( nixl_port >= 1 && nixl_port <= 65535 )) \
                || fail "Generated NIXL side-channel port ${nixl_port} exceeds the TCP port range."
            WORKER_NIXL_SIDE_CHANNEL_PORTS+=("${nixl_port}")
            role="${WORKER_ROLES[$worker_index]}"
            if [[ "${role}" == "prefill" ]]; then
                PREFILL_BACKEND_URLS+=("${backend_url}")
            else
                DECODE_BACKEND_URLS+=("${backend_url}")
            fi
        fi
    done

    if [[ "${topology_mode}" == "pd" ]]; then
        log "NUMA P/D topology: nodes=${NUMA_NODES[*]}; roles=${WORKER_ROLES[*]}; TP=${WORKER_PARALLEL_WIDTHS[*]}; groups=${WORKER_NUMA_GROUPS[*]}; ports=${WORKER_PORTS[*]}; NIXL ports=${WORKER_NIXL_SIDE_CHANNEL_PORTS[*]}."
    else
        log "NUMA replica topology: host profile=${HOST_CPU_PROFILE:-generic}; nodes=${NUMA_NODES[*]}; total ranks=${TOPOLOGY_TOTAL_RANKS}; TP=${WORKER_PARALLEL_WIDTHS[*]}; groups=${WORKER_NUMA_GROUPS[*]}; OMP bindings=${WORKER_OMP_THREADS_BINDS[*]}; ports=${WORKER_PORTS[*]}."
    fi
}

topology_index_group() {
    local start=$1
    local width=$2
    local group="" separator="" offset
    for ((offset = 0; offset < width; offset++)); do
        group+="${separator}$((start + offset))"
        separator=","
    done
    printf '%s' "${group}"
}

configure_gpu_topology() {
    local device_label="${HOST_GPU_NAME:-accelerator}"
    (( HOST_GPU_COUNT > 0 )) \
        || fail "The ${device_label} stack profile requires at least one ${device_label} device."

    platform_resolve_parallelism "${device_label}"
    local parallel_width="${ACCEL_TP_SIZE}"
    local replica_count="${ACCEL_REPLICA_COUNT}"
    NUMA_NODES=()
    initialize_worker_topology

    local worker_index port
    for ((worker_index = 0; worker_index < replica_count; worker_index++)); do
        port=$((WORKER_BASE_PORT + worker_index))
        (( port <= 65535 )) \
            || fail "Generated worker port ${port} exceeds the TCP port range."
        WORKER_IDS+=("worker${worker_index}")
        WORKER_ROLES+=("replica")
        WORKER_PARALLEL_WIDTHS+=("${parallel_width}")
        WORKER_GPU_IDS+=("$(topology_index_group \
            "$((worker_index * parallel_width))" "${parallel_width}")")
        WORKER_PORTS+=("${port}")
        BACKEND_URLS+=("http://127.0.0.1:${port}")
    done

    log "${device_label} replica topology: devices=${HOST_GPU_COUNT}x ${HOST_GPU_NAME:-unknown}; TP=${parallel_width}; replicas=${replica_count}; groups=${WORKER_GPU_IDS[*]}; ports=${WORKER_PORTS[*]}."
}

discover_topology() {
    local -a discovered_nodes=()
    if [[ "${TOPOLOGY_MODE:-replica}" == "gpu_replica" ]]; then
        platform_discover_devices
        configure_gpu_topology
        return
    fi

    platform_discover_devices
    mapfile -t discovered_nodes < <(
        lscpu --parse=NODE --online \
            | awk -F, '!/^#/ && $1 ~ /^[0-9]+$/ { print $1 }' \
            | sort -n -u
    )
    configure_topology "${discovered_nodes[@]}"
}
