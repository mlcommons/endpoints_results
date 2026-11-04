#!/usr/bin/env bash
# Global run state is initialized here and consumed by other sourced modules.
# shellcheck disable=SC2034

load_mlcflow_configuration() {
    local mlcflow_config="${SCRIPT_DIR}/configs/mlcflow.conf"
    [[ -f "${mlcflow_config}" ]] \
        || fail "MLCflow configuration does not exist: ${mlcflow_config}"
    # This is a trusted file shipped in the benchmark tarball.
    # shellcheck disable=SC1090
    source "${mlcflow_config}"

    [[ "${MLCFLOW_VERSION:-}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
        || fail "MLCflow configuration must define a pinned MLCFLOW_VERSION."
    [[ "${MLC_AUTOMATIONS_REF:-}" =~ ^[0-9a-f]{40}$ ]] \
        || fail "MLCflow configuration must define a pinned 40-character MLC_AUTOMATIONS_REF."
}

load_benchmark_configuration() {
    local benchmark_config="${SCRIPT_DIR}/configs/benchmark.conf"
    [[ -f "${benchmark_config}" ]] \
        || fail "Benchmark configuration does not exist: ${benchmark_config}"
    # This is a trusted file shipped in the benchmark tarball.
    # shellcheck disable=SC1090
    source "${benchmark_config}"
    load_mlcflow_configuration

    local required_variable
    for required_variable in \
        ENDPOINT_BASE_IMAGE ENDPOINT_IMAGE AGENTIC_DATASET_SHA256 \
        SWEBENCH_NETWORK_PROBE_IMAGE SERVED_MODEL_NAME \
        SWEBENCH_PORT; do
        [[ -n "${!required_variable:-}" ]] \
            || fail "Benchmark configuration does not define ${required_variable}."
    done
}

load_vllm_configuration() {
    local vllm_config="${SCRIPT_DIR}/configs/vllm.conf"
    [[ -f "${vllm_config}" ]] \
        || fail "vLLM configuration does not exist: ${vllm_config}"
    # This is a trusted file shipped in the benchmark tarball.
    # shellcheck disable=SC1090
    source "${vllm_config}"

    [[ -n "${VLLM_VERSION:-}" ]] \
        || fail "vLLM configuration does not define VLLM_VERSION."
}

apply_submission_point_topology_overrides() {
    local regular_override=false
    local pd_override=false
    local target_variable override_variable
    local -a regular_target_variables=(TP_SIZE REPLICA_COUNT)
    local -a pd_target_variables=(
        PREFILL_TP_SIZE
        PREFILL_REPLICA_COUNT
        DECODE_TP_SIZE
        DECODE_REPLICA_COUNT
    )
    local -a all_target_variables=(
        "${regular_target_variables[@]}"
        "${pd_target_variables[@]}"
    )
    local -a active_target_variables=()

    for target_variable in "${all_target_variables[@]}"; do
        override_variable="SUBMISSION_POINT_${target_variable}"
        if [[ -v "${override_variable}" ]]; then
            config_validate_positive_integer_value \
                "${!override_variable}" "${override_variable}"
            case "${target_variable}" in
                PREFILL_*|DECODE_*) pd_override=true ;;
                *) regular_override=true ;;
            esac
        fi
    done

    case "${TOPOLOGY_MODE:-replica}" in
        replica|gpu_replica)
            [[ "${pd_override}" == "false" ]] \
                || fail "P/D submission-point overrides require a P/D topology profile."
            active_target_variables=("${regular_target_variables[@]}")
            ;;
        pd)
            [[ "${regular_override}" == "false" ]] \
                || fail "TP/replica submission-point overrides cannot be used with a P/D topology profile."
            active_target_variables=("${pd_target_variables[@]}")
            ;;
        *)
            [[ "${regular_override}" == "false" && "${pd_override}" == "false" ]] \
                || fail "Submission-point topology overrides do not support TOPOLOGY_MODE=${TOPOLOGY_MODE}."
            return 0
            ;;
    esac

    for target_variable in "${active_target_variables[@]}"; do
        override_variable="SUBMISSION_POINT_${target_variable}"
        if [[ -v "${override_variable}" ]]; then
            printf -v "${target_variable}" '%s' "${!override_variable}"
        fi
    done
}

load_stack_profile() {
    load_benchmark_configuration
    load_vllm_configuration
    if [[ -z "${STACK_PROFILE:-}" ]]; then
        local candidate priority family default_profile
        while read -r priority family default_profile; do
            candidate="${SCRIPT_DIR}/platforms/${family}"
            if (
                # shellcheck disable=SC1090,SC1091
                source "${candidate}/platform.conf" || exit 1
                # shellcheck disable=SC1090,SC1091
                source "${candidate}/topology.sh" || exit 1
                platform_probe
            ); then
                STACK_PROFILE="${default_profile}"
                break
            fi
        done < <(
            for candidate in "${SCRIPT_DIR}"/platforms/*/platform.conf; do
                [[ -f "${candidate}" ]] || continue
                (
                    # shellcheck disable=SC1090
                    source "${candidate}"
                    printf '%s %s %s\n' "${FAMILY_PRIORITY}" \
                        "$(basename "$(dirname "${candidate}")")" \
                        "${FAMILY_DEFAULT_PROFILE}"
                )
            done | sort -n
        )
        [[ -n "${STACK_PROFILE:-}" ]] \
            || fail "No included hardware family matches this host. Set STACK_PROFILE to an included profile after checking its hardware requirements."
        log "Auto-selected included stack profile ${STACK_PROFILE}."
    fi
    [[ "${STACK_PROFILE}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] \
        || fail "Invalid STACK_PROFILE name: ${STACK_PROFILE}"

    local profile_path="" candidate_profile
    for candidate_profile in "${SCRIPT_DIR}"/platforms/*/profiles/"${STACK_PROFILE}.conf"; do
        [[ -f "${candidate_profile}" ]] || continue
        [[ -z "${profile_path}" ]] \
            || fail "Duplicate STACK_PROFILE '${STACK_PROFILE}' in included families."
        profile_path="${candidate_profile}"
    done
    [[ -n "${profile_path}" ]] \
        || fail "Unknown or omitted STACK_PROFILE '${STACK_PROFILE}'; inspect platforms/*/profiles/ for included profiles."
    PLATFORM_DIR="$(dirname "$(dirname "${profile_path}")")"
    STACK_FAMILY="${PLATFORM_DIR##*/}"
    # Clear optional hooks and observations from a previously selected family.
    unset -f platform_configure_cpu_workers platform_resolve_parallelism \
        platform_write_manifest_entries
    HOST_GPU_COUNT=0 HOST_GPU_NAME=""
    HOST_CPU_PROFILE=generic HOST_CPU_VENDOR="" HOST_CPU_MODEL=""
    unset CALIBRATION_RECIPE_VERSION CALIBRATION_IMAGE \
        VLLM_CPU_IMAGE \
        VLLM_PD_IMAGE VLLM_PD_NIXL_VERSION VLLM_PD_NIXL_COMMIT \
        VLLM_PD_UCX_COMMIT RUNTIME_BASE_IMAGE RUNTIME_DOCKERFILE \
        RUNTIME_CAPABILITY MODEL_CPU_REQUIREMENT RUNTIME_IMAGE_DISPLAY_TAG \
        LLMD_SIDECAR_IMAGE LLMD_SIDECAR_IMAGE_DISPLAY_TAG \
        LLMD_SIDECAR_PORT_OFFSET ROUTING_PROXY_IMAGE ROUTING_PROXY_IMAGE_DISPLAY_TAG
    # shellcheck disable=SC1090,SC1091
    source "${PLATFORM_DIR}/platform.conf"
    # shellcheck disable=SC1090,SC1091
    source "${PLATFORM_DIR}/topology.sh"
    # Profiles are trusted files shipped in the benchmark tarball.
    RUNTIME_DEVICE="cpu"
    # TP/replica dimensions are owned by the selected profile. Clear any
    # inherited environment or previously sourced profile values before
    # loading the direct file assignments.
    unset TP_SIZE REPLICA_COUNT CPU_TP_SIZE CPU_REPLICA_COUNT \
        PREFILL_TP_SIZE PREFILL_REPLICA_COUNT \
        DECODE_TP_SIZE DECODE_REPLICA_COUNT \
        CPU_KV_CACHE_PER_WORKER_GIB \
        PREFILL_KV_CACHE_PER_WORKER_GIB \
        DECODE_KV_CACHE_PER_WORKER_GIB \
        RUNTIME_EXTRA_ARGS
    # shellcheck disable=SC1090
    source "${profile_path}"
    [[ -n "${RUNTIME_DEFAULT_WEIGHT_PRECISION:-}" ]] \
        || fail "Family ${STACK_FAMILY} does not define RUNTIME_DEFAULT_WEIGHT_PRECISION."
    [[ -v TP_SIZE || ! -v CPU_TP_SIZE ]] || TP_SIZE="${CPU_TP_SIZE}"
    [[ -v REPLICA_COUNT || ! -v CPU_REPLICA_COUNT ]] \
        || REPLICA_COUNT="${CPU_REPLICA_COUNT}"
    apply_submission_point_topology_overrides

    local required_variable
    for required_variable in \
        RUNTIME_ADAPTER ROUTING_ADAPTER RUNTIME_DEVICE RUNTIME_IMAGE \
        ROUTING_IMAGE MODEL_NAME PUBLIC_ENDPOINT_PORT WORKER_BASE_PORT; do
        [[ -n "${!required_variable:-}" ]] \
            || fail "Profile ${STACK_PROFILE} does not define ${required_variable}."
    done
    # shellcheck source=lib/calibration.sh
    source "${SCRIPT_DIR}/lib/calibration.sh"
}

load_component_adapters() {
    [[ "${RUNTIME_ADAPTER}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] \
        || fail "Invalid runtime adapter name: ${RUNTIME_ADAPTER}"
    [[ "${ROUTING_ADAPTER}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] \
        || fail "Invalid routing adapter name: ${ROUTING_ADAPTER}"

    local runtime_path="${PLATFORM_DIR}/runtimes/${RUNTIME_ADAPTER}.sh"
    local routing_path="${SCRIPT_DIR}/components/routing/${ROUTING_ADAPTER}.sh"
    [[ -f "${runtime_path}" ]] \
        || fail "Runtime adapter does not exist: ${runtime_path}"
    [[ -f "${routing_path}" ]] \
        || fail "Routing adapter does not exist: ${routing_path}"

    # shellcheck disable=SC1090
    source "${runtime_path}"
    # shellcheck disable=SC1090
    source "${routing_path}"

    local function_name
    for function_name in \
        runtime_validate runtime_prepare_assets runtime_assert_prepared \
        runtime_start_workers routing_validate routing_prepare_assets \
        routing_assert_prepared routing_configure routing_start; do
        declare -F "${function_name}" >/dev/null \
            || fail "Selected component adapters do not implement ${function_name}."
    done
}

initialize_base_configuration() {
    MIN_FREE_GIB="${MIN_FREE_GIB:-10}"
    if [[ ! -v VLLM_KV_OFFLOADING_SIZE_GIB ]]; then
        VLLM_KV_OFFLOADING_SIZE_GIB=0
    fi
    WORKER_KV_CACHE_AGGREGATE_GIB=()
    WORKER_KV_CACHE_RANK_GIB=()
    OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/run_output}"
    MODEL_CACHE_DIR="${HF_HOME:-}"

    BENCHMARK_MODE=""
    HOST_DOCKER_CONFIG_DIR=""
    SERVICE_NO_PROXY=""
    PUBLIC_ENDPOINT_URL=""
    CLIENT_LOG_PID=""

    SERVICE_DOCKER_CONFIG_ARGS=()
    SERVICE_PROXY_ENV_ARGS=()
    initialize_service_registry
}

config_validate_positive_integer_value() {
    local value=$1
    local description=$2
    local max_value="${3:-2147483647}"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
        || fail "${description} must be a positive decimal integer."
    if (( ${#value} > ${#max_value} )) \
        || { (( ${#value} == ${#max_value} )) \
            && [[ "${value}" > "${max_value}" ]]; }; then
        fail "${description} must not exceed ${max_value}."
    fi
}

resolve_effective_target_concurrency() {
    local benchmark_yaml="${BENCHMARK_CONFIG_TEMPLATE:-${SCRIPT_DIR}/configs/qwen_agentic_benchmark.yaml}"
    local target_concurrency target_count

    if [[ -v TARGET_CONCURRENCY ]]; then
        target_concurrency="${TARGET_CONCURRENCY}"
    else
        [[ -f "${benchmark_yaml}" ]] \
            || fail "Benchmark configuration does not exist: ${benchmark_yaml}"
        target_count=$(grep -Ec \
            '^[[:space:]]+target_concurrency:[[:space:]]*' \
            "${benchmark_yaml}" || true)
        [[ "${target_count}" -eq 1 ]] \
            || fail "Benchmark config must contain exactly one target_concurrency entry (found ${target_count})."
        target_concurrency=$(awk '
            /^[[:space:]]+target_concurrency:[[:space:]]*/ {
                value = $0
                sub("^[[:space:]]+target_concurrency:[[:space:]]*", "", value)
                sub("[[:space:]]*#.*$", "", value)
                sub("[[:space:]]+$", "", value)
                print value
                exit
            }
        ' "${benchmark_yaml}")
    fi

    config_validate_positive_integer_value \
        "${target_concurrency}" "TARGET_CONCURRENCY"
    EFFECTIVE_TARGET_CONCURRENCY=$((10#${target_concurrency}))
}

resolve_cpu_kv_cache_allocation() {
    [[ "${RUNTIME_DEVICE:-cpu}" == "cpu" ]] || return 0
    [[ -v EFFECTIVE_TARGET_CONCURRENCY ]] \
        || resolve_effective_target_concurrency
    (( ${#WORKER_IDS[@]} > 0 )) \
        || fail "CPU KV-cache allocation requires at least one worker."

    local baseline_total_gib=384
    local baseline_concurrency=100
    local requested_total_gib
    requested_total_gib=$(((
        baseline_total_gib * EFFECTIVE_TARGET_CONCURRENCY
        + baseline_concurrency - 1
    ) / baseline_concurrency))
    # Long SWE-bench contexts still need the baseline pool at low concurrency.
    if (( requested_total_gib < baseline_total_gib )); then
        requested_total_gib="${baseline_total_gib}"
    fi

    WORKER_KV_CACHE_AGGREGATE_GIB=()
    WORKER_KV_CACHE_RANK_GIB=()
    CPU_KV_CACHE_BASELINE_TOTAL_GIB="${baseline_total_gib}"
    CPU_KV_CACHE_BASELINE_CONCURRENCY="${baseline_concurrency}"
    CPU_KV_CACHE_REQUESTED_TOTAL_GIB="${requested_total_gib}"
    CPU_KV_CACHE_EFFECTIVE_TOTAL_GIB=0
    PREFILL_KV_CACHE_EFFECTIVE_TOTAL_GIB=0
    DECODE_KV_CACHE_EFFECTIVE_TOTAL_GIB=0
    CPU_KV_CACHE_ALLOCATION_SOURCE="adaptive"
    CPU_KV_CACHE_ALLOCATION_MODE="adaptive"

    local global_override=""
    if [[ -v VLLM_CPU_KVCACHE_SPACE_GIB ]]; then
        global_override="${VLLM_CPU_KVCACHE_SPACE_GIB}"
        config_validate_positive_integer_value \
            "${global_override}" "VLLM_CPU_KVCACHE_SPACE_GIB"
        global_override=$((10#${global_override}))
        CPU_KV_CACHE_ALLOCATION_SOURCE="environment"
        CPU_KV_CACHE_ALLOCATION_MODE="explicit"
    fi

    local index role parallel_width worker_count worker_aggregate_gib
    local role_override role_override_set role_total
    local prefill_count=0 decode_count=0
    for role in "${WORKER_ROLES[@]}"; do
        case "${role}" in
            prefill)
                prefill_count=$((prefill_count + 1))
                ;;
            decode)
                decode_count=$((decode_count + 1))
                ;;
            replica)
                ;;
            *)
                fail "Unsupported CPU worker role for KV-cache allocation: ${role}"
                ;;
        esac
    done

    for index in "${!WORKER_IDS[@]}"; do
        role="${WORKER_ROLES[$index]}"
        parallel_width="${WORKER_PARALLEL_WIDTHS[$index]}"
        config_validate_positive_integer_value \
            "${parallel_width}" "TP for ${WORKER_IDS[$index]}"

        if [[ -n "${global_override}" ]]; then
            worker_aggregate_gib="${global_override}"
        else
            role_override=""
            role_override_set=false
            if [[ "${role}" == "replica" \
                && -v CPU_KV_CACHE_PER_WORKER_GIB ]]; then
                role_override="${CPU_KV_CACHE_PER_WORKER_GIB}"
                role_override_set=true
            elif [[ "${role}" == "prefill" \
                && -v PREFILL_KV_CACHE_PER_WORKER_GIB ]]; then
                role_override="${PREFILL_KV_CACHE_PER_WORKER_GIB}"
                role_override_set=true
            elif [[ "${role}" == "decode" \
                && -v DECODE_KV_CACHE_PER_WORKER_GIB ]]; then
                role_override="${DECODE_KV_CACHE_PER_WORKER_GIB}"
                role_override_set=true
            fi

            if [[ "${role_override_set}" == "true" ]]; then
                config_validate_positive_integer_value \
                    "${role_override}" \
                    "${role^} KV-cache per-worker override"
                worker_aggregate_gib=$((10#${role_override}))
                CPU_KV_CACHE_ALLOCATION_SOURCE="profile"
                CPU_KV_CACHE_ALLOCATION_MODE="explicit"
            else
                if [[ "${role}" == "prefill" ]]; then
                    worker_count="${prefill_count}"
                elif [[ "${role}" == "decode" ]]; then
                    worker_count="${decode_count}"
                else
                    worker_count="${#WORKER_IDS[@]}"
                fi
                worker_aggregate_gib=$(((
                    requested_total_gib + worker_count - 1
                ) / worker_count))
                worker_aggregate_gib=$(((
                    worker_aggregate_gib + parallel_width - 1
                ) / parallel_width * parallel_width))
            fi
        fi

        (( worker_aggregate_gib > 0 )) \
            || fail "Resolved KV-cache allocation for ${WORKER_IDS[$index]} is not positive."
        (( worker_aggregate_gib % parallel_width == 0 )) \
            || fail "KV-cache allocation for ${WORKER_IDS[$index]} must be divisible by TP=${parallel_width}."
        WORKER_KV_CACHE_AGGREGATE_GIB+=("${worker_aggregate_gib}")
        WORKER_KV_CACHE_RANK_GIB+=(
            "$((worker_aggregate_gib / parallel_width))"
        )
        CPU_KV_CACHE_EFFECTIVE_TOTAL_GIB=$((CPU_KV_CACHE_EFFECTIVE_TOTAL_GIB + worker_aggregate_gib))
        if [[ "${role}" == "prefill" ]]; then
            role_total="${PREFILL_KV_CACHE_EFFECTIVE_TOTAL_GIB}"
            PREFILL_KV_CACHE_EFFECTIVE_TOTAL_GIB="$((role_total + worker_aggregate_gib))"
        elif [[ "${role}" == "decode" ]]; then
            role_total="${DECODE_KV_CACHE_EFFECTIVE_TOTAL_GIB}"
            DECODE_KV_CACHE_EFFECTIVE_TOTAL_GIB="$((role_total + worker_aggregate_gib))"
        fi
    done

    log "Resolved CPU KV cache: mode=${CPU_KV_CACHE_ALLOCATION_MODE}, source=${CPU_KV_CACHE_ALLOCATION_SOURCE}, target concurrency=${EFFECTIVE_TARGET_CONCURRENCY}, requested pool=${CPU_KV_CACHE_REQUESTED_TOTAL_GIB} GiB, effective worker total=${CPU_KV_CACHE_EFFECTIVE_TOTAL_GIB} GiB."
}

configure_calibration_settings() {
    local weight_dtype_was_set=false
    local kv_cache_dtype_was_set=false
    [[ -v WEIGHT_DTYPE ]] && weight_dtype_was_set=true
    [[ -v KV_CACHE_DTYPE ]] && kv_cache_dtype_was_set=true
    MTP_ENABLED="${MTP_ENABLED:-false}"
    MTP_ENABLED="${MTP_ENABLED,,}"
    [[ "${MTP_ENABLED}" == "true" || "${MTP_ENABLED}" == "false" ]] \
        || fail "MTP_ENABLED must be true or false."

    CALIBRATION_REQUIRED=false
    CALIBRATION_IMAGE_ID=""
    CALIBRATION_SOURCE_MODEL="${MODEL_NAME}"
    CALIBRATION_SOURCE_REVISION=""
    CALIBRATION_MANIFEST=""
    CALIBRATED_MODEL_HOST_PATH=""
    EFFECTIVE_MODEL_NAME="${MODEL_NAME}"
    if [[ -n "${MODEL_LOCAL_PATH:-}" ]]; then
        [[ "${MODEL_LOCAL_PATH}" == /* && -f "${MODEL_LOCAL_PATH}/config.json" ]] \
            || fail "MODEL_LOCAL_PATH must be an absolute directory containing config.json: ${MODEL_LOCAL_PATH}"
        MODEL_REVISION=""
        CALIBRATED_MODEL_HOST_PATH="${MODEL_LOCAL_PATH}"
        EFFECTIVE_MODEL_NAME="${MODEL_LOCAL_PATH}"
    fi

    if [[ "${CALIBRATION_SUPPORTED:-false}" != "true" ]]; then
        if [[ "${MTP_ENABLED}" == "true" \
            || "${weight_dtype_was_set}" == "true" \
            || "${kv_cache_dtype_was_set}" == "true" ]]; then
            fail "MTP_ENABLED, WEIGHT_DTYPE, and KV_CACHE_DTYPE are only supported by the stock FP8 Qwen3.6 profiles."
        fi
        WEIGHT_DTYPE="profile"
        KV_CACHE_DTYPE="auto"
        return
    fi

    WEIGHT_DTYPE="${WEIGHT_DTYPE:-fp8}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
    WEIGHT_DTYPE="${WEIGHT_DTYPE,,}"
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE,,}"

    case "${WEIGHT_DTYPE}" in
        fp8|int8)
            ;;
        *)
            fail "WEIGHT_DTYPE must be fp8 or int8 (case-insensitive)."
            ;;
    esac
    case "${KV_CACHE_DTYPE}" in
        auto|fp8)
            ;;
        *)
            fail "KV_CACHE_DTYPE must be auto or fp8 (case-insensitive)."
            ;;
    esac

    if [[ "${WEIGHT_DTYPE}" == "int8" && -z "${MODEL_LOCAL_PATH:-}" ]]; then
        MODEL_NAME="Qwen/Qwen3.6-35B-A3B"
        MODEL_REVISION="${MODEL_REVISION:-995ad96eacd98c81ed38be0c5b274b04031597b0}"
        CALIBRATION_SOURCE_MODEL="${MODEL_NAME}"
        EFFECTIVE_MODEL_NAME="${MODEL_NAME}"
    fi

    if [[ "${WEIGHT_DTYPE}" == "int8" \
        || "${KV_CACHE_DTYPE}" == "fp8" ]]; then
        CALIBRATION_REQUIRED=true
    fi
    CALIBRATION_OUTPUT_BASE="${MODEL_CACHE_DIR}/endpoints-agentic/calibrated_models"
}

initialize_benchmark_configuration() {
    initialize_base_configuration
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
    RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
    LOG_DIR="${RUN_DIR}/logs"
    SWEBENCH_ARTIFACT_ROOT="${RUN_DIR}/swebench_service_artifacts"
    RUNTIME_CONFIG="${RUN_DIR}/qwen_agentic_benchmark.runtime.yaml"
    CONTAINER_PREFIX="endpoints-agentic-benchmark-${RUN_ID}"
    ENDPOINT_SOURCE=""
    ACTIVE_SERVER_RUN_ID=""
    ACTIVE_SERVER_MANIFEST=""
    ACTIVE_SERVER_STACK_PROFILE=""
}

initialize_server_configuration() {
    initialize_base_configuration
    SERVER_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
    SERVER_RUN_DIR="${OUTPUT_ROOT}/server_runs/${SERVER_RUN_ID}"
    RUN_DIR="${SERVER_RUN_DIR}"
    LOG_DIR="${SERVER_RUN_DIR}/logs"
    CONTAINER_PREFIX="endpoints-agentic-server-${SERVER_RUN_ID}"
    ACTIVE_SERVER_FILE="${OUTPUT_ROOT}/active-server.env"
}

initialize_benchmark_context() {
    initialize_benchmark_configuration
    load_benchmark_configuration
}

configure_benchmark_mode() {
    local requested_mode="${MODE:-Both}"
    requested_mode="${requested_mode,,}"

    case "${requested_mode}" in
        both)
            BENCHMARK_MODE="both"
            ;;
        accuracy)
            BENCHMARK_MODE="acc"
            ;;
        performance)
            BENCHMARK_MODE="perf"
            ;;
        smoke)
            BENCHMARK_MODE="both"
            if [[ ! -v PERFORMANCE_TRAJECTORY_COUNT ]]; then
                PERFORMANCE_TRAJECTORY_COUNT=20
            fi
            if [[ ! -v SWEBENCH_INSTANCE_COUNT ]]; then
                SWEBENCH_INSTANCE_COUNT=10
            fi
            ;;
        *)
            fail "MODE must be Accuracy, Performance, Both, or Smoke (case-insensitive)."
            ;;
    esac
    log "Benchmark mode: ${BENCHMARK_MODE}."
}

configure_model_cache() {
    if [[ -z "${MODEL_CACHE_DIR}" ]]; then
        [[ -n "${HOME:-}" ]] \
            || fail "HF_HOME or HOME is required for the Hugging Face cache."
        MODEL_CACHE_DIR="${HOME}/.cache/huggingface"
    fi
    [[ "${MODEL_CACHE_DIR}" == /* ]] \
        || fail "HF_HOME must be an absolute path when set."
}
