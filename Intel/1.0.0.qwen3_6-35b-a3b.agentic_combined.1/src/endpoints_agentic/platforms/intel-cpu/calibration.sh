#!/usr/bin/env bash
# shellcheck disable=SC2034

calibration_validate_model_revision() {
    local revision="${MODEL_REVISION:-}"
    [[ -z "${revision}" ]] && return 0
    [[ "${revision}" =~ ^[a-fA-F0-9]{40}$ ]] \
        || fail "MODEL_REVISION must be a full 40-character hexadecimal commit SHA."
}

calibration_resolve_paths() {
    calibration_validate_model_revision
    [[ "${CALIBRATION_REQUIRED}" == "true" ]] || return 0

    local requested_revision="${MODEL_REVISION:-}"
    local source_snapshot
    local revision_env_args=()
    if [[ "${WEIGHT_DTYPE}" == "int8" ]]; then
        [[ -n "${requested_revision}" ]] || fail "INT8 calibration requires a pinned BF16 MODEL_REVISION."
        source_snapshot="/root/.cache/huggingface/hub/models--${CALIBRATION_SOURCE_MODEL//\//--}/snapshots/${requested_revision,,}"
    else
        if [[ -n "${requested_revision}" ]]; then
            revision_env_args=(-e "ENDPOINTS_AGENTIC_MODEL_REVISION=${requested_revision}")
        fi
        source_snapshot=$(
            docker run --rm \
                --network none \
                -e "ENDPOINTS_AGENTIC_MODEL_NAME=${CALIBRATION_SOURCE_MODEL}" \
                "${revision_env_args[@]}" \
                -v "${MODEL_CACHE_DIR}:/root/.cache/huggingface" \
                --entrypoint python \
                "${RUNTIME_IMAGE}" \
                -c 'import os; from huggingface_hub import snapshot_download; model = os.environ["ENDPOINTS_AGENTIC_MODEL_NAME"]; kwargs = {"repo_id": model, "local_files_only": True}; revision = os.environ.get("ENDPOINTS_AGENTIC_MODEL_REVISION"); kwargs.update({"revision": revision} if revision else {}); print(snapshot_download(**kwargs))'
        ) || fail "The cached FP8 source checkpoint could not be resolved. Run bash preflight.sh first."
    fi

    [[ "${source_snapshot}" == /root/.cache/huggingface/hub/*/snapshots/* ]] \
        || fail "Unexpected cached model snapshot path: ${source_snapshot}"
    CALIBRATION_SOURCE_REVISION="${source_snapshot##*/}"
    [[ "${CALIBRATION_SOURCE_REVISION}" =~ ^[a-fA-F0-9]{7,64}$ ]] \
        || fail "Unexpected model revision in cached snapshot path: ${CALIBRATION_SOURCE_REVISION}"
    if [[ -n "${requested_revision}" ]]; then
        [[ "${CALIBRATION_SOURCE_REVISION,,}" == "${requested_revision,,}" ]] \
            || fail "Cached model revision ${CALIBRATION_SOURCE_REVISION} does not match requested MODEL_REVISION ${requested_revision}."
    fi

    local key
    key="qwen36-${CALIBRATION_SOURCE_REVISION:0:12}-w${WEIGHT_DTYPE}-kv${KV_CACHE_DTYPE}-v${CALIBRATION_RECIPE_VERSION}"
    CALIBRATED_MODEL_HOST_PATH="${CALIBRATION_OUTPUT_BASE}/${key}"
    EFFECTIVE_MODEL_NAME="/root/.cache/huggingface/endpoints-agentic/calibrated_models/${key}"
    CALIBRATION_MANIFEST="${CALIBRATED_MODEL_HOST_PATH}/calibration_manifest.json"
    CALIBRATION_SOURCE_SNAPSHOT="${source_snapshot}"
}

calibration_output_matches() {
    local output_path=${1:-${CALIBRATED_MODEL_HOST_PATH}}
    [[ -d "${output_path}" ]] || return 1
    local -a verify_args=()
    if [[ "${CALIBRATION_REQUIRED:-false}" == "true" \
        && "${MTP_ENABLED:-false}" == "true" ]]; then
        verify_args+=(--mtp-enabled)
    fi
    [[ "${2:-}" != "inspect-tensors" ]] \
        || verify_args+=(--inspect-tensors)
    if [[ "${WEIGHT_DTYPE}" == "int8" && -z "${MODEL_LOCAL_PATH:-}" ]]; then
        verify_args+=(--source-model-id "${CALIBRATION_SOURCE_MODEL}"
            --source-revision "${CALIBRATION_SOURCE_REVISION}")
    fi
    docker run --rm \
        --network none \
        -v "${output_path}:/model:ro" \
        -v "${SCRIPT_DIR}/platforms/intel-cpu/calibration:/calibration:ro" \
        --entrypoint python \
        "${RUNTIME_IMAGE}" \
        /calibration/calibrate.py verify-existing \
        --output-dir /model \
        --weight-dtype "${WEIGHT_DTYPE}" \
        --kv-cache-dtype "${KV_CACHE_DTYPE}" \
        "${verify_args[@]}" \
        >/dev/null 2>&1
}

calibration_assert_prepared() {
    [[ "${CALIBRATION_SUPPORTED:-false}" == "true" ]] || return 0
    if [[ -n "${MODEL_LOCAL_PATH:-}" ]]; then
        calibration_output_matches "${MODEL_LOCAL_PATH}" \
            || fail "Local checkpoint does not match the requested quantization: ${MODEL_LOCAL_PATH}"
        return 0
    fi
    [[ "${CALIBRATION_REQUIRED}" == "true" ]] || return 0
    calibration_resolve_paths
    calibration_output_matches \
        || fail "The requested calibrated checkpoint is missing or incompatible. Run bash preflight.sh first: ${CALIBRATED_MODEL_HOST_PATH}"
}

calibration_physical_cpus() {
    local node=$1
    LC_ALL=C lscpu --parse=CPU,CORE,NODE --online \
        | awk -F, -v requested_node="${node}" '
            /^#/ { next }
            $1 !~ /^[0-9]+$/ || $2 !~ /^[0-9]+$/ || $3 !~ /^[0-9]+$/ {
                exit 1
            }
            $3 == requested_node && !seen[$2]++ {
                cpus = cpus separator $1
                separator = ","
                count++
            }
            END {
                if (count == 0) exit 1
                print cpus
            }
        '
}

calibration_validate_output_space() {
    local free_kib
    free_kib=$(df -Pk "${CALIBRATION_OUTPUT_BASE}" | awk 'NR == 2 {print $4}')
    (( free_kib >= 45 * 1024 * 1024 )) \
        || fail "Calibration requires at least 45 GiB free under ${CALIBRATION_OUTPUT_BASE}; only $((free_kib / 1024 / 1024)) GiB is available."
}

calibration_remove_source() {
    [[ "${WEIGHT_DTYPE}" == "int8" ]] || return 0
    local source_path="${MODEL_CACHE_DIR}${CALIBRATION_SOURCE_SNAPSHOT#/root/.cache/huggingface}"
    [[ -d "${source_path}" ]] || return 0
    local snapshot
    for snapshot in "${MODEL_CACHE_DIR}"/hub/*/snapshots/"${CALIBRATION_SOURCE_REVISION}"; do
        [[ ! -d "${snapshot}" || "${snapshot}" == "${source_path}" ]] \
            || fail "BF16 cache cleanup refused an ambiguous revision hash; the published checkpoint is intact."
    done
    log "Removing BF16 source revision ${CALIBRATION_SOURCE_REVISION} with hf cache rm."
    docker run --rm --network none \
        -v "${MODEL_CACHE_DIR}:/root/.cache/huggingface" \
        --entrypoint hf "${RUNTIME_IMAGE_ID:-${RUNTIME_IMAGE}}" \
        cache rm "${CALIBRATION_SOURCE_REVISION}" --yes \
        --cache-dir /root/.cache/huggingface/hub \
        || fail "BF16 cache cleanup failed; the published checkpoint is intact. Rerun preflight to retry cleanup."
    [[ ! -d "${source_path}" ]] \
        || fail "hf cache rm did not remove the BF16 source; the published checkpoint is intact."
}

calibration_prepare_corpus() (
    [[ "${KV_CACHE_DTYPE}" == "fp8" ]] || return 0
    local corpus_dir="${MODEL_CACHE_DIR}/endpoints-agentic/calibration_corpora/${AGENTIC_DATASET_SHA256:0:12}"
    require_command flock
    require_command sha256sum
    mkdir -p "${corpus_dir%/*}" "${MODEL_CACHE_DIR}/endpoints-agentic/locks"
    local corpus_lock_fd
    exec {corpus_lock_fd}>"${MODEL_CACHE_DIR}/endpoints-agentic/locks/calibration-corpus.lock"
    flock "${corpus_lock_fd}"
    if [[ -f "${corpus_dir}/corpus_manifest.json" ]]; then
        log "Reusing calibration corpus ${corpus_dir}."
        return 0
    fi
    [[ ! -e "${corpus_dir}" ]] \
        || fail "Refusing to overwrite incomplete calibration corpus: ${corpus_dir}"

    local temporary_corpus
    temporary_corpus=$(mktemp -d "${corpus_dir}.partial.XXXXXX")
    # Invoked by the corpus preparation subprocess's signal/exit traps.
    # shellcheck disable=SC2329
    cleanup_corpus_partial() {
        rm -rf -- "${temporary_corpus}"
    }
    trap cleanup_corpus_partial EXIT INT TERM
    docker image inspect "${ENDPOINT_IMAGE}" >/dev/null 2>&1 \
        || benchmark_prepare_image
    log "Extracting the performance dataset from ${ENDPOINT_IMAGE}."
    docker run --rm --network none --entrypoint cat "${ENDPOINT_IMAGE}" \
        /workspace/dataset_cache/agentic_inference_conversations/mlperf_agentic_inference_dataset.jsonl \
        >"${temporary_corpus}/source.jsonl" \
        || fail "Could not read the performance dataset from ${ENDPOINT_IMAGE}."
    printf '%s  %s\n' "${AGENTIC_DATASET_SHA256}" "${temporary_corpus}/source.jsonl" \
        | sha256sum --check --status \
        || fail "Benchmark dataset does not match AGENTIC_DATASET_SHA256. Run bash preflight.sh to rebuild the image."
    log "Performance dataset checksum verified."
    local node="${NUMA_NODES[${#NUMA_NODES[@]}-1]}"
    local cpus
    cpus=$(calibration_physical_cpus "${node}") \
        || fail "Could not derive physical CPUs for corpus generation NUMA node ${node}."
    log "Generating calibration corpus from the pinned performance dataset."
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        --cpuset-cpus "${cpus}" \
        --cpuset-mems "${node}" \
        -e HOME=/tmp \
        -e HF_HOME=/models \
        -e TOKENIZERS_PARALLELISM=true \
        -e "RAYON_NUM_THREADS=$(awk -F, '{print NF}' <<<"${cpus}")" \
        -v "${MODEL_CACHE_DIR}:/models" \
        -v "${temporary_corpus}:/corpus" \
        -v "${SCRIPT_DIR}/platforms/intel-cpu/calibration:/calibration:ro" \
        --entrypoint python "${CALIBRATION_IMAGE}" \
        /calibration/generate_corpus.py /corpus/source.jsonl /corpus/generated \
        || fail "Calibration corpus generation failed."
    mv -n -T -- "${temporary_corpus}/generated" "${corpus_dir}"
    [[ ! -e "${temporary_corpus}/generated" ]] \
        || fail "Calibration corpus destination appeared during generation: ${corpus_dir}"
    log "Prepared calibration corpus ${corpus_dir}."
)

calibration_generate_and_publish() (
    local node_index calibration_node calibration_cpus omp_threads
    node_index=$((${#NUMA_NODES[@]} - 1))
    calibration_node="${NUMA_NODES[$node_index]}"
    calibration_cpus=$(calibration_physical_cpus "${calibration_node}") \
        || fail "Could not derive physical CPUs for calibration NUMA node ${calibration_node}."
    omp_threads=$(awk -F, '{print NF}' <<<"${calibration_cpus}")

    local temporary_output
    temporary_output=$(mktemp -d \
        "${CALIBRATION_OUTPUT_BASE}/.${CALIBRATED_MODEL_HOST_PATH##*/}.partial.XXXXXX")
    # Invoked by the publication subprocess's signal/exit traps.
    # shellcheck disable=SC2329
    cleanup_calibration_partial() {
        [[ -z "${temporary_output}" ]] \
            || rm -rf -- "${temporary_output}"
    }
    trap cleanup_calibration_partial EXIT INT TERM

    local container_output="/outputs/${temporary_output##*/}"
    local container_source
    container_source="/models${CALIBRATION_SOURCE_SNAPSHOT#/root/.cache/huggingface}"
    local -a corpus_mount=() corpus_args=()
    if [[ "${KV_CACHE_DTYPE}" == "fp8" ]]; then
        local corpus_dir="${MODEL_CACHE_DIR}/endpoints-agentic/calibration_corpora/${AGENTIC_DATASET_SHA256:0:12}"
        corpus_mount=(-v "${corpus_dir}:/corpus:ro")
        corpus_args=(--corpus-dir /corpus)
    fi
    log "Calibrating Qwen3.6 on NUMA node ${calibration_node}, CPUs ${calibration_cpus}."
    local calibration_status=0
    docker run --rm \
        --network none \
        --ipc host \
        --user "$(id -u):$(id -g)" \
        --cpuset-cpus "${calibration_cpus}" \
        --cpuset-mems "${calibration_node}" \
        -e HOME=/tmp \
        -e USER=calibration \
        -e "OMP_NUM_THREADS=${omp_threads}" \
        -e "CALIBRATION_IMAGE_ID=${CALIBRATION_IMAGE_ID}" \
        -v "${MODEL_CACHE_DIR}:/models:ro" \
        -v "${CALIBRATION_OUTPUT_BASE}:/outputs" \
        -v "${SCRIPT_DIR}/platforms/intel-cpu/calibration:/calibration:ro" \
        "${corpus_mount[@]}" \
        --entrypoint python \
        "${CALIBRATION_IMAGE}" \
        /calibration/calibrate.py calibrate \
        --source-model "${container_source}" \
        --source-model-id "${CALIBRATION_SOURCE_MODEL}" \
        --source-revision "${CALIBRATION_SOURCE_REVISION}" \
        --output-dir "${container_output}" \
        --weight-dtype "${WEIGHT_DTYPE}" \
        --kv-cache-dtype "${KV_CACHE_DTYPE}" \
        "${corpus_args[@]}" \
        || calibration_status=$?
    if (( calibration_status != 0 )); then
        fail "Qwen3.6 calibration failed with status ${calibration_status}."
    fi

    : >"${temporary_output}/COMPLETE"
    log "Validating calibrated checkpoint tensors."
    calibration_output_matches "${temporary_output}" inspect-tensors \
        || fail "Generated calibration output failed validation."
    mv -n -T -- "${temporary_output}" "${CALIBRATED_MODEL_HOST_PATH}"
    if [[ -e "${temporary_output}" ]]; then
        fail "Calibration destination appeared while calibration was running: ${CALIBRATED_MODEL_HOST_PATH}"
    fi
    temporary_output=""
    trap - EXIT INT TERM
    log "Prepared calibrated checkpoint ${CALIBRATED_MODEL_HOST_PATH}."
)

calibration_prepare_directories() {
    [[ "${CALIBRATION_SUPPORTED:-false}" == "true" \
        && "${CALIBRATION_REQUIRED:-false}" == "true" \
        && -z "${MODEL_LOCAL_PATH:-}" ]] || return 0

    prepare_writable_directory "${CALIBRATION_OUTPUT_BASE}" "calibration output" HF_HOME
    if [[ "${WEIGHT_DTYPE}" == "int8" || "${KV_CACHE_DTYPE}" == "fp8" ]]; then
        prepare_writable_directory \
            "${MODEL_CACHE_DIR}/endpoints-agentic/locks" "calibration locks" HF_HOME
    fi
    if [[ "${KV_CACHE_DTYPE}" == "fp8" ]]; then
        prepare_writable_directory \
            "${MODEL_CACHE_DIR}/endpoints-agentic/calibration_corpora" \
            "calibration corpus" HF_HOME
    fi
}

calibration_prepare_model() {
    [[ "${CALIBRATION_SUPPORTED:-false}" == "true" ]] || return 0
    if [[ -n "${MODEL_LOCAL_PATH:-}" ]]; then
        calibration_assert_prepared
        return 0
    fi
    [[ "${CALIBRATION_REQUIRED}" == "true" ]] || return 0
    calibration_prepare_directories
    calibration_resolve_paths
    local command
    for command in id mktemp mv rm; do
        require_command "${command}"
    done
    local calibration_lock_fd
    if [[ "${WEIGHT_DTYPE}" == "int8" ]]; then
        require_command flock
        exec {calibration_lock_fd}>"${MODEL_CACHE_DIR}/endpoints-agentic/locks/${CALIBRATION_SOURCE_MODEL//\//--}.lock"
        flock "${calibration_lock_fd}"
    fi

    if [[ -e "${CALIBRATED_MODEL_HOST_PATH}" ]]; then
        calibration_output_matches \
            || fail "Refusing to overwrite incomplete or mismatched calibration output: ${CALIBRATED_MODEL_HOST_PATH}"
        log "Reusing compatible checkpoint ${CALIBRATED_MODEL_HOST_PATH}."
        calibration_remove_source
        [[ -z "${calibration_lock_fd:-}" ]] || exec {calibration_lock_fd}>&-
        return 0
    fi

    if [[ "${WEIGHT_DTYPE}" == "int8" ]]; then
        vllm_download_model calibration || fail "BF16 source download failed."
    fi
    calibration_validate_output_space
    log "Building calibration image ${CALIBRATION_IMAGE}."
    docker build \
        --pull \
        --file "${SCRIPT_DIR}/platforms/intel-cpu/docker/Dockerfile.calibration" \
        --tag "${CALIBRATION_IMAGE}" \
        "${SCRIPT_DIR}"
    CALIBRATION_IMAGE_ID=$(
        docker image inspect --format '{{.Id}}' "${CALIBRATION_IMAGE}"
    )
    calibration_prepare_corpus
    calibration_generate_and_publish
    calibration_remove_source
    [[ -z "${calibration_lock_fd:-}" ]] || exec {calibration_lock_fd}>&-
}
