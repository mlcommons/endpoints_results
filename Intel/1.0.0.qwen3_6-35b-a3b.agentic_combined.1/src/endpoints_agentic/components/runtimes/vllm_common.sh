#!/usr/bin/env bash
# Shared state is consumed by the selected runtime adapter.
# shellcheck disable=SC2034

VLLM_MODEL_MOUNT_ARGS=()

vllm_assert_runtime_image_id() {
    local image_id
    image_id=$(docker image inspect --format '{{.Id}}' \
        "${RUNTIME_IMAGE}" 2>/dev/null) \
        || fail "Runtime image is not available locally: ${RUNTIME_IMAGE}. Run bash preflight.sh first."
    [[ -n "${image_id}" ]] \
        || fail "Docker returned an empty image ID for ${RUNTIME_IMAGE}."

    if [[ -n "${SUBMISSION_EXPECTED_RUNTIME_IMAGE_ID:-}" ]]; then
        [[ "${image_id}" == "${SUBMISSION_EXPECTED_RUNTIME_IMAGE_ID}" ]] \
            || fail "Runtime image ${RUNTIME_IMAGE} has ID ${image_id}, expected ${SUBMISSION_EXPECTED_RUNTIME_IMAGE_ID}."
    fi

    # Consumed by the persisted stack manifest and image-only metadata probe.
    # shellcheck disable=SC2034
    RUNTIME_IMAGE_ID="${image_id}"
}

vllm_runtime_worker_image_ref() {
    vllm_assert_runtime_image_id
    printf '%s\n' "${RUNTIME_IMAGE_ID}"
}

vllm_validate_model_revision() {
    local revision="${MODEL_REVISION:-}"
    [[ -z "${revision}" ]] && return 0
    [[ "${revision}" =~ ^[a-fA-F0-9]{40}$ ]] \
        || fail "MODEL_REVISION must be a full 40-character hexadecimal commit SHA."
}

vllm_download_model() {
    vllm_validate_model_revision
    [[ -z "${MODEL_LOCAL_PATH:-}" ]] || return 0
    if [[ "${CALIBRATION_REQUIRED:-false}" == "true" && "${WEIGHT_DTYPE:-}" == "int8" \
        && "${1:-}" != calibration ]]; then
        return 0
    fi
    local hf_token_args=()
    local revision_args=()
    if [[ -n "${HF_TOKEN:-}" ]]; then
        hf_token_args=(-e "HF_TOKEN=${HF_TOKEN}")
    fi
    if [[ -n "${MODEL_REVISION:-}" ]]; then
        revision_args=(--revision "${MODEL_REVISION,,}")
    fi
    log "Downloading model ${MODEL_NAME} into ${MODEL_CACHE_DIR}."
    docker run --rm \
        "${hf_token_args[@]}" \
        -v "${MODEL_CACHE_DIR}:/root/.cache/huggingface" \
        --entrypoint hf \
        "${RUNTIME_IMAGE_ID:-${RUNTIME_IMAGE}}" \
        download "${MODEL_NAME}" \
        "${revision_args[@]}" \
        --cache-dir /root/.cache/huggingface/hub \
        >/dev/null
}

vllm_assert_cached_model() {
    vllm_validate_model_revision
    [[ -z "${MODEL_LOCAL_PATH:-}" ]] || return 0
    local effective_model_name="${EFFECTIVE_MODEL_NAME:-${MODEL_NAME}}"
    local revision_env_args=()
    if [[ -n "${MODEL_REVISION:-}" && "${effective_model_name}" != /* ]]; then
        revision_env_args=(-e "ENDPOINTS_AGENTIC_MODEL_REVISION=${MODEL_REVISION}")
    fi
    docker run --rm \
        --network none \
        -e HF_HUB_OFFLINE=1 \
        -e TRANSFORMERS_OFFLINE=1 \
        -e "ENDPOINTS_AGENTIC_MODEL_NAME=${effective_model_name}" \
        "${revision_env_args[@]}" \
        -v "${MODEL_CACHE_DIR}:/root/.cache/huggingface" \
        --entrypoint "${VLLM_PYTHON_ENTRYPOINT}" \
        "${RUNTIME_IMAGE_ID:-${RUNTIME_IMAGE}}" \
        -c 'import os; from pathlib import Path; from huggingface_hub import snapshot_download; from transformers import AutoConfig; model = os.environ["ENDPOINTS_AGENTIC_MODEL_NAME"]; kwargs = {"repo_id": model, "local_files_only": True}; revision = os.environ.get("ENDPOINTS_AGENTIC_MODEL_REVISION"); kwargs.update({"revision": revision} if revision else {}); AutoConfig.from_pretrained(model, local_files_only=True) if Path(model).is_absolute() else snapshot_download(**kwargs)' \
        >/dev/null \
        || fail "Model ${effective_model_name} is not complete in ${MODEL_CACHE_DIR}. Run bash preflight.sh first."
}

# MODEL_LOCAL_PATH serves a checkpoint directory from local disk instead of the
# Hugging Face cache. Sets EFFECTIVE_MODEL_NAME and VLLM_MODEL_MOUNT_ARGS.
vllm_configure_local_model() {
    vllm_validate_model_revision
    VLLM_MODEL_MOUNT_ARGS=()
    [[ -n "${MODEL_LOCAL_PATH:-}" ]] || return 0
    EFFECTIVE_MODEL_NAME="${MODEL_LOCAL_PATH}"
    VLLM_MODEL_MOUNT_ARGS=(-v "${MODEL_LOCAL_PATH}:${MODEL_LOCAL_PATH}:ro")
    log "Using local checkpoint ${MODEL_LOCAL_PATH} (profile expects ${MODEL_NAME})."
}

vllm_build_derived_runtime() {
    local pull_arg=--pull
    if docker image inspect "${RUNTIME_BASE_IMAGE}" >/dev/null 2>&1; then
        pull_arg=--pull=false
    fi
    docker build \
        "${pull_arg}" \
        --build-arg "RUNTIME_BASE_IMAGE=${RUNTIME_BASE_IMAGE}" \
        "$@" \
        --file "${SCRIPT_DIR}/${RUNTIME_DOCKERFILE}" \
        --tag "${RUNTIME_IMAGE}" \
        "${SCRIPT_DIR}"
}

vllm_collect_runtime_metadata() {
    local metadata_code
    local -a device_args=()
    [[ -f "${PLATFORM_DIR}/runtime_metadata.py" ]] \
        || fail "Selected family runtime metadata probe is missing."

    if [[ "${TOPOLOGY_MODE:-replica}" == "gpu_replica" ]]; then
        accel_init_device
        accel_docker_all_device_args device_args
    fi

    metadata_code=$(cat <<'PY'
import ctypes
import ctypes.util
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import torch
import vllm

image_id, image_ref, runtime_adapter, runtime_device, capability = sys.argv[1:6]

def distribution_version(name):
    return importlib.metadata.version(name)

def container_os():
    try:
        return platform.freedesktop_os_release()["PRETTY_NAME"]
    except (AttributeError, OSError, KeyError):
        return platform.system()

result = {
    "image_id": image_id,
    "image_ref": image_ref,
    "vllm_version": vllm.__version__,
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "platform": runtime_device,
    "container_platform": f"{platform.system().lower()}/{platform.machine()}",
    "container_os": container_os(),
}

software = {"transformers": distribution_version("transformers")}
PY
    cat "${PLATFORM_DIR}/runtime_metadata.py"
    cat <<'PY'

result["software"] = software

json.dump(result, sys.stdout, separators=(",", ":"), sort_keys=True)
sys.stdout.write("\n")
PY
)
    docker run --rm \
        --network none \
        "${device_args[@]}" \
        --entrypoint "${VLLM_PYTHON_ENTRYPOINT}" \
        "${RUNTIME_IMAGE_ID}" \
        -c "${metadata_code}" \
        "${RUNTIME_IMAGE_ID}" \
        "${RUNTIME_IMAGE}" \
        "${RUNTIME_ADAPTER}" \
        "${RUNTIME_DEVICE}" \
        "${RUNTIME_CAPABILITY:-}"
}

vllm_build_server_args() {
    local output_name=$1
    local effective_model_name=$2
    local port=$3
    local parallel_width=$4
    local -n output_args="${output_name}"

    output_args=(
        "${effective_model_name}"
        --port "${port}"
        --served-model-name "${SERVED_MODEL_NAME}"
    )
    vllm_validate_model_revision
    if [[ -n "${MODEL_REVISION:-}" && "${effective_model_name}" != /* ]]; then
        output_args+=(--revision "${MODEL_REVISION}" --tokenizer-revision "${MODEL_REVISION}")
    fi
    if (( parallel_width > 1 )); then
        output_args+=(--tensor-parallel-size "${parallel_width}")
    fi
    output_args+=(
        --enable-auto-tool-choice
        --tool-call-parser qwen3_coder
        --language-model-only
        --reasoning-parser qwen3
        --enable-prefix-caching
    )
    if [[ "${MTP_ENABLED:-false}" == "true" ]]; then
        local speculative_config
        if [[ "${CALIBRATION_REQUIRED:-false}" == "true" ]]; then
            printf -v speculative_config \
                '{"method":"mtp","model":"%s/mtp","num_speculative_tokens":1,"kv_cache_dtype":"bfloat16","disable_eagle_block_drop":true}' \
                "${effective_model_name}"
        else
            speculative_config='{"method":"mtp","num_speculative_tokens":1,"kv_cache_dtype":"bfloat16","disable_eagle_block_drop":true}'
        fi
        output_args+=(--speculative-config "${speculative_config}")
    fi
}

vllm_validate_worker_kv_cache_allocations() {
    local worker_count=${#WORKER_IDS[@]}
    (( ${#WORKER_KV_CACHE_AGGREGATE_GIB[@]} == worker_count )) \
        || fail "CPU KV-cache aggregate allocation count does not match worker count."
    (( ${#WORKER_KV_CACHE_RANK_GIB[@]} == worker_count )) \
        || fail "CPU KV-cache per-rank allocation count does not match worker count."

    local index aggregate_gib rank_gib parallel_width
    for index in "${!WORKER_IDS[@]}"; do
        aggregate_gib="${WORKER_KV_CACHE_AGGREGATE_GIB[$index]}"
        rank_gib="${WORKER_KV_CACHE_RANK_GIB[$index]}"
        parallel_width="${WORKER_PARALLEL_WIDTHS[$index]}"
        [[ "${aggregate_gib}" =~ ^[1-9][0-9]*$ ]] \
            || fail "CPU KV-cache aggregate allocation for ${WORKER_IDS[$index]} must be a positive integer."
        [[ "${rank_gib}" =~ ^[1-9][0-9]*$ ]] \
            || fail "CPU KV-cache per-rank allocation for ${WORKER_IDS[$index]} must be a positive integer."
        [[ "${parallel_width}" =~ ^[1-9][0-9]*$ ]] \
            || fail "CPU worker TP for ${WORKER_IDS[$index]} must be a positive integer."
        (( aggregate_gib % parallel_width == 0 )) \
            || fail "CPU KV-cache aggregate allocation for ${WORKER_IDS[$index]} must be divisible by TP=${parallel_width}."
        (( aggregate_gib / parallel_width == rank_gib )) \
            || fail "CPU KV-cache per-rank allocation for ${WORKER_IDS[$index]} is inconsistent with its aggregate allocation."
    done
}
