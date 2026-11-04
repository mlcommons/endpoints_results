#!/usr/bin/env bash

# shellcheck source=components/runtimes/vllm_cpu_common.sh
source "${SCRIPT_DIR}/components/runtimes/vllm_cpu_common.sh"

runtime_validate_cpu() {
    if [[ -n "${HOST_CPU_VENDOR:-}" ]]; then
        [[ "${HOST_CPU_VENDOR}" == "GenuineIntel" ]] \
            || fail "The Intel CPU runtime requires an Intel host CPU."
    fi

    case "${MODEL_CPU_REQUIREMENT:-generic}" in
        generic|amx_fp8)
            ;;
        *)
            fail "Unsupported MODEL_CPU_REQUIREMENT: ${MODEL_CPU_REQUIREMENT}"
            ;;
    esac
}

vllm_validate_runtime_capability() {
    [[ -z "${RUNTIME_CAPABILITY:-}" ]] \
        || fail "Unsupported Intel CPU runtime capability: ${RUNTIME_CAPABILITY}"
}
