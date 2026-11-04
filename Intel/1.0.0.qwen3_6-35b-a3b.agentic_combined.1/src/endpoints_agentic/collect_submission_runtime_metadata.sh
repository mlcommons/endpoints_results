#!/usr/bin/env bash
# shellcheck disable=SC1091

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# Keep adapter and resolver diagnostics off the JSON output stream.
log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}
# shellcheck source=lib/config.sh
source "${SCRIPT_DIR}/lib/config.sh"
# shellcheck source=lib/topology.sh
source "${SCRIPT_DIR}/lib/topology.sh"

main() {
    (( $# == 0 )) \
        || fail "collect_submission_runtime_metadata.sh does not accept positional arguments."

    load_stack_profile
    # shellcheck disable=SC2034
    MODEL_CACHE_DIR="${HF_HOME:-}"
    configure_model_cache
    configure_calibration_settings
    discover_topology

    local runtime_path="${PLATFORM_DIR}/runtimes/${RUNTIME_ADAPTER}.sh"
    [[ -f "${runtime_path}" ]] \
        || fail "Runtime adapter does not exist: ${runtime_path}"
    # shellcheck disable=SC1090
    source "${runtime_path}"

    local local_only="${SUBMISSION_RUNTIME_LOCAL_ONLY:-0}"
    case "${local_only}" in
        0)
            runtime_prepare_image >&2
            ;;
        1)
            runtime_assert_image_prepared >&2
            ;;
        *)
            fail "SUBMISSION_RUNTIME_LOCAL_ONLY must be 0 or 1."
            ;;
    esac

    local metadata_json
    metadata_json=$(vllm_collect_runtime_metadata)
    [[ -n "${metadata_json}" ]] \
        || fail "Runtime metadata probe returned no JSON."
    python3 -c '
import json
import sys

value = json.loads(sys.stdin.read())
required = {
    "image_id", "image_ref", "vllm_version", "python_version",
    "torch_version", "platform", "container_os",
}
if not isinstance(value, dict) or not required.issubset(value):
    raise SystemExit("Runtime metadata probe omitted required fields.")
if "software" in value and not isinstance(value["software"], dict):
    raise SystemExit("Runtime metadata software field must be an object.")
' <<<"${metadata_json}"
    printf '%s\n' "${metadata_json}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
