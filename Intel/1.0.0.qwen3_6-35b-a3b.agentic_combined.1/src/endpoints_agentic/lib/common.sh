#!/usr/bin/env bash

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

fail() {
    log "ERROR: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

prepare_writable_directory() {
    local path=$1 purpose=$2 setting=$3
    local command uid gid probe remedy
    for command in id mkdir mktemp rmdir; do
        require_command "${command}"
    done
    uid=$(id -u)
    gid=$(id -g)
    remedy="Choose a writable location using ${setting}, or ask an administrator to grant UID ${uid}/GID ${gid} access to this directory."
    if [[ ! -e "${path}" && ! -L "${path}" ]]; then
        local create_command create_path="${path%/}" parent
        parent="${create_path%/*}"
        while [[ -n "${parent}" && ! -e "${parent}" && ! -L "${parent}" ]]; do
            create_path="${parent}"
            parent="${create_path%/*}"
        done
        if [[ -d "${parent:-/}" ]]; then
            printf -v create_command 'sudo install -d -o %q -g %q -m 0755 -- %q' \
                "${uid}" "${gid}" "${create_path}"
            remedy+=" If this is a private directory for this user, an administrator can create the first missing directory with: ${create_command}"
        fi
    fi
    if ! mkdir -p -- "${path}"; then
        fail "Could not create ${purpose} directory '${path}' for UID ${uid}/GID ${gid}. ${remedy}"
    fi
    if ! probe=$(mktemp -d -- "${path%/}/.endpoints-agentic-write-check.XXXXXX"); then
        fail "Cannot write to ${purpose} directory '${path}' as UID ${uid}/GID ${gid}. ${remedy}"
    fi
    if ! rmdir -- "${probe}"; then
        fail "Could not remove directory write probe '${probe}'. ${remedy}"
    fi
}

require_local_image() {
    local image=$1
    docker image inspect "${image}" >/dev/null 2>&1 \
        || fail "Required image is not available locally: ${image}. Run bash preflight.sh first."
}

validate_digest_image() {
    local variable_name=$1
    local owner=$2
    local image="${!variable_name:-}"
    [[ "${image}" =~ ^[^[:space:]]+@sha256:[0-9a-f]{64}$ ]] \
        || fail "${owner} requires ${variable_name} to be an immutable sha256 image reference."
}

pd_routed_backend_port() {
    local role=$1
    local model_server_port=$2
    if [[ "${TOPOLOGY_MODE:-replica}" == "pd" \
        && "${role}" == "decode" ]]; then
        printf '%s\n' "$((model_server_port + LLMD_SIDECAR_PORT_OFFSET))"
    else
        printf '%s\n' "${model_server_port}"
    fi
}

apply_image_display_tag() {
    docker image tag "$1" "$2"
}

require_single_literal() {
    local file=$1
    local literal=$2
    local count
    count=$( (grep -Fo "${literal}" "${file}" || true) | wc -l)
    [[ "${count}" -eq 1 ]] \
        || fail "Benchmark config must contain exactly one ${literal} placeholder (found ${count})."
}
