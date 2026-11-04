#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=lib/emon.sh
source "${SCRIPT_DIR}/lib/emon.sh"

WRAPPER_SERVER_PID=""
WRAPPER_BENCHMARK_PID=""
WRAPPER_READY_FILE=""

cleanup_wrapper() {
    local status=$?
    trap - EXIT INT TERM
    set +e
    if [[ -n "${WRAPPER_BENCHMARK_PID}" ]] \
        && kill -0 "${WRAPPER_BENCHMARK_PID}" 2>/dev/null; then
        kill -TERM "${WRAPPER_BENCHMARK_PID}" 2>/dev/null || true
        wait "${WRAPPER_BENCHMARK_PID}" 2>/dev/null || true
    fi
    if [[ -n "${WRAPPER_SERVER_PID}" ]] \
        && kill -0 "${WRAPPER_SERVER_PID}" 2>/dev/null; then
        kill -TERM "${WRAPPER_SERVER_PID}" 2>/dev/null || true
        wait "${WRAPPER_SERVER_PID}" 2>/dev/null || true
    fi
    [[ -z "${WRAPPER_READY_FILE}" ]] || rm -f -- "${WRAPPER_READY_FILE}"
    exit "${status}"
}

wait_for_wrapper_server() {
    local timeout_seconds="${SERVER_START_TIMEOUT_SECONDS:-7200}"
    [[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]] \
        || fail "SERVER_START_TIMEOUT_SECONDS must be a positive integer."
    local start_seconds
    start_seconds=$(date +%s)
    while [[ ! -s "${WRAPPER_READY_FILE}" ]]; do
        if ! kill -0 "${WRAPPER_SERVER_PID}" 2>/dev/null; then
            wait "${WRAPPER_SERVER_PID}" || true
            fail "Server exited before publishing readiness."
        fi
        (( $(date +%s) - start_seconds < timeout_seconds )) \
            || fail "Timed out after ${timeout_seconds}s waiting for server readiness."
        sleep 2
    done
}

run_composed_benchmark() {
    bash "${SCRIPT_DIR}/run_benchmark.sh" &
    WRAPPER_BENCHMARK_PID=$!

    local server_failed=false
    while kill -0 "${WRAPPER_BENCHMARK_PID}" 2>/dev/null; do
        if ! kill -0 "${WRAPPER_SERVER_PID}" 2>/dev/null; then
            server_failed=true
            kill -TERM "${WRAPPER_BENCHMARK_PID}" 2>/dev/null || true
            break
        fi
        sleep 5
    done

    local benchmark_status=0
    if wait "${WRAPPER_BENCHMARK_PID}"; then
        benchmark_status=0
    else
        benchmark_status=$?
    fi
    WRAPPER_BENCHMARK_PID=""

    if [[ "${server_failed}" == "true" ]]; then
        wait "${WRAPPER_SERVER_PID}" || true
        WRAPPER_SERVER_PID=""
        fail "Server supervisor exited during the benchmark."
    fi
    return "${benchmark_status}"
}

main() {
    (( $# == 0 )) \
        || fail "run_submission.sh does not accept positional arguments; configure it with environment variables."
    configure_emon
    if [[ "${SUBMISSION_SKIP_PREFLIGHT:-0}" != "1" ]]; then
        bash "${SCRIPT_DIR}/preflight.sh"
    fi

    require_command mktemp
    WRAPPER_READY_FILE=$(mktemp "${TMPDIR:-/tmp}/endpoints-agentic-ready.XXXXXX")
    rm -f -- "${WRAPPER_READY_FILE}"
    trap cleanup_wrapper EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    SERVER_READY_FILE="${WRAPPER_READY_FILE}" \
        bash "${SCRIPT_DIR}/start_server.sh" &
    WRAPPER_SERVER_PID=$!
    wait_for_wrapper_server

    local benchmark_status=0
    if run_composed_benchmark; then
        benchmark_status=0
    else
        benchmark_status=$?
    fi

    if kill -0 "${WRAPPER_SERVER_PID}" 2>/dev/null; then
        kill -TERM "${WRAPPER_SERVER_PID}" 2>/dev/null || true
        wait "${WRAPPER_SERVER_PID}" 2>/dev/null || true
    fi
    WRAPPER_SERVER_PID=""
    return "${benchmark_status}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
