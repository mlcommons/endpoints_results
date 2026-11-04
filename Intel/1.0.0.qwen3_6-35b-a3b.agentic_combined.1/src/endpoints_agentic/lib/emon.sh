#!/usr/bin/env bash

EMON_BINARY=/opt/intel/oneapi/vtune/latest/bin64/emon
EMON_DELAY_SECONDS=1800
EMON_DURATION_SECONDS=1800
EMON_MAX_TRACE_BYTES=1073741824
EMON_PID=""
EMON_LOCK_FD=""
EMON_STATE=disabled
EMON_REASON=""
EMON_STARTED_UTC=""
EMON_ENDED_UTC=""
EMON_EXIT_CODE=""
EMON_STOP_EXIT_CODE=""

configure_emon() {
    COLLECT_EMON="${COLLECT_EMON-0}"
    [[ "${COLLECT_EMON}" == 0 || "${COLLECT_EMON}" == 1 ]] \
        || fail "COLLECT_EMON must be 0 or 1."
}

write_emon_manifest_entries() {
    printf 'COLLECT_EMON=%q\n' "${COLLECT_EMON:-0}"
    printf 'EMON_BINARY=%q\n' "${EMON_BINARY}"
    printf 'EMON_INTERVAL_SECONDS=1\n'
    printf 'EMON_DELAY_SECONDS=%s\n' "${EMON_DELAY_SECONDS}"
    printf 'EMON_DURATION_SECONDS=%s\n' "${EMON_DURATION_SECONDS}"
    printf 'EMON_MAX_TRACE_BYTES=%s\n' "${EMON_MAX_TRACE_BYTES}"
}

write_emon_status() {
    local bytes=0 warnings=0
    [[ ! -f "${EMON_DIR}/emon.dat" ]] \
        || bytes=$(stat -c %s "${EMON_DIR}/emon.dat")
    [[ ! -f "${EMON_DIR}/collector.log" ]] \
        || warnings=$(grep -Eic 'warning' "${EMON_DIR}/collector.log" || true)
    {
        printf 'HOST=%q\n' "$(uname -n)"
        write_emon_manifest_entries
        printf 'STATE=%q\n' "${EMON_STATE}"
        printf 'STOP_REASON=%q\n' "${EMON_REASON}"
        printf 'START_TIME_UTC=%q\n' "${EMON_STARTED_UTC}"
        printf 'END_TIME_UTC=%q\n' "${EMON_ENDED_UTC}"
        printf 'TRACE_BYTES=%s\n' "${bytes}"
        printf 'COLLECTOR_EXIT_CODE=%q\n' "${EMON_EXIT_CODE}"
        printf 'STOP_EXIT_CODE=%q\n' "${EMON_STOP_EXIT_CODE}"
        printf 'WARNING_COUNT=%s\n' "${warnings}"
    } >"${EMON_DIR}/status.env" \
        || log "WARNING: Could not write EMON status: ${EMON_DIR}/status.env"
}

prepare_emon() {
    [[ "${COLLECT_EMON}" == 1 ]] || return 0
    EMON_DIR="${RUN_DIR}/emon"
    EMON_STATE=unavailable
    if ! mkdir -p "${EMON_DIR}"; then
        log "WARNING: Cannot create EMON artifact directory; continuing without collection."
        return 0
    fi
    local command
    for command in flock pgrep stat timeout; do
        if ! command -v "${command}" >/dev/null; then
            EMON_REASON="missing_${command}"
            log "WARNING: EMON requires ${command}; continuing without collection."
            write_emon_status
            return 0
        fi
    done
    if [[ ! -x "${EMON_BINARY}" ]] \
        || ! timeout 15 "${EMON_BINARY}" -v >"${EMON_DIR}/version.log" 2>&1 \
        || grep -Eiq 'error|failed|problem accessing' "${EMON_DIR}/version.log"; then
        EMON_REASON=driver_unavailable
        log "WARNING: EMON executable or driver access is unavailable; see ${EMON_DIR}/version.log and docs/emon/README.md."
    else
        EMON_STATE=waiting
    fi
    write_emon_status
}

begin_emon_run() {
    EMON_BENCHMARK_START_SECONDS=${SECONDS}
}

tick_emon() {
    [[ "${EMON_STATE}" == waiting || "${EMON_STATE}" == collecting ]] || return 0
    if [[ "${EMON_STATE}" == waiting ]]; then
        (( SECONDS - EMON_BENCHMARK_START_SECONDS >= EMON_DELAY_SECONDS )) || return 0
        if ! { exec {EMON_LOCK_FD}>"${TMPDIR:-/tmp}/endpoints-agentic-emon-${UID}.lock"; } \
            || ! flock -n "${EMON_LOCK_FD}" \
            || pgrep -x emon >/dev/null; then
            log "WARNING: EMON is already in use or its lock is unavailable; skipping collection."
            finish_emon collector_busy
            return 0
        fi
        EMON_STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        EMON_COLLECTION_START_SECONDS=${SECONDS}
        (
            set +u
            set --
            # The vendor script changes SCRIPT_DIR and references unset variables.
            # shellcheck disable=SC1091
            source "${EMON_BINARY%/bin64/emon}/sep_vars.sh" >/dev/null
            exec "${EMON_BINARY}" -collect-edp -t 1 -f "${EMON_DIR}/emon.dat"
        ) >"${EMON_DIR}/collector.log" 2>&1 &
        EMON_PID=$!
        EMON_STATE=collecting
        log "EMON collection started: ${EMON_DIR}/emon.dat"
        write_emon_status
    fi
    if ! kill -0 "${EMON_PID}" 2>/dev/null; then
        finish_emon collector_exited
    elif (( SECONDS - EMON_COLLECTION_START_SECONDS >= EMON_DURATION_SECONDS )); then
        finish_emon duration_limit
    elif [[ -f "${EMON_DIR}/emon.dat" ]] \
        && (( $(stat -c %s "${EMON_DIR}/emon.dat") >= EMON_MAX_TRACE_BYTES )); then
        finish_emon size_limit
    fi
}

finish_emon() {
    local reason=$1 attempt forced=false
    [[ "${EMON_STATE}" == waiting || "${EMON_STATE}" == collecting \
        || -n "${EMON_LOCK_FD}" ]] || return 0
    EMON_REASON="${reason}"
    if [[ -n "${EMON_PID}" ]]; then
        EMON_STOP_EXIT_CODE=0
        if kill -0 "${EMON_PID}" 2>/dev/null; then
            if timeout 10 "${EMON_BINARY}" -stop >"${EMON_DIR}/stop.log" 2>&1; then
                EMON_STOP_EXIT_CODE=0
            else
                EMON_STOP_EXIT_CODE=$?
            fi
            for ((attempt=0; attempt<100; attempt++)); do
                kill -0 "${EMON_PID}" 2>/dev/null || break
                sleep 0.1
            done
            if kill -0 "${EMON_PID}" 2>/dev/null; then
                forced=true
                kill -TERM "${EMON_PID}" 2>/dev/null || true
                for ((attempt=0; attempt<50; attempt++)); do
                    kill -0 "${EMON_PID}" 2>/dev/null || break
                    sleep 0.1
                done
                kill -KILL "${EMON_PID}" 2>/dev/null || true
            fi
        fi
        if wait "${EMON_PID}"; then
            EMON_EXIT_CODE=0
        else
            EMON_EXIT_CODE=$?
        fi
        EMON_PID=""
        EMON_STATE=completed
        if [[ "${EMON_EXIT_CODE}" != 0 || "${EMON_STOP_EXIT_CODE}" != 0 \
            || "${forced}" == true || "${reason}" == collector_exited \
            || ! -s "${EMON_DIR}/emon.dat" ]]; then
            EMON_STATE=failed
            log "WARNING: EMON collection failed or stopped unexpectedly; preserved artifacts: ${EMON_DIR}"
        elif grep -Eiq 'warning' "${EMON_DIR}/collector.log"; then
            EMON_STATE=completed_with_warnings
        fi
        grep -Ei 'warning|error|failed' "${EMON_DIR}/collector.log" >&2 || true
        EMON_ENDED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        log "EMON collection stopped (${EMON_REASON}): ${EMON_DIR}"
    elif [[ "${EMON_STATE}" == waiting ]]; then
        EMON_STATE=skipped
    fi
    if [[ -n "${EMON_LOCK_FD}" ]]; then
        exec {EMON_LOCK_FD}>&-
        EMON_LOCK_FD=""
    fi
    write_emon_status
}
