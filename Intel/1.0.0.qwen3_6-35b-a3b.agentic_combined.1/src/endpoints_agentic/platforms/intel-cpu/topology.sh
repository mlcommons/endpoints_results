#!/usr/bin/env bash

platform_probe() {
    command -v lscpu >/dev/null 2>&1 || return 1
    local cpu_summary vendor
    cpu_summary=$(LC_ALL=C lscpu 2>/dev/null) || return 1
    vendor=$(awk -F: '$1 == "Vendor ID" { sub(/^[[:space:]]+/, "", $2); print $2 }' \
        <<<"${cpu_summary}")
    [[ "${vendor}" == "GenuineIntel" ]]
}

discover_host_cpu_profile() {
    HOST_CPU_PROFILE="generic"
    HOST_CPU_VENDOR=""
    HOST_CPU_MODEL=""

    local cpu_summary
    cpu_summary=$(LC_ALL=C lscpu) \
        || fail "Could not read Intel CPU information from lscpu."
    HOST_CPU_VENDOR=$(awk -F: '$1 == "Vendor ID" { sub(/^[[:space:]]+/, "", $2); print $2 }' \
        <<<"${cpu_summary}")
    HOST_CPU_MODEL=$(awk -F: '$1 == "Model name" { sub(/^[[:space:]]+/, "", $2); print $2 }' \
        <<<"${cpu_summary}")
    [[ "${HOST_CPU_VENDOR}" == "GenuineIntel" ]] \
        || fail "The selected Intel CPU family requires an Intel CPU."
}

platform_discover_devices() {
    discover_host_cpu_profile
}
