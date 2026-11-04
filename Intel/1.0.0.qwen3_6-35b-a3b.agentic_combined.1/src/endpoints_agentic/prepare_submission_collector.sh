#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

assert_only=false
case $# in
    0) ;;
    1)
        [[ "$1" == "--assert-only" ]] || {
            printf 'Usage: prepare_submission_collector.sh [--assert-only]\n' >&2
            exit 2
        }
        assert_only=true
        ;;
    *)
        printf 'Usage: prepare_submission_collector.sh [--assert-only]\n' >&2
        exit 2
        ;;
esac

# shellcheck source=configs/mlcflow.conf
source "${SCRIPT_DIR}/configs/mlcflow.conf"
[[ "${MLCFLOW_VERSION:-}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
    printf 'MLCflow configuration must define a pinned MLCFLOW_VERSION.\n' >&2
    exit 1
}
[[ "${MLC_AUTOMATIONS_REF:-}" =~ ^[0-9a-f]{40}$ ]] || {
    printf 'MLCflow configuration must define a pinned 40-character MLC_AUTOMATIONS_REF.\n' >&2
    exit 1
}

if [[ -n "${MLCFLOW_VENV_DIR:-}" ]]; then
    venv_dir="${MLCFLOW_VENV_DIR}"
else
    [[ "${HOME:-}" == /* ]] || {
        printf 'HOME must be an absolute path when MLCFLOW_VENV_DIR is unset.\n' >&2
        exit 1
    }
    venv_dir="${HOME}/.cache/endpoints-agentic/mlcflow-${MLCFLOW_VERSION}-${MLC_AUTOMATIONS_REF}"
fi
[[ "${venv_dir}" == /* ]] || {
    printf 'MLCFLOW_VENV_DIR must be an absolute path.\n' >&2
    exit 1
}
automation_dir="${venv_dir}.mlperf-automations"
[[ "${automation_dir}" == /* ]] || {
    printf 'MLCflow automation checkout path must be absolute.\n' >&2
    exit 1
}
python_bin="${venv_dir}/bin/python"

required_utilities=(git jq pstree systemctl dmidecode lsblk df uname lscpu free)
missing_utilities=()
for utility in "${required_utilities[@]}"; do
    utility_path="$(type -P -- "${utility}" || true)"
    if [[ -z "${utility_path}" || ! -x "${utility_path}" ]]; then
        missing_utilities+=("${utility}")
    fi
done
if ((${#missing_utilities[@]})); then
    printf 'Required host collector utilities are missing: %s.\n' \
        "${missing_utilities[*]}" >&2
    printf 'Install these utilities with the host operating system package manager, then rerun this helper; it does not install host utilities.\n' >&2
    exit 1
fi

if [[ "${assert_only}" == "false" ]]; then
    flock_path="$(type -P -- flock || true)"
    [[ -n "${flock_path}" && -x "${flock_path}" ]] || {
        printf 'Required host collector preparation command is missing: flock. Install it with the host operating system package manager; this script does not install host utilities.\n' >&2
        exit 1
    }
    venv_parent="${venv_dir%/*}"
    [[ -n "${venv_parent}" ]] || venv_parent="/"
    umask 077
    mkdir -p -- "${venv_parent}"
    if ! { exec 9>"${venv_dir}.lock"; } 2>/dev/null; then
        printf 'Cannot open the MLCflow preparation lock at %s.lock.\n' \
            "${venv_dir}" >&2
        exit 1
    fi
    flock -x 9
fi

validate_automation_checkout() {
    local actual_head checkout_status
    if ! actual_head="$(
        GIT_OPTIONAL_LOCKS=0 git -C "${automation_dir}" \
            rev-parse --verify 'HEAD^{commit}' 2>/dev/null
    )"; then
        printf 'Cannot read the MLCflow automation Git HEAD at %s.\n' \
            "${automation_dir}" >&2
        return 1
    fi
    [[ "${actual_head}" == "${MLC_AUTOMATIONS_REF}" ]] || {
        printf 'MLCflow automation checkout is at %s; expected %s. Choose a clean pinned checkout or a fresh path.\n' \
            "${actual_head}" "${MLC_AUTOMATIONS_REF}" >&2
        return 1
    }

    if ! checkout_status="$(
        GIT_OPTIONAL_LOCKS=0 git -C "${automation_dir}" \
            status --porcelain=v1 --untracked-files=all 2>/dev/null
    )"; then
        printf 'Cannot read tracked and untracked status for the MLCflow automation checkout at %s.\n' \
            "${automation_dir}" >&2
        return 1
    fi
    [[ -z "${checkout_status}" ]] || {
        printf 'MLCflow automation checkout has tracked or untracked changes. Choose a clean checkout or a fresh path.\n' >&2
        return 1
    }
    automation_revision="${actual_head}"
}

validate_environment() {
    [[ -f "${venv_dir}/pyvenv.cfg" && -x "${python_bin}" ]] || {
        printf 'MLCflow Python environment is missing or incomplete at %s.\n' \
            "${venv_dir}" >&2
        return 1
    }

    local installed_version
    if ! installed_version="$(PYTHONDONTWRITEBYTECODE=1 "${python_bin}" -c '
import importlib.metadata
import sys

try:
    version = importlib.metadata.version("mlcflow")
except importlib.metadata.PackageNotFoundError:
    sys.exit(1)
print(version)
')"; then
        printf 'Cannot read the installed mlcflow package version using %s.\n' \
            "${python_bin}" >&2
        return 1
    fi
    [[ "${installed_version}" == "${MLCFLOW_VERSION}" ]] || {
        printf 'Installed mlcflow version is %s; expected %s in %s.\n' \
            "${installed_version:-missing}" "${MLCFLOW_VERSION}" "${venv_dir}" >&2
        return 1
    }

    local entrypoint
    for entrypoint in mlc mlcr; do
        [[ -f "${venv_dir}/bin/${entrypoint}" && -x "${venv_dir}/bin/${entrypoint}" ]] || {
            printf 'MLCflow entrypoint %s is missing or not executable at %s/bin/%s.\n' \
                "${entrypoint}" "${venv_dir}" "${entrypoint}" >&2
            return 1
        }
    done
}

if [[ "${assert_only}" == "false" ]]; then
    bash "${SCRIPT_DIR}/lib/prepare_mlcflow.sh" \
        "${venv_dir}" "${automation_dir}" >&2
fi

validate_environment
validate_automation_checkout

jq -cn \
    --arg venv "${venv_dir}" \
    --arg python "${python_bin}" \
    --arg automation "${automation_dir}" \
    --arg mlcflow_version "${MLCFLOW_VERSION}" \
    --arg automation_revision "${automation_revision}" \
    '{venv:$venv,python:$python,automation:$automation,mlcflow_version:$mlcflow_version,automation_revision:$automation_revision}'
