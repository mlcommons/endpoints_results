#!/usr/bin/env bash
set -euo pipefail

(( $# == 2 )) || {
    printf 'Usage: bash lib/prepare_mlcflow.sh <absolute-venv> <absolute-automations-checkout>\n' >&2
    exit 2
}
venv_dir=$1
automations_dir=$2
[[ "${venv_dir}" == /* && "${automations_dir}" == /* ]] || {
    printf 'MLCflow environment and automation paths must be absolute.\n' >&2
    exit 2
}
helper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=configs/mlcflow.conf
source "${helper_dir}/../configs/mlcflow.conf"

# Provision only new checkouts; Git supplies clone/fetch/checkout diagnostics.
if [[ ! -e "${automations_dir}" && ! -L "${automations_dir}" ]]; then
    git clone --quiet --no-checkout --depth 1 -- \
        https://github.com/mlcommons/mlperf-automations.git "${automations_dir}" >&2
    git -C "${automations_dir}" fetch --quiet --depth 1 origin "${MLC_AUTOMATIONS_REF}" >&2
    git -C "${automations_dir}" checkout --quiet --detach "${MLC_AUTOMATIONS_REF}" >&2
fi
actual_head="$(git -C "${automations_dir}" rev-parse --verify 'HEAD^{commit}')"
[[ "${actual_head}" == "${MLC_AUTOMATIONS_REF}" ]] || {
    printf 'Automation checkout is at %s; expected %s. Choose a clean pinned checkout or a fresh path.\n' \
        "${actual_head}" "${MLC_AUTOMATIONS_REF}" >&2
    exit 1
}
checkout_status="$(git -C "${automations_dir}" status --porcelain=v1 --untracked-files=all)"
[[ -z "${checkout_status}" ]] || {
    printf 'Automation checkout has tracked or untracked changes. Choose a clean checkout or a fresh path.\n' >&2
    exit 1
}

# Do not recreate an incomplete existing environment. Valid environments are
# reused, and the package manager enforces the requested version and dependencies.
if [[ -e "${venv_dir}" || -L "${venv_dir}" ]]; then
    [[ -f "${venv_dir}/pyvenv.cfg" && -x "${venv_dir}/bin/python" ]] || {
        printf 'Incomplete MLCflow environment at %s; choose a fresh path.\n' "${venv_dir}" >&2
        exit 1
    }
fi
if command -v uv >/dev/null 2>&1; then
    [[ -x "${venv_dir}/bin/python" ]] || uv venv "${venv_dir}" >&2
    uv pip install --python "${venv_dir}/bin/python" "mlcflow==${MLCFLOW_VERSION}" >&2
else
    [[ -x "${venv_dir}/bin/python" ]] || python3 -m venv "${venv_dir}" >&2
    "${venv_dir}/bin/python" -m pip install "mlcflow==${MLCFLOW_VERSION}" >&2
fi
