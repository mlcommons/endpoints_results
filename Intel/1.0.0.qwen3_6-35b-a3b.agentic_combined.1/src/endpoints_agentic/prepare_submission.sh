#!/usr/bin/env bash
set -euo pipefail

(( $# == 0 )) || { echo "prepare_submission.sh accepts no positional arguments." >&2; exit 1; }
# This prepares only the campaign checker. Host MLCflow collection has a
# separate bootstrap and remains gated by metadata validation.
SUBMISSION_CLI_URL=https://github.com/mlcommons/endpoints-submission-cli.git
for required_command in git flock; do
    command -v "${required_command}" >/dev/null 2>&1 || {
        printf 'Required command not found: %s\n' "${required_command}" >&2
        exit 1
    }
done

if ! submission_remote_head="$(
    GIT_TERMINAL_PROMPT=0 \
    GIT_ASKPASS=/bin/false \
    SSH_ASKPASS=/bin/false \
        git -c credential.interactive=never ls-remote --exit-code --refs -- \
            "${SUBMISSION_CLI_URL}" refs/heads/main
)"; then
    printf 'Failed to resolve the current submission checker commit from %s (refs/heads/main).\n' \
        "${SUBMISSION_CLI_URL}" >&2
    exit 1
fi
if [[ "${submission_remote_head}" == *$'\n'* ||
      "${submission_remote_head}" == *$'\r'* ||
      "${submission_remote_head}" != *$'\t'* ]]; then
    printf 'Invalid refs/heads/main response from %s; expected one 40-hex commit.\n' \
        "${SUBMISSION_CLI_URL}" >&2
    exit 1
fi
SUBMISSION_CLI_REF="${submission_remote_head%%$'\t'*}"
submission_remote_ref="${submission_remote_head#*$'\t'}"
if [[ ! "${SUBMISSION_CLI_REF}" =~ ^[[:xdigit:]]{40}$ ||
      "${submission_remote_ref}" != refs/heads/main ||
      "${submission_remote_head}" != "${SUBMISSION_CLI_REF}"$'\t'refs/heads/main ]]; then
    printf 'Invalid refs/heads/main response from %s; expected one 40-hex commit.\n' \
        "${SUBMISSION_CLI_URL}" >&2
    exit 1
fi

submission_venv="${SUBMISSION_VENV_DIR:-${HOME}/.cache/endpoints-agentic/submission-cli-${SUBMISSION_CLI_REF}}"
[[ "${submission_venv}" == /* ]] || { echo "SUBMISSION_VENV_DIR must be absolute." >&2; exit 1; }
umask 077
mkdir -p "$(dirname "${submission_venv}")"
exec 9>"${submission_venv}.lock"
flock -x 9

checker_identity=""
if [[ -f "${submission_venv}/pyvenv.cfg" && -x "${submission_venv}/bin/python" ]]; then
    if checker_identity="$("${submission_venv}/bin/python" - "${SUBMISSION_CLI_REF}" <<'PY'
import importlib.metadata
import json
import sys

try:
    distribution = importlib.metadata.distribution("endpoints-submission-cli")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if not isinstance(commit, str):
        sys.exit(2)
    print(commit)
    if commit != sys.argv[1]:
        sys.exit(3)
    import yaml
    import submission_checker
except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
    sys.exit(4)
PY
)"; then
        if [[ "${checker_identity}" == "${SUBMISSION_CLI_REF}" ]]; then
            printf '%s\n' "${submission_venv}"
            exit 0
        fi
    fi
fi
if [[ "${checker_identity}" =~ ^[[:xdigit:]]{40}$ &&
      "${checker_identity}" != "${SUBMISSION_CLI_REF}" ]]; then
    printf 'Existing submission checker environment is at commit %s; current main is %s. Choose a fresh directory.\n' \
        "${checker_identity}" "${SUBMISSION_CLI_REF}" >&2
    exit 1
fi

submission_marker="${submission_venv}.endpoints-submission-ref"
if [[ -e "${submission_venv}" || -L "${submission_venv}" ]]; then
    [[ -f "${submission_marker}" && "$(<"${submission_marker}")" == "${SUBMISSION_CLI_REF}" ]] || {
        echo "Existing SUBMISSION_VENV_DIR is not a prepared campaign environment; choose a fresh directory." >&2
        exit 1
    }
else
    printf '%s\n' "${SUBMISSION_CLI_REF}" >"${submission_marker}"
fi
if command -v uv >/dev/null 2>&1; then
    [[ -x "${submission_venv}/bin/python" ]] || uv venv "${submission_venv}" >&2
    uv pip install --python "${submission_venv}/bin/python" \
        "git+https://github.com/mlcommons/endpoints-submission-cli.git@${SUBMISSION_CLI_REF}" >&2
else
    [[ -x "${submission_venv}/bin/python" ]] || python3 -m venv "${submission_venv}" >&2
    "${submission_venv}/bin/python" -m pip install \
        "git+https://github.com/mlcommons/endpoints-submission-cli.git@${SUBMISSION_CLI_REF}" >&2
fi
printf '%s\n' "${submission_venv}"
