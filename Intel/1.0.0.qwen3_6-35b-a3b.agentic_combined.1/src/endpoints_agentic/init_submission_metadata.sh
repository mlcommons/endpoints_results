#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if (( $# != 0 )); then
    printf 'init_submission_metadata.sh accepts no positional arguments.\n' >&2
    exit 2
fi

submission_venv=$(bash "${SCRIPT_DIR}/prepare_submission.sh")
submission_python="${submission_venv}/bin/python"
[[ -x "${submission_python}" ]] || {
    printf 'Prepared submission Python is missing: %s\n' "${submission_python}" >&2
    exit 1
}
exec "${submission_python}" "${SCRIPT_DIR}/lib/submission_metadata.py"
