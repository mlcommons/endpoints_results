#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
(( $# == 0 )) || { echo "run_full_submission.sh accepts no positional arguments; use environment variables." >&2; exit 1; }
submission_venv="$(bash "${SCRIPT_DIR}/prepare_submission.sh")"
export PATH="${submission_venv}/bin:${PATH}"
exec "${submission_venv}/bin/python" "${SCRIPT_DIR}/lib/submission.py"
