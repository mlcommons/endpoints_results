#!/usr/bin/env bash
set -euo pipefail
POINT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$POINT_DIR/../run_point.sh" "$POINT_DIR" "$@"
