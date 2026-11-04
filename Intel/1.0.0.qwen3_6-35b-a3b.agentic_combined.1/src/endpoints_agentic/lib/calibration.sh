#!/usr/bin/env bash

calibration_prepare_directories() { :; }
calibration_prepare_model() { :; }
calibration_assert_prepared() { :; }

if [[ "${CALIBRATION_SUPPORTED:-false}" == "true" ]]; then
    [[ -f "${PLATFORM_DIR}/calibration.sh" ]] \
        || fail "Calibration is enabled but the selected family has no implementation: ${PLATFORM_DIR}/calibration.sh"
    # shellcheck disable=SC1090
    source "${PLATFORM_DIR}/calibration.sh"
fi
