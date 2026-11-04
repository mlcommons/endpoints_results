#!/usr/bin/env bash
# acc/both sweep of the Kimi-K3 v1.0.3 SWE recipes on GKE, with the drop's SWE-bench service.
#
#   v1.0.3/kimi-k3/swe/run_swe_sweep.sh               # MODE=acc, POINTS="24" (first acc run)
#   MODE=both POINTS="24 48" v1.0.3/kimi-k3/swe/run_swe_sweep.sh
#   GROUP=v1.0.3/kimi-k3/with-pull-secret/swe_acc/8nodes POINTS="24" \
#     v1.0.3/kimi-k3/swe/run_swe_sweep.sh
#   v1.0.3/kimi-k3/swe/run_swe_sweep.sh 48            # resume at point_48
#
# (paths relative to examples/self_contained/kubernetes)
#
# What it does:
#   0. Runs `pilot/swe_pilot.py --check` (offline pin, ConfigMap, image and SWE_CONTRACT check).
#   1. Checks that the SWE point recipes are current
#      (`generate_point_recipes.py [--image-pull-secret] --swe $MODE --check --points $POINTS`).
#   2. Starts `pilot/swe_pilot.py serve` in the background. For every NEW sflow run folder of
#      the watched points in `<group>/point_<C>/sflow_output/<run id>/` whose rendered
#      client.yaml has `eval_method: swe_bench_scorer`, it starts `kimi-swe-host-<alloc>` on
#      that run's client node (node 2 for C12..C80, node 4 for C192..C1024): the v1.0.1 drop's
#      swe_service task (swe-service.py and its Enroot sandbox transport, byte-identical). The
#      pod writes the connection file to `<node-local-ssd>/kimi-swe/
#      handoff/<run id>/swe-connection.json`, where the recipe's benchmark task waits for it.
#      When the run's benchmark pod ends, serve stops the host pod (teardown + leak scan
#      covering the run's durable report on the node) and copies the service's run directory
#      via `kimi-swe-collect-<alloc>` to `<run dir>/swe_service/`.
#   3. Once serve is watching, runs `tools/run_sweep.sh <group>` with `MODE_VAR=BENCHMARK_MODE`
#      (or, if `SKIP_WAIT_IDLE=1` for concurrent paired 8+8 runs on a 16-node pool, launches
#      each point directly without `wait_idle`).
#   4. Stops serve with SIGTERM; it finishes the current run's teardown and copy first.
#   Exit status: non-zero if the sweep or serve reported a problem.
#
# Prerequisites (once per cluster; see swe/README.md, "Production"):
#   python3 -B v1.0.3/kimi-k3/swe/pilot/swe_pilot.py configmaps
#   python3 -B v1.0.3/kimi-k3/swe/pilot/swe_pilot.py secret
#
# Stop everything: kill -TERM -- -<process group id> (serve tears the SWE pod down first).
set -uo pipefail

export PYTHONDONTWRITEBYTECODE=1

SWE="$(cd "$(dirname "$0")" && pwd)"
K8S_ROOT="$(cd "${SWE}/../../.." && pwd)"
SFLOW_VENV="${SFLOW_VENV:<workdir>/sflow_workspace/.venv}"
if [ -x "${SFLOW_VENV}/bin/python3" ]; then
  PATH="${SFLOW_VENV}/bin:${PATH}"
fi

MODE="${MODE:-acc}"
case "${MODE}" in
  acc|both) ;;
  *) echo "MODE must be acc or both (got '${MODE}')" >&2; exit 2 ;;
esac

GROUP="${GROUP:-v1.0.3/kimi-k3/swe_${MODE}/8nodes}"
case "${GROUP}" in
  /*) GROUP_DIR="${GROUP}" ;;
  *)  GROUP_DIR="${K8S_ROOT}/${GROUP}" ;;
esac
if [ ! -d "${GROUP_DIR}" ] || [ -L "${GROUP_DIR}" ]; then
  echo "no such node group directory: ${GROUP_DIR}" >&2
  exit 2
fi
GROUP_BASE="$(basename "${GROUP_DIR}")"
MODE_DIR="$(dirname "${GROUP_DIR}")"
MODE_BASE="$(basename "${MODE_DIR}")"
if ! [[ "${GROUP_BASE}" =~ ^[0-9]{1,3}nodes$ ]] || [ "${MODE_BASE}" != "swe_${MODE}" ]; then
  echo "${GROUP_DIR} is not a SWE node group for MODE=${MODE} (expected .../swe_${MODE}/<N>nodes; never point run_swe_sweep.sh at a perf group)" >&2
  exit 2
fi

if [ -z "${POINTS:-}" ]; then
  if [ "${GROUP_BASE}" = "8nodes" ] && [ -d "${GROUP_DIR}/point_24" ]; then
    POINTS="24"
  else
    POINTS="$(ls -d "${GROUP_DIR}"/point_* 2>/dev/null | sed 's#.*/point_##' | sort -n | tr '\n' ' ' | sed 's/ $//')"
  fi
fi
if [ -z "${POINTS}" ]; then
  echo "no points selected in ${GROUP_DIR}" >&2
  exit 2
fi
for C in ${POINTS}; do
  if ! [[ "${C}" =~ ^[0-9]{1,5}$ ]] || [ ! -d "${GROUP_DIR}/point_${C}" ] || [ -L "${GROUP_DIR}/point_${C}" ]; then
    echo "no point folder point_${C} in ${GROUP_DIR}" >&2
    exit 2
  fi
  RECIPE_FILE="${GROUP_DIR}/point_${C}/kimi_k3_gke_disagg_point${C}_swe_${MODE}.yaml"
  if [ ! -f "${RECIPE_FILE}" ] || [ -L "${RECIPE_FILE}" ] || [ ! -x "${GROUP_DIR}/point_${C}/launch_test.sh" ]; then
    echo "missing SWE recipe or executable launch_test.sh in ${GROUP_DIR}/point_${C}" >&2
    exit 2
  fi
done
if [ -n "${1:-}" ]; then
  if ! [[ "$1" =~ ^[0-9]{1,5}$ ]] || ! [[ " ${POINTS} " == *" $1 "* ]]; then
    echo "start point '$1' is not in POINTS='${POINTS}'" >&2
    exit 2
  fi
fi

ts()  { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
say() { echo "=== SWE-SWEEP $(ts) $*"; }

GEN="${KIMI_SWE_GENERATOR:-${SWE}/../generate_point_recipes.py}"
RECIPES_ROOT="${KIMI_SWE_RECIPES_ROOT:-${SWE}/..}"
PS_ARGS=()
if [ "$(basename "$(dirname "${MODE_DIR}")")" = "with-pull-secret" ]; then
  PS_ARGS=(--image-pull-secret)
fi

# 0. Driver & contract self-check (offline, no kubectl).
if ! python3 -B "${SWE}/pilot/swe_pilot.py" --check \
    --generator "${GEN}" --recipes-root "${RECIPES_ROOT}"; then
  echo "swe_pilot.py --check failed" >&2
  exit 2
fi

# 1. Recipes must match the generator and drops before anything starts.
# KIMI_SWE_GEN_ARGS: only "--seed-fix" is accepted (the seedfix/ variant tree).
GEN_EXTRA=()
case "${KIMI_SWE_GEN_ARGS:-}" in
  "") ;;
  --seed-fix) GEN_EXTRA=(--seed-fix) ;;
  *) echo "KIMI_SWE_GEN_ARGS must be empty or --seed-fix" >&2; exit 2 ;;
esac
# shellcheck disable=SC2086  # POINTS is a validated list of numbers
if ! python3 -B "${GEN}" --dest "${RECIPES_ROOT}" "${PS_ARGS[@]}" "${GEN_EXTRA[@]}" \
    --swe "${MODE}" --check --points ${POINTS}; then
  echo "SWE recipes out of date; regenerate with generate_point_recipes.py --swe ${MODE} (then --check)" >&2
  exit 2
fi

# 2. serve, logging to a private file next to the sweep summary.
mkdir -p "${GROUP_DIR}/sweep"
SERVE_LOG="${GROUP_DIR}/sweep/swe-serve-$(date -u +%Y%m%dT%H%M%SZ).log"
( umask 077 && : > "${SERVE_LOG}" ) || exit 1
SERVE_PID=
stop_serve() {
  if [ -n "${SERVE_PID}" ] && kill -0 "${SERVE_PID}" 2>/dev/null; then
    say "stopping serve (pid ${SERVE_PID}); it finishes the current run's teardown and copy first"
    kill -TERM "${SERVE_PID}" 2>/dev/null
  fi
  if [ -n "${SERVE_PID}" ]; then
    wait "${SERVE_PID}"
    SERVE_RC=$?
    SERVE_PID=
    say "serve exited with ${SERVE_RC}; log: ${SERVE_LOG}"
  fi
}
SERVE_RC=0
trap stop_serve EXIT
trap 'exit 143' TERM INT

say "BEGIN MODE=${MODE} group=${GROUP_DIR} points=${POINTS} serve log=${SERVE_LOG} pid=$$"
# shellcheck disable=SC2086  # POINTS is a validated list of numbers
python3 -B "${SWE}/pilot/swe_pilot.py" serve --group "${GROUP_DIR}" --points ${POINTS} \
  >> "${SERVE_LOG}" 2>&1 &
SERVE_PID=$!
for _ in $(seq 1 120); do
  grep -q "serve: watching" "${SERVE_LOG}" && break
  kill -0 "${SERVE_PID}" 2>/dev/null || break
  sleep 1
done
if ! grep -q "serve: watching" "${SERVE_LOG}"; then
  echo "serve did not start watching; last lines of ${SERVE_LOG}:" >&2
  tail -n 20 "${SERVE_LOG}" >&2
  exit 1
fi
say "serve is watching (pid ${SERVE_PID})"

# 3. The sweep itself.
if [ "${SKIP_WAIT_IDLE:-0}" = "1" ]; then
  # Concurrent paired mode (e.g. 8-node SWE run alongside another 8-node run on a 16-node pool):
  # tools/run_sweep.sh's wait_idle would block on the partner run's pods in namespace default.
  START_AT="${1:-}"
  STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
  SUMMARY="${GROUP_DIR}/sweep/summary-${MODE}.tsv"
  if [ ! -s "${SUMMARY}" ]; then
    python3 -B "${K8S_ROOT}/tools/extract_summary.py" --header > "${SUMMARY}"
  fi
  EXTRA=()
  if [ -n "${SFLOW_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${SFLOW_EXTRA_ARGS})
  fi
  SWEEP_RC=0
  STARTED=0
  for C in ${POINTS}; do
    if [ -n "${START_AT}" ] && [ "${STARTED}" -eq 0 ]; then
      [ "${C}" = "${START_AT}" ] && STARTED=1 || continue
    fi
    PDIR="${GROUP_DIR}/point_${C}"
    mkdir -p "${PDIR}/sflow_output"
    STAMP=$(date -u +"%Y%m%dT%H%M%SZ")
    LOG="${PDIR}/sflow_output/launch-${MODE}-${STAMP}.log"
    BEFORE=$(ls -1d "${PDIR}"/sflow_output/*/ 2>/dev/null | sort || true)
    say "START C${C} (SKIP_WAIT_IDLE=1) -> ${LOG}"
    ( cd "${PDIR}" && ./launch_test.sh -s "BENCHMARK_MODE=${MODE}" "${EXTRA[@]}" ) > "${LOG}" 2>&1
    RC=$?
    AFTER=$(ls -1d "${PDIR}"/sflow_output/*/ 2>/dev/null | sort || true)
    RUN_DIR=$(comm -13 <(printf '%s\n' "${BEFORE}") <(printf '%s\n' "${AFTER}") | tail -n 1)
    [ -z "${RUN_DIR}" ] && RUN_DIR=$(ls -1dt "${PDIR}"/sflow_output/*/ 2>/dev/null | head -n 1 || true)
    RUN_DIR="${RUN_DIR%/}"
    say "END C${C} exit=${RC} run_dir=${RUN_DIR:-none}"
    if [ -n "${RUN_DIR}" ] && [ -d "${RUN_DIR}" ]; then
      ROW=$(python3 -B "${K8S_ROOT}/tools/extract_summary.py" "${RUN_DIR}" --point "${C}" --mode "${MODE}" --rc "${RC}")
      printf '%s\n' "${ROW}" | tee -a "${SUMMARY}"
      STATUS=$(printf '%s' "${ROW}" | awk -F'\t' '{print $NF}')
    else
      STATUS="NO_RUN_DIR(rc=${RC})"
      printf 'C%s\t%s\t-\t-\t%s\n' "${C}" "${MODE}" "${STATUS}" | tee -a "${SUMMARY}"
    fi
    if [ "${STATUS}" != "OK" ]; then
      SWEEP_RC=1
      if [ "${STOP_ON_FAIL}" = "1" ]; then
        say "STOP on C${C} (${STATUS})"
        break
      fi
    fi
  done
else
  MODE="${MODE}" MODE_VAR=BENCHMARK_MODE POINTS="${POINTS}" \
    "${K8S_ROOT}/tools/run_sweep.sh" "${GROUP_DIR}" "$@"
  SWEEP_RC=$?
fi
say "sweep exited with ${SWEEP_RC}"

# 4.
stop_serve
trap - EXIT
if [ "${SWEEP_RC}" -ne 0 ] || [ "${SERVE_RC}" -ne 0 ]; then
  say "DONE with problems (sweep ${SWEEP_RC}, serve ${SERVE_RC})"
  exit 1
fi
say "DONE"
