#!/usr/bin/env bash
# DRAM clock lock for VR-NVL72 point runs (GPU_MEM_CLOCK_LOCK=<MHz>, 4752 by default): lock
# with `nvidia-smi -lmc V,V`, reset with `-lmc 3186,4752`, and leave SM clocks unlocked.
#
# The point's server.sh runs `start` in the serve_endpoint step, after it creates RUN_DIR.
# SFlow stops that step through srun, which forwards the SIGTERM as SIGKILL, so
# a trap there cannot reset the clocks. `start` therefore launches `guard` as a
# separate job step and waits until the lock is confirmed. The guard holds the
# lock until Slurm ends the job; completion, scancel and the time limit all send
# it SIGTERM (KillWait before SIGKILL), and it resets on exit. Files in RUN_DIR:
#   gpu-clocks-start.txt  lock attempts, `nvidia-smi -q -d CLOCK` and the CSV query
#   gpu-clocks.csv        the CSV query at lock time, then every 30 s until exit
#   gpu-clocks-end.txt    reset attempts and the same snapshot after the reset
#   gpu-clock-lock.log    guard and srun messages
set -euo pipefail

readonly RESET_MEM_CLOCKS=3186,4752
readonly QUERY=timestamp,index,clocks.mem,clocks.max.mem,clocks.sm,clocks.max.sm,clocks_event_reasons.active
readonly LOG_INTERVAL_S=30
lock=${GPU_MEM_CLOCK_LOCK:?Set GPU_MEM_CLOCK_LOCK}
[[ "$lock" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid GPU_MEM_CLOCK_LOCK: $lock" >&2; exit 2; }
out=${RUN_DIR:?Set RUN_DIR}
log=$out/gpu-clock-lock.log

smi() { TZ=UTC timeout 60 nvidia-smi "$@"; }
now() { date -u +%FT%TZ; }
snapshot() {
  echo "# nvidia-smi -q -d CLOCK ($(now))"
  smi -q -d CLOCK
  echo "# nvidia-smi --query-gpu=$QUERY --format=csv"
  smi --query-gpu="$QUERY" --format=csv
}
# Every visible GPU reports the requested memory clock.
mem_clocks_at() {
  local clocks
  clocks=$(smi --query-gpu=clocks.mem --format=csv,noheader,nounits) || return 1
  echo "clocks.mem: ${clocks//$'\n'/ }"
  [[ -n "$clocks" ]] && ! grep -qvx "$1" <<< "$clocks"
}

start() {
  srun --overlap --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --nodelist="${SLURMD_NODENAME:-$(hostname -s)}" \
    --cpu-bind=none --job-name=gpu_mem_clock bash "${BASH_SOURCE[0]}" guard < /dev/null >> "$log" 2>&1 &
  local guard_srun=$! waited
  for ((waited = 0; waited < 300; waited++)); do
    if [[ -e "$out/gpu-clocks-start.txt" ]]; then
      echo "GPU memory clocks locked at $lock MHz; guard step holds the lock until the job ends ($out/gpu-clocks-start.txt)"
      return 0
    fi
    kill -0 "$guard_srun" 2> /dev/null || break
    sleep 1
  done
  echo "GPU memory clock lock at $lock MHz not confirmed after ${waited} s; see $log" >&2
  tail -n 20 "$log" >&2 || true
  return 1
}

guard() {
  exec >> "$log" 2>&1 < /dev/null
  echo "$(now) guard step $SLURM_JOB_ID.${SLURM_STEP_ID:-?} on $(hostname -s): lock $lock,$lock"
  sleeper= locked=0
  trap 'finish $?' EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
  trap 'exit 129' HUP
  # Mark first so a partial lock is still reset.
  locked=1
  local attempt confirmed=0 partial=$out/gpu-clocks-start.partial
  for attempt in 1 2 3 4 5 6; do
    echo "# attempt $attempt ($(now)): sudo -n nvidia-smi -lmc $lock,$lock"
    if timeout 60 sudo -n nvidia-smi -lmc "$lock,$lock" && mem_clocks_at "$lock"; then
      confirmed=1
      break
    fi
    sleep 5
  done > "$partial" 2>&1
  if (( ! confirmed )); then
    mv "$partial" "$out/gpu-clocks-lock-failed.txt"
    echo "$(now) memory clocks not confirmed at $lock MHz:"
    cat "$out/gpu-clocks-lock-failed.txt"
    exit 1
  fi
  snapshot >> "$partial" 2>&1
  # A failed sample must not end the guard and release the lock.
  set +e
  smi --query-gpu="$QUERY" --format=csv > "$out/gpu-clocks.csv"
  mv "$partial" "$out/gpu-clocks-start.txt"
  echo "$(now) locked; logging every $LOG_INTERVAL_S s"
  while :; do
    sleep "$LOG_INTERVAL_S" &
    sleeper=$!
    wait "$sleeper"
    smi --query-gpu="$QUERY" --format=csv,noheader >> "$out/gpu-clocks.csv"
  done
}

finish() {
  local rc=$1 attempt reset=none
  trap - EXIT TERM INT HUP
  set +e
  # A signal during a redirected sample leaves stdout on gpu-clocks.csv; log from here on.
  exec >> "$log" 2>&1
  [[ -n "$sleeper" ]] && kill "$sleeper" 2> /dev/null
  [[ -e "$out/gpu-clocks.csv" ]] && smi --query-gpu="$QUERY" --format=csv,noheader >> "$out/gpu-clocks.csv"
  if (( locked )); then
    reset=failed
    {
      echo "# guard ending on status $rc ($(now))"
      for attempt in 1 2 3; do
        echo "# reset attempt $attempt ($(now)): sudo -n nvidia-smi -lmc $RESET_MEM_CLOCKS"
        if timeout 60 sudo -n nvidia-smi -lmc "$RESET_MEM_CLOCKS"; then reset=ok; break; fi
        sleep 3
      done
      snapshot
    } > "$out/gpu-clocks-end.txt" 2>&1
  fi
  # Slurm's SIGTERM at job end is the guard's normal exit; a failed reset is not.
  (( rc == 143 )) && rc=0
  [[ "$reset" == failed ]] && rc=1
  echo "$(now) guard exit status $rc (ended on $1, reset $reset)"
  exit "$rc"
}

case "${1:-start}" in
  start) start ;;
  guard) guard ;;
  *) echo "usage: $0 [start|guard]" >&2; exit 2 ;;
esac
