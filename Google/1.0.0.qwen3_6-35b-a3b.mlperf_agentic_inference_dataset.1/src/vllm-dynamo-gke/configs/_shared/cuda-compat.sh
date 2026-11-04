#!/usr/bin/env bash
# Decide CUDA forward compatibility explicitly, inside the server container, then exec the command.
#
# NGC CUDA images ship a forward-compat libcuda under $_CUDA_COMPAT_PATH (/usr/local/cuda/compat; the real files in
# lib.real, or lib) and normally let their entrypoint choose it. Pyxis/enroot launches skip that entrypoint, so
# the choice is made here:
#   - host kernel driver OLDER than the compat driver (CUDA_DRIVER_VERSION, e.g. a 580.x host driver vs a 615.x compat driver):
#     put the compat libcuda first on LD_LIBRARY_PATH;
#   - host driver the same or NEWER (e.g. a 620.x host driver): strip every compat entry, so the host libcuda is used
#     (a newer host driver with the older compat libcuda fails CUDA init with error 803).
# Nothing is baked into the image ENV. QWEN_DRIVER_VERSION_FILE / QWEN_HOST_DRIVER_VERSION override the host probe
# (tests); the decision is printed to stderr and, if QWEN_CUDA_COMPAT_LOG is set, written there.
set -euo pipefail
compat_root="${_CUDA_COMPAT_PATH:-/usr/local/cuda/compat}"
compat_version="${CUDA_DRIVER_VERSION:-}"
host="${QWEN_HOST_DRIVER_VERSION:-}"
if [[ -z "$host" ]]; then
  version_file="${QWEN_DRIVER_VERSION_FILE:-/proc/driver/nvidia/version}"
  if [[ -r "$version_file" ]]; then
    host="$(grep -oE 'Kernel Module( for [A-Za-z0-9_]+)? +[0-9]+(\.[0-9]+)+' "$version_file" | grep -oE '[0-9]+(\.[0-9]+)+$' | head -n1 || true)"
  fi
  if [[ -z "$host" ]] && command -v nvidia-smi > /dev/null 2>&1; then
    host="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
  fi
fi
compat_dir=""
# lib.real = discrete/datacenter GPUs (GB300, VR200); lib.real-nvgpu is the integrated-GPU (nvgpu) variant, never used here.
for candidate in "$compat_root/lib.real" "$compat_root/lib"; do
  if [[ -e "$candidate/libcuda.so.1" ]]; then compat_dir="$candidate"; break; fi
done
# Strip every compat entry first; the decision below may add exactly one back.
kept=()
IFS=':' read -r -a entries <<< "${LD_LIBRARY_PATH:-}"
for entry in ${entries[@]+"${entries[@]}"}; do
  [[ -z "$entry" || "$entry" == "$compat_root"* || "$entry" == /usr/local/cuda*/compat* ]] && continue
  kept+=("$entry")
done
older() {  # older A B: version A < version B (numeric, dotted)
  awk -v a="$1" -v b="$2" 'BEGIN { n = split(a, x, "."); m = split(b, y, "."); k = (n > m ? n : m)
    for (i = 1; i <= k; i++) { if (x[i] + 0 < y[i] + 0) exit 0; if (x[i] + 0 > y[i] + 0) exit 1 }; exit 1 }'
}
if [[ -z "$host" ]]; then
  decision="host driver unknown: compat off"
elif [[ -z "$compat_version" || -z "$compat_dir" ]]; then
  decision="host driver $host, no compat libcuda in the image: compat off"
elif older "$host" "$compat_version"; then
  kept=("$compat_dir" ${kept[@]+"${kept[@]}"})
  decision="host driver $host < compat $compat_version: compat ON ($compat_dir)"
else
  decision="host driver $host >= compat $compat_version: compat off"
fi
LD_LIBRARY_PATH="$(IFS=':'; printf '%s' "${kept[*]-}")"
export LD_LIBRARY_PATH VLLM_ENABLE_CUDA_COMPATIBILITY=0
printf 'cuda-compat: %s\n' "$decision" >&2
if [[ -n "${QWEN_CUDA_COMPAT_LOG:-}" ]]; then
  printf '%s\nLD_LIBRARY_PATH=%s\n' "$decision" "$LD_LIBRARY_PATH" > "$QWEN_CUDA_COMPAT_LOG"
fi
exec "$@"
