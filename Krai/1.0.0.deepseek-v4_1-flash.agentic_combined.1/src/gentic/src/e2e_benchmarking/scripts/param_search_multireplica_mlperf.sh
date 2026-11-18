#!/bin/bash
set -eo pipefail
umask 002
REPO_ROOT="$HOME/gentic"
VERSION="v1.0.0"
UBUNTU_IMG="ubuntu:24.04"
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)

echo "==============="
echo "Running Version: ${VERSION}"

# Arguments supplied by prepare_runs.py. todo: improve arg passing for maintainability
ARGS=("$@")

if (( ${#ARGS[@]} < 20 )); then
  echo "Received only ${#ARGS[@]} arguments." >&2
  exit 2
fi

BENCH_SIZE=${ARGS[0]}
SCENARIO=${ARGS[1]}
PLATFORM=${ARGS[2]}
BENCH_MODE=${ARGS[3]}
PATCH_NUM=${ARGS[4]}
SIGNATURE=${ARGS[5]}
MODEL=${ARGS[6]}
NUM_REPLICAS=${ARGS[7]}
TOTAL_GPUS=${ARGS[8]}
PROFILE=${ARGS[9]}
# todo: current no flag does not run vanilla flags in this script, to be impl'd
NOFLAG=${ARGS[10]}
VLLM_IMAGE=${ARGS[11]}
CONTAINER_IMG=${ARGS[12]}
DATA_FOLDER=${ARGS[13]}
KERNEL_RESULTS_DIR=${ARGS[14]}
BENCH_RESULTS=${ARGS[15]}
PROFILES_DIR=${ARGS[16]}
DOCKER_BENCH_RESULTS=${ARGS[17]}
DOCKER_PROFILES_DIR=${ARGS[18]}

ENV_COUNT=${ARGS[19]}
if [[ ! $ENV_COUNT =~ ^[0-9]+$ ]]; then
  echo "Environment entry count must be a non-negative integer: $ENV_COUNT" >&2
  exit 2
fi

VLLM_ARGS_START=$((20 + ENV_COUNT))
if (( ${#ARGS[@]} < VLLM_ARGS_START )); then
  echo "Expected $ENV_COUNT environment entries, but the argument list ended early." >&2
  exit 2
fi

VLLM_ENVS=("${ARGS[@]:20:ENV_COUNT}")
VLLM_ARGS=("${ARGS[@]:VLLM_ARGS_START}")

# GET PATCH NUM AND CREATE PATCH COMMAND
[ "$PATCH_NUM" = "no_patch" ] && PATCH_NUM=""
if [ -n "$PATCH_NUM" ]; then
  # TODO: make this flexible w.r.t. model selection
  PATCH_FILE="${REPO_ROOT}/engines/diffs/vllm_gentic_Qwen_Qwen3-VL-235B-A22B-Instruct_offline_vision_${PATCH_NUM}/integration.patch"
  PATCH_CMD="patch -p0 -d / < $PATCH_FILE && "
else
  PATCH_CMD=""
fi

# PROFILING VS NORMAL RUN - EASY TO DEBUG
if [ "$PROFILE" = "true" ]; then
  echo "=========================="
  echo "PROFILING RUN"
else
  echo "=========================="
  echo "Normal RUN"
fi

# SETUP =========================================

BASE_FOLDER="$HOME/gentic"
DOCKER_DATA_FOLDER="/root/mlperf_bench"

BENCH_MOUNTS="\
  -v $BASE_FOLDER:/gentic \
  -v $DATA_FOLDER:/root/mlperf_bench \
  -v $DATA_FOLDER/hf_cache:/root/.cache/huggingface \
  -v $DATA_FOLDER/hf_cache:/dataset_cache"

BASE_PORT=8000
MODEL_CONTAINER_PREFIX=gentic-mlperf-server
BENCH_CONTAINER=gentic-mlperf-bench
ROUTER_CONTAINER=gentic-mlperf-router
ROUTER_IMAGE=nginx:1.27-alpine

# Data-parallel attention (--data-parallel-size > 1): vLLM listens on SERVER_PORT_BASE and an nginx router on
# BASE_PORT pins every conversation (client X-Session-ID) to one DP rank via the X-data-parallel-rank header
DP=1
TP=$TOTAL_GPUS
for ((k = 0; k < ${#VLLM_ARGS[@]} - 1; k++)); do
  case ${VLLM_ARGS[k]} in
    --data-parallel-size)   DP=${VLLM_ARGS[k + 1]} ;;
    --tensor-parallel-size) TP=${VLLM_ARGS[k + 1]} ;;
  esac
done

ROUTED=0
if [ "$SCENARIO" = "endpoints" ]; then ROUTED=1; fi

SERVER_PORT_BASE=$BASE_PORT
if (( ROUTED )); then
  SERVER_PORT_BASE=9000
  if (( NUM_REPLICAS > 1 )); then
    echo "Router supports one server per node (got ${NUM_REPLICAS} servers with DP=${DP})" >&2
    exit 2
  fi
fi

if [ "$SCENARIO" = "endpoints" ]; then
  SWEBENCH_CONTAINER=gentic-mlperf-swebench
  # SWE-bench task images: one tar per image on shared storage, plus this run's image list
  SWEBENCH_IMAGE_CACHE="$DATA_FOLDER/docker_cache/swebench"
  SWEBENCH_IMAGES="$BENCH_RESULTS/swebench_images.txt"
  SWEBENCH_IMAGES_PID=""
fi

BENCH_YAML="${DOCKER_BENCH_RESULTS}/bench_config_resolved.yaml"
echo "Using run-specific container YAML (as container-mount-specific path): $BENCH_YAML"

# PROFILING CONFIG =========================================

DOCKER_PROFILER_DIR="/profiles"
NSYS_WRAP=""
if [ "$PROFILE" = "true" ]; then
  case "$PLATFORM" in
    (nvidia)
      NSYS_WRAP="nsys profile --sample=none --trace=cuda,nvtx --output /profiles/trace --force-overwrite true --trace-fork-before-exec=true --cuda-graph-trace=node -- "
      ;;
    (amd)
      # TODO: add ROCm profiling?
      ;;
  esac
fi

# UTILS =========================================

function cleanup() {
  if (( ROUTED )); then
    # Per-rank counters (prefix-cache hits/queries, running) for checking routing; no file if the server is down
    curl -sf --max-time 10 "http://localhost:${SERVER_PORT_BASE}/metrics" -o "$BENCH_RESULTS/vllm_metrics_final.prom" || true
  fi
  docker rm -f ${ROUTER_CONTAINER} >/dev/null 2>&1 || true

  for i in $(seq 0 $((NUM_REPLICAS - 1))); do
    docker rm -f ${MODEL_CONTAINER_PREFIX}-${i} >/dev/null 2>&1 || true
  done
  docker rm -f ${BENCH_CONTAINER} >/dev/null 2>&1 || true
  
  if [ "$SCENARIO" = "endpoints" ]; then
    docker rm -f ${SWEBENCH_CONTAINER} >/dev/null 2>&1 || true
    docker ps -aq --filter label=com.mlcommons.endpoints.swebench-run | xargs -r docker rm -f >/dev/null 2>&1 || true
    if [ -n "$SWEBENCH_IMAGES_PID" ]; then
      pkill -P "$SWEBENCH_IMAGES_PID" >/dev/null 2>&1 || true
      kill "$SWEBENCH_IMAGES_PID" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT
cleanup

# RUN MODEL =========================================

"$REPO_ROOT/src/docker_pull_w_cache.sh" "$VLLM_IMAGE"
"$REPO_ROOT/src/docker_pull_w_cache.sh" "$CONTAINER_IMG"
if (( ROUTED )); then
  "$REPO_ROOT/src/docker_pull_w_cache.sh" "$ROUTER_IMAGE"
fi

for i in $(seq 0 $((NUM_REPLICAS - 1))); do
  PORT=$((SERVER_PORT_BASE + i))
  GPU_START=$((i * TOTAL_GPUS))
  GPUS="${GPU_START}"
  if [ "$TOTAL_GPUS" -gt 1 ]; then
    for j in $(seq 1 $((TOTAL_GPUS - 1))); do
      GPUS+=",$((GPU_START + j))"
    done
  fi
  echo "Starting replica ${i} on port ${PORT} with GPUs ${GPUS}..."

  if [ "$PROFILE" = "true" ]; then
    LOG_FILE="$BENCH_RESULTS/model_server_profile_tp${TP}_dp${DP}_nr${NUM_REPLICAS}_${i}.log"
  else
    LOG_FILE="$BENCH_RESULTS/model_server_tp${TP}_dp${DP}_nr${NUM_REPLICAS}_${i}.log"
  fi

  DOCKER_CMD=(
    "docker" "run" "-d" "--rm"
    "--name" "${MODEL_CONTAINER_PREFIX}-${i}"
    "--entrypoint" "bash"
    "--ipc=host"
    "--network=host"
    "-v" "${DATA_FOLDER}/root_cache/replica-${i}:/root/.cache"
    "-v" "${DATA_FOLDER}/hf_cache:/root/.cache/huggingface"
  )
  case "$PLATFORM" in
    (nvidia)
      DOCKER_CMD+=("--gpus" "\"device=$GPUS\"")
      ;;
    (amd)
      DOCKER_CMD+=("--device=/dev/kfd" "--device=/dev/dri" "--group-add=video" "-e" "HIP_VISIBLE_DEVICES=${GPUS}")
      ;;
  esac

  for ENV_SPEC in "${VLLM_ENVS[@]}"; do
    DOCKER_CMD+=("-e" "$ENV_SPEC")
  done

  if [ "$PROFILE" = "true" ]; then
    DOCKER_CMD+=("-v" "${PROFILES_DIR}/replica-${i}:${DOCKER_PROFILER_DIR}")
  fi

  if [[ -n $PATCH_FILE ]]; then
    DOCKER_CMD+=("-v" "${PATCH_FILE}:${PATCH_FILE}")
  fi

  DOCKER_CMD+=("${VLLM_IMAGE}")

  SERVE_CMD=(
    "vllm" "serve" "$MODEL"
    "--host" "0.0.0.0"
    "--port" "$PORT"
    "${VLLM_ARGS[@]}"
  )
  printf -v SERVE_CMD_STRING '%q ' "${SERVE_CMD[@]}"
  DOCKER_CMD+=("-c" "umask 002 && ${PATCH_CMD}${NSYS_WRAP}${SERVE_CMD_STRING}")

  echo "=== Running Docker Command ==="
  printf '%s ' "${DOCKER_CMD[@]}"
  echo
  echo "=============================="

  "${DOCKER_CMD[@]}"

  docker logs -f ${MODEL_CONTAINER_PREFIX}-${i} 2>&1 |
    tee "$KERNEL_RESULTS_DIR/logs/current_model_server_tp${TP}_dp${DP}_nr${NUM_REPLICAS}_${i}.log" \
        "$LOG_FILE" \
        >/dev/null 2>&1 &
  echo "Replica ${i} starting on port ${PORT}, logs: $LOG_FILE"
done

# PREPARE SWE-BENCH IMAGES (in the background, while vLLM loads) =====
if [ "$BENCH_MODE" = "acc" ] || [ "$BENCH_MODE" = "both" ]; then
  echo "Preparing SWE-bench images in the background, log: $BENCH_RESULTS/swebench_images.log"
  (
    set -eo pipefail
    # 1. Image list from the client's own code: the run's bench yaml, the client's SWE-bench loader and
    #    option defaults, the scorer's first-N rule, and mini-swe-agent's image naming. The default
    #    workdir keeps the client's dataset cache inside this throwaway container, so the dataset is
    #    fetched from Hugging Face fresh, exactly like the benchmark container does.
    docker run --rm --network host \
      ${BENCH_MOUNTS} \
      --entrypoint bash \
      "$CONTAINER_IMG" \
      -c "umask 002 && python /gentic/src/e2e_benchmarking/scripts/build_swebench_image_manifest.py $BENCH_YAML $DOCKER_BENCH_RESULTS/swebench_images.txt"
    # 2. Make every image local: already present -> skip, tar on Lustre -> load, neither -> pull (with the
    #    submitter's docker login) and save. 4 at a time.
    xargs -a "$SWEBENCH_IMAGES" -P 4 -I{} "$REPO_ROOT/src/docker_pull_w_cache.sh" {} "$SWEBENCH_IMAGE_CACHE"
  ) > "$BENCH_RESULTS/swebench_images.log" 2>&1 &
  SWEBENCH_IMAGES_PID=$!
fi

# WAIT FOR HEALTHY SERVERS =====================================

echo "Waiting for all ${NUM_REPLICAS} replicas to become healthy..."
DEADLINE=$((SECONDS + 2700))

for i in $(seq 0 $((NUM_REPLICAS - 1))); do
  PORT=$((SERVER_PORT_BASE + i))
  until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    if [ $SECONDS -ge $DEADLINE ]; then
      echo "Timed out waiting for replica ${i} (port ${PORT})."
      docker logs ${MODEL_CONTAINER_PREFIX}-${i} --tail 200
      exit 1
    fi
    sleep 20
    echo "  replica ${i} (port ${PORT}) still waiting..."
  done
  echo "Replica ${i} healthy."
done

echo "Servers are running."

# START DP ROUTER =========================================

if (( ROUTED )); then
  ROUTER_TEMPLATE="$REPO_ROOT/src/e2e_benchmarking/configs/nginx_dp_router.conf"
  ROUTER_CONF="$BENCH_RESULTS/nginx_dp_router.conf"

  SPLIT=""
  for ((r = 0; r < DP - 1; r++)); do
    SPLIT+="    $(awk -v d="$DP" 'BEGIN { printf "%.2f", 100 / d }')% ${r};"$'\n'
  done
  SPLIT+="    * $((DP - 1));"
  CONF=$(<"$ROUTER_TEMPLATE")
  CONF="${CONF//@DP_SPLIT@/$SPLIT}"
  CONF="${CONF//@SERVER_PORT@/$SERVER_PORT_BASE}"
  CONF="${CONF//@ROUTER_PORT@/$BASE_PORT}"
  printf '%s\n' "$CONF" > "$ROUTER_CONF"

  docker run -d --rm --name ${ROUTER_CONTAINER} --network host \
    -v "$ROUTER_CONF:/etc/nginx/nginx.conf:ro" \
    -v "$BENCH_RESULTS:/logs" \
    "$ROUTER_IMAGE"
  
  ROUTER_DEADLINE=$((SECONDS + 120))
  until curl -sf "http://localhost:${BASE_PORT}/health" >/dev/null 2>&1; do
    if ! docker inspect ${ROUTER_CONTAINER} >/dev/null 2>&1; then
      echo "DP router exited during startup, see $BENCH_RESULTS/nginx_dp_router_error.log" >&2
      exit 1
    fi
    if (( SECONDS >= ROUTER_DEADLINE )); then
      echo "Router not healthy after 120 s (upstream 502?), see $BENCH_RESULTS/nginx_dp_router_error.log" >&2
      exit 1
    fi
    sleep 2
  done

  echo "Router healthy: ${DP} rank(s) behind port ${BASE_PORT}."
fi

{ echo "host: $(hostname)  $(date --iso-8601=seconds)"
  for img in "$VLLM_IMAGE" "$CONTAINER_IMG" "$ROUTER_IMAGE"; do
    echo "image: $img $(docker image inspect --format '{{.Id}} {{join .RepoDigests ","}}' "$img" 2>/dev/null)"; done
  nvidia-smi --query-gpu=index,name,driver_version,power.limit,clocks.max.sm,memory.total --format=csv || echo "nvidia-smi unavailable"
  (cd "$REPO_ROOT/src/e2e_benchmarking/tokenizers/deepseek-v4.1-flash" && sha256sum -c SHA256SUMS) || echo "TOKENIZER CHECK FAILED"
  ls "$DATA_FOLDER/hf_cache/hub/datasets--princeton-nlp--SWE-bench_Verified/snapshots/" 2>/dev/null || echo "SWE-bench Verified not cached yet (the client downloads it)"
} > "$BENCH_RESULTS/provenance.txt" 2>&1 || true

# Wait for the images before anything else: loads must not overlap the measured perf phase.
if [ -n "$SWEBENCH_IMAGES_PID" ]; then
  echo "Waiting for SWE-bench images..."
  if ! wait "$SWEBENCH_IMAGES_PID"; then
    echo "SWE-bench image preparation failed, see $BENCH_RESULTS/swebench_images.log" >&2
    exit 1
  fi
  SWEBENCH_IMAGES_PID=""
  echo "SWE-bench images ready: $(wc -l < "$SWEBENCH_IMAGES") images."
fi

for i in $(seq 20); do curl -s -o /dev/null -w '%{time_total}\n' "http://localhost:${BASE_PORT}/health" || echo "fail"; done > "$BENCH_RESULTS/net_baseline_s.txt"
echo "SERVER_READY $(date --iso-8601=ns)" >> "$BENCH_RESULTS/timeline.txt"

# RUN BENCHMARK =========================================

echo "All replicas healthy. Starting benchmark."



if [ "$SCENARIO" = "endpoints" ]; then
  mkdir -p "$BENCH_RESULTS/swebench"
  docker run -d --rm \
    --name ${SWEBENCH_CONTAINER} \
    --network host \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$BENCH_RESULTS/swebench:/artifacts" \
    --entrypoint bash \
    ${CONTAINER_IMG} \
    -c "umask 002 && cd /artifacts && export PATH=/opt/endpoints/src/inference_endpoint/evaluation/swebench_service/.venv/bin:\$PATH && python -m swebench_service --host 127.0.0.1 --port 18080 --artifact-root /artifacts --allow-unauthenticated"
  docker logs -f ${SWEBENCH_CONTAINER} > "$BENCH_RESULTS/swebench_service.log" 2>&1 &
  until curl -sf http://127.0.0.1:18080/health >/dev/null 2>&1; do
    if ! docker inspect ${SWEBENCH_CONTAINER} >/dev/null 2>&1; then
      echo "SWE-bench service exited during startup, see $BENCH_RESULTS/swebench_service.log" >&2
      exit 1
    fi
    sleep 5
  done
  echo "SWE-bench service healthy."
fi

BENCH_COMMAND="umask 002 && inference-endpoint benchmark from-config -c $BENCH_YAML"
if [ "$SCENARIO" = "endpoints" ]; then
  BENCH_COMMAND+=" --mode $BENCH_MODE"
fi

echo "BENCH_START $(date --iso-8601=ns)" >> "$BENCH_RESULTS/timeline.txt"

docker run --rm \
  --name ${BENCH_CONTAINER} \
  --network host \
  --shm-size=128g \
  ${BENCH_MOUNTS} \
  -e HF_TOKEN="${HF_TOKEN}" \
  --entrypoint bash \
  ${CONTAINER_IMG} \
  -c "$BENCH_COMMAND"
 
echo "FINISHED BENCHMARKING"

ls "$DATA_FOLDER/hf_cache/hub/datasets--princeton-nlp--SWE-bench_Verified/snapshots/" >> "$BENCH_RESULTS/provenance.txt" 2>&1 || true

if ! grep -qE '^Total samples failed: 0$' "$BENCH_RESULTS/report.txt"; then
  echo "Benchmark had failed samples or no report, see $BENCH_RESULTS" >&2
  exit 1
fi

# # Copy results ===============================================
# mkdir -p "${KERNEL_RESULTS_DIR}"
# cp -r "${BENCH_RESULTS}/." "${KERNEL_RESULTS_DIR}/"

# # Collect statistics =========================================

# if [ "$PROFILE" = "true" ]; then
#   echo "Stopping model containers (waiting for nsys to flush traces)..."
#   for i in $(seq 0 $((NUM_REPLICAS - 1))); do
#     docker stop --time=600 ${MODEL_CONTAINER_PREFIX}-${i} &
#   done
#   wait
#   echo "Model containers stopped."

#   echo "Generating kernel stats for replica-0..."
#       docker run --rm \
#     --entrypoint bash \
#     -v "${PROFILES_DIR}/replica-0:/profiles" \
#     -v "${KERNEL_RESULTS_DIR}:/results" \
#     ${VLLM_IMAGE} \
#     -c "nsys stats --report cuda_gpu_kern_sum --format csv --force-overwrite true --force-export=true \
#         --output /profiles/kernels /profiles/trace.nsys-rep &&
#       python3 -c '
# import csv, json, io
# with open(\"/profiles/kernels_cuda_gpu_kern_sum.csv\") as f:
#     raw = f.read()
# kernels = []
# for i, row in enumerate(csv.DictReader(io.StringIO(raw))):
#     if i >= 10: break
#     row = {k.strip(): v.strip() for k, v in row.items()}
#     kernels.append({
#         \"name\": row[\"Name\"],
#         \"call_count\": int(row[\"Instances\"]),
#         \"ext_id\": None,
#         \"bottleneck_rank\": i + 1,
#         \"pct_gpu_time\": float(row[\"Time (%)\"]),
#         \"total_dur\": round(float(row[\"Total Time (ns)\"]) / 1e6, 4)
#     })
# with open(\"/results/ref_kernels.json\", \"w\") as f:
#     json.dump(kernels, f, indent=2)
# print(\"Top 10 kernels -> /results/ref_kernels.json\")
# '
#     "
# fi
