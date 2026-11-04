#!/usr/bin/env bash

append_no_proxy_host() {
    local host=$1
    case ",${SERVICE_NO_PROXY}," in
        *",${host},"*) ;;
        *)
            if [[ -n "${SERVICE_NO_PROXY}" ]]; then
                SERVICE_NO_PROXY+=","
            fi
            SERVICE_NO_PROXY+="${host}"
            ;;
    esac
}

create_docker_proxy_config() {
    local config_dir=$1
    local config_path="${config_dir}/config.json"
    local effective_http_proxy=$2
    local effective_https_proxy=$3
    local temporary_config=""
    local command

    for command in chmod jq ln mktemp rm; do
        require_command "${command}"
    done

    if [[ -e "${config_dir}" || -L "${config_dir}" ]]; then
        [[ -d "${config_dir}" ]] \
            || fail "Docker client configuration path is not a directory: ${config_dir}"
    else
        if ! mkdir -p -- "${config_dir}"; then
            fail "Could not create Docker client configuration directory: ${config_dir}"
        fi
        if ! chmod 700 "${config_dir}"; then
            fail "Could not protect Docker client configuration directory ${config_dir}."
        fi
    fi
    [[ -w "${config_dir}" ]] \
        || fail "Docker client configuration directory is not writable: ${config_dir}"

    temporary_config=$(mktemp "${config_dir}/.config.json.tmp.XXXXXX") \
        || fail "Could not create a temporary Docker client configuration in ${config_dir}."
    if ! chmod 600 "${temporary_config}"; then
        rm -f -- "${temporary_config}"
        fail "Could not protect temporary Docker client configuration ${temporary_config}."
    fi

    if ! ENDPOINTS_AGENTIC_HTTP_PROXY="${effective_http_proxy}" \
        ENDPOINTS_AGENTIC_HTTPS_PROXY="${effective_https_proxy}" \
        ENDPOINTS_AGENTIC_NO_PROXY="${SERVICE_NO_PROXY}" \
        jq -n '
            def add_nonempty($key; $value):
                if ($value // "") == "" then . else . + {($key): $value} end;
            {
                proxies: {
                    default: (
                        {}
                        | add_nonempty("httpProxy"; $ENV.ENDPOINTS_AGENTIC_HTTP_PROXY)
                        | add_nonempty("httpsProxy"; $ENV.ENDPOINTS_AGENTIC_HTTPS_PROXY)
                        | add_nonempty("noProxy"; $ENV.ENDPOINTS_AGENTIC_NO_PROXY)
                    )
                }
            }
        ' >"${temporary_config}"
    then
        rm -f -- "${temporary_config}"
        fail "Could not generate Docker client proxy configuration."
    fi

    if ln -- "${temporary_config}" "${config_path}" 2>/dev/null; then
        rm -f -- "${temporary_config}"
        log "Created Docker client proxy configuration at ${config_path} (mode 0600)."
        return 0
    fi

    rm -f -- "${temporary_config}"
    if [[ -f "${config_path}" && -r "${config_path}" ]]; then
        log "Docker client configuration appeared concurrently; preserving ${config_path}."
        return 0
    fi
    if [[ -e "${config_path}" || -L "${config_path}" ]]; then
        fail "Docker client configuration appeared concurrently but is not a readable regular file: ${config_path}"
    fi
    fail "Could not atomically create Docker client configuration: ${config_path}"
}

configure_swebench_docker_client() {
    if [[ -n "${DOCKER_CONFIG:-}" ]]; then
        HOST_DOCKER_CONFIG_DIR="${DOCKER_CONFIG}"
    elif [[ -n "${HOME:-}" ]]; then
        HOST_DOCKER_CONFIG_DIR="${HOME}/.docker"
    else
        fail "HOME or an absolute DOCKER_CONFIG directory is required."
    fi
    [[ "${HOST_DOCKER_CONFIG_DIR}" == /* ]] \
        || fail "DOCKER_CONFIG must be an absolute directory path when set."

    if [[ -e "${HOST_DOCKER_CONFIG_DIR}" || -L "${HOST_DOCKER_CONFIG_DIR}" ]]; then
        [[ -d "${HOST_DOCKER_CONFIG_DIR}" ]] \
            || fail "Docker client configuration path is not a directory: ${HOST_DOCKER_CONFIG_DIR}"
    fi

    local effective_http_proxy="${HTTP_PROXY:-${http_proxy:-}}"
    local effective_https_proxy="${HTTPS_PROXY:-${https_proxy:-}}"
    local config_path="${HOST_DOCKER_CONFIG_DIR}/config.json"

    SERVICE_NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
    append_no_proxy_host localhost
    append_no_proxy_host 127.0.0.1
    append_no_proxy_host ::1

    SERVICE_DOCKER_CONFIG_ARGS=()
    if [[ -f "${config_path}" && -r "${config_path}" ]]; then
        SERVICE_DOCKER_CONFIG_ARGS=(
            --volume "${HOST_DOCKER_CONFIG_DIR}:/root/.docker:ro"
        )
        log "SWE-bench will use the host Docker client configuration (mounted read-only)."
    elif [[ -e "${config_path}" || -L "${config_path}" ]]; then
        fail "Docker client configuration exists but is not a readable regular file: ${config_path}"
    elif [[ -n "${effective_http_proxy}${effective_https_proxy}" ]]; then
        create_docker_proxy_config \
            "${HOST_DOCKER_CONFIG_DIR}" \
            "${effective_http_proxy}" \
            "${effective_https_proxy}"
        SERVICE_DOCKER_CONFIG_ARGS=(
            --volume "${HOST_DOCKER_CONFIG_DIR}:/root/.docker:ro"
        )
    else
        log "No host Docker client configuration found; SWE-bench task containers will use direct bridge egress."
    fi

    SERVICE_PROXY_ENV_ARGS=()
    local variable
    for variable in HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; do
        if [[ -n "${!variable:-}" ]]; then
            SERVICE_PROXY_ENV_ARGS+=(--env "${variable}")
        fi
    done
    SERVICE_PROXY_ENV_ARGS+=(
        --env "NO_PROXY=${SERVICE_NO_PROXY}"
        --env "no_proxy=${SERVICE_NO_PROXY}"
    )
}

check_swebench_task_network() {
    log "Checking nested SWE-bench task-container package access over Docker's default bridge."
    if ! docker run --rm \
        --user root \
        --volume /var/run/docker.sock:/var/run/docker.sock \
        "${SERVICE_DOCKER_CONFIG_ARGS[@]}" \
        --entrypoint docker \
        "${ENDPOINT_IMAGE}" \
        run \
        --rm \
        --entrypoint /bin/bash \
        "${SWEBENCH_NETWORK_PROBE_IMAGE}" \
        -ceu '
            if [ -n "${HTTP_PROXY:-}${HTTPS_PROXY:-}${http_proxy:-}${https_proxy:-}" ]; then
                echo "Nested task proxy configuration: present"
            else
                echo "Nested task proxy configuration: direct egress"
            fi
            if ! PIP_DEFAULT_TIMEOUT=15 python -m pip index versions asgiref \
                >/dev/null 2>/tmp/pip-network-probe.log
            then
                cat /tmp/pip-network-probe.log >&2
                exit 1
            fi
        '
    then
        fail "Nested SWE-bench task-container network preflight failed. Verify the host Docker client proxy configuration and default-bridge egress."
    fi
    log "Nested SWE-bench task-container package access is ready."
}

prepull_swebench_task_images() {
    [[ "${BENCHMARK_MODE}" == "acc" || "${BENCHMARK_MODE}" == "both" ]] \
        || return 0

    local config_template="${SCRIPT_DIR}/configs/qwen_agentic_benchmark.yaml"
    local instance_count image_list image completed_pid pid
    local -a task_images=() pull_pids=() remaining_pids=()
    local pull_failed=false
    local max_parallel_pulls=8
    if [[ -v SWEBENCH_INSTANCE_COUNT ]]; then
        instance_count="${SWEBENCH_INSTANCE_COUNT}"
    else
        require_single_yaml_key "${config_template}" num_instances
        instance_count=$(read_yaml_scalar "${config_template}" num_instances)
    fi
    config_validate_positive_integer_value "${instance_count}" SWEBENCH_INSTANCE_COUNT

    log "Resolving SWE-bench task images for the first ${instance_count} Verified instances."
    if ! image_list=$(docker run --rm -i \
        --network host \
        --user "$(id -u):$(id -g)" \
        --env HOME=/tmp \
        "${SERVICE_PROXY_ENV_ARGS[@]}" \
        --entrypoint /opt/swebench-venv/bin/python \
        "${ENDPOINT_IMAGE}" \
        - "${instance_count}" <<'PY'
import contextlib
import sys

with contextlib.redirect_stdout(sys.stderr):
    from datasets import load_dataset
    from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

selected = dataset.select(range(min(int(sys.argv[1]), len(dataset))))
if not selected:
    raise RuntimeError("SWE-bench Verified/test contains no instances")
for instance in selected:
    print(get_swebench_docker_image_name(instance))
PY
    ); then
        fail "Could not resolve the selected SWE-bench task images."
    fi
    [[ -n "${image_list}" ]] \
        || fail "No SWE-bench task images were selected."

    mapfile -t task_images <<<"${image_list}"
    for image in "${task_images[@]}"; do
        [[ -n "${image}" && ! "${image}" =~ [[:space:]] ]] \
            || fail "Invalid SWE-bench task image reference in resolved list."
    done

    for image in "${task_images[@]}"; do
        if docker image inspect "${image}" >/dev/null 2>&1; then
            continue
        fi

        if (( ${#pull_pids[@]} >= max_parallel_pulls )); then
            completed_pid=""
            if ! wait -n -p completed_pid "${pull_pids[@]}"; then
                pull_failed=true
            fi
            if [[ -z "${completed_pid}" ]]; then
                pull_failed=true
                break
            fi
            remaining_pids=()
            for pid in "${pull_pids[@]}"; do
                [[ "${pid}" == "${completed_pid}" ]] \
                    || remaining_pids+=("${pid}")
            done
            pull_pids=("${remaining_pids[@]}")
            [[ "${pull_failed}" == "false" ]] || break
        fi

        (
            log "Pulling SWE-bench task image ${image}."
            if docker pull "${image}" >/dev/null; then
                log "Pulled SWE-bench task image ${image}."
            else
                log "ERROR: Could not pull SWE-bench task image ${image}." >&2
                exit 1
            fi
        ) &
        pull_pids+=("$!")
    done

    for pid in "${pull_pids[@]}"; do
        if ! wait "${pid}"; then
            pull_failed=true
        fi
    done
    [[ "${pull_failed}" == "false" ]] \
        || fail "Could not pre-pull all selected SWE-bench task images."
    log "Selected SWE-bench task images are available locally."
}

start_swebench_service() {
    local token=$1
    local container_name="${CONTAINER_PREFIX}-swebench"
    register_ready_service \
        "swebench-service" \
        "${container_name}" \
        "http://127.0.0.1:${SWEBENCH_PORT}/health" \
        120

    log "Starting SWE-bench accuracy service on port ${SWEBENCH_PORT}."
    docker run -d \
        --name "${container_name}" \
        --network host \
        --user root \
        --env UV_PROJECT_ENVIRONMENT=/opt/swebench-venv \
        "${SERVICE_DOCKER_CONFIG_ARGS[@]}" \
        "${SERVICE_PROXY_ENV_ARGS[@]}" \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v "${SWEBENCH_ARTIFACT_ROOT}:${SWEBENCH_ARTIFACT_ROOT}" \
        --entrypoint uv \
        "${ENDPOINT_IMAGE}" \
        run \
        --no-sync \
        --project /mnt/inference-endpoint/src/inference_endpoint/evaluation/swebench_service \
        python -m swebench_service \
        --port "${SWEBENCH_PORT}" \
        --artifact-root "${SWEBENCH_ARTIFACT_ROOT}" \
        --auth-token "${token}" \
        >/dev/null

    wait_for_service "swebench-service" "SWE-bench service"
}

cleanup_swebench_task_containers() {
    [[ -d "${SWEBENCH_ARTIFACT_ROOT:-}" ]] || return 0

    local artifact_dir swebench_run_id
    local -a task_containers=()
    local -a artifact_dirs=()
    shopt -s nullglob
    artifact_dirs=("${SWEBENCH_ARTIFACT_ROOT}"/*)
    shopt -u nullglob

    for artifact_dir in "${artifact_dirs[@]}"; do
        [[ -d "${artifact_dir}" ]] || continue
        swebench_run_id=${artifact_dir##*/}
        [[ "${swebench_run_id}" =~ ^[0-9a-f]{32}$ ]] || continue
        mapfile -t task_containers < <(
            docker ps -aq \
                --filter "label=com.mlcommons.endpoints.swebench-run=${swebench_run_id}"
        )
        if (( ${#task_containers[@]} > 0 )); then
            log "Removing ${#task_containers[@]} run-owned SWE-bench task containers."
            docker rm -f "${task_containers[@]}" >/dev/null 2>&1 || true
        fi
    done
}
