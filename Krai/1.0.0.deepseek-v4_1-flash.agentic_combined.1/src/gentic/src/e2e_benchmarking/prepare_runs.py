import argparse
import copy
import hashlib
import itertools
import json
import os
import re
import shlex
import shutil
from os import fdopen
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent

# PREFLIGHT_SCRIPT = SCRIPT_DIR / "param_search_preflight.sh"
SBATCH_SCRIPT = SCRIPT_DIR / "scripts" / "execute_runs.sh"
SHELL_SCRIPT = SCRIPT_DIR / "scripts" / "param_search_multireplica_mlperf.sh"
SERVER_CONFIGS_DIR = SCRIPT_DIR / "configs" / "server_configs"
BENCH_CONFIGS_DIR = SCRIPT_DIR / "configs" / "bench_configs"

# ENDPOINT_CONFIG_DIR = SCRIPT_DIR / "endpoint_configs"
# CONTAINER_CONFIG_DIR = SCRIPT_DIR / ".." / "configs" / "multireplica"

# Large-storage data folder (results, image cache, HF cache, dataset). Override with GENTIC_DATA_DIR on another
# cluster; unset, it is the folder the submitted runs used.
DATA_FOLDERS = {
    "nvidia": Path(os.environ.get("GENTIC_DATA_DIR", "/hpelustre/krai/krai_gentic")).resolve(),
    "amd": Path(os.environ.get("GENTIC_DATA_DIR", "/mnt/llm_data/krai/krai-gentic")).resolve(),
}
DOCKER_DATA_FOLDER = Path("/root/mlperf_bench")

VLLM_IMAGE = {
    "nvidia": "wongcannan/vllm-v0.30.0-nsys-withsrc:v1.0", # vllm v0.30.0, compatible with Qwen3-vlm and DeepSeek-V4 series
    # "nvidia": "kagni/vllm-nsys-withsrc",
    "amd": "vllm/vllm-openai-rocm:v0.24.0", # todo: to be updated once amd work comes in?
}

# CONTAINER_IMG = "ainikolai/mlperf_vlm_v6.1:latest" #maybe we can update this image too?
CONTAINER_IMG = "wongcannan/mlperf-endpoints:v1.1"

MAX_GPUS = 8

def slug(value):
    if isinstance(value, float):
        value = format(value, "g").replace(".", "p")
    else:
        value = str(value).lower()
    return re.sub(r"[^a-z0-9_-]+", "-", value).strip("-")

def value_label(value):
    """Filesystem-safe run_id label; long dict values (e.g. speculative-config) get a compact tag plus a short hash."""
    full = slug(value)
    if not isinstance(value, dict) or len(full) <= 48:
        return full
    tag = [str(value.get("method", "cfg"))]
    if "num_speculative_tokens" in value:
        tag.append(f"k{value['num_speculative_tokens']}")
    if "enable_adaptive_verification" in value:
        tag.append(f"av{int(bool(value['enable_adaptive_verification']))}")
    tag.append(hashlib.sha1(json.dumps(value, sort_keys=True).encode()).hexdigest()[:6])
    return slug("-".join(tag))

# vllm flag cfgs extractor
def parameter_args(parameters):
    local_args = []
    for name, value in parameters.items():
        flag = f"--{name.lstrip('-')}"
        if value is True:
            local_args.append(flag)
        elif value is False:
            continue
        elif isinstance(value, (dict, list)):
            local_args.extend([flag, json.dumps(value, separators=(",", ":"))])
        else:
            local_args.extend([flag, str(value)])
    return local_args

# per-trial spec builder
def build_runs(config, mode):
    vllm_config = config["vllm_config"]
    var_params = vllm_config.get("var_params") or []
    if not var_params:
        # Explicit/resolved config: everything is already in fixed_params, one run.
        yield "0001", []
        return

    # * For search mode, grid (cartesian product) iterator across range arrays defined in yaml
    # * For eval mode, each index across all arrays represent one full trial
    if mode == "search":
        param_iterator = enumerate(
            itertools.product(*(parameter["values"] for parameter in var_params)), 
            start=1,
        )
    elif mode == "eval":
        if not all(len(parameter["values"]) == len(var_params[0]["values"]) for parameter in var_params):
            raise ValueError("All var_params must have the same length for eval mode. Refer to prepare_runs.py/build_runs for more info.")
        param_iterator = enumerate(
            zip(*(parameter["values"] for parameter in var_params)),
            start=1,
        )

    for index, values in param_iterator:
        selected_params = [
            {
                "name": parameter["name"],
                "value": value,
                "append_to": parameter.get("append_to"),
            }
            for parameter, value in zip(var_params, values)
        ]
        labels = [
            f"{slug(parameter.get('label', parameter['name']))}-{value_label(value)}"
            for parameter, value in zip(var_params, values)
        ]
        run_id = f"{index:04d}__{'__'.join(labels)}"
        if len(run_id) > 200:  # stay well under the 255-byte path-component limit
            run_id = f"{run_id[:180]}__{hashlib.sha1(run_id.encode()).hexdigest()[:10]}"
        yield run_id, selected_params

def resolved_configs(server_config, bench_config, selected_params, patched_yaml_name, run_docker_mnt_bench_results_dir):
    server_config_resolved = copy.deepcopy(server_config)
    bench_config_resolved = copy.deepcopy(bench_config)
    vllm_config = server_config_resolved["vllm_config"]

    for parameter in selected_params:
        target = vllm_config["fixed_params"]
        if parameter.get("append_to"):
            target = target[parameter["append_to"]]
        target[parameter["name"]] = parameter["value"]
    vllm_config.pop("var_params", None)

    fixed_params = vllm_config["fixed_params"]
    max_tokens = fixed_params.get("max-num-batched-tokens")
    compilation_config = fixed_params.get("compilation-config")

    # * Runtime checks/caps on some params to prevent startup issues

    # * 1. Ensuring num-replicas * TP * DP = MAX_GPUS
    tp = fixed_params.get("tensor-parallel-size", 1)
    gpus_per_server = tp * fixed_params.get("data-parallel-size", 1)
    if gpus_per_server < 1 or gpus_per_server > MAX_GPUS or MAX_GPUS % gpus_per_server != 0:
        raise ValueError(
            f"Invalid tensor-parallel-size * data-parallel-size: {gpus_per_server}. Must divide MAX_GPUS={MAX_GPUS}."
        )
    num_replicas = MAX_GPUS // gpus_per_server


    # * 2. Capping cudagraph max sizes to max-tokens we are currently testing
    if max_tokens is not None and isinstance(compilation_config, dict):
        max_capture = compilation_config.get("max_cudagraph_capture_size")

        if max_capture is not None and max_tokens < max_capture:
            compilation_config["max_cudagraph_capture_size"] = max_tokens

            capture_sizes = compilation_config.get("cudagraph_capture_sizes")
            if capture_sizes is not None:
                compilation_config["cudagraph_capture_sizes"] = [
                    size for size in capture_sizes if size <= max_tokens
                ]

    # * 3. Patch model, endpoint ports to create per-run container yaml
    # (this is done in case TP becomes a grid param in the future)
    bench_config_resolved["model_params"]["name"] = server_config_resolved["launch"]["model"]
    bench_config_resolved["endpoint_config"]["endpoints"] = [
        f"http://localhost:{8000 + replica_index}"
        for replica_index in range(num_replicas)
    ]
    

    # * 4. Patch yaml name + report_dir for consistent performance report dumping from benchmarking container
    bench_config_resolved["name"] = patched_yaml_name
    bench_config_resolved["report_dir"] = str(run_docker_mnt_bench_results_dir.resolve())

    # * 5. If fixed_params include target qps (which should come from var_params during server search/eval), patch container to enable higher serving load
    target_qps = fixed_params.pop("target_qps", None)
    if target_qps is not None:
        bench_config_resolved["settings"]["load_pattern"]["target_qps"] = target_qps

    # * 6. If fixed_params include target concurrency (for endpoints), patch container to enable different throughput regions
    target_concurrency = server_config_resolved["launch"].get("target_concurrency")
    if target_concurrency is not None:
        bench_config_resolved["settings"]["load_pattern"]["target_concurrency"] = target_concurrency

    return server_config_resolved, bench_config_resolved, num_replicas, gpus_per_server

def claim_experiment(lustre_parent, local_parent, signature):
    n = 0
    while True:
        name = f"experiment_{n}"
        (lustre_parent / signature).mkdir(parents=True, exist_ok=True)
        try:
            (lustre_parent / signature / name).mkdir()             # the claim
        except FileExistsError:
            n += 1
            continue
        try:
            (local_parent / signature / name).mkdir(parents=True)  # must also be new locally
            return name
        except FileExistsError:
            (lustre_parent  / signature / name).rmdir()             # give the number back
            n += 1

def swap_docker_dir(org_dir, platform):
    return (DOCKER_DATA_FOLDER
        / org_dir.relative_to(DATA_FOLDERS[platform]))

def main():
    print("Entered main...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--bench_size", type=str, choices=["short", "long"])
    parser.add_argument("--scenario", type=str, choices=["offline", "server", "endpoints"])
    parser.add_argument("--platform", type=str, default="nvidia", choices=["nvidia", "amd"])
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "search"]) #need to change this later...
    parser.add_argument("--custom_run", action="store_true", help="Use the default config")
    # Only for reproducing results.
    parser.add_argument("--server_config", type=Path, help="For reproducing results, explicit server config YAML, e.g. a point's server_configs/server_config_resolved.yaml")
    parser.add_argument("--bench_config", type=Path, help="For reproducing results, explicit bench config YAML; required with --server_config")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()
    if bool(args.server_config) != bool(args.bench_config):
        parser.error("--server_config and --bench_config must be given together")

    if args.server_config:
        print("Explicit config files")
        server_config = yaml.safe_load(args.server_config.read_text())
    elif args.custom_run:
        print("Custom")
        server_config = yaml.safe_load(
            (
                SERVER_CONFIGS_DIR
                / f"mlperf_{args.model}_custom.yaml"
            ).read_text()
        )
    else:
        print("Default with params")
        server_config = yaml.safe_load(
            (
                SERVER_CONFIGS_DIR
                / f"mlperf_{args.model}_{args.bench_size}_{args.scenario}_{args.platform}_{args.mode}.yaml"
            ).read_text()
        )

    # Extracting launch cfgs from yaml
    launch = server_config["launch"]

    if args.bench_config:
        bench_config = yaml.safe_load(args.bench_config.read_text())
    else:
        bench_config = yaml.safe_load(
        (
            BENCH_CONFIGS_DIR
            / (
                f"mlperf_{args.model}_"
                f"{'short_' if launch["bench_size"] == 'short' else 'long_'}"
                f"{launch["scenario"]}.yaml"
            )
        ).read_text()
    )
    runs = list(build_runs(server_config, args.mode))

    patch_num = launch.get("patch_num")
    patch_num = "no_patch" if patch_num is None else str(patch_num)

    # * Preparing experiment + search specific signature for directory prefixes
    if args.server_config:
        signature_prefix = f"repro_{args.version}__" if args.version else "repro__"
    elif args.custom_run:
        signature_prefix = f"custom__"
    else:
        signature_prefix = f"{args.version}__"
    
    signature_prefix += f"M_{launch['model_label']}__BS_{launch['bench_size']}__SC_{launch['scenario']}__P_{args.platform}"

    if launch["scenario"] == "endpoints":
        concurrency = launch.get("target_concurrency") or bench_config["settings"]["load_pattern"]["target_concurrency"]
        signature_prefix += f"__C_{concurrency}"

    if patch_num == "no_patch":
        signature_prefix += f"__vanilla"
    else:
        signature_prefix += f"__patch_{patch_num}"

    if launch.get("no_flag"):
        signature_prefix += "__no_flag"

    # ! Since this script is currently still only for grid search
    # ! default to appending param search modes to signature
    signature_prefix += f"__param_{args.mode}"

    kernels_dir_prefix = (
        SCRIPT_DIR / ".." / ".."
        / "engines/krai_gentic/mlperf/runs"
    )
    bench_results_prefix = (
        DATA_FOLDERS[args.platform] / "results/mlperf"
    )
    if not args.dry_run:
        experiment_name = claim_experiment(bench_results_prefix, kernels_dir_prefix, signature_prefix)
    else:
        experiment_name = "experiment_test_dry_run"
    signature = signature_prefix + f"/{experiment_name}"

    # * Preparing parent dirs for snapshots of current config/code for consistent sbatch execution

    # kernel_results_dir is the "local" = within-repo path to store per-search snapshots + important results
    kernel_results_dir = kernels_dir_prefix / signature

    # bench_results_dir is the per-search large volume storage directory, for all docker image mounts, logs, cache, outputs etc.
    bench_results_dir = DATA_FOLDERS[args.platform] / "results/mlperf" / signature / f"results_multireplica"

    # profiles_dir is where the profiling logs live on the large volume storage dir
    profiles_dir = DATA_FOLDERS[args.platform] / "multireplica_profiles" / signature / f"profiles_multireplica"

    # calls preflight script to check/pull for docker images
    print(f"Sweep: {signature}")
    print(f"Combinations: {len(runs)}")
    if not args.dry_run:
        kernel_results_dir.mkdir(parents=True, exist_ok=True)
        (kernel_results_dir / "logs").mkdir(parents=True, exist_ok=True)
        (kernel_results_dir / "configs").mkdir(parents=True, exist_ok=True)
        print("Created local results dir: ", kernel_results_dir)

        bench_results_dir.mkdir(parents=True, exist_ok=True)
        print("Created large storage log/cache dir: ", bench_results_dir)

        if launch["profile"]:
            profiles_dir.mkdir(parents=True, exist_ok=True)
            print("Created large storage profile logs dir: ", profiles_dir)

        # Dump the current search/eval config, base container yaml & shell script into the local snapshot dir (per-search level)
        (kernel_results_dir / f"configs/server_config.yaml").write_text(yaml.safe_dump(server_config, sort_keys=False))
        (kernel_results_dir / f"configs/bench_config.yaml").write_text(yaml.safe_dump(bench_config, sort_keys=False))
        shutil.copy(SHELL_SCRIPT, kernel_results_dir / "param_search_multireplica_mlperf.sh")
        shutil.copy(SBATCH_SCRIPT, kernel_results_dir / "execute_runs.sh")        
        print(f"Dumped endpoint {args.mode} config incl. var_params and base container config at {kernel_results_dir}")
        
    commands = []
    for run_id, selected_params in runs:
        run_results_dir = bench_results_dir / run_id
        run_profiles_dir = profiles_dir / run_id
        # per-run report_dir to be patched into container yaml
        patched_yaml_name = f"multireplica-{signature}-{run_id}"
        run_server_config_path = run_results_dir / "server_config_resolved.yaml"
        run_bench_config_path = run_results_dir / "bench_config_resolved.yaml"

        # Parsing + splitting the per-trial params into original multireplica shell script compatible part and the docker command part
        server_config_resolved, bench_config_resolved, num_replicas, gpus_per_server = resolved_configs(server_config, bench_config, selected_params, patched_yaml_name, swap_docker_dir(run_results_dir, args.platform))
        if not args.dry_run:
            # store per-run yaml snapshots
            run_results_dir.mkdir(parents=True, exist_ok=True)
            run_server_config_path.write_text(yaml.safe_dump(server_config_resolved, sort_keys=False))
            run_bench_config_path.write_text(yaml.safe_dump(bench_config_resolved, sort_keys=False))
            print(f"Initiated run and log dir at {run_results_dir}, dumped endpoint/container config as [server|bench]_config_resolved.yaml")

            if launch["profile"]:
                for i in range(num_replicas):
                    (run_profiles_dir / f"replica-{i}").mkdir(parents=True, exist_ok=True)

        parameters = server_config_resolved["vllm_config"]["fixed_params"]
        vllm_args = parameter_args(parameters)

        env_config = server_config_resolved["vllm_config"].get("env") or {}

        env_args = []
        for name, value in env_config.items():
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                raise ValueError(f"Invalid environment variable name: {name!r}")

            env_args.append(f"{name}={value}")

        command = [
            "bash",
            str((kernel_results_dir / "param_search_multireplica_mlperf.sh").resolve()),
            # launch configs
            launch["bench_size"],
            launch["scenario"],
            args.platform,
            launch.get("bench_mode", "perf") if launch["scenario"] == "endpoints" else "",
            patch_num,
            signature,
            launch["model"],
            # flags
            str(num_replicas),
            str(gpus_per_server),
            str(launch["profile"]).lower(),
            str(launch["no_flag"]).lower(),
            VLLM_IMAGE[args.platform],
            CONTAINER_IMG,
            # directories
            str(DATA_FOLDERS[args.platform]),
            str(kernel_results_dir.resolve()),
            str(run_results_dir.resolve()),
            str(run_profiles_dir.resolve()),
            str(swap_docker_dir(run_results_dir, args.platform).resolve()),
            str(swap_docker_dir(run_profiles_dir, args.platform).resolve()),
            # vllm env + args
            str(len(env_args)),
            *env_args,
            *vllm_args,
        ]

        if args.dry_run:
            print(f"Dry run mode: Command for {run_id}: {' '.join(command)}")
        else:
            commands.append((run_id, command))
    if not args.dry_run:
        with (kernel_results_dir / "manifest.txt").open("w", encoding="utf-8") as f:
            print("writing manifest.txt")
            for run_id, command in commands:
                command_string = shlex.join(command)
                f.write(f"{run_id}\t{command_string}\n")

        with fdopen(3, "w") as result_fd:   # handoff channel opened by entry_*.sh
            print(kernel_results_dir.resolve(), file=result_fd)
        # tmp_dir = SCRIPT_DIR / "tmp"
        # tmp_dir.mkdir(parents=True, exist_ok=True)
        # with ( tmp_dir / args.tmp_file_name ).open("w", encoding="utf-8") as f:
        #     f.write(str(kernel_results_dir.resolve()))

if __name__ == "__main__":
    main()
