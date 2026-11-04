#!/usr/bin/env python3
"""Generate the GKE sflow recipes for the 16 GB300 Kimi-K3 points of the NVIDIA
MLPerf Endpoints v1.0.3 drop (TensorRT-LLM + Dynamo disaggregated, DSpark draft).

Source of truth (read-only; nothing about the workload is typed in here)
  <slurm-root> = nv-mlpinf-partner-partner-endpoints-v1.0.3/NVIDIA/src/kimi-k3
    configs/GB300-NVL72_GB300-288GB_aarch64x48_TRTLLM-Dynamo/point_<C>.yaml
        variables .......................... emitted VERBATIM (raw text; SWE overrides
                                             stripped in perf tree)
        PREFILL_CONFIG DECODE_CONFIG PREFILL_ENV DECODE_ENV ROUTER_ENV
                                             emitted VERBATIM (raw text, same names)
        CLIENT_CONFIG ...................... verbatim + CLIENT_REWRITES (2 lines; D13)
        workflow.tasks ..................... point-specific overrides merged onto
                                             setup.yaml + disagg.yaml
    disagg.yaml ............................ client_gate/control_plane/prefill/decode/
                                             frontend/benchmark base tasks + variables
    setup.yaml ............................. workflow variables (VERBATIM), preflight
                                             task, fabric check (moved into server pods)
    slurm_env.yaml ......................... images (pinned by digest), MODE,
                                             BENCHMARK_MODE, container mounts (asserted)
    scripts/node.py, scripts/control_plane.json, ../sflow/tools/run_with_env.py
                                             embedded VERBATIM as <workdir> artifacts
    scripts/swe_service.py ................. SWE-bench service & token redaction rule
    README.md, ../../../dataset/README.md .. provenance pins (asserted)

The GKE layer lives only in this file. Outputs: <dest>/<N>nodes/point_<C>/
{kimi_k3_gke_disagg_point<C>.yaml, launch_test.sh}.

Opt-in variant (--image-pull-secret; default OFF, default outputs unaffected):
<dest>/with-pull-secret/<N>nodes/point_<C>/, the same two files where every
operator that runs the private server image also carries
image_pull_secrets: [nv-gitlab-pull] and launch_test.sh checks that Secret.
Only for when the prepull-images.yaml canary hits ErrImagePull.

Usage
  ./generate_point_recipes.py                  regenerate all 16 points
  ./generate_point_recipes.py --points 24      regenerate one point
  ./generate_point_recipes.py --check          verify drop + outputs; writes nothing
  ./generate_point_recipes.py --image-pull-secret [--check] [--points ...]
                                               write / verify the variant tree only
  ./generate_point_recipes.py --swe {acc,both} [--image-pull-secret] [--check] [--points ...]
                                               write / verify the opt-in SWE-bench tree
                                               (<dest>/[with-pull-secret/]swe_<mode>/<N>nodes/point_<C>/)
  ./generate_point_recipes.py --dump-scripts D [--swe {acc,both}]
                                               write each task script as D/<point>/<task>.sh
                                               (for an external `bash -n`)
  --slurm-root PATH     v1.0.3 drop dir (default: the v1.0.3 drop next to nv-sflow)
  --swe-drop-root PATH  v1.0.3 drop dir carrying the SWE-bench wiring (used only with --swe)
  --dest PATH           output root (default: this script's dir)

No network, no subprocess. Any drift fails closed with SystemExit.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    raise SystemExit("generate_point_recipes: PyYAML is required (pip install pyyaml)")

HERE = Path(__file__).resolve().parent
_DROP_BASE = HERE.parents[5] / "code-drop-nvidia"
if not _DROP_BASE.is_dir():
    _DROP_BASE = Path.home() / "<workdir>" / "code-drop-nvidia"
DEFAULT_SLURM_ROOT = (_DROP_BASE / "nv-mlpinf-partner-partner-endpoints-v1.0.3"
                      / "NVIDIA" / "src" / "kimi-k3")
DEFAULT_SWE_DROP_ROOT = DEFAULT_SLURM_ROOT
POINTS_SUBDIR = "configs/GB300-NVL72_GB300-288GB_aarch64x48_TRTLLM-Dynamo"
DROP_REL = "nv-mlpinf-partner-partner-endpoints-v1.0.3/NVIDIA/src/kimi-k3"

# --------------------------------------------------------------------------------------
# Pins. Each is asserted against the drop; a mismatch stops generation.
# --------------------------------------------------------------------------------------
GPUS_PER_NODE = 4
CTX_BASE_PORT = 9210          # drop: export DYN_SYSTEM_PORT=$((9210 + SFLOW_REPLICA_INDEX))
GEN_BASE_PORT = 9310          # drop: export DYN_SYSTEM_PORT=$((9310 + SFLOW_REPLICA_INDEX))
FRONTEND_SYSTEM_PORT = 9200   # drop: export DYN_SYSTEM_PORT=9200
FRONTEND_PORT = 8000          # drop: setup.yaml FRONTEND_PORT, dynamo.frontend --http-port 8000
TRAJ_PER_PASS = 613           # setup.yaml: NUM_TRAJECTORIES % 613 == 0

SERVER_IMAGE_DROP = ("registry.gitlab.com#nvidia/mlperf-inference-partner/nv-mlpinf-partner/kimi-k3-dynamo"
                     "@sha256:985260a01a2dfb66bdd0c68f0bd482e78f69c1b101120c7341782d101e2602a4")
SERVER_IMAGE = SERVER_IMAGE_DROP.replace("#", "/", 1)     # enroot '#' -> registry '/'
CLIENT_IMAGE_DROP = ("ghcr.io#mlcommons/endpoints"
                     "@sha256:1523051b141437b25083924d6428082f435713516eee71c54ae207b393900b2d")
CLIENT_IMAGE = CLIENT_IMAGE_DROP.replace("#", "/", 1)     # commit d3c57f2
TRTLLM_VERSION = "1.3.0rc25.post1"
DYNAMO_VERSION = "1.6.0.dev20260918"
MODEL_REPO, MODEL_REV = "nvidia/Kimi-K3-NVFP4", "b2428a0b83a8b712ff2e1a8448a103d4175341f1"
DRAFT_REPO, DRAFT_REV = "RadixArk/Kimi-K3-DSpark", "3c5bac301d9cf392706189d82ed947feca6c2f0f"
# git blob sha1 of config.json at MODEL_REV (HF tree API); checked on every node.
MODEL_CONFIG_GITSHA1 = "0e4b41bbe8d1e6ca545853ec60f12601a84e5fe5"
DATASET_DROP_NAME = "mlperf_agentic_inference_dataset.jsonl"
DATASET_SHA256 = "1beb24c882122df96571cf11b390acbea388944038bc55c78b891475459014ae"
SFLOW_PIN = "b8afb5b86b3f46d9fa771531990fc7afaefc69b9"
DYNAMO_FRONTEND_CORE_SHA256 = "7041b8df80accb230c5ecb4dd3be59785e70f2636d9891c107cb31a86bf826d9"

NODE_SSD = "<node-local-ssd>"
MODEL_PATH = f"{NODE_SSD}/model-cache/Kimi-K3-NVFP4"
DRAFT_PATH = f"{NODE_SSD}/model-cache/Kimi-K3-DSpark"
CONTROL_PLANE_DIR = f"{NODE_SSD}/model-cache/kimi-control-plane"
DATASET_DIR = f"{NODE_SSD}/mlperf-data"
DATASET_PATH = f"{DATASET_DIR}/mlperf_agentic_inference_dataset.jsonl"
REPORTS_DIR = f"{NODE_SSD}/kimi-reports"
METRICS_TOKENIZER_WORKERS = 32
HANG_DETECTOR = "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/pyexecutor/hang_detector.py"
PYEXECUTOR_UTIL = "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/pyexecutor/_util.py"
KV_CACHE_MANAGER_V2 = "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py"
FLASHINFER_UTILS = "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/flashinfer_utils.py"
# Opt-in --image-pull-secret variant. PULL_SECRET is the Secret prepull-images.yaml creates.
PULL_SECRET = "nv-gitlab-pull"
PULL_SECRET_SUBDIR = "with-pull-secret"
SERVER_IMAGE_OPERATORS = ("server_ctx", "server_gen", "server_cpu")  # the ones running SERVER_IMAGE

POINT_CORE_VARS = (
    "ALLOCATION_SETUP", "ALLOCATION_CLEANUP", "NUMA_BIND", "SEGMENT_NODES",
    "MEMORY_CLOCK_MHZ", "MEMORY_CLOCK_RESTORE", "CONCURRENCY", "NUM_TRAJECTORIES",
    "NUM_CTX", "NUM_GEN", "GEN_GPUS", "TOTAL_GPUS", "TOTAL_NODES", "CTX_GPUS",
    "DRAFT_HEAD", "FRONTEND_NODE", "CLIENT_NODE", "DYNAMO_REQUEST_PLANE",
    "DYNAMO_DECODE_ENTRY", "DYNAMO_DECODE_ROUTER_MODE", "DYNAMO_DECODE_EXTRA_ARGS",
)
POINT_OPTIONAL_VARS = ("REFERENCE_SYSTEM",)
POINT_SWE_ONLY_VARS = ("BENCHMARK_MODE", "SWE_ROOT", "SWE_IMAGE_REGISTRY")
ACCURACY_DROP_POINTS = (12, 16, 32, 1024)
HIGH_CONCURRENCY_POINTS = (192, 288, 384, 448, 512, 576, 640, 1024)

DROP_ARTIFACTS = (
    ("CLIENT_CONFIG", "<workdir>"),
    ("PREFILL_CONFIG", "<workdir>"),
    ("DECODE_CONFIG", "<workdir>"),
    ("PREFILL_ENV", "<workdir>"),
    ("DECODE_ENV", "<workdir>"),
    ("ROUTER_ENV", "<workdir>"),
)
VERBATIM_ARTIFACTS = tuple(n for n, _ in DROP_ARTIFACTS if n != "CLIENT_CONFIG")
HELPERS = (  # (artifact name, uri, path relative to the drop's NVIDIA/src)
    ("RUN_WITH_ENV_PY", "<workdir>/run_with_env.py", "sflow/tools/run_with_env.py"),
    ("NODE_PY", "<workdir>/scripts/node.py", "kimi-k3/scripts/node.py"),
    ("CONTROL_PLANE_JSON", "<workdir>/scripts/control_plane.json", "kimi-k3/scripts/control_plane.json"),
)

# C -> (NUM_TRAJECTORIES, NUM_CTX, CTX_GPUS, NUM_GEN, GEN_GPUS, TOTAL_GPUS, TOTAL_NODES,
#       FRONTEND_NODE, CLIENT_NODE). Asserted against every point file.
EXPECTED = {
    12: (613, 1, 8, 5, 8, 48, 12, 2, 2),
    13: (613, 1, 8, 3, 8, 32, 8, 2, 2),
    16: (613, 1, 8, 3, 8, 32, 8, 2, 2),
    24: (613, 1, 8, 3, 8, 32, 8, 2, 2),
    32: (613, 1, 8, 3, 8, 32, 8, 2, 2),
    48: (1226, 1, 8, 3, 8, 32, 8, 2, 2),
    64: (1839, 1, 8, 3, 8, 32, 8, 2, 2),
    80: (1839, 1, 8, 3, 8, 32, 8, 2, 2),
    192: (3065, 16, 1, 2, 16, 48, 12, 4, 4),
    288: (4904, 16, 1, 2, 16, 48, 12, 4, 4),
    384: (4291, 16, 1, 2, 16, 48, 12, 4, 4),
    448: (4291, 16, 1, 1, 16, 32, 8, 4, 4),
    512: (4904, 16, 1, 1, 16, 32, 8, 4, 4),
    576: (4904, 16, 1, 1, 16, 32, 8, 4, 4),
    640: (4904, 16, 1, 1, 16, 32, 8, 4, 4),
    1024: (4904, 16, 1, 1, 16, 32, 8, 4, 4),
}

GKE_UCX_TLS = "cuda_ipc,cuda_copy,sm,self,tcp"

# GKE server env, exported after the drop env eval so GKE's MNNVL + eth0 settings win over
# the Slurm drop's IB settings (UCX_TLS=all, UCX_CUDA_IPC_ENABLE_MNNVL=n, unset UCX_NET_DEVICES).
GKE_SERVER_ENV = (
    ("UCX_NET_DEVICES", "eth0"),
    ("UCX_TLS", GKE_UCX_TLS),
    ("NCCL_NET_PLUGIN", "none"),
    ("NCCL_MNNVL_ENABLE", "1"), ("NCCL_CUMEM_ENABLE", "1"), ("NCCL_DEBUG", "INFO"),
    ("NCCL_SOCKET_IFNAME", "eth0"), ("GLOO_SOCKET_IFNAME", "eth0"), ("TP_SOCKET_IFNAME", "eth0"),
    ("OMPI_MCA_pml", "ob1"), ("OMPI_MCA_btl", "tcp,self"), ("OMPI_MCA_btl_tcp_if_include", "eth0"),
    ("OMPI_MCA_oob_tcp_if_include", "eth0"), ("OMPI_MCA_hwloc_base_binding_policy", "none"),
    ("PRTE_MCA_hwloc_default_binding_policy", "none"),
    ("OMPI_MCA_coll", "^ucc"),
    ("UCX_CUDA_IPC_ENABLE_MNNVL", "y"), ("UCX_RNDV_SCHEME", "put_zcopy"), ("UCX_MEMTYPE_CACHE", "n"),
    ("UCX_PROTO_INFO", "y"),
    ("TLLM_LOG_LEVEL", "INFO"), ("TRTLLM_HANG_DETECTOR_TIMEOUT", "3600"),
    ("TRTLLM_TORCH_MEM_SAFETY_MARGIN_MB", "4096"),
)

DEVIATIONS = """\
D1  Launcher: srun + pyxis (operators host/server/client, --segment=SEGMENT_NODES,
    mpi: pmix) -> backend `cluster` (kubernetes; host network/IPC; device-plugin
    GPUs; rdma: disable; compute_domain create = the MNNVL domain) and operators
    server_ctx/server_gen (k8s_mpi, privileged, sshd on 2222, ensure_sshd
    apt-installs openssh-server at pod start), server_cpu (k8s, not privileged:
    preflight + control_plane + frontend; frontend symlinks
    /usr/local/cuda/compat/lib.real/libcuda.so.1 over the image's 0-byte
    /usr/lib/aarch64-linux-gnu/libcuda.so.620.43 and guards flashinfer_utils.py's
    import-time get_device_capability() when no GPU is mounted), client (k8s:
    client_gate + client_preflight + benchmark). Images by digest, imagePullPolicy
    IfNotPresent (pre-pulled; see README).
D2  Server launch: `exec trtllm-llmapi-launch env ... python3 -m dynamo.trtllm ...`
    (or `python3 /opt/k3-dynamo/helpers/dynamo_entry_v4.py ...`) ->
    `exec mpirun -np <CTX_GPUS|GEN_GPUS> --allow-run-as-root /tmp/kimi-rank-env.sh
    <role> trtllm-llmapi-launch env UCX_NET_DEVICES="$UCX_NET_DEVICES"
    UCX_TLS="$UCX_TLS" python3 ...`. Everything from `python3 ` onward is the
    drop's exact text.
D3  Cache dirs & Dynamo namespace:
    /tmp/kimi-${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID:-0} ->
    /tmp/kimi-<role>-${SFLOW_REPLICA_INDEX}-${KIMI_RANK:-0}, and
    DYN_NAMESPACE="k3-${SLURM_JOB_ID}" ->
    DYN_NAMESPACE="k3-$(basename "${SFLOW_WORKFLOW_OUTPUT_DIR}")".
    /tmp/kimi-rank-env.sh redoes the per-rank cache-dir exports, unsets NGC 26.08
    gIB plugin exports, restores full-node CPU affinity, sets
    garbage_collection_threshold:0.75 on PYTORCH_CUDA_ALLOC_CONF, and sets a
    disjoint per-pod /proc/sys/kernel/ns_last_pid base so UCX's remote VMM cache
    never collides across containers.
D4  ${SFLOW_TOOLS}/${RECIPE} (fs:// into a repo checkout on shared storage) ->
    run_with_env.py, node.py, control_plane.json embedded verbatim as <workdir>
    artifacts; scripts export SFLOW_TOOLS/RECIPE to their in-pod directories.
    control_plane caches verified etcd v3.6.5 and nats-server v2.11.9 under
    CONTROL_PLANE_DIR on node 0's local SSD.
D5  Model/draft: container mounts MODEL_PATH:/models/kimi-k3:ro and
    DRAFT_PATH:/models/kimi-k3-dspark:ro -> node-local SSD hostPaths (fs://) plus
    atomic symlinks /models/kimi-k3 and /models/kimi-k3-dspark in every server and
    frontend pod, and /models/kimi-k3 in every client pod. Each server/frontend pod
    exits unless both dirs hold config.json and a .sflow-staged marker equal to the
    pinned revision (stage-kimi-k3-assets.yaml) and the model's config.json is
    nvidia/Kimi-K3-NVFP4@b2428a0b (git sha1).
D6  Dataset: client container mount DATASET_PATH:/data/agentic_combined_v6.jsonl:ro
    -> fs:// on DATASET_DIR plus symlink /data/agentic_combined_v6.jsonl ->
    ${DATASET_PATH} in every client task. The benchmark refuses to start unless the
    file's sha256 matches dataset/README.md.
D7  Dynamo control plane & discovery: control_plane runs `node.py control-plane` on
    node 0 (HEAD_NODE_IP:2379 etcd + nats://HEAD_NODE_IP:4222); prefill, decode, and
    frontend share DYN_NAMESPACE="k3-$(basename "${SFLOW_WORKFLOW_OUTPUT_DIR}")" and
    export DYN_SYSTEM_HOST/DYN_TCP_RPC_HOST/DYN_TCP_RESPONSE_STREAM_HOST/DYN_EVENT_PLANE_HOST="$(hostname -i)"
    so Dynamo's ip_resolver advertises the node's eth0 IPv4 instead of GKE's
    <ip> link-local metadata address on lo.
D8  UCX_NET_DEVICES=eth0 and UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp forced after the
    drop env eval (the Slurm drop sets UCX_TLS=all / UCX_CUDA_IPC_ENABLE_MNNVL=n for
    InfiniBand on C12-C80/C1024 and eth0 on C192-C640).
D9  GKE env block in the server pods (see GKE_SERVER_ENV): MNNVL NCCL, eth0 socket
    ifnames, Open MPI ob1/tcp on eth0, PRTE binding policy none, NGC 26.08
    /opt/gcp/nccl-plugins/lib64 disabled so /etc/shinit_v2 cannot force NCCL_NET=gIB,
    UCX_CUDA_IPC_ENABLE_MNNVL=y, UCX_RNDV_SCHEME=put_zcopy, UCX_MEMTYPE_CACHE=n,
    TLLM_LOG_LEVEL=INFO.
D10 hang_detector.py, _util.py, and kv_cache_manager_v2.py patched in-pod (atomic;
    exits if the pattern is not found) so TRTLLM_HANG_DETECTOR_TIMEOUT=3600 replaces
    the 300 s default, TRTLLM_TORCH_MEM_SAFETY_MARGIN_MB=4096 replaces
    _adjust_torch_mem_fraction's 32 MiB margin, and the mirrored draft KV cache
    manager sets max_util_for_resume=1.0 in _build_base_config.
D11 Best-effort sysctl socket-buffer raise in the server pods.
D12 Fabric: setup.yaml `fabric` (one srun per node) + `node.py check ... preflight` ->
    the drop's `node.py probe` fabric line runs in every server pod and prints
    KIMI_FABRIC; the ComputeDomain admits only one IMEX domain. The preflight task
    keeps the drop's arithmetic asserts and server-configs snapshot.
D13 Client: report_dir /results/report -> ${REPORT_DIR} (node SSD, survives pod loss)
    and settings.metrics_tokenizer_workers: 32 (post-run tokenization speed only).
    Benchmark: client import check, dataset sha256 + tokenizer checks, REPORT_DIR
    printed first, small-file report mirror into the sflow output.
D14 Workflow name kimi_k3_trtllm_v1_0_3_gke_c<C> (drop: kimi-k3-dynamo).
Unchanged: server/env YAMLs, probes, timeouts, ports, depends_on, Dynamo router/decode flags.
"""


def die(msg: str) -> None:
    raise SystemExit(f"generate_point_recipes: {msg}")


def read_text(path: Path) -> str:
    if not path.is_file():
        die(f"missing input file: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        die(f"cannot read {path}: {exc}")
    return ""  # unreachable


def parse_yaml(text: str, where: str):
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        die(f"{where}: invalid YAML: {exc}")
    return None  # unreachable


def ntasks_per_node_expr(gvar: str) -> str:
    """The drop's srun ntasks_per_node expression; reused verbatim as slots_per_worker.
    sflow variables are strings inside expressions, hence the drop's `| int`."""
    return (f"${{{{ (variables.{gvar} | int) if (variables.{gvar} | int) < {GPUS_PER_NODE} "
            f"else {GPUS_PER_NODE} }}}}")


# --------------------------------------------------------------------------------------
# SWE-only blocks in D3 point_{12,16,32,1024}.yaml (stripped in perf tree, applied to all
# 16 points in --swe acc|both trees).
# --------------------------------------------------------------------------------------
DROP_SWE_VARS_RAW = """\
  BENCHMARK_MODE:
    value: both
  SWE_ROOT:
    value: ${{ variables.RESULTS_PATH }}/swe
  SWE_IMAGE_REGISTRY:
    description: The partner registry's ARM64 SWE-bench images; set your own mirror with --set.
    value: registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/swebench-arm64
"""

DROP_SWE_DATASET_CONTENT = """\
- name: swe_bench
  type: accuracy
  accuracy_config:
    eval_method: swe_bench_scorer
    extras:
      swebench_service_url: ${SWEBENCH_SERVICE_URL}
      swebench_service_auth_token: ${SWEBENCH_SERVICE_AUTH_TOKEN}
      num_instances: 200
      workers: 16
"""
DROP_SWE_DATASET_RAW = "".join("    " + ln + "\n" for ln in DROP_SWE_DATASET_CONTENT.splitlines())
DROP_SWE_EVAL_LINE = (
    'eval "$(python3 -c \'import json,os,shlex;'
    'd=json.load(open(os.environ["SFLOW_WORKFLOW_OUTPUT_DIR"]+"/swe-connection.json"));'
    'print("export SWEBENCH_SERVICE_URL="+shlex.quote(d["url"])+'
    '" SWEBENCH_SERVICE_AUTH_TOKEN="+shlex.quote(d["token"]))\')"'
)


# --------------------------------------------------------------------------------------
# Drop loading
# --------------------------------------------------------------------------------------
class Drop:
    def __init__(self, root: Path):
        self.root = root
        if not root.is_dir():
            die(f"--slurm-root {root} is not a directory")
        self.src = root.parent                      # NVIDIA/src
        self.readme = read_text(root / "README.md")
        self.dataset_readme = read_text(root.parents[2] / "dataset" / "README.md")
        self.setup_raw = read_text(root / "setup.yaml")
        self.setup = parse_yaml(self.setup_raw, "setup.yaml")
        self.env = parse_yaml(read_text(root / "slurm_env.yaml"), "slurm_env.yaml")
        self.disagg_raw = read_text(root / "disagg.yaml")
        self.template = parse_yaml(self.disagg_raw, "disagg.yaml")
        self.helpers = {name: read_text(self.src / rel) for name, _, rel in HELPERS}
        self.points = {}
        pdir = root / POINTS_SUBDIR
        if not pdir.is_dir():
            die(f"missing {pdir}")
        found = sorted(int(m.group(1)) for p in pdir.iterdir()
                       if (m := re.fullmatch(r"point_(\d+)\.yaml", p.name)))
        if found != sorted(EXPECTED):
            die(f"{POINTS_SUBDIR}: points {found} != expected {sorted(EXPECTED)}")
        for c in found:
            self.points[c] = DropPoint(c, pdir / f"point_{c}.yaml")
        self.tasks = self._tasks()
        self.setup_tasks = {t["name"]: t for t in self.setup["workflow"]["tasks"]}
        self.wf_vars_raw = self._wf_vars_raw()
        self._assert_global()

    def _tasks(self) -> dict:
        wf = self.template.get("workflow") or {}
        tasks = wf.get("tasks") or []
        names = [t.get("name") for t in tasks]
        want_names = ["client_gate", "control_plane", "prefill", "decode", "frontend", "benchmark"]
        if names != want_names:
            die(f"disagg.yaml tasks {names} != {want_names}")
        if wf.get("name") != "kimi-k3-dynamo":
            die("disagg.yaml workflow name changed")
        by_name = {t["name"]: t for t in tasks}
        for role, gvar in (("prefill", "CTX_GPUS"), ("decode", "GEN_GPUS")):
            want = {"name": "server", "ntasks": f"${{{{ variables.{gvar} }}}}",
                    "ntasks_per_node": ntasks_per_node_expr(gvar)}
            if by_name[role].get("operator") != want:
                die(f"disagg.yaml {role} operator changed: {by_name[role].get('operator')}")
        if by_name["frontend"].get("operator") != {"name": "server", "ntasks": 1}:
            die("disagg.yaml frontend operator changed")
        if by_name["control_plane"].get("operator") != "host":
            die("disagg.yaml control_plane operator changed")
        for cl_task in ("client_gate", "benchmark"):
            if by_name[cl_task].get("operator") != "client":
                die(f"disagg.yaml {cl_task} operator changed")
        return by_name

    def _wf_vars_raw(self) -> str:
        m = re.search(r"(?ms)^  variables:\n(.*?)^  tasks:\n", self.setup_raw)
        if not m:
            die("setup.yaml: workflow.variables block not found")
        raw = m.group(1)
        want = {"HEAD_NODE_IP": "${{ backends.cluster.nodes[0].ip_address }}",
                "NODE_NAMES": "${{ backends.cluster.nodes | map(attribute='name') | join(',') }}",
                "FRONTEND_PORT": str(FRONTEND_PORT),
                "FRONTEND_NODE_IP": "${{ backends.cluster.nodes[variables.FRONTEND_NODE | int].ip_address }}"}
        got = {k: str(v.get("value")) for k, v in (self.setup["workflow"].get("variables") or {}).items()}
        if got != want:
            die(f"setup.yaml workflow.variables changed: {got}")
        if parse_yaml("variables:\n" + raw, "setup.yaml variables block")["variables"] != \
                self.setup["workflow"]["variables"]:
            die("setup.yaml variables raw block does not round-trip")
        return raw

    def _assert_global(self) -> None:
        v = {k: (d or {}).get("value") for k, d in (self.env.get("variables") or {}).items()}
        if v.get("SERVER_IMAGE") != SERVER_IMAGE_DROP:
            die(f"slurm_env.yaml SERVER_IMAGE {v.get('SERVER_IMAGE')!r} != pinned {SERVER_IMAGE_DROP!r}")
        if v.get("CLIENT_IMAGE") != CLIENT_IMAGE_DROP:
            die(f"slurm_env.yaml CLIENT_IMAGE {v.get('CLIENT_IMAGE')!r} != pinned {CLIENT_IMAGE_DROP!r}")
        if v.get("MODE") != "disagg" or v.get("BENCHMARK_MODE") != "perf":
            die("slurm_env.yaml MODE/BENCHMARK_MODE changed")
        ops = {o["name"]: o for o in self.env.get("operators") or []}
        mounts = (ops.get("server") or {}).get("container_mounts")
        if mounts != ["${{ variables.MODEL_PATH }}:/models/kimi-k3:ro",
                      "${{ variables.DRAFT_PATH }}:/models/kimi-k3-dspark:ro"]:
            die(f"slurm_env.yaml server container_mounts changed: {mounts}")
        cl_mounts = (ops.get("client") or {}).get("container_mounts")
        if cl_mounts != [
            "${{ variables.MODEL_PATH }}:/models/kimi-k3:ro",
            "${{ variables.DATASET_PATH }}:/data/agentic_combined_v6.jsonl:ro",
            "${{ variables.RESULTS_PATH }}:/results",
            "${{ variables.RESULTS_PATH }}:${{ variables.RESULTS_PATH }}",
        ]:
            die(f"slurm_env.yaml client container_mounts changed: {cl_mounts}")
        dvars = {k: (d or {}).get("value") for k, d in (self.template.get("variables") or {}).items()}
        if dvars.get("DYNAMO_DECODE_METRICS_FLAG") != "--no-publish-metrics":
            die("disagg.yaml DYNAMO_DECODE_METRICS_FLAG changed")
        if dvars.get("DYNAMO_FRONTEND_CORE_SHA256") != DYNAMO_FRONTEND_CORE_SHA256:
            die("disagg.yaml DYNAMO_FRONTEND_CORE_SHA256 changed")
        for needle, what in ((MODEL_REV, "model revision"), (DRAFT_REV, "draft revision"),
                             (MODEL_REPO, "model repo"), (DRAFT_REPO, "draft repo"),
                             (SERVER_IMAGE_DROP, "server image"), (CLIENT_IMAGE_DROP, "client image"),
                             (SFLOW_PIN, "sflow pin"), (DATASET_DROP_NAME, "dataset name")):
            if needle not in self.readme:
                die(f"README.md no longer mentions the pinned {what} {needle}")
        if DATASET_SHA256 not in self.dataset_readme or DATASET_DROP_NAME not in self.dataset_readme:
            die("dataset/README.md no longer pins the Kimi dataset sha256")
        arts = [(a["name"], a["uri"]) for a in self.setup.get("artifacts") or []]
        if arts != [("RECIPE", "fs://${{ variables.REPO_ROOT }}/NVIDIA/src/kimi-k3"),
                    ("MODEL", "fs://${{ variables.MODEL_PATH }}"),
                    ("DATASET", "fs://${{ variables.DATASET_PATH }}"),
                    ("DRAFT", "fs://${{ variables.DRAFT_PATH }}"),
                    ("SFLOW_TOOLS", "fs://${{ variables.REPO_ROOT }}/NVIDIA/src/sflow/tools")]:
            die(f"setup.yaml artifacts changed: {arts}")
        if list(self.setup_tasks) != ["fabric", "preflight"]:
            die(f"setup.yaml tasks {list(self.setup_tasks)} != [fabric, preflight]")
        if FABRIC_CMD not in self.setup_tasks["fabric"]["script"]:
            die("setup.yaml fabric task no longer runs the node.py probe fabric line")
        want_srv_probes = {
            "readiness": {"log_watch": {"regex_pattern": "Set runtime config data_parallel_size"},
                          "timeout": 3600},
            "failure": {"log_watch": {
                "regex_pattern": "re:OutOfMemoryError|RuntimeError: Executor|EngineDeadError|CUDA error:|CUDA out of memory"
            }},
        }
        for role in ("prefill", "decode"):
            if self.tasks[role].get("probes") != want_srv_probes:
                die(f"disagg.yaml {role} probes changed: {self.tasks[role].get('probes')}")
        want_fe_probes = {
            "readiness": {"log_watch": {"regex_pattern": "DYNAMO_FRONTEND_STARTING"}, "timeout": 3600},
            "failure": {"log_watch": {"regex_pattern": "re:Traceback|panicked at"}},
        }
        if self.tasks["frontend"].get("probes") != want_fe_probes:
            die(f"disagg.yaml frontend probes changed: {self.tasks['frontend'].get('probes')}")
        want_cp_probes = {
            "readiness": {"log_watch": {"regex_pattern": "DYNAMO_CONTROL_READY"}, "timeout": 120},
        }
        if self.tasks["control_plane"].get("probes") != want_cp_probes:
            die(f"disagg.yaml control_plane probes changed: {self.tasks['control_plane'].get('probes')}")
        for name, text in self.helpers.items():
            if re.search(r"\{\{|\{%|\{#", text):
                die(f"helper {name} contains Jinja markers; sflow would rewrite it")
            if not text or text.endswith("\n\n"):
                die(f"helper {name} must be non-empty and not end with multiple newlines")

    def composed_tasks(self, c: int, swe: bool = False) -> dict[str, dict]:
        """Return composed workflow tasks for point `c` (setup.yaml + disagg.yaml + point_<c>.yaml),
        with Slurm-only `fabric` and `swe_service` omitted and SWE overrides stripped when swe=False."""
        pt = self.points[c]
        merged: dict[str, dict] = {}
        for src_tasks in (self.setup["workflow"]["tasks"],
                          self.template["workflow"]["tasks"],
                          (pt.data.get("workflow") or {}).get("tasks") or []):
            for t in src_tasks:
                name = t["name"]
                if name in ("fabric", "swe_service"):
                    continue
                if name not in merged:
                    merged[name] = copy.deepcopy(t)
                else:
                    merged[name].update(copy.deepcopy(t))
        for t in merged.values():
            if "depends_on" in t:
                t["depends_on"] = [d for d in t["depends_on"] if d not in ("fabric", "swe_service")]
        if not swe and c in ACCURACY_DROP_POINTS:
            # Strip the drop's SWE-only lines from client_gate, client_preflight, and benchmark
            for cl_task in ("client_gate", "client_preflight"):
                if cl_task in merged:
                    merged[cl_task]["script"] = [
                        x for x in merged[cl_task]["script"] if x != DROP_SWE_EVAL_LINE
                    ]
            bm_script = []
            for x in merged["benchmark"]["script"]:
                if x == DROP_SWE_EVAL_LINE:
                    continue
                x = x.replace("HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0",
                              "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
                x = x.replace("--mode ${{ variables.BENCHMARK_MODE }}", "--mode perf")
                bm_script.append(x)
            merged["benchmark"]["script"] = bm_script
        return merged


class DropPoint:
    def __init__(self, c: int, path: Path):
        self.c = c
        self.path = path
        self.raw = read_text(path)
        self.data = parse_yaml(self.raw, path.name)
        if not isinstance(self.data, dict) or self.data.get("version") != "0.1":
            die(f"{path.name}: unexpected structure")
        all_vars = {k: (v or {}).get("value") for k, v in (self.data.get("variables") or {}).items()}
        want_keys = set(POINT_CORE_VARS)
        if c in HIGH_CONCURRENCY_POINTS:
            want_keys |= set(POINT_OPTIONAL_VARS)
        if c in ACCURACY_DROP_POINTS:
            want_keys |= set(POINT_SWE_ONLY_VARS)
        if set(all_vars) != want_keys:
            die(f"{path.name}: variables {sorted(all_vars)} != expected {sorted(want_keys)}")
        self.vars = {k: v for k, v in all_vars.items() if k not in POINT_SWE_ONLY_VARS}
        self.var_keys = tuple(self.vars.keys())
        m = re.search(r"(?ms)^variables:\n(.*?)^artifacts:\n", self.raw)
        if not m:
            die(f"{path.name}: variables block not found")
        full_vars_raw = m.group(1)
        if c in ACCURACY_DROP_POINTS:
            if not full_vars_raw.endswith(DROP_SWE_VARS_RAW):
                die(f"{path.name}: expected SWE variables tail not found verbatim")
            self.vars_raw = full_vars_raw[:-len(DROP_SWE_VARS_RAW)]
        else:
            self.vars_raw = full_vars_raw
        if not self.vars_raw.endswith("\n") or self.vars_raw.endswith("\n\n"):
            die(f"{path.name}: variables block must end with exactly one newline")
        parsed_vars = {k: (v or {}).get("value")
                       for k, v in parse_yaml("variables:\n" + self.vars_raw, path.name)["variables"].items()}
        if parsed_vars != self.vars:
            die(f"{path.name}: variables raw block does not round-trip")
        arts = self.data.get("artifacts") or []
        if [(a.get("name"), a.get("uri")) for a in arts] != list(DROP_ARTIFACTS):
            die(f"{path.name}: artifacts {[(a.get('name'), a.get('uri')) for a in arts]} changed")
        self.content = {a["name"]: a["content"] for a in arts}
        m_art = re.search(r"(?ms)^artifacts:\n(.*?)^workflow:\n", self.raw)
        if not m_art:
            die(f"{path.name}: artifacts..workflow block not found")
        body = m_art.group(1)
        blocks = re.split(r"(?m)^(?=- name: )", body)
        if blocks[0] != "":
            die(f"{path.name}: unexpected text before the first artifact")
        self.raw_art = {}
        for blk in blocks[1:]:
            parsed = parse_yaml(blk, path.name)
            if not isinstance(parsed, list) or len(parsed) != 1:
                die(f"{path.name}: cannot isolate artifact block")
            name = parsed[0]["name"]
            if parsed[0] != next(a for a in arts if a["name"] == name):
                die(f"{path.name}: raw block of {name} does not round-trip")
            self.raw_art[name] = blk if blk.endswith("\n") else blk + "\n"
        # Perf CLIENT_CONFIG (strips the 9-line swe_bench block on C12, C16, C32, C1024)
        if c in ACCURACY_DROP_POINTS:
            if ("\n" + DROP_SWE_DATASET_CONTENT) not in self.content["CLIENT_CONFIG"]:
                die(f"{path.name}: CLIENT_CONFIG missing expected swe_bench block")
            if ("\n" + DROP_SWE_DATASET_RAW) not in self.raw_art["CLIENT_CONFIG"]:
                die(f"{path.name}: raw CLIENT_CONFIG missing expected swe_bench block")
            self.perf_client_content = self.content["CLIENT_CONFIG"].replace(
                "\n" + DROP_SWE_DATASET_CONTENT, "\n", 1
            )
            self.perf_client_raw_art = self.raw_art["CLIENT_CONFIG"].replace(
                "\n" + DROP_SWE_DATASET_RAW, "\n", 1
            )
        else:
            if "swe_bench" in self.content["CLIENT_CONFIG"]:
                die(f"{path.name}: unexpected swe_bench in perf point CLIENT_CONFIG")
            self.perf_client_content = self.content["CLIENT_CONFIG"]
            self.perf_client_raw_art = self.raw_art["CLIENT_CONFIG"]
        self._assert_point()

    def _assert_point(self) -> None:
        v, c, n = self.vars, self.c, self.path.name
        got = (v["NUM_TRAJECTORIES"], v["NUM_CTX"], v["CTX_GPUS"], v["NUM_GEN"], v["GEN_GPUS"],
               v["TOTAL_GPUS"], v["TOTAL_NODES"], v["FRONTEND_NODE"], v["CLIENT_NODE"])
        if v["CONCURRENCY"] != c:
            die(f"{n}: CONCURRENCY {v['CONCURRENCY']} != {c}")
        if got != EXPECTED[c]:
            die(f"{n}: topology {got} != EXPECTED {EXPECTED[c]}")
        if v["DRAFT_HEAD"] != "sglang":
            die(f"{n}: DRAFT_HEAD changed")
        if v["NUMA_BIND"] != 0:
            die(f"{n}: NUMA_BIND {v['NUMA_BIND']} != 0")
        for k in ("CONCURRENCY", "NUM_TRAJECTORIES", "NUM_CTX", "NUM_GEN", "GEN_GPUS",
                  "TOTAL_GPUS", "TOTAL_NODES", "CTX_GPUS", "FRONTEND_NODE", "CLIENT_NODE"):
            if not isinstance(v[k], int) or isinstance(v[k], bool):
                die(f"{n}: {k} is not an integer")
        if v["TOTAL_GPUS"] != v["NUM_CTX"] * v["CTX_GPUS"] + v["NUM_GEN"] * v["GEN_GPUS"]:
            die(f"{n}: TOTAL_GPUS != NUM_CTX*CTX_GPUS + NUM_GEN*GEN_GPUS")
        if v["TOTAL_GPUS"] != v["TOTAL_NODES"] * GPUS_PER_NODE:
            die(f"{n}: TOTAL_GPUS != TOTAL_NODES*4")
        if v["NUM_TRAJECTORIES"] <= 0 or v["NUM_TRAJECTORIES"] % TRAJ_PER_PASS:
            die(f"{n}: NUM_TRAJECTORIES not a positive multiple of {TRAJ_PER_PASS}")
        allowed_overlap = {"UCX_NET_DEVICES", "UCX_TLS", "UCX_CUDA_IPC_ENABLE_MNNVL", "UCX_RNDV_SCHEME"}
        for envname in ("PREFILL_ENV", "DECODE_ENV"):
            env = parse_yaml(self.content[envname], f"{n} {envname}")
            if not isinstance(env, dict):
                die(f"{n}: {envname} is not a flat mapping")
            overlap = {k for k, _ in GKE_SERVER_ENV} & set(env)
            if not overlap.issubset(allowed_overlap) or "UCX_TLS" not in overlap:
                die(f"{n}: {envname} unexpected overlap with GKE_SERVER_ENV: {sorted(overlap)}")
        for cfg_name, gpus in (("PREFILL_CONFIG", v["CTX_GPUS"]), ("DECODE_CONFIG", v["GEN_GPUS"])):
            cfg = parse_yaml(self.content[cfg_name], f"{n} {cfg_name}")
            world = (cfg.get("tensor_parallel_size", 1) * cfg.get("pipeline_parallel_size", 1)
                     * cfg.get("context_parallel_size", 1))
            if world != gpus:
                die(f"{n}: {cfg_name} TP*PP*CP={world} != {gpus} GPUs")
            if cfg.get("gpus_per_node") != GPUS_PER_NODE:
                die(f"{n}: {cfg_name} gpus_per_node changed")
        dec = parse_yaml(self.content["DECODE_CONFIG"], n)
        if (dec.get("speculative_config") or {}).get("speculative_model") != "/models/kimi-k3-dspark":
            die(f"{n}: DECODE_CONFIG speculative_model moved")
        cl = self.perf_client_content
        for line in CLIENT_ANCHORS:
            if cl.count(line) != 1:
                die(f"{n}: CLIENT_CONFIG anchor {line!r} not found exactly once")
        if "metrics_tokenizer_workers" in cl:
            die(f"{n}: drop CLIENT_CONFIG now sets metrics_tokenizer_workers; drop rewrite 2")
        clp = parse_yaml(cl, f"{n} CLIENT_CONFIG")
        if "seed" in (clp.get("model_params") or {}):
            die(f"{n}: drop CLIENT_CONFIG now sets model_params.seed")
        if (clp["model_params"]["name"] != "/models/kimi-k3"
                or clp["model_params"]["tokenizer_name"] != "/models/kimi-k3"
                or clp["datasets"][0]["path"] != "/data/agentic_combined_v6.jsonl"):
            die(f"{n}: CLIENT_CONFIG model/dataset references changed")
        if clp["endpoint_config"]["endpoints"] != ["http://127.0.0.1:8000"]:
            die(f"{n}: CLIENT_CONFIG endpoints changed")


FABRIC_CMD = 'python3 "${RECIPE}/scripts/node.py" probe "${SFLOW_WORKFLOW_OUTPUT_DIR}/fabric"'
CLIENT_ANCHORS = ("\nsettings:\n", "\nreport_dir: /results/report\n")
CLIENT_REWRITES = (
    ("report_dir: /results/report\n", "report_dir: ${REPORT_DIR}\n"),
    ("settings:\n", f"settings:\n  metrics_tokenizer_workers: {METRICS_TOKENIZER_WORKERS}\n"),
)


def client_content(drop_text: str) -> str:
    out = drop_text
    for old, new in CLIENT_REWRITES:
        key = "\n" + old
        if out.count(key) != 1:
            die(f"CLIENT_CONFIG: {old!r} not found exactly once")
        out = out.replace(key, "\n" + new)
    return out


# --------------------------------------------------------------------------------------
# YAML emission helpers
# --------------------------------------------------------------------------------------
_PLAIN_BAD_START = set("!&*-?{}[]|>'\"%@`#,:")


def scalar(s: str) -> str:
    if "\n" in s:
        die("internal: scalar() got a multi-line string")
    if (s and s == s.strip() and s[0] not in _PLAIN_BAD_START and ": " not in s and " #" not in s
            and not s.endswith(":")):
        try:
            if yaml.safe_load(s) == s:
                return s
        except yaml.YAMLError:
            pass
    q = "'" + s.replace("'", "''") + "'"
    if yaml.safe_load(q) != s:
        die(f"internal: cannot emit {s!r}")
    return q


def item_lines(s: str, pad: str) -> list:
    """YAML block-sequence item for string s at indentation pad."""
    if "\n" not in s:
        return [f"{pad}- {scalar(s)}"]
    if s.endswith("\n\n"):
        die("internal: multi-line item with several trailing newlines")
    chomp, body = ("", s[:-1]) if s.endswith("\n") else ("-", s)
    lines = body.split("\n")
    if lines[0].startswith(" ") or any(ln != ln.rstrip() for ln in lines):
        die(f"internal: item not emittable as a literal block: {lines[0]!r}")
    return [f"{pad}- |{chomp}"] + [(f"{pad}  {ln}" if ln else "") for ln in lines]


def content_lines(name: str, uri: str, text: str) -> list:
    if not text or text.endswith("\n\n"):
        die(f"internal: artifact {name} must be non-empty and not end with multiple newlines")
    chomp, raw = ("", text[:-1]) if text.endswith("\n") else ("-", text)
    body = raw.split("\n")
    if body[0].startswith(" ") or any(ln != ln.rstrip() for ln in body):
        die(f"internal: artifact {name} not emittable as a literal block")
    return [f"- name: {name}", f"  uri: {uri}", f"  content: |{chomp}"] + [(f"    {ln}" if ln else "") for ln in body]


# --------------------------------------------------------------------------------------
# Script plans. Each task script = drop items in order with these exact changes.
#   ("D", item)            drop item, verbatim
#   ("R", drop, new, why)  drop item rewritten
#   ("X", drop, why)       drop item removed
#   ("G", new, why)        GKE insertion
# --------------------------------------------------------------------------------------
def _cache_rewrites(role: str) -> dict:
    tag = "${SLURM_JOB_ID}-${SLURM_STEP_ID}-${SLURM_PROCID:-0}"
    new = f"{role}-${{SFLOW_REPLICA_INDEX}}-${{KIMI_RANK:-0}}"
    why = "D3 per-replica/per-rank cache dir (no SLURM_* on GKE; set -u)"
    return {f'export HF_MODULES_CACHE="/tmp/kimi-{tag}/hf-modules"':
            (f'export HF_MODULES_CACHE="/tmp/kimi-{new}/hf-modules"', why),
            f'export XDG_CACHE_HOME="/tmp/kimi-{tag}/cache"':
            (f'export XDG_CACHE_HOME="/tmp/kimi-{new}/cache"', why)}


DYN_NS_DROP = 'export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379" NATS_SERVER="nats://${HEAD_NODE_IP}:4222" DYN_NAMESPACE="k3-${SLURM_JOB_ID}"'
DYN_NS_GKE = ('export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379" NATS_SERVER="nats://${HEAD_NODE_IP}:4222" '
              'DYN_NAMESPACE="k3-$(basename "${SFLOW_WORKFLOW_OUTPUT_DIR}")"')
DYN_NS_WHY = "D3/D7 per-run Dynamo namespace (no SLURM_JOB_ID on GKE; set -u)"

G_TOOLS = (
    'export SFLOW_TOOLS="$(dirname "${RUN_WITH_ENV_PY}")"\n'
    'export RECIPE="$(dirname "$(dirname "${NODE_PY}")")"\n'
    'for f in "${SFLOW_TOOLS}/run_with_env.py" "${RECIPE}/scripts/node.py" "${RECIPE}/scripts/control_plane.json"; do\n'
    '  [ -f "$f" ] || { echo "FATAL: helper artifact $f missing" >&2; exit 1; }\n'
    'done\n'
)

G_CLIENT_MOUNTS = (
    'export REPORT_DIR="${REPORTS_DIR}/$(basename "${SFLOW_WORKFLOW_OUTPUT_DIR}")"\n'
    'mkdir -p /models /data "${REPORT_DIR}"\n'
    'ln -sfn "${MODEL_PATH}" /models/kimi-k3\n'
    'ln -sfn "${DATASET_PATH}" /data/agentic_combined_v6.jsonl\n'
    'ln -sfn "${REPORTS_DIR}" /results\n'
)

G_SWE_GATE_PLACEHOLDER = "export SWEBENCH_SERVICE_URL=http://127.0.0.1:1 SWEBENCH_SERVICE_AUTH_TOKEN=disabled\n"


def g_env() -> str:
    lines = [
        'echo "GKE EDIT D8: UCX_NET_DEVICES=${UCX_NET_DEVICES:-<unset>} UCX_TLS=${UCX_TLS:-<unset>} (drop env) -> eth0 / ' + GKE_UCX_TLS + '"',
        "unset BASH_ENV ENV NCCL_ENV_PLUGIN NCCL_TUNER_PLUGIN NCCL_PROFILER_PLUGIN NCCL_TUNER_CONFIG_PATH",
        "if [ -d /opt/gcp/nccl-plugins/lib64 ]; then",
        "  mv /opt/gcp/nccl-plugins/lib64 /opt/gcp/nccl-plugins/lib64.disabled 2>/dev/null || true",
        "fi",
    ]
    for k, val in GKE_SERVER_ENV:
        q = val if re.fullmatch(r"[A-Za-z0-9_.,/-]+", val) else "'" + val + "'"
        lines.append(f"export {k}={q}")
    lines.append('export DYN_SYSTEM_HOST="$(hostname -i)" DYN_TCP_RPC_HOST="$(hostname -i)" DYN_TCP_RESPONSE_STREAM_HOST="$(hostname -i)" DYN_EVENT_PLANE_HOST="$(hostname -i)"')
    return "\n".join(lines) + "\n"


G_HANG = f"""python3 - <<'EOF_HANG_PATCH'
import os, sys, tempfile
for label, path, old, new in (
    (
        "hang_detector",
        "{HANG_DETECTOR}",
        "self.timeout = timeout if timeout is not None else 300",
        'self.timeout = int(os.environ.get("TRTLLM_HANG_DETECTOR_TIMEOUT", "3600")) if timeout is None else timeout',
    ),
    (
        "pyexecutor_util_safety_margin",
        "{PYEXECUTOR_UTIL}",
        "    safety_margin = 32 * 1024**2",
        '    safety_margin = int(os.environ.get("TRTLLM_TORCH_MEM_SAFETY_MARGIN_MB", "4096")) * 1024**2',
    ),
    (
        "kv_cache_manager_v2_draft_resume",
        "{KV_CACHE_MANAGER_V2}",
        "            max_util_for_resume=kv_cache_config.max_util_for_resume,",
        "            max_util_for_resume=(1.0 if self.is_draft else kv_cache_config.max_util_for_resume),",
    ),
):
    src = open(path).read()
    if new in src:
        print("GKE_PATCH " + label + ": already patched")
        continue
    if src.count(old) != 1 or "\\nimport os\\n" not in src:
        sys.exit("GKE_PATCH FATAL: " + label + " pattern not found exactly once in " + path)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".gke-tmp")
    with os.fdopen(fd, "w") as f:
        f.write(src.replace(old, new))
    os.chmod(tmp, os.stat(path).st_mode)
    os.replace(tmp, path)
    print("GKE_PATCH " + label + ": patched (atomic)")
EOF_HANG_PATCH
"""

G_SYSCTL = (
    "sysctl -w net.core.rmem_max=134217728 >/dev/null 2>&1 || true\n"
    "sysctl -w net.core.wmem_max=134217728 >/dev/null 2>&1 || true\n"
    'sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728" >/dev/null 2>&1 || true\n'
    'sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728" >/dev/null 2>&1 || true\n'
)

G_MODELS = f"""mkdir -p /models
for spec in "kimi-k3|${{MODEL_PATH}}|{MODEL_REV}" "kimi-k3-dspark|${{DRAFT_PATH}}|{DRAFT_REV}"; do
  IFS='|' read -r name target rev <<< "${{spec}}"
  if [ ! -f "${{target}}/config.json" ]; then
    echo "FATAL: ${{target}}/config.json missing on $(hostname): stage the checkpoints on every node" >&2
    exit 1
  fi
  staged="$(head -c 128 "${{target}}/.sflow-staged" 2>/dev/null || true)"
  if [ "${{staged}}" != "${{rev}}" ]; then
    echo "FATAL: ${{target}}/.sflow-staged is '${{staged//[^A-Za-z0-9._-]/?}}', want ${{rev}} on $(hostname): run stage-kimi-k3-assets.yaml to completion" >&2
    exit 1
  fi
  echo "STAGED ${{target}} @ ${{rev}}"
  ln -sfn "${{target}}" "/models/.${{name}}.$$"
  mv -Tf "/models/.${{name}}.$$" "/models/${{name}}"
  if [ "$(readlink "/models/${{name}}")" != "${{target}}" ]; then
    echo "FATAL: /models/${{name}} does not point at ${{target}}" >&2
    exit 1
  fi
done
python3 - "${{MODEL_PATH}}/config.json" <<'EOF_MODEL_SHA'
import hashlib, sys
p = sys.argv[1]
b = open(p, "rb").read()
h = hashlib.sha1(b"blob %d\\0" % len(b) + b).hexdigest()
want = "{MODEL_CONFIG_GITSHA1}"
print("MODEL config.json git-sha1 " + h + " (want " + want + " = {MODEL_REPO}@{MODEL_REV[:8]})")
sys.exit(0 if h == want else "FATAL: " + p + " is not {MODEL_REPO}@{MODEL_REV[:8]}")
EOF_MODEL_SHA
"""

G_FABRIC_PROBE = 'flock -x "${SFLOW_WORKFLOW_OUTPUT_DIR}/.fabric-probe.lock" env SLURMD_NODENAME="$(hostname | cut -d. -f1)" ' + FABRIC_CMD
G_FABRIC_PRINT = ('echo "KIMI_FABRIC node=$(hostname | cut -d. -f1) '
                  '$(cat "${SFLOW_WORKFLOW_OUTPUT_DIR}/fabric/$(hostname | cut -d. -f1).json")"')

G_BINS = (
    "for b in mpirun trtllm-llmapi-launch python3; do\n"
    '  command -v "$b" >/dev/null || { echo "FATAL: $b not on PATH=$PATH" >&2; exit 1; }\n'
    "done\n"
)

G_RANK_ENV = """cat > "/tmp/kimi-rank-env.sh.$$" <<'EOF_RANK_ENV'
#!/usr/bin/env bash
# GKE D3: the drop's per-rank cache-dir exports, redone for this MPI rank.
set -euo pipefail
unset BASH_ENV ENV NCCL_ENV_PLUGIN NCCL_TUNER_PLUGIN NCCL_PROFILER_PLUGIN NCCL_TUNER_CONFIG_PATH
export NCCL_NET_PLUGIN=none
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:+${PYTORCH_CUDA_ALLOC_CONF},}garbage_collection_threshold:0.75"
export PYTORCH_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
taskset -pc "0-$(($(nproc --all 2>/dev/null || echo 144) - 1))" "$$" >/dev/null 2>&1 || true
role="$1"; shift
export KIMI_RANK="${OMPI_COMM_WORLD_RANK:?kimi-rank-env.sh must be started by mpirun}"
if [ "$role" = "prefill" ]; then
  pod_slot=$(( SFLOW_REPLICA_INDEX * 2 + KIMI_RANK / 4 ))
else
  pod_slot=$(( 32 + SFLOW_REPLICA_INDEX * 4 + KIMI_RANK / 4 ))
fi
pod_pid_base=$(( 100000 + pod_slot * 30000 ))
(
  flock -x 9
  cur_pid=$(cat /proc/sys/kernel/ns_last_pid)
  if [ "$cur_pid" -lt "$pod_pid_base" ]; then
    echo "$pod_pid_base" > /proc/sys/kernel/ns_last_pid
  fi
) 9>/tmp/kimi-pid-ns.lock
export HF_MODULES_CACHE="/tmp/kimi-${role}-${SFLOW_REPLICA_INDEX}-${KIMI_RANK}/hf-modules"
export XDG_CACHE_HOME="/tmp/kimi-${role}-${SFLOW_REPLICA_INDEX}-${KIMI_RANK}/cache"
export TLLM_AUTOTUNER_CACHE_PATH="$XDG_CACHE_HOME/autotuner.json"
export DYN_SYSTEM_HOST="$(hostname -i)" DYN_TCP_RPC_HOST="$(hostname -i)" DYN_TCP_RESPONSE_STREAM_HOST="$(hostname -i)" DYN_EVENT_PLANE_HOST="$(hostname -i)"
mkdir -p "$HF_MODULES_CACHE" "$XDG_CACHE_HOME"
bash -c 'exec "$@"' _ "$@"
EOF_RANK_ENV
chmod 0755 "/tmp/kimi-rank-env.sh.$$"
mv -f "/tmp/kimi-rank-env.sh.$$" /tmp/kimi-rank-env.sh
"""

G_FRONTEND_CPU = f"""if [ ! -s /usr/lib/aarch64-linux-gnu/libcuda.so.1 ]; then
  if [ ! -s /usr/local/cuda/compat/lib.real/libcuda.so.1 ]; then
    echo "FATAL: neither /usr/lib/aarch64-linux-gnu/libcuda.so.1 nor /usr/local/cuda/compat/lib.real/libcuda.so.1 is non-empty" >&2
    exit 1
  fi
  ln -sfn /usr/local/cuda/compat/lib.real/libcuda.so.1 /usr/lib/aarch64-linux-gnu/libcuda.so.1
fi
python3 - <<'EOF_FRONTEND_PATCH'
import os, sys, tempfile
path = "{FLASHINFER_UTILS}"
old = "major, minor = torch.cuda.get_device_capability()"
new = "major, minor = torch.cuda.get_device_capability() if torch.cuda.is_available() else (10, 0)"
src = open(path).read()
if new in src:
    print("GKE_PATCH flashinfer_utils: already patched")
    sys.exit(0)
if src.count(old) != 1:
    sys.exit("GKE_PATCH FATAL: flashinfer_utils pattern not found exactly once in " + path)
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".gke-tmp")
with os.fdopen(fd, "w") as f:
    f.write(src.replace(old, new))
os.chmod(tmp, os.stat(path).st_mode)
os.replace(tmp, path)
print("GKE_PATCH flashinfer_utils: patched (atomic) for CPU-only frontend")
EOF_FRONTEND_PATCH
export DYN_SYSTEM_HOST="$(hostname -i)" DYN_TCP_RPC_HOST="$(hostname -i)" DYN_TCP_RESPONSE_STREAM_HOST="$(hostname -i)" DYN_EVENT_PLANE_HOST="$(hostname -i)"
"""

G_BENCH_IMPORT = (
    'echo "client interpreter: $(command -v python3)"\n'
    "if ! python3 -c 'import inference_endpoint' 2>/dev/null; then\n"
    '  echo "FATAL: inference_endpoint not importable with $(command -v python3) (CLIENT_IMAGE=${CLIENT_IMAGE})" >&2\n'
    "  exit 1\n"
    "fi\n"
)

G_BENCH_ASSETS = f"""if [ ! -f "${{DATASET_PATH}}" ]; then
  echo "FATAL: dataset not found at ${{DATASET_PATH}} on $(hostname)" >&2
  exit 1
fi
got_sha="$(python3 - "${{DATASET_PATH}}" <<'EOF_DATASET_SHA'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print(h.hexdigest())
EOF_DATASET_SHA
)"
if [ "${{got_sha}}" != "{DATASET_SHA256}" ]; then
  echo "FATAL: ${{DATASET_PATH}} sha256 ${{got_sha}} != {DATASET_SHA256} ({DATASET_DROP_NAME})" >&2
  exit 1
fi
echo "dataset ${{DATASET_PATH}} sha256 ${{got_sha}} OK"
for f in config.json tokenizer_config.json; do
  [ -f "${{MODEL_PATH}}/${{f}}" ] || {{ echo "FATAL: ${{MODEL_PATH}}/${{f}} missing (client tokenizer)" >&2; exit 1; }}
done
"""

G_BENCH_REPORT = """echo "KIMI_REPORT_DIR node=${SFLOW_TASK_NODE_NAME:-$(hostname)} path=${REPORT_DIR}"
for v in FRONTEND_NODE_IP FRONTEND_PORT MODEL_PATH DATASET_PATH REPORT_DIR; do
  [ -n "${!v:-}" ] || { echo "FATAL: ${v} is empty; client.yaml expands it" >&2; exit 1; }
done
echo "client endpoint: http://127.0.0.1:${FRONTEND_PORT} (FRONTEND_NODE_IP=${FRONTEND_NODE_IP})"
"""

G_BENCH_MIRROR = """DEST="${SFLOW_WORKFLOW_OUTPUT_DIR}/report"
mkdir -p "${DEST}"
( cd "${REPORT_DIR}" && find . -type f ! -name 'events.jsonl' -print0 | tar cf - --null -T - ) 2>/dev/null \\
  | ( cd "${DEST}" && tar xf - ) 2>/dev/null || true
ls -la "${DEST}" || true
echo "durable report copy on node ${SFLOW_TASK_NODE_NAME:-$(hostname)}: ${REPORT_DIR}"
"""

G_BENCH_EXIT = ('printf "%s\\n" "${KIMI_BENCH_RC}" > "${SFLOW_WORKFLOW_OUTPUT_DIR}/benchmark-client-exit-code"; '
                'echo "benchmark exit code ${KIMI_BENCH_RC}"; exit "${KIMI_BENCH_RC}"')


def gke_launch(role: str, drop_item: str) -> str:
    gpus = "CTX_GPUS" if role == "prefill" else "GEN_GPUS"
    idx = drop_item.find("python3 ")
    if idx < 0 or not drop_item.startswith("exec "):
        die(f"{role}: unexpected launch line {drop_item!r}")
    tail = drop_item[idx:]
    return (f"exec mpirun -np ${{{{ variables.{gpus} }}}} --allow-run-as-root /tmp/kimi-rank-env.sh {role} "
            f'trtllm-llmapi-launch env UCX_NET_DEVICES="$UCX_NET_DEVICES" UCX_TLS="$UCX_TLS" '
            + tail)


def server_plan(drop: Drop, c: int, role: str) -> list:
    script = drop.composed_tasks(c)[role]["script"]
    rw = _cache_rewrites(role)
    envname = "PREFILL_ENV" if role == "prefill" else "DECODE_ENV"
    port = CTX_BASE_PORT if role == "prefill" else GEN_BASE_PORT
    anchors = {
        "eval": 'eval "$ENV_EXPORTS"',
        "env": f'ENV_EXPORTS=$(python3 "${{SFLOW_TOOLS}}/run_with_env.py" "${{{envname}}}")',
        "port": f"export DYN_SYSTEM_PORT=$(({port} + SFLOW_REPLICA_INDEX))",
        "dyn_ns": DYN_NS_DROP,
    }
    for k, a in anchors.items():
        if script.count(a) != 1:
            die(f"C{c} {role}: anchor {k} {a!r} not found exactly once")
    launch_item = script[-1]
    if not launch_item.startswith("exec ") or "trtllm-llmapi-launch" not in launch_item:
        die(f"C{c} {role}: unexpected final launch item {launch_item!r}")
    has_unset_ucx = "unset UCX_NET_DEVICES" in script
    gke_server_blocks = [
        ("G", g_env(), "D8/D9 GKE env (after the drop env so UCX_NET_DEVICES/UCX_TLS win)"),
        ("G", G_HANG, "D10 hang detector + mem margin + draft resume patches (fails closed)"),
        ("G", G_SYSCTL, "D11 socket buffers, best effort"),
        ("G", G_MODELS, "D5 /models symlinks + checkpoint pin (fails closed)"),
        ("G", G_FABRIC_PROBE, "D12 the drop's setup.yaml node.py probe fabric line, per server pod"),
        ("G", G_FABRIC_PRINT, "D12 fabric identity for the runbook cross-node check"),
        ("G", G_BINS, "D1 launcher binaries present"),
    ]
    plan, used = [], set()
    for it in script:
        if it in rw:
            plan.append(("R", it, rw[it][0], rw[it][1]))
            used.add(it)
        elif it == DYN_NS_DROP:
            plan.append(("R", it, DYN_NS_GKE, DYN_NS_WHY))
        elif it == anchors["env"]:
            plan.append(("G", G_TOOLS, "D4 helper artifacts: SFLOW_TOOLS / RECIPE"))
            plan.append(("D", it))
        elif it == anchors["eval"]:
            plan.append(("D", it))
            if not has_unset_ucx:
                plan += gke_server_blocks
        elif it == "unset UCX_NET_DEVICES":
            plan.append(("X", it, "D8 GKE sets UCX_NET_DEVICES=eth0 in the GKE env block"))
            plan += gke_server_blocks
        elif it == launch_item:
            plan.append(("G", G_RANK_ENV, "D3 per-rank cache-dir wrapper"))
            plan.append(("R", it, gke_launch(role, it), "D2 mpirun launch + GKE UCX_NET_DEVICES/UCX_TLS"))
        else:
            plan.append(("D", it))
    if used != set(rw):
        die(f"C{c} {role}: cache-dir lines not found: {sorted(set(rw) - used)}")
    return plan


def frontend_plan(drop: Drop, c: int) -> list:
    script = drop.composed_tasks(c)["frontend"]["script"]
    rw = _cache_rewrites("frontend")
    start_echo = "echo DYNAMO_FRONTEND_STARTING"
    for a in (DYN_NS_DROP, start_echo):
        if script.count(a) != 1:
            die(f"C{c} frontend: {a!r} not found exactly once")
    plan, used = [], set()
    for it in script:
        if it in rw:
            plan.append(("R", it, rw[it][0], rw[it][1]))
            used.add(it)
        elif it == DYN_NS_DROP:
            plan.append(("R", it, DYN_NS_GKE, DYN_NS_WHY))
        elif it == start_echo:
            plan.append(("G", G_MODELS, "D5 /models symlinks + checkpoint pin for frontend tokenizer"))
            plan.append(("G", G_FRONTEND_CPU, "D1 CPU-only frontend libcuda.so.1 symlink + flashinfer_utils GPU guard"))
            plan.append(("D", it))
        else:
            plan.append(("D", it))
    if used != set(rw):
        die(f"C{c} frontend: cache-dir lines not found: {sorted(set(rw) - used)}")
    return plan


def control_plane_plan(drop: Drop, c: int) -> list:
    script = drop.composed_tasks(c)["control_plane"]["script"]
    cp_cmd = 'exec python3 "${RECIPE}/scripts/node.py" control-plane'
    if script != ["set -euo pipefail", cp_cmd]:
        die(f"C{c} control_plane: unexpected script {script}")
    return [
        ("D", "set -euo pipefail"),
        ("G", G_TOOLS, "D4 helper artifacts: SFLOW_TOOLS / RECIPE"),
        ("R", cp_cmd,
         'SLURM_JOB_ID="$(basename "${SFLOW_WORKFLOW_OUTPUT_DIR}")" exec python3 "${RECIPE}/scripts/node.py" control-plane',
         "D3/D7 per-run control-plane state dir"),
    ]


def client_gate_plan(drop: Drop, c: int, swe: bool = False) -> list:
    script = drop.composed_tasks(c)["client_gate"]["script"]
    path_line = 'export PATH="/opt/venv/bin:$PATH"'
    if script.count(path_line) != 1:
        die(f"C{c} client_gate: {path_line!r} not found")
    plan = []
    for it in script:
        plan.append(("D", it))
        if it == path_line:
            plan.append(("G", G_CLIENT_MOUNTS, "D5/D6/D13 client container mount symlinks + REPORT_DIR"))
            if swe:
                plan.append(("G", G_SWE_GATE_PLACEHOLDER,
                             "S5 schema validation placeholder (real SWE handoff runs in benchmark)"))
    return plan


def client_preflight_plan(drop: Drop, c: int, swe: bool = False) -> list:
    tasks = drop.composed_tasks(c)
    if "client_preflight" not in tasks:
        return []
    script = tasks["client_preflight"]["script"]
    unset_line = "unset PYTHONPATH PYTHONHOME UV_PROJECT_ENVIRONMENT"
    if script.count(unset_line) != 1:
        die(f"C{c} client_preflight: {unset_line!r} not found")
    plan = []
    for it in script:
        plan.append(("D", it))
        if it == unset_line:
            plan.append(("G", G_CLIENT_MOUNTS, "D5/D6/D13 client container mount symlinks + REPORT_DIR"))
            if swe:
                plan.append(("G", G_SWE_GATE_PLACEHOLDER,
                             "S5 schema validation placeholder (real SWE handoff runs in benchmark)"))
    return plan


PERF_RUN_CMD = '/opt/venv/bin/python3 -m inference_endpoint.main benchmark from-config -c "${CLIENT_CONFIG}" --mode perf'
ALT_PERF_RUN_CMD = 'inference-endpoint benchmark from-config -c "${CLIENT_CONFIG}" --mode perf'


def benchmark_plan(drop: Drop, c: int) -> list:
    script = drop.composed_tasks(c)["benchmark"]["script"]
    hf = ("export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
          "PYTHONDONTWRITEBYTECODE=1")
    if script.count(hf) != 1:
        die(f"C{c} benchmark: {hf!r} not found")
    plan = []
    for it in script:
        if it == hf:
            plan.append(("D", it))
            plan += [
                ("G", G_TOOLS, "D4 helper artifacts: SFLOW_TOOLS / RECIPE"),
                ("G", G_CLIENT_MOUNTS, "D5/D6/D13 client container mount symlinks + REPORT_DIR"),
                ("G", G_BENCH_IMPORT, "D13 client import check"),
                ("G", G_BENCH_ASSETS, "D6/D13 dataset sha256 + tokenizer present"),
                ("G", G_BENCH_REPORT, "D13 durable REPORT_DIR, printed first"),
            ]
        elif it == PERF_RUN_CMD:
            plan.append(("D", it))
        elif it == ALT_PERF_RUN_CMD:
            plan.append(("G", "set +e", "D13 keep the exit code; mirror the report either way"))
            plan.append(("R", it, PERF_RUN_CMD, "D13 canonical venv python3 -m inference_endpoint.main invocation"))
            plan.append(("G", "KIMI_BENCH_RC=$?; set -e", "D13 benchmark exit code"))
            plan.append(("G", G_BENCH_MIRROR, "D13 small-file report mirror"))
            plan.append(("G", G_BENCH_EXIT, "D13 propagate the benchmark exit code"))
        elif it == "CLIENT_RC=$?":
            plan.append(("R", it, "KIMI_BENCH_RC=$?; set -e", "D13 benchmark exit code"))
            plan.append(("G", G_BENCH_MIRROR, "D13 small-file report mirror"))
        elif it == 'printf "%s\\n" "$CLIENT_RC" > "${SFLOW_WORKFLOW_OUTPUT_DIR}/benchmark-client-exit-code"':
            plan.append(("X", it, "D13 folded into post-mirror exit line"))
        elif it == 'exit "$CLIENT_RC"':
            plan.append(("R", it, G_BENCH_EXIT, "D13 propagate the benchmark exit code"))
        else:
            plan.append(("D", it))
    return plan


def preflight_plan(drop: Drop, c: int) -> list:
    script = drop.composed_tasks(c)["preflight"]["script"]
    pre = 'python3 "${SEGMENTED_TOOLS}/node.py" check "${SEGMENTED_PLAN}/placement.json" preflight'
    ln_s = 'ln -s "${RESULTS_PATH}/report" "${SFLOW_WORKFLOW_OUTPUT_DIR}/report"'
    if script.count(pre) != 1:
        die(f"C{c} preflight: node.py preflight line not found")
    plan = []
    for it in script:
        if it == pre:
            plan.append(("X", it, "D12 Slurm segmented placement check replaced by ComputeDomain + per-server-pod node.py probe"))
            plan.append(("G", 'echo "GKE D12: node.py preflight placement check replaced by the per-server-pod '
                              'fabric probe (grep KIMI_FABRIC; one ClusterUUID/CliqueId expected)"', "D12 note"))
        elif it == ln_s:
            plan.append(("X", it, "D13 benchmark mirrors REPORT_DIR into ${SFLOW_WORKFLOW_OUTPUT_DIR}/report"))
        else:
            plan.append(("D", it))
    return plan


def plans(drop: Drop, c: int, swe: bool = False) -> dict:
    out = {
        "preflight": preflight_plan(drop, c),
        "client_gate": client_gate_plan(drop, c, swe=swe),
    }
    if "client_preflight" in drop.composed_tasks(c):
        out["client_preflight"] = client_preflight_plan(drop, c, swe=swe)
    out.update({
        "control_plane": control_plane_plan(drop, c),
        "prefill": server_plan(drop, c, "prefill"),
        "decode": server_plan(drop, c, "decode"),
        "frontend": frontend_plan(drop, c),
        "benchmark": benchmark_plan(drop, c),
    })
    return out


def plan_items(plan: list) -> list:
    out = []
    for p in plan:
        if p[0] == "D":
            out.append(p[1])
        elif p[0] == "R":
            out.append(p[2])
        elif p[0] == "G":
            out.append(p[1])
    return out


def emit_script(plan: list, pad: str) -> list:
    lines = []
    for p in plan:
        if p[0] == "D":
            lines += item_lines(p[1], pad)
        elif p[0] == "R":
            lines.append(f"{pad}# GKE EDIT {p[3]}")
            lines += item_lines(p[2], pad)
        elif p[0] == "G":
            lines.append(f"{pad}# GKE ADD {p[2]}")
            lines += item_lines(p[1], pad)
        elif p[0] == "X":
            lines.append(f"{pad}# GKE REMOVED ({p[2]}): {p[1]}")
    return lines


# --------------------------------------------------------------------------------------
# Topology / placement (sflow resource_planner: sub-node servers pack contiguously from
# node 0; a server larger than a node starts on a whole-node boundary).
# --------------------------------------------------------------------------------------
def placement(v: dict) -> dict:
    nc, cg, ng, gg = v["NUM_CTX"], v["CTX_GPUS"], v["NUM_GEN"], v["GEN_GPUS"]
    ctx_gpus = nc * cg
    gen_base = ctx_gpus if gg <= GPUS_PER_NODE else -(-ctx_gpus // GPUS_PER_NODE) * GPUS_PER_NODE

    def nodes(start, count):
        return list(range(start // GPUS_PER_NODE, (start + count - 1) // GPUS_PER_NODE + 1))

    ctx = [nodes(i * cg, cg) for i in range(nc)]
    gen = [nodes(gen_base + i * gg, gg) for i in range(ng)]
    used = [n for s in ctx + gen for n in s]
    if max(used) + 1 != v["TOTAL_NODES"]:
        die(f"C{v['CONCURRENCY']}: placement uses {max(used) + 1} nodes != TOTAL_NODES {v['TOTAL_NODES']}")
    if gen_base % GPUS_PER_NODE and gg > GPUS_PER_NODE:
        die("internal: gen_base not node aligned")
    for idx in (v["FRONTEND_NODE"], v["CLIENT_NODE"]):
        if not 0 <= idx < v["TOTAL_NODES"]:
            die(f"C{v['CONCURRENCY']}: node index {idx} outside the allocation")
    return {"ctx": ctx, "gen": gen, "gen_base": gen_base}


def topo_text(v: dict) -> list:
    pl = placement(v)

    def fmt(groups, base):
        return ", ".join(f"{i}:{base + i}@n{g[0]}" + (f"-{g[-1]}" if len(g) > 1 else "")
                         for i, g in enumerate(groups))

    return [f"#   ctx  {v['NUM_CTX']} x {v['CTX_GPUS']} GPU  (replica:sysport@nodes) {fmt(pl['ctx'], CTX_BASE_PORT)}",
            f"#   gen  {v['NUM_GEN']} x {v['GEN_GPUS']} GPU  (replica:sysport@nodes) {fmt(pl['gen'], GEN_BASE_PORT)}",
            f"#   = {v['TOTAL_GPUS']} GPUs = {v['TOTAL_NODES']} nodes; control_plane node 0 (etcd :2379, nats :4222), "
            f"frontend node {v['FRONTEND_NODE']} (:{FRONTEND_PORT}), client node {v['CLIENT_NODE']}"]


def dest_dir(dest: Path, v: dict) -> Path:
    return dest / f"{v['TOTAL_NODES']}nodes" / f"point_{v['CONCURRENCY']}"


def recipe_name(c: int) -> str:
    return f"kimi_k3_gke_disagg_point{c}.yaml"


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
def render_recipe(drop: Drop, c: int) -> str:
    pt = drop.points[c]
    v = pt.vars
    pl = plans(drop, c)
    ctasks = drop.composed_tasks(c)
    L = []
    a = L.append
    a('version: "0.1"')
    a("# ============================================================================")
    a(f"# Kimi-K3 -- MLPerf Endpoints v1.0.3, point C{c}, TensorRT-LLM + Dynamo")
    a("# disaggregated + DSpark, on GKE A4X-max (GB300 NVL72).")
    a("# GENERATED by ../../generate_point_recipes.py -- do not hand-edit; `--check`")
    a("# verifies this file against the drop byte for byte.")
    a("#")
    a(f"# Port of {DROP_REL}/")
    a(f"#   {POINTS_SUBDIR}/point_{c}.yaml   variables + 6 artifacts + task overrides")
    a("#   disagg.yaml  setup.yaml  slurm_env.yaml  scripts/  ../sflow/tools/")
    a(f"# Server image {SERVER_IMAGE}")
    a(f"#   (TensorRT-LLM {TRTLLM_VERSION}, Dynamo {DYNAMO_VERSION}); client image {CLIENT_IMAGE}")
    a(f"#   (drop digest {CLIENT_IMAGE_DROP.replace('#', '/')}).")
    a(f"# Checkpoint {MODEL_REPO}@{MODEL_REV}")
    a(f"# Draft      {DRAFT_REPO}@{DRAFT_REV}")
    a(f"# Dataset    {DATASET_DROP_NAME} sha256 {DATASET_SHA256}")
    a("#")
    a("# TOPOLOGY (drop values; node math = sflow placement):")
    L.extend(topo_text(v))
    a("#")
    a("# GKE deviations (each is also marked GKE EDIT / ADD / REMOVED at its line):")
    for ln in DEVIATIONS.rstrip("\n").split("\n"):
        a(f"#   {ln}" if ln else "#")
    a("# ============================================================================")
    a("")
    a("variables:")
    a(f"  # ---- VERBATIM: point_{c}.yaml variables ----")
    L.extend(pt.vars_raw.rstrip("\n").split("\n"))
    a("  # ---- VERBATIM: disagg.yaml variables ----")
    a("  DYNAMO_DECODE_METRICS_FLAG:")
    a("    value: --no-publish-metrics")
    a("  DYNAMO_FRONTEND_CORE_SHA256:")
    a(f"    value: {DYNAMO_FRONTEND_CORE_SHA256}")
    a("  # ---- VERBATIM: slurm_env.yaml ----")
    a("  BENCHMARK_MODE:")
    a("    value: perf")
    a("  MODE:")
    a("    value: disagg")
    a("  # ---- GKE EDIT (D1/D4/D5/D6): site values replacing slurm_env.yaml ----")
    a("  MODEL_PATH:")
    a(f"    description: {scalar(f'{MODEL_REPO}@{MODEL_REV[:8]} staged on every node (stage-kimi-k3-assets.yaml).')}")
    a(f"    value: {MODEL_PATH}")
    a("  DRAFT_PATH:")
    a(f"    description: {scalar(f'{DRAFT_REPO}@{DRAFT_REV[:8]} staged on every node (drop: SGL_DRAFT_PATH).')}")
    a(f"    value: {DRAFT_PATH}")
    a("  DATASET_PATH:")
    a(f"    description: {scalar(f'{DATASET_DROP_NAME} (same sha256), staged on every node.')}")
    a(f"    value: {DATASET_PATH}")
    a("  CONTROL_PLANE_DIR:")
    a("    description: 'Cached etcd v3.6.5 and nats-server v2.11.9 binaries on node SSD.'")
    a(f"    value: {CONTROL_PLANE_DIR}")
    a("  SERVER_IMAGE:")
    a(f"    description: {scalar(f'drop {SERVER_IMAGE_DROP.split(chr(64))[0]} by its pinned digest')}")
    a(f"    value: {SERVER_IMAGE}")
    a("  CLIENT_IMAGE:")
    a(f"    description: {scalar(f'drop {CLIENT_IMAGE_DROP} by its pinned digest (commit d3c57f2)')}")
    a(f"    value: {CLIENT_IMAGE}")
    a("  K8S_NAMESPACE:")
    a("    value: default")
    a("  GKE_NODEPOOL:")
    a("    value: <nodepool>")
    a("")
    a("artifacts:")
    a(f"# ---- VERBATIM: point_{c}.yaml CLIENT_CONFIG except two lines (D13) ----")
    a("#   report_dir: /results/report -> ${REPORT_DIR} (exported by the")
    a("#     client tasks: a per-run dir on the node SSD that survives the pod)")
    a(f"#   settings.metrics_tokenizer_workers: {METRICS_TOKENIZER_WORKERS} added (drop: client default;")
    a("#     only speeds up the post-run tokenization, never the measured tokens)")
    cl = client_content(pt.perf_client_content)
    new_raw_cl = pt.perf_client_raw_art
    for old, new in CLIENT_REWRITES:
        o = "\n" + "".join("    " + x + "\n" for x in old.rstrip("\n").split("\n"))
        nw = "\n" + "".join("    " + x + "\n" for x in new.rstrip("\n").split("\n"))
        if new_raw_cl.count(o) != 1:
            die(f"point_{c}: raw CLIENT_CONFIG line {old!r} not found exactly once")
        new_raw_cl = new_raw_cl.replace(o, nw)
    if parse_yaml(new_raw_cl, "client")[0]["content"] != cl:
        die(f"point_{c}: CLIENT_CONFIG raw edit diverges from the content edit")
    L.extend(new_raw_cl.rstrip("\n").split("\n"))
    a(f"# ---- VERBATIM: point_{c}.yaml server/env/router artifacts ----")
    for name in VERBATIM_ARTIFACTS:
        L.extend(pt.raw_art[name].rstrip("\n").split("\n"))
    a("# ---- GKE ADD (D4): drop helpers, VERBATIM ----")
    for name, uri, rel in HELPERS:
        a(f"# {rel}")
        L.extend(content_lines(name, uri, drop.helpers[name]))
    a("# ---- GKE EDIT (D4/D5/D6): node-local hostPaths (setup.yaml MODEL/DRAFT/DATASET + control plane) ----")
    a("- name: MODEL")
    a("  uri: fs://${{ variables.MODEL_PATH }}")
    a("- name: DRAFT")
    a("  uri: fs://${{ variables.DRAFT_PATH }}")
    a("- name: DATASET_DIR")
    a(f"  uri: fs://{DATASET_DIR}")
    a("- name: CONTROL_PLANE_DIR")
    a("  uri: fs://${{ variables.CONTROL_PLANE_DIR }}")
    a("# ---- GKE ADD: durable reports (mk-report-dir.yaml) and the IMEX channels ----")
    a("- name: REPORTS_DIR")
    a(f"  uri: fs://{REPORTS_DIR}")
    a("- name: IMEX_CHANNELS")
    a("  uri: fs:///dev/nvidia-caps-imex-channels")
    a("")
    a("operators:")
    a("# GKE EDIT (D1): srun operators -> Kubernetes. slots_per_worker = the pod's GPUs;")
    a("# sflow's default cpu_bind=core maps ranks --map-by ppr:<slots>:node, i.e. packed")
    a("# per pod, which TRT-LLM needs (device = global rank % GPUs per pod).")
    for opname, gvar in (("server_ctx", "CTX_GPUS"), ("server_gen", "GEN_GPUS")):
        a(f"- name: {opname}")
        a("  type: k8s_mpi")
        a("  image: ${{ variables.SERVER_IMAGE }}")
        a("  image_pull_policy: IfNotPresent")
        a("  shm_size: 64Gi")
        a("  run_as_root: true")
        a("  security_context:")
        a("    privileged: true")
        a("  mpi:")
        a("    route: auto")
        a(f"    # = the drop's srun ntasks_per_node for {gvar}, verbatim")
        a(f"    slots_per_worker: {ntasks_per_node_expr(gvar)}")
        a("    run_launcher_as_worker: true")
        a("    launcher_creation_policy: WaitForWorkersReady")
        a("    ssh_port: 2222")
        a("    ensure_sshd: true")
    a("# CPU-only server pod (preflight, control_plane, frontend): no GPU, not privileged.")
    a("- name: server_cpu")
    a("  type: k8s")
    a("  image: ${{ variables.SERVER_IMAGE }}")
    a("  image_pull_policy: IfNotPresent")
    a("  shm_size: 16Gi")
    a("  run_as_root: true")
    a("- name: client")
    a("  type: k8s")
    a("  image: ${{ variables.CLIENT_IMAGE }}")
    a("  image_pull_policy: IfNotPresent")
    a("  shm_size: 16Gi")
    a("  run_as_root: true")
    a("")
    a("backends:")
    a("# GKE EDIT (D1): Slurm backend -> Kubernetes. Named `cluster` like the drop so the")
    a("# setup.yaml workflow variables (backends.cluster.nodes[...]) stay verbatim.")
    a("- name: cluster")
    a("  type: kubernetes")
    a("  default: true")
    a("  namespace: ${{ variables.K8S_NAMESPACE }}")
    a("  nodes: ${{ variables.TOTAL_NODES }}")
    a(f"  gpus_per_node: {GPUS_PER_NODE}")
    a("  host_network: true")
    a("  host_ipc: true")
    a("  scheduling: device_plugin")
    a("  node_selector:")
    a("    cloud.google.com/gke-nodepool: ${{ variables.GKE_NODEPOOL }}")
    a("  # No usable in-pod RDMA NIC on A4X-max/GKE; MNNVL is P2P, not a NET transport.")
    a("  rdma: disable")
    a("  # The NVLink ComputeDomain: GKE's replacement for --segment=SEGMENT_NODES.")
    a("  compute_domain:")
    a("    create: true")
    a("  tolerations:")
    a("  - key: kubernetes.io/arch")
    a("    operator: Equal")
    a("    value: arm64")
    a("    effect: NoSchedule")
    a("  - key: nvidia.com/gpu")
    a("    operator: Exists")
    a("    effect: NoSchedule")
    a("  reservation:")
    a("    timeout: 1800")
    a("  collect_max_file_size: 1Gi")
    a("  collect_grace_seconds: 900")
    a("")
    a("workflow:")
    a(f"  name: kimi_k3_trtllm_v1_0_3_gke_c{c}")
    a("  # ---- VERBATIM: setup.yaml workflow variables ----")
    a("  variables:")
    L.extend(drop.wf_vars_raw.rstrip("\n").split("\n"))
    a("  tasks:")
    # preflight
    a("  # ---- setup.yaml preflight (D12) ----")
    a("  - name: preflight")
    a("    # GKE EDIT (D1): operator host (srun) -> server_cpu")
    a("    operator: server_cpu")
    a("    resources:")
    a("      nodes:")
    a("        indices:")
    a("        - 0")
    a("    # GKE EDIT (D12): depends_on fabric removed (fabric runs in the server pods)")
    a("    script:")
    L.extend(emit_script(pl["preflight"], "    "))
    # client_gate
    a("  # ---- disagg.yaml client_gate ----")
    a("  - name: client_gate")
    a("    operator: client")
    a("    resources:")
    a("      nodes:")
    a("        indices:")
    a("        - ${{ variables.CLIENT_NODE }}")
    a("    depends_on:")
    for d in ctasks["client_gate"]["depends_on"]:
        a(f"    - {d}")
    a("    script:")
    L.extend(emit_script(pl["client_gate"], "    "))
    # client_preflight (C192..C1024)
    if "client_preflight" in ctasks:
        a(f"  # ---- point_{c}.yaml client_preflight ----")
        a("  - name: client_preflight")
        a("    operator: client")
        a("    resources:")
        a("      nodes:")
        a("        indices:")
        a("        - ${{ variables.FRONTEND_NODE }}")
        a("    depends_on:")
        for d in ctasks["client_preflight"]["depends_on"]:
            a(f"    - {d}")
        a("    script:")
        L.extend(emit_script(pl["client_preflight"], "    "))
    # control_plane
    a("  # ---- disagg.yaml control_plane (D1/D7) ----")
    a("  - name: control_plane")
    a("    # GKE EDIT (D1): operator host (srun) -> server_cpu")
    a("    operator: server_cpu")
    a("    resources:")
    a("      nodes:")
    a("        indices:")
    a("        - 0")
    a("    depends_on:")
    for d in ctasks["control_plane"]["depends_on"]:
        a(f"    - {d}")
    a("    script:")
    L.extend(emit_script(pl["control_plane"], "    "))
    L.extend(control_plane_probe_lines("    "))
    # servers
    for role in ("prefill", "decode"):
        t = ctasks[role]
        cnt, gvar, op = (("NUM_CTX", "CTX_GPUS", "server_ctx") if role == "prefill"
                         else ("NUM_GEN", "GEN_GPUS", "server_gen"))
        a(f"  # ---- disagg.yaml + point_{c}.yaml {role} ----")
        a(f"  - name: {role}")
        a("    replicas:")
        a(f"      count: ${{{{ variables.{cnt} }}}}")
        a("      policy: parallel")
        a(f"    # GKE EDIT (D1): srun operator server (ntasks {gvar}) -> {op} (mpirun -np {gvar})")
        a(f"    operator: {op}")
        a("    resources:")
        a("      gpus:")
        a(f"        count: ${{{{ variables.{gvar} }}}}")
        a("    depends_on:")
        for d in t["depends_on"]:
            a(f"    - {d}")
        a("    script:")
        L.extend(emit_script(pl[role], "    "))
        L.extend(server_probe_lines("    "))
    # frontend
    t = ctasks["frontend"]
    a(f"  # ---- disagg.yaml + point_{c}.yaml frontend ----")
    a("  - name: frontend")
    a("    # GKE EDIT (D1): srun operator server (ntasks 1) -> server_cpu")
    a("    operator: server_cpu")
    a("    resources:")
    a("      nodes:")
    a("        indices:")
    a("        - ${{ variables.FRONTEND_NODE }}")
    a("    depends_on:")
    for d in t["depends_on"]:
        a(f"    - {d}")
    a("    script:")
    L.extend(emit_script(pl["frontend"], "    "))
    L.extend(frontend_probe_lines("    "))
    # benchmark
    t = ctasks["benchmark"]
    a(f"  # ---- disagg.yaml + point_{c}.yaml benchmark ----")
    a("  - name: benchmark")
    a("    operator: client")
    a("    resources:")
    a("      nodes:")
    a("        indices:")
    a("        - ${{ variables.CLIENT_NODE }}")
    a("    depends_on:")
    for d in t["depends_on"]:
        a(f"    - {d}")
    a("    script:")
    L.extend(emit_script(pl["benchmark"], "    "))
    return "\n".join(L) + "\n"


def control_plane_probe_lines(pad: str) -> list:
    return [f"{pad}# VERBATIM: disagg.yaml control_plane probes",
            f"{pad}probes:",
            f"{pad}  readiness:",
            f"{pad}    log_watch:",
            f"{pad}      regex_pattern: DYNAMO_CONTROL_READY",
            f"{pad}    timeout: 120"]


def server_probe_lines(pad: str) -> list:
    return [f"{pad}# VERBATIM: disagg.yaml prefill/decode probes",
            f"{pad}probes:",
            f"{pad}  readiness:",
            f"{pad}    log_watch:",
            f"{pad}      regex_pattern: Set runtime config data_parallel_size",
            f"{pad}    timeout: 3600",
            f"{pad}  failure:",
            f"{pad}    log_watch:",
            f"{pad}      regex_pattern: 're:OutOfMemoryError|RuntimeError: Executor|EngineDeadError|CUDA error:|CUDA out of memory'"]


def frontend_probe_lines(pad: str) -> list:
    return [f"{pad}# VERBATIM: disagg.yaml frontend probes",
            f"{pad}probes:",
            f"{pad}  readiness:",
            f"{pad}    log_watch:",
            f"{pad}      regex_pattern: DYNAMO_FRONTEND_STARTING",
            f"{pad}    timeout: 3600",
            f"{pad}  failure:",
            f"{pad}    log_watch:",
            f"{pad}      regex_pattern: re:Traceback|panicked at"]


LAUNCH_TEMPLATE = r"""#!/usr/bin/env bash
# Launch Kimi-K3 MLPerf Endpoints v1.0.3 point C@@C@@ (TensorRT-LLM + Dynamo disaggregated)
# on GKE A4X-max (GB300). @@TOPO@@
# GENERATED by ../../generate_point_recipes.py -- do not hand-edit.
#
# Before the first point (../../README.md): stage-kimi-k3-assets.yaml,
# mk-report-dir.yaml, prepull-images.yaml (temporary pull secret), then one point
# at a time.
#
#   NAMESPACE=<ns> NODEPOOL=<pool> ./launch_test.sh [sflow run flags]
#   REQUIRED_NODES=<n>   node-gate threshold (default @@N@@)
#   SKIP_NODE_CHECK=1    skip the node gate
#   ./launch_test.sh --dry-run
set -euo pipefail

cd "$(dirname "$0")"

SFLOW_VENV="${SFLOW_VENV:<workdir>/sflow_workspace/.venv}"
if [ -x "${SFLOW_VENV}/bin/sflow" ]; then
  PATH="${SFLOW_VENV}/bin:${PATH}"
fi
if ! command -v sflow >/dev/null 2>&1; then
  echo "sflow not found. Set SFLOW_VENV=/path/to/venv or install sflow." >&2
  exit 1
fi

RECIPE="@@RECIPE@@"
NAMESPACE="${NAMESPACE:-default}"
NODEPOOL="${NODEPOOL:<nodepool>}"
REQUIRED_NODES="${REQUIRED_NODES:-@@N@@}"

# Fail fast instead of letting the reservation sit for its 1800 s timeout:
# count Ready, uncordoned nodes in the pool that advertise 4 GPUs.
if [ "${SKIP_NODE_CHECK:-0}" != "1" ] && command -v kubectl >/dev/null 2>&1; then
  if ! NODE_STATES=$(kubectl get nodes -l "cloud.google.com/gke-nodepool=${NODEPOOL}" \
      -o jsonpath='{range .items[*]}{.spec.unschedulable}{"|"}{.status.allocatable.nvidia\.com/gpu}{"|"}{range .status.conditions[?(@.type=="Ready")]}{.status}{end}{"\n"}{end}'); then
    echo "kubectl get nodes failed (see above): fix the kubectl context/credentials," >&2
    echo "or set SKIP_NODE_CHECK=1 to launch without the node check." >&2
    exit 1
  fi
  HEALTHY=$(printf '%s' "${NODE_STATES}" | awk -F'|' '$1!="true" && $2=="4" && $3=="True"' | wc -l)
  echo "schedulable 4-GPU nodes in ${NODEPOOL}: ${HEALTHY} (need ${REQUIRED_NODES})"
  if [ "${HEALTHY}" -lt "${REQUIRED_NODES}" ]; then
    echo "Not enough healthy nodes; refusing to launch. Set SKIP_NODE_CHECK=1 to override." >&2
    exit 1
  fi
fi

sflow run -f "${RECIPE}" \
  --kube-namespace "${NAMESPACE}" \
  --kube-node-selector "cloud.google.com/gke-nodepool=${NODEPOOL}" \
  "$@"
"""


def render_launch(drop: Drop, c: int) -> str:
    v = drop.points[c].vars
    topo = (f"{v['NUM_CTX']} ctx x {v['CTX_GPUS']} GPU + {v['NUM_GEN']} gen x {v['GEN_GPUS']} GPU = "
            f"{v['TOTAL_GPUS']} GPUs = {v['TOTAL_NODES']} nodes.")
    out = (LAUNCH_TEMPLATE.replace("@@C@@", str(c)).replace("@@TOPO@@", topo)
           .replace("@@N@@", str(v["TOTAL_NODES"])).replace("@@RECIPE@@", recipe_name(c)))
    if "@@" in out:
        die("internal: unreplaced launch_test.sh placeholder")
    return out


# --------------------------------------------------------------------------------------
# Opt-in --image-pull-secret variant = the default render plus anchored insertions.
# Each anchor must occur exactly once; check(pull_secret=True) re-derives the delta.
# --------------------------------------------------------------------------------------
PS_RECIPE_HEADER_OLD = ("# GENERATED by ../../generate_point_recipes.py -- do not hand-edit; `--check`\n"
                        "# verifies this file against the drop byte for byte.\n")
PS_OP_LINES = (f"  # GKE ADD ({PULL_SECRET_SUBDIR} variant): registry credentials for the private\n"
               "  # server image, for kubelets that re-check them even for cached images.\n"
               "  image_pull_secrets:\n"
               f"  - {PULL_SECRET}\n")
PS_LAUNCH_HEADER_OLD = "# GENERATED by ../../generate_point_recipes.py -- do not hand-edit.\n"
PS_LAUNCH_README_OLD = "# Before the first point (../../README.md): stage-kimi-k3-assets.yaml,\n"
PS_LAUNCH_README_NEW = "# Before the first point (../../../README.md): stage-kimi-k3-assets.yaml,\n"
PS_LAUNCH_RUN = '\nsflow run -f "${RECIPE}" \\\n'
PS_LAUNCH_GATE = f"""# VARIANT {PULL_SECRET_SUBDIR}: the server pods name the registry Secret {PULL_SECRET};
# without it in NAMESPACE they sit in ErrImagePull until the reservation times
# out. Reads only the Secret's type.
if [ "${{SKIP_SECRET_CHECK:-0}}" != "1" ]; then
  if ! SECRET_TYPE=$(kubectl get secret {PULL_SECRET} -n "${{NAMESPACE}}" -o jsonpath='{{.type}}'); then
    echo "Secret {PULL_SECRET} not readable in namespace ${{NAMESPACE}}: create it there as in" >&2
    echo "../../../prepull-images.yaml (and keep it), or set SKIP_SECRET_CHECK=1." >&2
    exit 1
  fi
  if [ "${{SECRET_TYPE}}" != "kubernetes.io/dockerconfigjson" ]; then
    echo "Secret {PULL_SECRET} in ${{NAMESPACE}} has type '${{SECRET_TYPE}}', want kubernetes.io/dockerconfigjson" >&2
    exit 1
  fi
  echo "pull secret {PULL_SECRET} present in ${{NAMESPACE}}"
fi

"""


def ps_recipe_header(v: dict) -> str:
    base = f"../../../{v['TOTAL_NODES']}nodes/point_{v['CONCURRENCY']}/"
    return ("# GENERATED by ../../../generate_point_recipes.py --image-pull-secret -- do not\n"
            "# hand-edit; `--image-pull-secret --check` verifies it against the drop and\n"
            f"# against the default recipe in {base}.\n"
            f"# VARIANT {PULL_SECRET_SUBDIR}: identical to that default recipe except these\n"
            f"# header lines and `image_pull_secrets: [{PULL_SECRET}]` on {', '.join(SERVER_IMAGE_OPERATORS)}\n"
            "# (the operators running the private server image). Use it only if the\n"
            "# prepull-images.yaml canary hit ErrImagePull; the Secret must exist in the run\n"
            "# namespace (README.md).\n")


def ps_launch_header(v: dict) -> str:
    return ("# GENERATED by ../../../generate_point_recipes.py --image-pull-secret -- do not hand-edit.\n"
            f"# VARIANT {PULL_SECRET_SUBDIR}: ../../../{v['TOTAL_NODES']}nodes/point_{v['CONCURRENCY']}/launch_test.sh plus a\n"
            f"# check that Secret {PULL_SECRET} exists in NAMESPACE (SKIP_SECRET_CHECK=1 skips it).\n")


def ps_op_anchor(op: str) -> str:
    kind = "k8s" if op == "server_cpu" else "k8s_mpi"
    return (f"- name: {op}\n  type: {kind}\n  image: ${{{{ variables.SERVER_IMAGE }}}}\n"
            "  image_pull_policy: IfNotPresent\n")


def replace_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        die(f"internal: {what} anchor occurs {text.count(old)} times, want exactly 1")
    return text.replace(old, new)


def render_recipe_ps(drop: Drop, c: int) -> str:
    out = replace_once(render_recipe(drop, c), PS_RECIPE_HEADER_OLD, ps_recipe_header(drop.points[c].vars),
                       "variant recipe header")
    for op in SERVER_IMAGE_OPERATORS:
        out = replace_once(out, ps_op_anchor(op), ps_op_anchor(op) + PS_OP_LINES, f"variant operator {op}")
    return out


def render_launch_ps(drop: Drop, c: int) -> str:
    out = replace_once(render_launch(drop, c), PS_LAUNCH_HEADER_OLD, ps_launch_header(drop.points[c].vars),
                       "variant launch header")
    out = replace_once(out, PS_LAUNCH_README_OLD, PS_LAUNCH_README_NEW, "variant launch README path")
    return replace_once(out, PS_LAUNCH_RUN, "\n" + PS_LAUNCH_GATE + PS_LAUNCH_RUN[1:], "variant launch gate")


def line_delta(old: str, new: str) -> tuple:
    """(added, removed) lines between two texts (unified diff, no context)."""
    add, rem = [], []
    for ln in list(difflib.unified_diff(old.splitlines(), new.splitlines(), n=0, lineterm=""))[2:]:
        if not ln.startswith("@@"):
            (add if ln[:1] == "+" else rem).append(ln[1:])
    return add, rem


def verify_ps(drop: Drop, c: int, text: str, launch: str, label: str) -> list:
    """The variant differs from the default render only by the declared lines."""
    P = []
    v = drop.points[c].vars
    base = render_recipe(drop, c)
    add, rem = line_delta(base, text)
    want_add = ps_recipe_header(v).splitlines() + PS_OP_LINES.splitlines() * len(SERVER_IMAGE_OPERATORS)
    want_rem = PS_RECIPE_HEADER_OLD.splitlines()
    if sorted(add) != sorted(want_add) or sorted(rem) != sorted(want_rem):
        P.append(f"{label}: recipe delta vs the default render is +{add} -{rem}")
    doc, bdoc = yaml.safe_load(text), yaml.safe_load(base)
    for o in (doc or {}).get("operators") or []:
        o.pop("image_pull_secrets", None)
    if doc != bdoc:
        P.append(f"{label}: parsed recipe differs from the default render beyond image_pull_secrets")
    server_ops = [o["name"] for o in bdoc["operators"] if o.get("image") == "${{ variables.SERVER_IMAGE }}"]
    if server_ops != list(SERVER_IMAGE_OPERATORS):
        P.append(f"{label}: operators running SERVER_IMAGE are {server_ops}, the variant covers "
                 f"{list(SERVER_IMAGE_OPERATORS)}")
    add, rem = line_delta(render_launch(drop, c), launch)
    want_add = ps_launch_header(v).splitlines() + PS_LAUNCH_README_NEW.splitlines() + PS_LAUNCH_GATE.splitlines()
    want_rem = PS_LAUNCH_HEADER_OLD.splitlines() + PS_LAUNCH_README_OLD.splitlines()
    if sorted(add) != sorted(want_add) or sorted(rem) != sorted(want_rem):
        P.append(f"{label}: launch_test.sh delta vs the default is +{add} -{rem}")
    return P


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------
def verify_recipe(drop: Drop, c: int, text: str, label: str, pull_secret: bool = False) -> list:
    """Structural checks of one generated recipe against the drop. Returns problems."""
    P = []
    pt = drop.points[c]
    v = pt.vars
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        return [f"{label}: not a YAML mapping"]
    # variables: drop point vars verbatim, plus the disagg/slurm_env/GKE ones
    vars_ = {k: (d or {}).get("value") for k, d in (doc.get("variables") or {}).items()}
    for k in pt.var_keys:
        if vars_.get(k) != v[k]:
            P.append(f"{label}: variable {k}={vars_.get(k)!r} != drop {v[k]!r}")
    if (f"\nvariables:\n  # ---- VERBATIM: point_{c}.yaml variables ----\n" + pt.vars_raw.rstrip("\n") + "\n"
            + "  # ---- VERBATIM: disagg.yaml variables ----\n") not in text:
        P.append(f"{label}: point variables block is not byte-identical to the drop")
    want_extra = {
        "DYNAMO_DECODE_METRICS_FLAG": "--no-publish-metrics",
        "DYNAMO_FRONTEND_CORE_SHA256": DYNAMO_FRONTEND_CORE_SHA256,
        "BENCHMARK_MODE": "perf", "MODE": "disagg",
        "MODEL_PATH": MODEL_PATH, "DRAFT_PATH": DRAFT_PATH,
        "DATASET_PATH": DATASET_PATH, "CONTROL_PLANE_DIR": CONTROL_PLANE_DIR,
        "SERVER_IMAGE": SERVER_IMAGE, "CLIENT_IMAGE": CLIENT_IMAGE,
        "K8S_NAMESPACE": "default", "GKE_NODEPOOL": "<nodepool>",
    }
    for k, want in want_extra.items():
        if vars_.get(k) != want:
            P.append(f"{label}: variable {k}={vars_.get(k)!r} != {want!r}")
    if set(vars_) != set(pt.var_keys) | set(want_extra):
        P.append(f"{label}: unexpected variables {sorted(set(vars_) - set(pt.var_keys) - set(want_extra))}")
    for img in (SERVER_IMAGE, CLIENT_IMAGE):
        if "@sha256:" not in img:
            P.append(f"{label}: image {img} not pinned by digest")
    # artifacts
    arts = {a["name"]: a for a in doc.get("artifacts") or []}
    for name in VERBATIM_ARTIFACTS:
        got = (arts.get(name) or {}).get("content")
        if got != pt.content[name]:
            P.append(f"{label}: artifact {name} content differs from the drop")
        if arts.get(name, {}).get("uri") != dict(DROP_ARTIFACTS)[name]:
            P.append(f"{label}: artifact {name} uri changed")
        if pt.raw_art[name] not in text:
            P.append(f"{label}: artifact {name} raw block is not byte-identical to the drop")
    got_cl = (arts.get("CLIENT_CONFIG") or {}).get("content")
    if got_cl is None:
        P.append(f"{label}: CLIENT_CONFIG missing")
    else:
        diff = [d for d in difflib.ndiff(pt.perf_client_content.splitlines(), got_cl.splitlines())
                if d[:1] in "+-"]
        want_diff = ["- report_dir: /results/report", "+ report_dir: ${REPORT_DIR}",
                     f"+   metrics_tokenizer_workers: {METRICS_TOKENIZER_WORKERS}"]
        if sorted(diff) != sorted(want_diff):
            P.append(f"{label}: CLIENT_CONFIG delta vs drop is {diff}, allowed {want_diff}")
        cfg = yaml.safe_load(got_cl)
        if (cfg.get("report_dir") != "${REPORT_DIR}"
                or (cfg.get("settings") or {}).get("metrics_tokenizer_workers") != METRICS_TOKENIZER_WORKERS
                or "seed" in (cfg.get("model_params") or {})):
            P.append(f"{label}: CLIENT_CONFIG rewrites not effective")
        drop_cfg = yaml.safe_load(pt.perf_client_content)
        drop_cfg["report_dir"] = "${REPORT_DIR}"
        drop_cfg["settings"] = {"metrics_tokenizer_workers": METRICS_TOKENIZER_WORKERS, **drop_cfg["settings"]}
        if cfg != drop_cfg:
            P.append(f"{label}: CLIENT_CONFIG parsed form differs beyond the two rewrites")
    for name, uri, _ in HELPERS:
        if (arts.get(name) or {}).get("content") != drop.helpers[name] or arts.get(name, {}).get("uri") != uri:
            P.append(f"{label}: helper artifact {name} differs from the drop file")
    expected_arts = ([n for n, _ in DROP_ARTIFACTS] + [n for n, _, _ in HELPERS]
                     + ["MODEL", "DRAFT", "DATASET_DIR", "CONTROL_PLANE_DIR", "REPORTS_DIR", "IMEX_CHANNELS"])
    if list(arts) != expected_arts:
        P.append(f"{label}: artifacts {list(arts)} != {expected_arts}")
    want_fs = {"MODEL": "fs://${{ variables.MODEL_PATH }}", "DRAFT": "fs://${{ variables.DRAFT_PATH }}",
               "DATASET_DIR": f"fs://{DATASET_DIR}",
               "CONTROL_PLANE_DIR": "fs://${{ variables.CONTROL_PLANE_DIR }}",
               "REPORTS_DIR": f"fs://{REPORTS_DIR}",
               "IMEX_CHANNELS": "fs:///dev/nvidia-caps-imex-channels"}
    for k, uri in want_fs.items():
        if arts.get(k, {}).get("uri") != uri:
            P.append(f"{label}: artifact {k} uri {arts.get(k, {}).get('uri')!r} != {uri!r}")
    if not DATASET_PATH.startswith(DATASET_DIR + "/"):
        P.append(f"{label}: DATASET_PATH not under DATASET_DIR")
    # backend / operators
    be = (doc.get("backends") or [{}])[0]
    for k, want in (("name", "cluster"), ("type", "kubernetes"), ("nodes", "${{ variables.TOTAL_NODES }}"),
                    ("gpus_per_node", GPUS_PER_NODE), ("host_network", True), ("host_ipc", True),
                    ("rdma", "disable"), ("compute_domain", {"create": True}),
                    ("scheduling", "device_plugin")):
        if be.get(k) != want:
            P.append(f"{label}: backend {k}={be.get(k)!r} != {want!r}")
    ops = {o["name"]: o for o in doc.get("operators") or []}
    if list(ops) != ["server_ctx", "server_gen", "server_cpu", "client"]:
        P.append(f"{label}: operators {list(ops)}")
    for name, o in ops.items():
        if o.get("image_pull_policy") != "IfNotPresent":
            P.append(f"{label}: operator {name} image_pull_policy != IfNotPresent")
        want_ps = [PULL_SECRET] if (pull_secret and name in SERVER_IMAGE_OPERATORS) else None
        if o.get("image_pull_secrets") != want_ps:
            P.append(f"{label}: operator {name} image_pull_secrets={o.get('image_pull_secrets')!r} != {want_ps!r}")
    for name, gvar in (("server_ctx", "CTX_GPUS"), ("server_gen", "GEN_GPUS")):
        o = ops.get(name, {})
        if o.get("type") != "k8s_mpi" or (o.get("security_context") or {}).get("privileged") is not True:
            P.append(f"{label}: {name} must be a privileged k8s_mpi operator")
        if (o.get("mpi") or {}).get("slots_per_worker") != ntasks_per_node_expr(gvar):
            P.append(f"{label}: {name} slots_per_worker")
        if (o.get("mpi") or {}).get("ensure_sshd") is not True:
            P.append(f"{label}: {name} ensure_sshd")
    if "security_context" in ops.get("server_cpu", {}) or ops.get("server_cpu", {}).get("type") != "k8s":
        P.append(f"{label}: server_cpu must be a non-privileged k8s operator")
    # workflow
    wf = doc.get("workflow") or {}
    if wf.get("name") != f"kimi_k3_trtllm_v1_0_3_gke_c{c}":
        P.append(f"{label}: workflow.name {wf.get('name')!r} != 'kimi_k3_trtllm_v1_0_3_gke_c{c}'")
    if wf.get("variables") != drop.setup["workflow"]["variables"]:
        P.append(f"{label}: workflow variables differ from setup.yaml")
    if ("  variables:\n" + drop.wf_vars_raw) not in text:
        P.append(f"{label}: workflow variables block is not byte-identical to setup.yaml")
    tasks = {t["name"]: t for t in wf.get("tasks") or []}
    pl = plans(drop, c)
    if list(tasks) != list(pl):
        P.append(f"{label}: tasks {list(tasks)} != {list(pl)}")
        return P
    ctasks = drop.composed_tasks(c)
    for name, plan in pl.items():
        got = tasks[name].get("script")
        if got != plan_items(plan):
            P.append(f"{label}: {name} script differs from its plan")
            continue
        src = ctasks[name]["script"]
        kept = [p[1] for p in plan if p[0] == "D"]
        rewritten = {p[1]: p[2] for p in plan if p[0] == "R"}
        removed = {p[1] for p in plan if p[0] == "X"}
        inserted = [p[1] for p in plan if p[0] == "G"]
        expect_order = [rewritten.get(x, x) for x in src if x not in removed]
        it = iter(got)
        if not all(any(g == e for g in it) for e in expect_order):
            P.append(f"{label}: {name} drop items out of order or missing")
        if [x for x in src if x not in rewritten and x not in removed] != kept:
            P.append(f"{label}: {name} verbatim drop items changed")
        extra = [g for g in got if g not in expect_order]
        if extra != [g for g in inserted if g not in expect_order]:
            P.append(f"{label}: {name} has undeclared items")
        for old, new in rewritten.items():
            if old in got:
                P.append(f"{label}: {name} still contains rewritten drop item {old!r}")
    for role in ("prefill", "decode"):
        s = tasks[role]["script"]
        if G_FABRIC_PROBE not in s:
            P.append(f"{label}: {role} lacks the drop fabric check")
        src_launch = ctasks[role]["script"][-1]
        launch = s[-1]
        if launch != gke_launch(role, src_launch):
            P.append(f"{label}: {role} launch line is not mpirun + the drop's command")
        joined = "\n".join(s)
        if "sys.exit(\"GKE_PATCH FATAL" not in joined:
            P.append(f"{label}: {role} hang patch does not fail closed")
        if re.search(r"(?m)^\s*export SLURM_", joined):
            P.append(f"{label}: {role} exports SLURM_* (would break trtllm-llmapi-launch rank detection)")
        if tasks[role].get("depends_on") != ctasks[role]["depends_on"]:
            P.append(f"{label}: {role} depends_on changed")
        if tasks[role].get("probes") != ctasks[role]["probes"]:
            P.append(f"{label}: {role} probes changed")
        if tasks[role].get("resources") != ctasks[role]["resources"]:
            P.append(f"{label}: {role} resources changed")
        if tasks[role].get("replicas") != ctasks[role]["replicas"]:
            P.append(f"{label}: {role} replicas changed")
    for name in ("client_gate", "control_plane", "frontend", "benchmark") + (
            ("client_preflight",) if "client_preflight" in ctasks else ()):
        if tasks[name].get("resources") != ctasks[name]["resources"]:
            P.append(f"{label}: {name} resources changed")
        if tasks[name].get("depends_on") != ctasks[name]["depends_on"]:
            P.append(f"{label}: {name} depends_on changed")
    for name in ("control_plane", "frontend"):
        if tasks[name].get("probes") != ctasks[name]["probes"]:
            P.append(f"{label}: {name} probes changed")
    if tasks["preflight"].get("resources") != ctasks["preflight"]["resources"]:
        P.append(f"{label}: preflight resources changed")
    if any(x.startswith("exec ") for x in tasks["benchmark"]["script"]):
        P.append(f"{label}: benchmark uses exec (kills sflow's collect trap)")
    # topology comments match the placement math
    for ln in topo_text(v):
        if ln not in text:
            P.append(f"{label}: topology comment missing: {ln}")
    return P


def check(drop: Drop, dest: Path, points: list, pull_secret: bool = False) -> int:
    problems = []
    root = dest / PULL_SECRET_SUBDIR if pull_secret else dest
    renders = (((recipe_name, render_recipe_ps), (lambda c: "launch_test.sh", render_launch_ps)) if pull_secret
               else ((recipe_name, render_recipe), (lambda c: "launch_test.sh", render_launch)))
    tag = f" [{PULL_SECRET_SUBDIR}]" if pull_secret else ""
    if pull_secret and not root.is_dir():
        problems.append(f"variant tree {root} missing (run --image-pull-secret)")
    for c in points:
        v = drop.points[c].vars
        d = dest_dir(root, v)
        for fname_of, render in renders:
            p = d / fname_of(c)
            want = render(drop, c)
            if not p.is_file():
                problems.append(f"C{c}{tag}: missing {p}")
                continue
            got = p.read_text(encoding="utf-8")
            if got != want:
                diff = list(difflib.unified_diff(want.splitlines(), got.splitlines(), "generated", str(p), n=0,
                                                 lineterm=""))[:12]
                problems.append(f"C{c}{tag}: {p} differs from a fresh render (hand edit or stale):\n    "
                                + "\n    ".join(diff))
        recipe = d / recipe_name(c)
        lt = d / "launch_test.sh"
        if recipe.is_file():
            problems += verify_recipe(drop, c, recipe.read_text(encoding="utf-8"), f"C{c}{tag}", pull_secret)
        if pull_secret and recipe.is_file() and lt.is_file():
            problems += verify_ps(drop, c, recipe.read_text(encoding="utf-8"), lt.read_text(encoding="utf-8"),
                                  f"C{c}{tag}")
        if lt.is_file():
            txt = lt.read_text(encoding="utf-8")
            if f'REQUIRED_NODES="${{REQUIRED_NODES:-{v["TOTAL_NODES"]}}}"' not in txt:
                problems.append(f"C{c}{tag}: launch_test.sh REQUIRED_NODES != {v['TOTAL_NODES']}")
            if f'RECIPE="{recipe_name(c)}"' not in txt:
                problems.append(f"C{c}{tag}: launch_test.sh RECIPE mismatch")
            if not (lt.stat().st_mode & 0o111):
                problems.append(f"C{c}{tag}: launch_test.sh not executable")
            if pull_secret != (f"kubectl get secret {PULL_SECRET} " in txt):
                problems.append(f"C{c}{tag}: launch_test.sh Secret check {'missing' if pull_secret else 'present'}")
        if d.parent.name != f"{v['TOTAL_NODES']}nodes":
            problems.append(f"C{c}{tag}: directory {d.parent.name} != {v['TOTAL_NODES']}nodes")
    # no stray point dirs
    for nd in sorted(root.glob("*nodes")):
        for pd in sorted(nd.glob("point_*")):
            m = re.fullmatch(r"point_(\d+)", pd.name)
            if not m or int(m.group(1)) not in EXPECTED or dest_dir(root, drop.points[int(m.group(1))].vars) != pd:
                problems.append(f"stray output directory {pd}")
    if problems:
        print("CHECK FAILED:")
        for p in problems:
            print("  - " + p)
        return 1
    print(f"CHECK OK{tag}: {len(points)} point(s); drop {drop.root}")
    for c in points:
        v = drop.points[c].vars
        pl = placement(v)
        print(f"  C{c:<5} {v['TOTAL_NODES']:>2} nodes  ctx {v['NUM_CTX']}x{v['CTX_GPUS']} "
              f"gen {v['NUM_GEN']}x{v['GEN_GPUS']}  gen nodes {[g[0] for g in pl['gen']]}  "
              f"frontend n{v['FRONTEND_NODE']} client n{v['CLIENT_NODE']}  traj {v['NUM_TRAJECTORIES']}")
    return 0


def _resolve_script_exprs(body: str, v: dict) -> str:
    body = re.sub(
        r"\$\{\{\s*'[^']*'\s+if\s+\(variables\.NUMA_BIND\s*\|\s*int\)\s+else\s+'true'\s*\}\}",
        "true",
        body,
    )
    body = re.sub(r"\$\{\{\s*variables\.(\w+)\s*\}\}", lambda m: str(v.get(m.group(1), "0")), body)
    return body


def dump_scripts(drop: Drop, out: Path, points: list) -> None:
    for c in points:
        v = {**drop.points[c].vars, "DYNAMO_DECODE_METRICS_FLAG": "--no-publish-metrics"}
        doc = yaml.safe_load(render_recipe(drop, c))
        for t in doc["workflow"]["tasks"]:
            body = _resolve_script_exprs("\n".join(t["script"]) + "\n", v)
            if "${{" in body:
                die(f"C{c} {t['name']}: unresolved sflow expression in script")
            d = out / f"point_{c}"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{t['name']}.sh").write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    print(f"dumped task scripts for {len(points)} point(s) under {out}")


# --------------------------------------------------------------------------------------
# Opt-in SWE-bench (--swe acc|both): same-config v1.0.3 server recipe + v1.0.3 drop SWE
# client/benchmark wiring, emitted into a parallel tree <root>/swe_<mode>/<N>nodes/point_<C>/.
# --------------------------------------------------------------------------------------
SWE_DROP_REL = "nv-mlpinf-partner-partner-endpoints-v1.0.3/NVIDIA/src/kimi-k3"
SWE_SERVICE_SHA256 = "edd3d17efe377074965f042b4678663a552fa9c1b9e14154ff09b4ee3f71aa1f"
SWE_MODES = ("acc", "both")
SWE_HANDOFF_DIR = f"{NODE_SSD}/kimi-swe/handoff"
SWE_HANDOFF_ARTIFACT = "SWE_HANDOFF_DIR"
SWE_CONNECTION_FILE = "swe-connection.json"
SWE_WAIT_S = 2700
SWE_URL_PREFIX = "http://127.0.0.1:"
SWE_MODE_MARKER = "eval_method: swe_bench_scorer"
DROP_SWE_PREFIX = 'case "${BENCHMARK_MODE}" in perf|acc|both)'

SWE_CONTRACT = {
    "handoff_dir": SWE_HANDOFF_DIR,
    "handoff_artifact": SWE_HANDOFF_ARTIFACT,
    "connection_file": SWE_CONNECTION_FILE,
    "wait_s": SWE_WAIT_S,
    "reports_dir": REPORTS_DIR,
    "client_node": {c: v[8] for c, v in EXPECTED.items()},
    "frontend_node": {c: v[7] for c, v in EXPECTED.items()},
    "benchmark_task": "benchmark",
    "client_task": "benchmark",
    "frontend_task": "frontend",
    "modes": SWE_MODES,
    "mode_marker": SWE_MODE_MARKER,
    "url_prefix": SWE_URL_PREFIX,
    "workflow_name": "kimi_k3_trtllm_v1_0_3_gke_c{c}_swe_{mode}",
    "recipe_path": "swe_{mode}/{n}nodes/point_{c}/kimi_k3_gke_disagg_point{c}_swe_{mode}.yaml",
    "launch_path": "swe_{mode}/{n}nodes/point_{c}/launch_test.sh",
    "pull_secret_subdir": PULL_SECRET_SUBDIR,
}

SWE_DEVIATIONS = """\
S1  BENCHMARK_MODE set to the requested SWE mode (acc or both; v1.0.3 slurm_env.yaml
    pins perf, and point_{12,16,32,1024}.yaml pin both). The benchmark script refuses
    any other mode at start before the import/asset checks.
S2  SWE_WAIT_S=2700 (integer variable): seconds the benchmark waits after frontend
    readiness for this run's SWE service connection file.
S3  CLIENT_CONFIG: the `swe_bench` accuracy dataset block from v1.0.3
    point_{12,16,32,1024}.yaml spliced verbatim before `settings:`.
S4  Artifact SWE_HANDOFF_DIR (`fs://<node-local-ssd>/kimi-swe/handoff`)
    carries the per-run `<run_id>/swe-connection.json` written by `kimi-swe-host`.
    No SWE_SERVICE_URL, SWE_SERVICE_TOKEN, SWEBENCH_SERVICE_URL, or
    SWEBENCH_SERVICE_AUTH_TOKEN workflow/top-level variables exist.
S5  Benchmark script: SWE_WAIT_BLOCK (waits for <SWE_HANDOFF_DIR>/<run_id>/swe-connection.json
    and GET http://127.0.0.1:<port>/health == 200) + the SWE handoff block +
    `--mode "${BENCHMARK_MODE}"` (replacing `--mode perf`, without `exec`).
S6  Benchmark script: SWE_REDACT_BLOCK after the client command and before the report
    mirror, redacting the run's SWE service token to `<redacted>` in all `*.yaml`
    under SFLOW_WORKFLOW_OUTPUT_DIR and REPORT_DIR (port of swe_service.py `finally:`).
"""

SWE_WAIT_BLOCK = r"""# GKE port of the drop's `depends_on: [swe_service]` and its swe-connection.json file.
# On Slurm, swe_service.py writes {"url", "token"} into SFLOW_WORKFLOW_OUTPUT_DIR/swe-connection.json
# and then prints SWE_READY. On GKE the service runs in the kimi-swe-host pod on this node,
# started for this run by swe/run_swe_sweep.sh, and writes that file under the SWE_HANDOFF_DIR
# hostPath, keyed by this run's id. Wait for it the way swe_service.py itself waits before
# SWE_READY (url set, GET /health == 200; the token is not read here), then the SWE block
# below reads it.
if [ "${BENCHMARK_MODE}" != perf ]; then
  export SWE_CONNECTION="${{ artifacts.SWE_HANDOFF_DIR.path }}/$(basename "${SFLOW_WORKFLOW_OUTPUT_DIR}")/swe-connection.json"
  echo "waiting up to ${{ variables.SWE_WAIT_S }} s for this run's SWE service: ${SWE_CONNECTION}"
  SWE_DEADLINE=$(( $(date +%s) + ${{ variables.SWE_WAIT_S }} ))
  until python3 -c 'import json, os, sys, urllib.request
url = json.load(open(os.environ["SWE_CONNECTION"])).get("url", "")
if not url.startswith("http://127.0.0.1:"):
    sys.exit(1)
with urllib.request.urlopen(url + "/health", timeout=3) as response:
    sys.exit(0 if response.status == 200 else 1)' 2>/dev/null; do
    if [ "$(date +%s)" -ge "${SWE_DEADLINE}" ]; then
      echo "FATAL: no SWE service for this run after ${{ variables.SWE_WAIT_S }} s." >&2
      echo "  acc/both on GKE must be launched with swe/run_swe_sweep.sh, which runs the" >&2
      echo "  drop's swe_service (kimi-swe-host pod) on the client node for each run. See swe/README.md." >&2
      exit 1
    fi
    sleep 10
  done
  echo "SWE service is ready"
fi
"""

DROP_SWE_PREFIX = 'case "${BENCHMARK_MODE}" in perf|acc|both) ;; *) echo "Invalid BENCHMARK_MODE=${BENCHMARK_MODE}" >&2; exit 2;; esac\n'
SWE_HANDOFF_BLOCK = """\
case "${BENCHMARK_MODE}" in perf|acc|both) ;; *) echo "Invalid BENCHMARK_MODE=${BENCHMARK_MODE}" >&2; exit 2;; esac
if [ "${BENCHMARK_MODE}" = perf ]; then
  export SWEBENCH_SERVICE_URL=http://127.0.0.1:1 SWEBENCH_SERVICE_AUTH_TOKEN=disabled
  export SWE_SERVICE_URL=http://127.0.0.1:1 SWE_SERVICE_TOKEN=disabled
else
  eval "$(python3 -c 'import json, os, shlex; d = json.load(open(os.environ["SWE_CONNECTION"])); u, t = shlex.quote(d["url"]), shlex.quote(d["token"]); print(f"export SWEBENCH_SERVICE_URL={u} SWEBENCH_SERVICE_AUTH_TOKEN={t} SWE_SERVICE_URL={u} SWE_SERVICE_TOKEN={t}")')"
  export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
fi
"""

SWE_REDACT_BLOCK = r"""# GKE port of the drop's end-of-run token redaction (swe_service.py, `finally:`): in
# every *.yaml under the run's output roots, this run's SWE service token becomes
# "<redacted>". Same rule as the drop. The pinned client already writes "<redacted>" for
# *_token fields in report/config.yaml, so this normally changes no file and only confirms
# that no YAML holds the token. It runs here because on GKE sflow collects report/ before
# the service pod stops, and that pod cannot write the root-owned REPORT_DIR. Roots:
# SFLOW_WORKFLOW_OUTPUT_DIR, as in the drop, and REPORT_DIR in place of the drop's RESULTS_PATH.
# Prints file names, never the token. Runs after `KIMI_BENCH_RC=$?; set -e` and before the
# report mirror so any partial YAML written before a client failure is also redacted.
if [ "${BENCHMARK_MODE}" != perf ]; then
  python3 -c 'import os, pathlib, sys
token = os.environ.get("SWEBENCH_SERVICE_AUTH_TOKEN") or os.environ.get("SWE_SERVICE_TOKEN", "")
if token in ("", "disabled"):
    sys.exit("FATAL: SWE_SERVICE_TOKEN is not set; cannot redact the report")
roots = [pathlib.Path(os.environ["SFLOW_WORKFLOW_OUTPUT_DIR"]), pathlib.Path(os.environ["REPORT_DIR"])]
redacted = []
for root in roots:
    for path in root.rglob("*.yaml"):
        content = path.read_text()
        if token in content:
            path.write_text(content.replace(token, "<redacted>"))
            redacted.append(str(path))
left = [str(p) for root in roots for p in root.rglob("*.yaml") if token in p.read_text()]
print("SWE token redacted in %d file(s): %s" % (len(redacted), " ".join(redacted) or "none"))
if left:
    sys.exit("FATAL: SWE token still present in: " + " ".join(left))'
fi
"""


class SweDrop:
    """Loads and verifies the v1.0.3 drop files carrying the SWE-bench wiring."""

    def __init__(self, root: Path):
        self.root = root
        if not root.is_dir():
            die(f"--swe-drop-root {root} is not a directory")
        self.service_path = root / "scripts" / "swe_service.py"
        if not self.service_path.is_file():
            die(f"missing v1.0.3 SWE file: {self.service_path}")
        got = hashlib.sha256(self.service_path.read_bytes()).hexdigest()
        if got != SWE_SERVICE_SHA256:
            die(f"{self.service_path}: sha256 {got} != expected v1.0.3 pin {SWE_SERVICE_SHA256}")
        self.service_raw = read_text(self.service_path)
        for needle in ('REVISION = "d3c57f2eba127c819e5a8fd83101d9e5f4e15fce"',
                       'json.dump({"url": url, "token": token}, f)',
                       'print("SWE_READY", flush=True)'):
            if needle not in self.service_raw:
                die(f"v1.0.3 scripts/swe_service.py: expected {needle!r} missing")
        p12 = root / POINTS_SUBDIR / "point_12.yaml"
        if not p12.is_file():
            die(f"missing {p12}")
        p12_raw = read_text(p12)
        if ("\n" + DROP_SWE_DATASET_RAW) not in p12_raw or DROP_SWE_EVAL_LINE not in p12_raw:
            die("v1.0.3 point_12.yaml: SWE dataset or eval line changed")
        self.swe_dataset_content = DROP_SWE_DATASET_CONTENT
        self.swe_dataset_raw = DROP_SWE_DATASET_RAW
        self.bench_swe = SWE_HANDOFF_BLOCK
        self.bench_run = ('exec /opt/venv/bin/python3 -m inference_endpoint.main benchmark from-config '
                          '-c "${CLIENT_CONFIG}" --mode "${BENCHMARK_MODE}"')


def swe_mode_lock(mode: str) -> str:
    if mode not in SWE_MODES:
        die(f"internal: unknown SWE mode {mode!r}")
    return (
        f'if [ "${{BENCHMARK_MODE}}" != "{mode}" ]; then\n'
        f'  echo "FATAL: this recipe is generated for --swe {mode} '
        f'(BENCHMARK_MODE={mode}, got ${{BENCHMARK_MODE}})" >&2\n'
        "  exit 2\n"
        "fi\n"
    )


def swe_benchmark_plan(drop: Drop, swe_drop: SweDrop, c: int, mode: str) -> list:
    script = drop.composed_tasks(c, swe=False)["benchmark"]["script"]
    hf = ("export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
          "PYTHONDONTWRITEBYTECODE=1")
    if script.count(hf) != 1:
        die(f"C{c} benchmark: {hf!r} not found")
    swe_run = swe_drop.bench_run[len("exec "):]
    plan = []
    for it in script:
        if it == hf:
            plan.append(("D", it))
            plan += [
                ("G", swe_mode_lock(mode), f"S1 lock BENCHMARK_MODE={mode} for --swe {mode}"),
                ("G", G_TOOLS, "D4 helper artifacts: SFLOW_TOOLS / RECIPE"),
                ("G", G_CLIENT_MOUNTS, "D5/D6/D13 client container mount symlinks + REPORT_DIR"),
                ("G", G_BENCH_IMPORT, "D13 client import check"),
                ("G", G_BENCH_ASSETS, "D6/D13 dataset sha256 + tokenizer present"),
                ("G", G_BENCH_REPORT, "D13 durable REPORT_DIR, printed first"),
            ]
        elif it == "set +e":
            plan += [
                ("G", SWE_WAIT_BLOCK, "S5 wait for this run's SWE service connection file (SWE_HANDOFF_DIR)"),
                ("G", swe_drop.bench_swe, "S5 v1.0.3 SWE connection handoff block"),
                ("D", it),
            ]
        elif it == PERF_RUN_CMD:
            plan.append(("R", it, swe_run, 'D13/S5 --mode "${BENCHMARK_MODE}"'))
        elif it == ALT_PERF_RUN_CMD:
            plan += [
                ("G", SWE_WAIT_BLOCK, "S5 wait for this run's SWE service connection file (SWE_HANDOFF_DIR)"),
                ("G", swe_drop.bench_swe, "S5 v1.0.3 SWE connection handoff block"),
                ("G", "set +e", "D13 keep the exit code; mirror the report either way"),
                ("R", it, swe_run, 'D13/S5 canonical venv python3 -m inference_endpoint.main + --mode "${BENCHMARK_MODE}"'),
                ("G", "KIMI_BENCH_RC=$?; set -e", "D13 benchmark exit code"),
                ("G", SWE_REDACT_BLOCK, "S6 v1.0.3 drop swe_service.py finally token redaction"),
                ("G", G_BENCH_MIRROR, "D13 small-file report mirror"),
                ("G", G_BENCH_EXIT, "D13 propagate the benchmark exit code"),
            ]
        elif it == "CLIENT_RC=$?":
            plan.append(("R", it, "KIMI_BENCH_RC=$?; set -e", "D13 benchmark exit code"))
            plan.append(("G", SWE_REDACT_BLOCK, "S6 v1.0.3 drop swe_service.py finally token redaction"))
            plan.append(("G", G_BENCH_MIRROR, "D13 small-file report mirror"))
        elif it == 'printf "%s\\n" "$CLIENT_RC" > "${SFLOW_WORKFLOW_OUTPUT_DIR}/benchmark-client-exit-code"':
            plan.append(("X", it, "D13 folded into post-mirror exit line"))
        elif it == 'exit "$CLIENT_RC"':
            plan.append(("R", it, G_BENCH_EXIT, "D13 propagate the benchmark exit code"))
        else:
            plan.append(("D", it))
    return plan


def swe_tree_root(dest: Path, mode: str, pull_secret: bool = False) -> Path:
    if mode not in SWE_MODES:
        die(f"unknown SWE mode {mode!r}; known: {SWE_MODES}")
    base = dest / PULL_SECRET_SUBDIR if pull_secret else dest
    return base / f"swe_{mode}"


def recipe_name_swe(c: int, mode: str) -> str:
    return f"kimi_k3_gke_disagg_point{c}_swe_{mode}.yaml"


def swe_recipe_header(v: dict, mode: str, pull_secret: bool = False) -> str:
    gen_rel = "../../../../generate_point_recipes.py" if pull_secret else "../../../generate_point_recipes.py"
    ps_flag = " --image-pull-secret" if pull_secret else ""
    perf_rel = f"../../../{v['TOTAL_NODES']}nodes/point_{v['CONCURRENCY']}/{recipe_name(v['CONCURRENCY'])}"
    lines = [
        f"# GENERATED by {gen_rel}{ps_flag} --swe {mode} -- do not hand-edit;",
        f"# `{ps_flag.lstrip()} --swe {mode} --check` verifies this file against the v1.0.3 drop".replace("` --", "`--"),
        f"# and against the same-config perf recipe {perf_rel}.",
        f"# SWE-BENCH VARIANT (swe_{mode}): server_ctx/server_gen/server_cpu, server YAMLs, env,",
        f"# images, and topology are byte-identical to {perf_rel}; only the",
        "# benchmark/client SWE wiring (S1-S6 below) is added from the v1.0.3 drop.",
    ]
    if pull_secret:
        lines += [
            f"# VARIANT {PULL_SECRET_SUBDIR}: also carries `image_pull_secrets: [{PULL_SECRET}]` on",
            f"# {', '.join(SERVER_IMAGE_OPERATORS)} (see README.md).",
        ]
    return "\n".join(lines) + "\n"


def swe_deviations_comment() -> str:
    lines = ["# SWE-bench (--swe acc|both; v1.0.3 drop SWE wiring, same server config as perf):"]
    for ln in SWE_DEVIATIONS.rstrip("\n").split("\n"):
        lines.append(f"#   {ln}" if ln else "#")
    return "\n".join(lines) + "\n"


SWE_BENCH_VARS_OLD = (
    "  # ---- VERBATIM: slurm_env.yaml ----\n"
    "  BENCHMARK_MODE:\n"
    "    value: perf\n"
)


def swe_bench_vars_new(mode: str) -> str:
    return (
        f"  # ---- GKE EDIT (S1): --swe {mode} (v1.0.3 slurm_env.yaml has perf; accuracy points have both) ----\n"
        "  BENCHMARK_MODE:\n"
        f"    value: {mode}\n"
    )


SWE_NODEPOOL_OLD = (
    "  GKE_NODEPOOL:\n"
    "    value: <nodepool>\n"
)
SWE_NODEPOOL_NEW = (
    "  GKE_NODEPOOL:\n"
    "    value: <nodepool>\n"
    "  # ---- GKE ADD (S2): SWE service handoff timeout (seconds after frontend ready) ----\n"
    "  SWE_WAIT_S:\n"
    "    type: integer\n"
    f"    value: {SWE_WAIT_S}\n"
)

SWE_IMEX_OLD = (
    "- name: IMEX_CHANNELS\n"
    "  uri: fs:///dev/nvidia-caps-imex-channels\n"
)
SWE_IMEX_NEW = (
    "- name: IMEX_CHANNELS\n"
    "  uri: fs:///dev/nvidia-caps-imex-channels\n"
    "# ---- GKE ADD (S4): per-run SWE connection handoff dir (swe/run_swe_sweep.sh) ----\n"
    f"- name: {SWE_HANDOFF_ARTIFACT}\n"
    f"  uri: fs://{SWE_HANDOFF_DIR}\n"
)


def render_recipe_swe(drop: Drop, swe_drop: SweDrop, c: int, mode: str, pull_secret: bool = False) -> str:
    v = drop.points[c].vars
    base = render_recipe_ps(drop, c) if pull_secret else render_recipe(drop, c)
    hdr_old = ps_recipe_header(v) if pull_secret else PS_RECIPE_HEADER_OLD
    out = replace_once(base, hdr_old, swe_recipe_header(v, mode, pull_secret), "SWE recipe header")
    dev_anchor = "#   Unchanged: server/env YAMLs, probes, timeouts, ports, depends_on, Dynamo router/decode flags.\n"
    out = replace_once(out, dev_anchor, dev_anchor + "#\n" + swe_deviations_comment(), "SWE deviations comment")
    out = replace_once(out, SWE_BENCH_VARS_OLD, swe_bench_vars_new(mode), "SWE BENCHMARK_MODE variable")
    out = replace_once(out, SWE_NODEPOOL_OLD, SWE_NODEPOOL_NEW, "SWE_WAIT_S variable")
    cl_hdr_old = f"# ---- VERBATIM: point_{c}.yaml CLIENT_CONFIG except two lines (D13) ----\n"
    cl_hdr_new = f"# ---- VERBATIM: point_{c}.yaml CLIENT_CONFIG except D13 + v1.0.3 SWE block (S3) ----\n"
    out = replace_once(out, cl_hdr_old, cl_hdr_new, "SWE CLIENT_CONFIG comment")
    ds_anchor = f"\n    settings:\n      metrics_tokenizer_workers: {METRICS_TOKENIZER_WORKERS}\n"
    out = replace_once(out, ds_anchor, "\n" + swe_drop.swe_dataset_raw + ds_anchor[1:],
                       "SWE CLIENT_CONFIG swe_bench block")
    out = replace_once(out, SWE_IMEX_OLD, SWE_IMEX_NEW, "SWE_HANDOFF_DIR artifact")
    wf_old = f"\nworkflow:\n  name: kimi_k3_trtllm_v1_0_3_gke_c{c}\n"
    wf_new = f"\nworkflow:\n  name: kimi_k3_trtllm_v1_0_3_gke_c{c}_swe_{mode}\n"
    out = replace_once(out, wf_old, wf_new, "SWE workflow name")
    perf_pl = plans(drop, c, swe=False)
    swe_pl = plans(drop, c, swe=True)
    for cl_task in ("client_gate", "client_preflight"):
        if cl_task in perf_pl:
            p_block = "\n".join(emit_script(perf_pl[cl_task], "    ")) + "\n"
            s_block = "\n".join(emit_script(swe_pl[cl_task], "    ")) + "\n"
            out = replace_once(out, f"  - name: {cl_task}\n" + out.split(f"  - name: {cl_task}\n", 1)[1].split("    script:\n", 1)[0] + "    script:\n" + p_block,
                               f"  - name: {cl_task}\n" + out.split(f"  - name: {cl_task}\n", 1)[1].split("    script:\n", 1)[0] + "    script:\n" + s_block,
                               f"SWE {cl_task} task script")
    perf_bench = "\n".join(emit_script(perf_pl["benchmark"], "    ")) + "\n"
    swe_bench = "\n".join(emit_script(swe_benchmark_plan(drop, swe_drop, c, mode), "    ")) + "\n"
    out = replace_once(out, f"  # ---- disagg.yaml + point_{c}.yaml benchmark ----\n"
                            "  - name: benchmark\n"
                            "    operator: client\n"
                            "    resources:\n"
                            "      nodes:\n"
                            "        indices:\n"
                            "        - ${{ variables.CLIENT_NODE }}\n"
                            "    depends_on:\n"
                            "    - frontend\n"
                            "    script:\n" + perf_bench,
                       f"  # ---- disagg.yaml + point_{c}.yaml benchmark ----\n"
                       "  - name: benchmark\n"
                       "    operator: client\n"
                       "    resources:\n"
                       "      nodes:\n"
                       "        indices:\n"
                       "        - ${{ variables.CLIENT_NODE }}\n"
                       "    depends_on:\n"
                       "    - frontend\n"
                       "    script:\n" + swe_bench,
                       "SWE benchmark task script")
    return out


def swe_launch_header(v: dict, mode: str, pull_secret: bool = False) -> str:
    gen_rel = "../../../../generate_point_recipes.py" if pull_secret else "../../../generate_point_recipes.py"
    ps_flag = " --image-pull-secret" if pull_secret else ""
    perf_rel = f"../../../{v['TOTAL_NODES']}nodes/point_{v['CONCURRENCY']}/launch_test.sh"
    lines = [
        f"# GENERATED by {gen_rel}{ps_flag} --swe {mode} -- do not hand-edit.",
        f"# SWE-BENCH VARIANT (swe_{mode}): {perf_rel} targeting",
        f"# {recipe_name_swe(v['CONCURRENCY'], mode)}. Launch via swe/run_swe_sweep.sh so the",
        "# per-run kimi-swe-host service pod is started on the client node.",
    ]
    if pull_secret:
        lines.append(f"# VARIANT {PULL_SECRET_SUBDIR}: checks that Secret {PULL_SECRET} exists in NAMESPACE.")
    return "\n".join(lines) + "\n"


def render_launch_swe(drop: Drop, c: int, mode: str, pull_secret: bool = False) -> str:
    v = drop.points[c].vars
    base = render_launch_ps(drop, c) if pull_secret else render_launch(drop, c)
    hdr_old = ps_launch_header(v) if pull_secret else PS_LAUNCH_HEADER_OLD
    out = replace_once(base, hdr_old, swe_launch_header(v, mode, pull_secret), "SWE launch header")
    readme_old = PS_LAUNCH_README_NEW if pull_secret else PS_LAUNCH_README_OLD
    readme_new = ("# Before the first point (../../../../README.md): stage-kimi-k3-assets.yaml,\n" if pull_secret
                  else "# Before the first point (../../../README.md): stage-kimi-k3-assets.yaml,\n")
    out = replace_once(out, readme_old, readme_new, "SWE launch README path")
    if pull_secret:
        out = replace_once(out, 'echo "../../../prepull-images.yaml (and keep it), or set SKIP_SECRET_CHECK=1." >&2\n',
                           'echo "../../../../prepull-images.yaml (and keep it), or set SKIP_SECRET_CHECK=1." >&2\n',
                           "SWE launch prepull path")
    out = replace_once(out, f'RECIPE="{recipe_name(c)}"\n', f'RECIPE="{recipe_name_swe(c, mode)}"\n',
                       "SWE launch RECIPE")
    return out


def _strip_swe_to_perf_doc(doc: dict, swe_drop: SweDrop, c: int, mode: str) -> dict:
    """Return a copy of parsed SWE recipe `doc` with the SWE-only sections reverted to perf."""
    d = copy.deepcopy(doc)
    vdict = d.get("variables") or {}
    if "BENCHMARK_MODE" in vdict and isinstance(vdict["BENCHMARK_MODE"], dict):
        vdict["BENCHMARK_MODE"]["value"] = "perf"
    vdict.pop("SWE_WAIT_S", None)
    arts = d.get("artifacts") or []
    d["artifacts"] = [a for a in arts if a.get("name") != SWE_HANDOFF_ARTIFACT]
    for a in d["artifacts"]:
        if a.get("name") == "CLIENT_CONFIG" and isinstance(a.get("content"), str):
            a["content"] = a["content"].replace("\n" + swe_drop.swe_dataset_content, "\n", 1)
    wf = d.get("workflow") or {}
    if wf.get("name") == f"kimi_k3_trtllm_v1_0_3_gke_c{c}_swe_{mode}":
        wf["name"] = f"kimi_k3_trtllm_v1_0_3_gke_c{c}"
    for t in wf.get("tasks") or []:
        if t.get("name") in ("client_gate", "client_preflight") and isinstance(t.get("script"), list):
            t["script"] = [x for x in t["script"] if x != G_SWE_GATE_PLACEHOLDER]
    return d


def verify_swe(drop: Drop, swe_drop: SweDrop, c: int, mode: str, text: str, launch: str,
               label: str, pull_secret: bool = False) -> list:
    """Structural and byte-delta checks for one generated SWE recipe + launch_test.sh."""
    P = []
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        return [f"{label}: not a YAML mapping"]
    perf_text = render_recipe_ps(drop, c) if pull_secret else render_recipe(drop, c)
    pdoc = yaml.safe_load(perf_text)

    # 1. Variables
    vars_raw = doc.get("variables") or {}
    if (vars_raw.get("BENCHMARK_MODE") or {}).get("value") != mode:
        P.append(f"{label}: variable BENCHMARK_MODE is {(vars_raw.get('BENCHMARK_MODE') or {}).get('value')!r}, "
                 f"want {mode!r}")
    if (vars_raw.get("SWE_WAIT_S") or {}).get("value") != SWE_WAIT_S or \
            (vars_raw.get("SWE_WAIT_S") or {}).get("type") != "integer":
        P.append(f"{label}: variable SWE_WAIT_S is not integer {SWE_WAIT_S}")
    for gone in ("SWE_SERVICE_URL", "SWE_SERVICE_TOKEN", "SWEBENCH_SERVICE_URL", "SWEBENCH_SERVICE_AUTH_TOKEN"):
        if gone in vars_raw:
            P.append(f"{label}: variable {gone} is back (the URL and token come from "
                     "the run's handoff file, as on Slurm)")

    # 2. Artifacts
    arts = {a.get("name"): a for a in (doc.get("artifacts") or []) if isinstance(a, dict)}
    if (arts.get(SWE_HANDOFF_ARTIFACT) or {}).get("uri") != f"fs://{SWE_HANDOFF_DIR}":
        P.append(f"{label}: artifact {SWE_HANDOFF_ARTIFACT} is not fs://{SWE_HANDOFF_DIR}")
    if "SWE_CONNECTION" in arts:
        P.append(f"{label}: a SWE_CONNECTION artifact is back (<workdir> is a read-only "
                 "ConfigMap on Kubernetes; the handoff file replaces it)")
    cl_content = (arts.get("CLIENT_CONFIG") or {}).get("content") or ""
    if SWE_MODE_MARKER not in cl_content:
        P.append(f"{label}: CLIENT_CONFIG lacks {SWE_MODE_MARKER!r}")
    if ("\n" + swe_drop.swe_dataset_content) not in cl_content:
        P.append(f"{label}: CLIENT_CONFIG lacks the v1.0.3 swe_bench dataset block")
    if "${SWEBENCH_SERVICE_AUTH_TOKEN}" not in cl_content or "${SWEBENCH_SERVICE_URL}" not in cl_content:
        P.append(f"{label}: CLIENT_CONFIG does not reference ${{SWEBENCH_SERVICE_URL}} / ${{SWEBENCH_SERVICE_AUTH_TOKEN}}")

    # 3. Workflow & benchmark script
    wf = doc.get("workflow") or {}
    want_wf_name = f"kimi_k3_trtllm_v1_0_3_gke_c{c}_swe_{mode}"
    if wf.get("name") != want_wf_name:
        P.append(f"{label}: workflow.name is {wf.get('name')!r}, want {want_wf_name!r}")
    tasks = {t.get("name"): t for t in (wf.get("tasks") or []) if isinstance(t, dict)}
    if "benchmark" not in tasks:
        P.append(f"{label}: benchmark task missing")
        return P
    bench = tasks["benchmark"].get("script") or []
    want_bench = plan_items(swe_benchmark_plan(drop, swe_drop, c, mode))
    if bench != want_bench:
        P.append(f"{label}: benchmark script differs from swe_benchmark_plan")
    lock = swe_mode_lock(mode)
    if lock not in bench:
        P.append(f"{label}: benchmark script lacks the BENCHMARK_MODE={mode} lock")
    elif G_BENCH_IMPORT in bench and bench.index(lock) > bench.index(G_BENCH_IMPORT):
        P.append(f"{label}: BENCHMARK_MODE lock must precede G_BENCH_IMPORT")
    at = [i for i, item in enumerate(bench) if item == swe_drop.bench_swe]
    if len(at) != 1:
        P.append(f"{label}: the drop's SWE block appears {len(at)}x in the benchmark (want once, verbatim)")
    elif at[0] == 0 or bench[at[0] - 1] != SWE_WAIT_BLOCK:
        P.append(f"{label}: the drop's SWE block is not directly preceded by SWE_WAIT_BLOCK")
    if bench.count(SWE_WAIT_BLOCK) != 1 or bench.count(SWE_REDACT_BLOCK) != 1:
        P.append(f"{label}: SWE_WAIT_BLOCK and SWE_REDACT_BLOCK must each appear once")
    swe_run = swe_drop.bench_run[len("exec "):]
    finals = [i for i in bench if "inference_endpoint.main benchmark" in i]
    if finals != [swe_run]:
        P.append(f"{label}: benchmark command is {finals}, want [{swe_run!r}]")
    elif SWE_REDACT_BLOCK in bench:
        f_at, r_at = bench.index(swe_run), bench.index(SWE_REDACT_BLOCK)
        if not (r_at > f_at and bench[r_at - 1:r_at] == ["KIMI_BENCH_RC=$?; set -e"]
                and bench[r_at + 1:r_at + 2] == [G_BENCH_MIRROR]):
            P.append(f"{label}: SWE_REDACT_BLOCK must follow the client command and "
                     "directly precede the mirror")
    users = [i for i, item in enumerate(bench) if "SWE_CONNECTION" in item]
    if users != [i for i, item in enumerate(bench) if item in (SWE_WAIT_BLOCK, swe_drop.bench_swe)]:
        P.append(f"{label}: SWE_CONNECTION is used outside SWE_WAIT_BLOCK and the drop's SWE block")
    if tasks["benchmark"].get("depends_on") != ["frontend"]:
        P.append(f"{label}: benchmark depends_on is {tasks['benchmark'].get('depends_on')!r}, want ['frontend']")
    if any(isinstance(x, str) and x.startswith("exec ") for x in bench):
        P.append(f"{label}: benchmark uses exec (kills sflow's collect trap)")
    for t in (wf.get("tasks") or []):
        for item in (t.get("script") or []):
            if isinstance(item, str) and re.search(
                    r"(?m)^\s*set\s+-[A-Za-z]*x\b|set\s+-o\s+xtrace|bash\s+-[A-Za-z]*x\b", item):
                P.append(f"{label}: task {t.get('name')!r} enables xtrace, which would print the SWE token")

    # 4. Same-config equivalence against the perf recipe (server configs, env, operators, backends, tasks)
    stripped = _strip_swe_to_perf_doc(doc, swe_drop, c, mode)
    ptasks = {t["name"]: t for t in pdoc["workflow"]["tasks"]}
    stasks = {t["name"]: t for t in (stripped.get("workflow") or {}).get("tasks") or []}
    for tname in ptasks:
        if tname == "benchmark":
            continue
        if stasks.get(tname) != ptasks.get(tname):
            P.append(f"{label}: task {tname!r} differs from the perf recipe (same-config violation)")
    if "benchmark" in stasks and "benchmark" in ptasks:
        stasks["benchmark"]["script"] = ptasks["benchmark"]["script"]
    if stripped != pdoc:
        for sec in ("variables", "artifacts", "operators", "backends", "workflow"):
            if stripped.get(sec) != pdoc.get(sec):
                P.append(f"{label}: section {sec!r} differs from the perf recipe beyond declared SWE edits")

    # 5. Token-absence guard across recipe and launch_test.sh
    for m in re.finditer(r"swebench_service_auth_token:\s*(\S+)", text):
        if m.group(1) not in ("${SWEBENCH_SERVICE_AUTH_TOKEN}", "${SWE_SERVICE_TOKEN}"):
            P.append(f"{label}: literal or unexpected swebench_service_auth_token value {m.group(1)!r}")
    for var in ("SWEBENCH_SERVICE_AUTH_TOKEN", "SWE_SERVICE_TOKEN"):
        for m in re.finditer(rf'{var}=(?:"([^"]*)"|\'([^\']*)\'|(\S+))', text):
            val = m.group(1) or m.group(2) or m.group(3) or ""
            if val not in ("disabled", "$(python3") and not val.startswith("{"):
                P.append(f"{label}: literal {var} assignment {val!r}")

    # 6. Launch script checks
    v = drop.points[c].vars
    if f'REQUIRED_NODES="${{REQUIRED_NODES:-{v["TOTAL_NODES"]}}}"' not in launch:
        P.append(f"{label}: launch_test.sh REQUIRED_NODES != {v['TOTAL_NODES']}")
    if f'RECIPE="{recipe_name_swe(c, mode)}"' not in launch:
        P.append(f"{label}: launch_test.sh RECIPE != {recipe_name_swe(c, mode)}")
    if pull_secret != (f"kubectl get secret {PULL_SECRET} " in launch):
        P.append(f"{label}: launch_test.sh Secret check {'missing' if pull_secret else 'present'}")
    return P


def check_swe(drop: Drop, swe_drop: SweDrop, dest: Path, points: list, mode: str,
              pull_secret: bool = False) -> int:
    problems = []
    base_root = dest / PULL_SECRET_SUBDIR if pull_secret else dest
    root = swe_tree_root(dest, mode, pull_secret)
    tag = f" [{PULL_SECRET_SUBDIR}/swe_{mode}]" if pull_secret else f" [swe_{mode}]"
    if not root.is_dir():
        problems.append(f"SWE tree {root} missing (run {'--image-pull-secret ' if pull_secret else ''}--swe {mode})")
    for c in points:
        v = drop.points[c].vars
        d = dest_dir(root, v)
        rname = recipe_name_swe(c, mode)
        recipe = d / rname
        lt = d / "launch_test.sh"
        want_r = render_recipe_swe(drop, swe_drop, c, mode, pull_secret)
        want_l = render_launch_swe(drop, c, mode, pull_secret)
        for p, want in ((recipe, want_r), (lt, want_l)):
            if not p.is_file():
                problems.append(f"C{c}{tag}: missing {p}")
                continue
            got = p.read_text(encoding="utf-8")
            if got != want:
                diff = list(difflib.unified_diff(want.splitlines(), got.splitlines(), "generated", str(p), n=0,
                                                 lineterm=""))[:12]
                problems.append(f"C{c}{tag}: {p} differs from a fresh render (hand edit or stale):\n    "
                                + "\n    ".join(diff))
        if recipe.is_file() and lt.is_file():
            problems += verify_swe(drop, swe_drop, c, mode, recipe.read_text(encoding="utf-8"),
                                   lt.read_text(encoding="utf-8"), f"C{c}{tag}", pull_secret)
        elif recipe.is_file():
            problems += verify_swe(drop, swe_drop, c, mode, recipe.read_text(encoding="utf-8"),
                                   want_l, f"C{c}{tag}", pull_secret)
        if lt.is_file() and not (lt.stat().st_mode & 0o111):
            problems.append(f"C{c}{tag}: launch_test.sh not executable")
        if d.parent.name != f"{v['TOTAL_NODES']}nodes":
            problems.append(f"C{c}{tag}: directory {d.parent.name} != {v['TOTAL_NODES']}nodes")
        # Cross-check the corresponding perf recipe in the parent tree if present or expected
        perf_p = dest_dir(base_root, v) / recipe_name(c)
        want_perf = render_recipe_ps(drop, c) if pull_secret else render_recipe(drop, c)
        if not perf_p.is_file():
            problems.append(f"C{c}{tag}: corresponding perf recipe {perf_p} is missing")
        elif perf_p.read_text(encoding="utf-8") != want_perf:
            problems.append(f"{c}{tag}: corresponding perf recipe {perf_p} differs from a fresh perf render")
    # Stray directories under swe_<mode> or unknown swe_* under base_root
    if root.is_dir():
        for ch in sorted(root.iterdir()):
            if not re.fullmatch(r"\d+nodes", ch.name):
                problems.append(f"stray SWE directory {ch}")
        for nd in sorted(root.glob("*nodes")):
            for pd in sorted(nd.glob("*")):
                if pd.name == "sweep":
                    continue
                m = re.fullmatch(r"point_(\d+)", pd.name)
                if not m or int(m.group(1)) not in EXPECTED or dest_dir(root, drop.points[int(m.group(1))].vars) != pd:
                    problems.append(f"stray SWE output directory {pd}")
    if base_root.is_dir():
        for sd in sorted(base_root.glob("swe_*")):
            if sd.name not in {f"swe_{m}" for m in SWE_MODES}:
                problems.append(f"stray SWE mode directory {sd}")
    if problems:
        print("CHECK FAILED:")
        for p in problems:
            print("  - " + p)
        return 1
    print(f"CHECK OK{tag}: {len(points)} point(s); drop {drop.root}; swe-drop {swe_drop.root}")
    for c in points:
        v = drop.points[c].vars
        pl = placement(v)
        print(f"  C{c:<5} {v['TOTAL_NODES']:>2} nodes  ctx {v['NUM_CTX']}x{v['CTX_GPUS']} "
              f"gen {v['NUM_GEN']}x{v['GEN_GPUS']}  gen nodes {[g[0] for g in pl['gen']]}  "
              f"frontend n{v['FRONTEND_NODE']} client n{v['CLIENT_NODE']}  traj {v['NUM_TRAJECTORIES']}  "
              f"mode {mode}")
    return 0


def dump_scripts_swe(drop: Drop, swe_drop: SweDrop, out: Path, points: list, mode: str) -> None:
    for c in points:
        v = {**drop.points[c].vars, "SWE_WAIT_S": SWE_WAIT_S, "BENCHMARK_MODE": mode,
             "DYNAMO_DECODE_METRICS_FLAG": "--no-publish-metrics"}
        doc = yaml.safe_load(render_recipe_swe(drop, swe_drop, c, mode))
        for t in doc["workflow"]["tasks"]:
            body = _resolve_script_exprs("\n".join(t["script"]) + "\n", v)
            body = re.sub(r"\$\{\{\s*artifacts\.SWE_HANDOFF_DIR\.path\s*\}\}", SWE_HANDOFF_DIR, body)
            if "${{" in body:
                die(f"C{c} [swe_{mode}] {t['name']}: unresolved sflow expression in script")
            d = out / f"point_{c}"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{t['name']}.sh").write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    print(f"dumped SWE ({mode}) task scripts for {len(points)} point(s) under {out}")


def _resolve_swe_drop_root(slurm_root: Path, swe_drop_root: Path | None) -> Path:
    if swe_drop_root is not None:
        return swe_drop_root.resolve()
    return slurm_root.resolve()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slurm-root", type=Path, default=DEFAULT_SLURM_ROOT)
    ap.add_argument("--swe-drop-root", type=Path, default=None,
                    help="v1.0.3 drop dir carrying the SWE-bench wiring (used only with --swe)")
    ap.add_argument("--dest", type=Path, default=HERE)
    ap.add_argument("--points", type=int, nargs="+")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dump-scripts", type=Path, metavar="DIR")
    ap.add_argument("--image-pull-secret", action="store_true",
                    help=f"write/check the opt-in variant tree <dest>/{PULL_SECRET_SUBDIR}/ "
                         f"(image_pull_secrets: [{PULL_SECRET}] on the server-image operators)")
    ap.add_argument("--swe", choices=SWE_MODES, default=None,
                    help="write/check the opt-in SWE-bench tree (<dest>/[with-pull-secret/]swe_<mode>/)")
    args = ap.parse_args(argv)
    slurm_root = args.slurm_root.resolve()
    drop = Drop(slurm_root)
    points = sorted(args.points) if args.points else sorted(EXPECTED)
    for c in points:
        if c not in EXPECTED:
            die(f"unknown point C{c}; known: {sorted(EXPECTED)}")
    dest = args.dest.resolve()
    if args.swe:
        swe_drop = SweDrop(_resolve_swe_drop_root(slurm_root, args.swe_drop_root))
        if args.check:
            return check_swe(drop, swe_drop, dest, points, args.swe, args.image_pull_secret)
        if args.dump_scripts:
            if args.image_pull_secret:
                die("--dump-scripts: task scripts are identical in the variant; drop --image-pull-secret")
            dump_scripts_swe(drop, swe_drop, args.dump_scripts.resolve(), points, args.swe)
            return 0
        ps = args.image_pull_secret
        root = swe_tree_root(dest, args.swe, ps)
        tag = f"C{{c}} [{PULL_SECRET_SUBDIR}/swe_{args.swe}]" if ps else f"C{{c}} [swe_{args.swe}]"
        for c in points:
            v = drop.points[c].vars
            text = render_recipe_swe(drop, swe_drop, c, args.swe, ps)
            launch = render_launch_swe(drop, c, args.swe, ps)
            problems = verify_swe(drop, swe_drop, c, args.swe, text, launch, tag.format(c=c), ps)
            if problems:
                die("SWE self-check failed:\n  " + "\n  ".join(problems))
            d = dest_dir(root, v)
            d.mkdir(parents=True, exist_ok=True)
            rpath = d / recipe_name_swe(c, args.swe)
            rpath.write_text(text, encoding="utf-8")
            lt = d / "launch_test.sh"
            lt.write_text(launch, encoding="utf-8")
            lt.chmod(0o755)
            print(f"wrote {rpath}")
        return 0
    if args.check:
        return check(drop, dest, points, args.image_pull_secret)
    if args.dump_scripts:
        if args.image_pull_secret:
            die("--dump-scripts: task scripts are identical in the variant; drop --image-pull-secret")
        dump_scripts(drop, args.dump_scripts.resolve(), points)
        return 0
    ps = args.image_pull_secret
    root = dest / PULL_SECRET_SUBDIR if ps else dest
    for c in points:
        v = drop.points[c].vars
        text = render_recipe_ps(drop, c) if ps else render_recipe(drop, c)
        launch = render_launch_ps(drop, c) if ps else render_launch(drop, c)
        problems = verify_recipe(drop, c, text, f"C{c}", ps)
        if ps:
            problems += verify_ps(drop, c, text, launch, f"C{c} [{PULL_SECRET_SUBDIR}]")
        if problems:
            die("self-check failed:\n  " + "\n  ".join(problems))
        d = dest_dir(root, v)
        d.mkdir(parents=True, exist_ok=True)
        (d / recipe_name(c)).write_text(text, encoding="utf-8")
        lt = d / "launch_test.sh"
        lt.write_text(launch, encoding="utf-8")
        lt.chmod(0o755)
        print(f"wrote {d / recipe_name(c)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
