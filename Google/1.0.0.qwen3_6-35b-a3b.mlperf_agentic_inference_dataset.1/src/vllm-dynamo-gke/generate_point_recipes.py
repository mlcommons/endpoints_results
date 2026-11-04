#!/usr/bin/env python3
"""Generate the Qwen3.6-35B-A3B GKE recipes from the NVIDIA v1.0.3 Slurm drop.

Every value that can differ between points is READ FROM THE DROP at run time:

  configs/dependencies.json                         version pins
  configs/_shared/dynamo-server.py                  -> DYNAMO_LAUNCHER artifact (verbatim)
  configs/_shared/dynamo_cpu_affinity.py            -> DYNAMO_CPU_AFFINITY artifact (verbatim)
  configs/_shared/cuda-compat.sh                    -> CUDA_COMPAT_SH artifact (verbatim)
  configs/_shared/autotune/GB300-NVL72.json         -> AUTOTUNE_JSON artifact (verbatim)
  configs/<SYSTEM_ID>/point_<C>/workflow.yaml       CONCURRENCY, DP, PORT, probes, Slurm nodes
  configs/<SYSTEM_ID>/point_<C>/server.yaml         -> SERVER_ENV + SERVER_CONFIG artifacts (verbatim)
  configs/<SYSTEM_ID>/point_<C>/client.yaml         -> CLIENT_CONFIG artifact (verbatim + 2 rewrites)
  configs/<SYSTEM_ID>/point_<C>/server.sh           -> serve_endpoint script (runtime lines verbatim)
  configs/<SYSTEM_ID>/point_<C>/client.sh           -> benchmark script (runtime lines verbatim)

All 9 GB300 points (32, 36, 64, 128, 144, 192, 256, 512, 768) come directly
from the v1.0.3 drop with no synthesized points needed.

    python3 generate_point_recipes.py            # (re)write 1nodes/point_<C>/
    python3 generate_point_recipes.py --check    # SLURM_TO_GKE.md section 6.2 checks
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import subprocess
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required: <workdir>/sflow_workspace/.venv/bin/python " + __file__
    )

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_DROP = (
    HERE
    if (HERE / "configs").is_dir()
    else (
        HERE.parents[5]
        / "code-drop-nvidia"
        / "nv-mlpinf-partner-partner-endpoints-v1.0.3"
        / "NVIDIA"
        / "src"
        / "qwen3.6-35b-a3b"
    )
)
TEMPLATE = HERE / "recipe_template.yaml"
SYSTEM_ID = "GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo"
GPUS_PER_NODE = 4  # A4X-max

DROP_POINTS = (32, 36, 64, 128, 144, 192, 256, 512, 768)

# Assertion table (SLURM_TO_GKE.md 6.1). Transcribed from the drop on intake;
# the generator reads the real values from the drop and aborts on any mismatch.
_ROW = dict(
    slurm_nodes=1, gpus_per_node=4, dp=4, tp=1,
    max_num_batched_tokens=32768, port=30003, client_cpus=69,
)
EXPECTED = {
    32: dict(_ROW, trajectories=2452, drop_mode="both"),
    36: dict(_ROW, trajectories=2452, drop_mode="both"),
    64: dict(_ROW, trajectories=3678, drop_mode="both"),
    128: dict(_ROW, trajectories=6130, drop_mode="perf"),
    144: dict(_ROW, trajectories=6130, drop_mode="perf"),
    192: dict(_ROW, trajectories=6743, drop_mode="both"),
    256: dict(_ROW, trajectories=6743, drop_mode="perf"),
    512: dict(_ROW, trajectories=8582, drop_mode="perf"),
    768: dict(_ROW, trajectories=14712, drop_mode="perf"),
}

# ---------------------------------------------------------------------------
# Documented edits to the drop's scripts. Each literal must match EXACTLY ONCE
# (sub1), so any upstream change aborts generation instead of being ported
# half-right.
# ---------------------------------------------------------------------------
SCRIPT_PREAMBLE = "#!/usr/bin/env bash\nset -euo pipefail\n"

SERVER_SITE_CHECKS = """\
for value in "$STORAGE_ROOT" "$SHARED_ROOT" "$REPO_ROOT" "$MODEL_PATH" "$DATASET_PATH" "$SYSTEM_ID"; do
  case "$value" in ''|*'<'*|*'>'*) echo 'Supply the placeholder variables with sflow --set NAME=value' >&2; exit 1;; esac
done
for path in "$REPO_ROOT" "$SHARED_ROOT" "$MODEL_PATH" "$DATASET_PATH"; do
  if [[ "$STORAGE_ROOT" != / && "$path" != "$STORAGE_ROOT" && "$path" != "$STORAGE_ROOT/"* ]]; then
    echo 'Code, run root, model and dataset must be within STORAGE_ROOT' >&2; exit 1
  fi
done
for path in "$SFLOW_WORKSPACE_DIR" "$SFLOW_WORKFLOW_OUTPUT_DIR"; do
  case "$path" in "$SHARED_ROOT"/*) ;; *) echo 'Set --workspace-dir and --output-dir below SHARED_ROOT' >&2; exit 1;; esac
done
"""
SERVER_SITE_CHECKS_GKE = (
    "# GKE: dropped the Slurm placeholder / STORAGE_ROOT / --workspace-dir checks"
    " (replaced by the asset checks above).\n"
)
SERVER_CACHE_ROOT = 'cache_root="$SHARED_ROOT/caches/$SYSTEM_ID/$GPU_UUID"\n'
SERVER_CACHE_ROOT_GKE = 'cache_root="$CACHE_ROOT/$SYSTEM_ID/$GPU_UUID"\n'
SERVER_OUT_OLD, SERVER_OUT_NEW, SERVER_OUT_COUNT = "$SFLOW_TASK_OUTPUT_DIR", "$SERVER_OUT", 13

CLIENT_SLURM_ENV = (
    'export SLURM_JOB_ID="$SFLOW_BACKEND_JOB_ID"\n'
    'export SLURMD_NODENAME="$(hostname -s)"\n'
)
CLIENT_SLURM_ENV_GKE = (
    "# GKE: dropped SLURM_JOB_ID / SLURMD_NODENAME (read only by the Pyxis SWE service).\n"
)
CLIENT_EXEC = (
    "exec /opt/venv/bin/inference-endpoint benchmark from-config \\\n"
    '  -c "$SFLOW_TASK_OUTPUT_DIR/client.yaml" --report-dir "$SFLOW_TASK_OUTPUT_DIR/report" --mode "$MODE"\n'
)
CLIENT_EXEC_GKE = (
    "# GKE: `exec inference-endpoint ...` moved to the next script item, run WITHOUT exec\n"
    '# and with --report-dir "${REPORT_DIR}" (SLURM_TO_GKE.md 5.8).\n'
)

# client.yaml: the ONLY two rewrites allowed (SLURM_TO_GKE.md 5.1 / 6.2).
CLIENT_REWRITES = (
    ("\n      report_dir: logs/qwen_agentic\n", "\n      report_dir: ${REPORT_DIR}\n"),
    ("\n        metrics_tokenizer_workers: 2\n", "\n        metrics_tokenizer_workers: 32\n"),
)

FABRIC_VARS = re.compile(r"\b(UCX_\w+|OMPI_\w+|NCCL_IB_\w+|NCCL_MNNVL_ENABLE|MC_FORCE_MNNVL|IMEX_CHANNELS)\b")


LAUNCH_SH = """\
#!/usr/bin/env bash
# Launch Qwen3.6-35B-A3B MLPerf-endpoints v1.0.3 point C{c} on a GB300 (A4X-max)
# GKE cluster: {dp} x TP{tp} dynamo.vllm workers + 1 Dynamo frontend on ONE node,
# exactly NVIDIA's Slurm topology (backends.cluster.nodes = {slurm_nodes}).
# GENERATED by ../../generate_point_recipes.py -- do not hand-edit.
#
# PREREQUISITES (../../README.md section 3):
#   1. {required} schedulable A4X-max node(s) (counted below before launching).
#   2. kubectl apply -f ../../stage-assets.yaml    # model @ {rev_short} + dataset, EVERY node
#   3. kubectl apply -f ../../mk-report-dir.yaml   # durable report hostPath
#   4. kubectl apply -f ../../prepull-images.yaml  # server image is private + large
#   5. Pull access to SERVER_IMAGE: the `gitlab-partner-registry` docker-registry
#      Secret, or an Artifact Registry mirror passed as -s SERVER_IMAGE=<mirror>@sha256:...
#   6. kubectl context pointing at the target cluster.
#
# Site overrides:
#   NAMESPACE=<ns>  NODEPOOL=<a4x-max nodepool>   (env vars)
#   CLIENT_RESERVE_NODES=1    put the client pod on its own node (needs 2 nodes)
#   SKIP_NODE_CHECK=1         skip the node count below
#   anything else via sflow flags, e.g.
#     ./launch_test.sh --dry-run
#     ./launch_test.sh -s MODE=perf -s SERVER_IMAGE=<mirror>@sha256:...
set -euo pipefail

cd "$(dirname "$0")"

# sflow is normally installed in a venv rather than on PATH.
SFLOW_VENV="${{SFLOW_VENV:<workdir>/sflow_workspace/.venv}}"
if [ -x "${{SFLOW_VENV}}/bin/sflow" ]; then
  PATH="${{SFLOW_VENV}}/bin:${{PATH}}"
fi
if ! command -v sflow >/dev/null 2>&1; then
  echo "sflow not found. Set SFLOW_VENV=/path/to/venv or install sflow." >&2
  exit 1
fi

RECIPE="{recipe}"
NAMESPACE="${{NAMESPACE:-default}}"
NODEPOOL="${{NODEPOOL:<nodepool>}}"
CLIENT_RESERVE_NODES="${{CLIENT_RESERVE_NODES:-0}}"
case "${{CLIENT_RESERVE_NODES}}" in 0|1) ;; *) echo "CLIENT_RESERVE_NODES must be 0 or 1" >&2; exit 2 ;; esac
REQUIRED_NODES={required}
REQUIRED_NODES=$(( REQUIRED_NODES + CLIENT_RESERVE_NODES ))

# Fail fast instead of letting the reservation sit for its 1800 s timeout:
# count Ready, uncordoned nodes in the pool that advertise 4 GPUs.
if [ "${{SKIP_NODE_CHECK:-0}}" != "1" ] && command -v kubectl >/dev/null 2>&1; then
  if ! NODE_STATES=$(kubectl get nodes -l "cloud.google.com/gke-nodepool=${{NODEPOOL}}" \\
      -o jsonpath='{{range .items[*]}}{{.spec.unschedulable}}{{"|"}}{{.status.allocatable.nvidia\\.com/gpu}}{{"|"}}{{range .status.conditions[?(@.type=="Ready")]}}{{.status}}{{end}}{{"\\n"}}{{end}}'); then
    echo "kubectl get nodes failed (see above): fix the kubectl context/credentials," >&2
    echo "or set SKIP_NODE_CHECK=1 to launch without the node check." >&2
    exit 1
  fi
  HEALTHY=$(printf '%s' "${{NODE_STATES}}" | awk -F'|' '$1!="true" && $2=="4" && $3=="True"' | wc -l)
  echo "schedulable 4-GPU nodes in ${{NODEPOOL}}: ${{HEALTHY}} (need ${{REQUIRED_NODES}})"
  if [ "${{HEALTHY}}" -lt "${{REQUIRED_NODES}}" ]; then
    echo "Not enough healthy nodes; refusing to launch. Set SKIP_NODE_CHECK=1 to override." >&2
    exit 1
  fi
fi

sflow run -f "${{RECIPE}}" \\
  --kube-namespace "${{NAMESPACE}}" \\
  --kube-node-selector "cloud.google.com/gke-nodepool=${{NODEPOOL}}" \\
  --set "CLIENT_RESERVE_NODES=${{CLIENT_RESERVE_NODES}}" \\
  "$@"
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sub1(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{what}: expected exactly 1 match, found {count}")
    return text.replace(old, new)


def suball(text: str, old: str, new: str, want: int, what: str) -> str:
    count = text.count(old)
    if count != want:
        raise SystemExit(f"{what}: expected {want} matches, found {count}")
    return text.replace(old, new)


def indent(text: str, n: int) -> str:
    """Indent every non-empty line by n spaces (block-scalar body)."""
    if text.endswith("\n"):
        text = text[:-1]
    pad = " " * n
    return "\n".join(pad + line if line else "" for line in text.split("\n"))


def content_block(raw: str, name: str, artifact: str) -> str:
    """Return the raw `content: |` body of `artifact` in a sflow fragment, indentation intact."""
    lines = raw.split("\n")
    headers = [i for i, line in enumerate(lines) if line == f"  {artifact}:"]
    if len(headers) != 1:
        raise SystemExit(f"{name}: expected one '  {artifact}:' line, found {len(headers)}")
    start = None
    for i in range(headers[0] + 1, len(lines)):
        line = lines[i]
        if line == "    content: |":
            start = i
            break
        if line and not line.startswith("    ") and not line.startswith("#"):
            break
    if start is None:
        raise SystemExit(f"{name}/{artifact}: expected '    content: |' line")
    body = []
    for line in lines[start + 1:]:
        if line == "" or line.startswith("      "):
            body.append(line)
        else:
            break
    while body and body[-1] == "":
        body.pop()
    return "\n".join(body)


def fragment_content(raw: str, artifact: str) -> str:
    doc = yaml.safe_load(raw)
    arts = doc["artifacts"]
    if isinstance(arts, dict):
        return arts[artifact]["content"]
    return next(a for a in arts if a["name"] == artifact)["content"]


def recipe_dir(dest: pathlib.Path, nodes: int, c: int) -> pathlib.Path:
    return dest / f"{nodes}nodes" / f"point_{c}"


def recipe_name(c: int) -> str:
    return f"qwen36_35b_a3b_gke_dynamo_point{c}.yaml"


def pyxis_to_ref(image: str) -> str:
    """docker://host#path@sha256:... (Pyxis) -> host/path@sha256:... (OCI ref)."""
    if not image.startswith("docker://") or image.count("#") != 1:
        raise SystemExit(f"unexpected Pyxis image reference: {image}")
    return image[len("docker://"):].replace("#", "/", 1)


# ---------------------------------------------------------------------------
# intake
# ---------------------------------------------------------------------------
def load_drop(drop: pathlib.Path) -> dict:
    configs = drop / "configs"
    deps_doc = json.loads((configs / "dependencies.json").read_text())
    deps = {k: v["value"] for k, v in deps_doc["variables"].items()}
    shared = {
        "dynamo_server": (configs / "_shared" / "dynamo-server.py").read_text(),
        "cpu_affinity": (configs / "_shared" / "dynamo_cpu_affinity.py").read_text(),
        "cuda_compat": (configs / "_shared" / "cuda-compat.sh").read_text(),
        "autotune_json": (configs / "_shared" / "autotune" / "GB300-NVL72.json").read_text(),
    }
    for key, text in shared.items():
        if "${{" in text or "{%" in text or "{#" in text:
            raise SystemExit(f"_shared {key}: contains a Jinja delimiter; cannot ship verbatim")
    json.loads(shared["autotune_json"])
    system = configs / SYSTEM_ID
    found = sorted(int(p.name.split("_", 1)[1]) for p in system.glob("point_*"))
    if found != sorted(DROP_POINTS):
        raise SystemExit(f"points in drop {found} != DROP_POINTS {sorted(DROP_POINTS)}")
    points = {}
    for c in found:
        pdir = system / f"point_{c}"
        raw = {n: (pdir / n).read_text() for n in
               ("workflow.yaml", "server.yaml", "client.yaml", "server.sh", "client.sh")}
        points[c] = derive_point(c, raw, deps)
    if sorted(points) != sorted(EXPECTED):
        raise SystemExit(f"derived points {sorted(points)} != EXPECTED {sorted(EXPECTED)}")
    return dict(deps=deps, shared=shared, points=points, drop=drop)


def derive_point(c: int, raw: dict, deps: dict) -> dict:
    wf = yaml.safe_load(raw["workflow.yaml"])
    var = {k: v.get("value") for k, v in wf["variables"].items()}
    cluster = wf["backends"]["cluster"]
    tasks = {t["name"]: t for t in wf["workflow"]["tasks"]}
    server_cfg = yaml.safe_load(fragment_content(raw["server.yaml"], "SERVER_CONFIG"))
    client_cfg = yaml.safe_load(fragment_content(raw["client.yaml"], "CLIENT_CONFIG"))

    tp = int(server_cfg["tensor-parallel-size"])
    inner_dp = int(server_cfg["data-parallel-size"])
    dp = int(var["DP"])
    total_gpus = dp * tp * inner_dp
    server_nodes = math.ceil(total_gpus / GPUS_PER_NODE)
    p = dict(
        concurrency=int(var["CONCURRENCY"]),
        slurm_nodes=int(cluster["nodes"]),
        gpus_per_node=int(cluster["gpus_per_node"]),
        dp=dp,
        tp=tp,
        total_gpus=total_gpus,
        server_nodes=server_nodes,
        trajectories=int(client_cfg["datasets"][0]["agentic_inference"]["num_trajectories_to_issue"]),
        max_num_batched_tokens=int(var["MAX_NUM_BATCHED_TOKENS"]),
        port=int(var["PORT"]),
        client_cpus=int(var["CLIENT_CPUS"]),
        drop_mode=var["MODE"],
        system_id=var["SYSTEM_ID"],
        model_dirname=pathlib.PurePosixPath(var["MODEL_PATH"]).name,
        dataset_filename=pathlib.PurePosixPath(var["DATASET_PATH"]).name,
        workflow_name=wf["workflow"]["name"],
        server_failure=tasks["serve_endpoint"]["probes"]["failure"]["log_watch"]["regex_pattern"],
        server_ready=tasks["serve_endpoint"]["probes"]["readiness"]["log_watch"]["regex_pattern"],
        server_ready_timeout=int(tasks["serve_endpoint"]["probes"]["readiness"]["timeout"]),
        client_failure=tasks["benchmark_0"]["probes"]["failure"]["log_watch"]["regex_pattern"],
        metrics_tokenizer_workers=client_cfg["settings"]["metrics_tokenizer_workers"],
        raw=raw,
    )
    # ---- cross-checks against the assertion table and internal consistency
    exp = EXPECTED[c]
    for key, want in exp.items():
        if p[key] != want:
            raise SystemExit(f"point_{c}: {key}={p[key]!r} but EXPECTED {want!r}")
    checks = {
        "CONCURRENCY == folder": p["concurrency"] == c,
        "SYSTEM_ID": p["system_id"] == SYSTEM_ID,
        "workflow name": p["workflow_name"] == f"{SYSTEM_ID}-c{c}",
        "GPUs == Slurm nodes x gpus_per_node": total_gpus == p["slurm_nodes"] * p["gpus_per_node"],
        "server nodes == Slurm nodes": server_nodes == p["slurm_nodes"],
        "gpus_per_node == A4X-max": p["gpus_per_node"] == GPUS_PER_NODE,
        "server.yaml port == PORT var": server_cfg["port"] == "${{ variables.PORT }}",
        "benchmark_0 depends on serve_endpoint": "serve_endpoint" in tasks["benchmark_0"]["depends_on"],
        "server image var": var["SERVER_IMAGE"] == "${{ variables.GB300_SERVER_IMAGE }}",
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise SystemExit(f"point_{c}: drop cross-checks failed: {bad}")
    return p


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------
def server_sh_block(raw: str) -> str:
    if not raw.startswith(SCRIPT_PREAMBLE):
        raise SystemExit("server.sh: unexpected preamble")
    text = raw[len(SCRIPT_PREAMBLE):]
    text = sub1(text, SERVER_SITE_CHECKS, SERVER_SITE_CHECKS_GKE, "server.sh site checks")
    text = sub1(text, SERVER_CACHE_ROOT, SERVER_CACHE_ROOT_GKE, "server.sh cache_root")
    text = suball(text, SERVER_OUT_OLD, SERVER_OUT_NEW, SERVER_OUT_COUNT, "server.sh $SFLOW_TASK_OUTPUT_DIR")
    return indent(text, 10)


def client_sh_block(raw: str) -> str:
    if not raw.startswith(SCRIPT_PREAMBLE):
        raise SystemExit("client.sh: unexpected preamble")
    text = raw[len(SCRIPT_PREAMBLE):]
    text = sub1(text, CLIENT_SLURM_ENV, CLIENT_SLURM_ENV_GKE, "client.sh SLURM env")
    text = sub1(text, CLIENT_EXEC, CLIENT_EXEC_GKE, "client.sh exec benchmark")
    return indent(text, 10)


def client_config_block(raw: str) -> str:
    body = "\n" + content_block(raw, "client.yaml", "CLIENT_CONFIG") + "\n"
    for old, new in CLIENT_REWRITES:
        body = sub1(body, old, new, f"client.yaml rewrite {old.strip()!r}")
    return body[1:-1]


def render(template: str, c: int, drop: dict) -> str:
    p, deps = drop["points"][c], drop["deps"]
    client_image = (
        pyxis_to_ref(deps["CLIENT_IMAGE"]).rsplit(":", 1)[0] + "@" + deps["CLIENT_IMAGE_DIGEST"]
    )
    values = {
        "CONCURRENCY": str(c),
        "SYSTEM_ID": p["system_id"],
        "TRAJECTORIES": str(p["trajectories"]),
        "SLURM_NODES": str(p["slurm_nodes"]),
        "DP": str(p["dp"]),
        "TP": str(p["tp"]),
        "TOTAL_GPUS": str(p["total_gpus"]),
        "SERVER_NODES": str(p["server_nodes"]),
        "GPUS_PER_NODE": str(p["gpus_per_node"]),
        "MODEL_REVISION_SHORT": deps["MODEL_REVISION"][:12],
        "MAX_NUM_BATCHED_TOKENS": str(p["max_num_batched_tokens"]),
        "PORT": str(p["port"]),
        "DROP_MODE": p["drop_mode"],
        "MODEL_REPOSITORY": deps["MODEL_REPOSITORY"],
        "MODEL_REVISION": deps["MODEL_REVISION"],
        "DATASET_SHA256": deps["DATASET_SHA256"],
        "DYNAMO_VERSION": deps["DYNAMO_VERSION"],
        "VLLM_BRANCH": deps["VLLM_BRANCH"],
        "VLLM_COMMIT": deps["VLLM_COMMIT"],
        "ENDPOINTS_REVISION": deps["ENDPOINTS_REVISION"],
        "SERVER_IMAGE": pyxis_to_ref(deps["GB300_SERVER_IMAGE"]),
        "CLIENT_IMAGE_NOTE": (
            f"Public; tag {deps['CLIENT_IMAGE'].rsplit(':', 1)[1]} is a multi-arch index"
            " (linux/amd64 + linux/arm64) at this digest, so no local build or mirror is needed."
        ),
        "CLIENT_IMAGE": client_image,
        "CLIENT_IMAGE_DIGEST": deps["CLIENT_IMAGE_DIGEST"],
        "CLIENT_CPUS": str(p["client_cpus"]),
        "MODEL_DIRNAME": p["model_dirname"],
        "DATASET_FILENAME": p["dataset_filename"],
        "SERVER_ENV_BLOCK": content_block(p["raw"]["server.yaml"], "server.yaml", "SERVER_ENV"),
        "SERVER_CONFIG_BLOCK": content_block(p["raw"]["server.yaml"], "server.yaml", "SERVER_CONFIG"),
        "CLIENT_CONFIG_BLOCK": client_config_block(p["raw"]["client.yaml"]),
        "DYNAMO_SERVER_PY_BLOCK": indent(drop["shared"]["dynamo_server"], 6),
        "DYNAMO_CPU_AFFINITY_PY_BLOCK": indent(drop["shared"]["cpu_affinity"], 6),
        "CUDA_COMPAT_SH_BLOCK": indent(drop["shared"]["cuda_compat"], 6),
        "AUTOTUNE_JSON_BLOCK": indent(drop["shared"]["autotune_json"], 6),
        "SERVER_SH_BLOCK": server_sh_block(p["raw"]["server.sh"]),
        "CLIENT_SH_BLOCK": client_sh_block(p["raw"]["client.sh"]),
        "SERVER_FAILURE_REGEX": json.dumps(p["server_failure"]),
        "SERVER_READY_REGEX": json.dumps(p["server_ready"]),
        "SERVER_READY_TIMEOUT": str(p["server_ready_timeout"]),
        "CLIENT_FAILURE_REGEX": json.dumps(p["client_failure"]),
    }
    out = template
    for key in sorted(values, key=len, reverse=True):
        token = f"@@{key}@@"
        if token not in out:
            raise SystemExit(f"template: placeholder {token} not found")
        out = out.replace(token, values[key])
    left = re.findall(r"@@[A-Z0-9_]+@@", out)
    if left:
        raise SystemExit(f"template: unfilled placeholders {sorted(set(left))}")
    return out


def render_launcher(c: int, drop: dict) -> str:
    p = drop["points"][c]
    return LAUNCH_SH.format(
        c=c, dp=p["dp"], tp=p["tp"], slurm_nodes=p["slurm_nodes"],
        required=p["server_nodes"], rev_short=drop["deps"]["MODEL_REVISION"][:12],
        recipe=recipe_name(c),
    )


# ---------------------------------------------------------------------------
# --check (SLURM_TO_GKE.md 6.2)
# ---------------------------------------------------------------------------
class Checker:
    def __init__(self) -> None:
        self.n = 0
        self.failures: list[str] = []

    def ok(self, cond: bool, what: str) -> None:
        self.n += 1
        if not cond:
            self.failures.append(what)
            print(f"  FAIL  {what}")


def bash_n(script: str) -> tuple[bool, str]:
    # ${{ ... }} is resolved by sflow before bash sees it; stand in a word.
    text = re.sub(r"\$\{\{.*?\}\}", "SFLOW_EXPR", script)
    proc = subprocess.run(["bash", "-n"], input=text, text=True, capture_output=True)
    return proc.returncode == 0, proc.stderr.strip()


def walk_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from walk_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk_strings(v)


def check_point(ck: Checker, c: int, drop: dict, dest: pathlib.Path, template: str) -> None:
    p, deps = drop["points"][c], drop["deps"]
    d = recipe_dir(dest, p["server_nodes"], c)
    recipe_path, launcher_path = d / recipe_name(c), d / "launch_test.sh"
    tag = f"point_{c}"

    # 1. regenerated == on disk
    for path, want in ((recipe_path, render(template, c, drop)), (launcher_path, render_launcher(c, drop))):
        rel = path.relative_to(dest)
        if not path.exists():
            print(f"  MISSING  {rel}")
            ck.ok(False, f"{rel} MISSING")
            continue
        same = path.read_text() == want
        print(f"  {'match' if same else 'DIFFERS'}  {rel}")
        ck.ok(same, f"{rel} DIFFERS from a fresh render")
    ck.ok(launcher_path.exists() and bool(launcher_path.stat().st_mode & 0o111), f"{tag}: launch_test.sh executable")
    if not recipe_path.exists():
        return
    text = recipe_path.read_text()
    doc = yaml.safe_load(text)
    arts = {a["name"]: a for a in doc["artifacts"]}
    variables = {k: v.get("value") for k, v in doc["variables"].items()}
    tasks = {t["name"]: t for t in doc["workflow"]["tasks"]}

    # 2. server.yaml (SERVER_ENV + SERVER_CONFIG) byte-identical (parsed content and raw block)
    ck.ok(arts["SERVER_ENV"]["content"] == fragment_content(p["raw"]["server.yaml"], "SERVER_ENV"),
          f"{tag}: SERVER_ENV content == drop server.yaml")
    ck.ok(content_block(p["raw"]["server.yaml"], "server.yaml", "SERVER_ENV") in text,
          f"{tag}: SERVER_ENV block pasted byte-for-byte")
    ck.ok(arts["SERVER_ENV"]["uri"] == "<workdir>", f"{tag}: SERVER_ENV uri")
    ck.ok(arts["SERVER_CONFIG"]["content"] == fragment_content(p["raw"]["server.yaml"], "SERVER_CONFIG"),
          f"{tag}: SERVER_CONFIG content == drop server.yaml")
    ck.ok(content_block(p["raw"]["server.yaml"], "server.yaml", "SERVER_CONFIG") in text,
          f"{tag}: SERVER_CONFIG block pasted byte-for-byte")
    ck.ok(arts["SERVER_CONFIG"]["uri"] == "<workdir>", f"{tag}: SERVER_CONFIG uri")

    # 3. client.yaml == drop except the two documented rewrites
    drop_client = fragment_content(p["raw"]["client.yaml"], "CLIENT_CONFIG")
    got_client = arts["CLIENT_CONFIG"]["content"]
    want_client = "\n" + drop_client
    for old, new in CLIENT_REWRITES:  # same rewrites, minus the 6-space block indent
        old_d, new_d = old.replace("\n      ", "\n"), new.replace("\n      ", "\n")
        ck.ok(want_client.count(old_d) == 1, f"{tag}: drop client.yaml has {old_d.strip()!r} once")
        want_client = want_client.replace(old_d, new_d)
    want_client = want_client[1:]
    ck.ok(got_client == want_client, f"{tag}: CLIENT_CONFIG == drop client.yaml + 2 rewrites")
    dl, gl = drop_client.split("\n"), got_client.split("\n")
    ck.ok(len(gl) == len(dl), f"{tag}: client.yaml line count unchanged")
    gc, dc = yaml.safe_load(got_client), yaml.safe_load(drop_client)
    ck.ok(
        gc["report_dir"] == "${REPORT_DIR}"
        and gc["settings"]["metrics_tokenizer_workers"] == 32
        and "seed" not in gc.get("model_params", {}),
        f"{tag}: rewrites landed (report_dir, settings.metrics_tokenizer_workers, no model_params.seed)",
    )
    ck.ok("metrics_tokenizer_workers" not in gc["settings"]["client"], f"{tag}: tokenizer workers under settings.")
    dc["report_dir"], dc["settings"]["metrics_tokenizer_workers"] = gc["report_dir"], 32
    ck.ok(gc == dc, f"{tag}: client.yaml parsed == drop parsed + 2 rewrites")
    ck.ok(arts["CLIENT_CONFIG"]["uri"] == "<workdir>", f"{tag}: CLIENT_CONFIG uri")

    # 4. _shared helpers byte-identical
    ck.ok(arts["DYNAMO_LAUNCHER"]["content"] == drop["shared"]["dynamo_server"], f"{tag}: dynamo-server.py verbatim")
    ck.ok(arts["DYNAMO_CPU_AFFINITY"]["content"] == drop["shared"]["cpu_affinity"],
          f"{tag}: dynamo_cpu_affinity.py verbatim")
    ck.ok(arts["CUDA_COMPAT_SH"]["content"] == drop["shared"]["cuda_compat"], f"{tag}: cuda-compat.sh verbatim")
    ck.ok(arts["AUTOTUNE_JSON"]["content"] == drop["shared"]["autotune_json"],
          f"{tag}: autotune/GB300-NVL72.json verbatim")
    try:
        compile(arts["FABRIC_CHECK"]["content"], "qwen_fabric_check.py", "exec")
        ck.ok(True, "")
    except SyntaxError as exc:
        ck.ok(False, f"{tag}: FABRIC_CHECK does not compile: {exc}")

    # 5. scripts: strings only, bash -n, runtime lines verbatim, no Slurm fabric env
    for name, task in tasks.items():
        items = task["script"]
        ck.ok(all(isinstance(i, str) for i in items), f"{tag}/{name}: every script item is a string")
        good, err = bash_n("\n".join(i for i in items if isinstance(i, str)))
        ck.ok(good, f"{tag}/{name}: bash -n: {err}")
        ck.ok(not any(re.search(r"^\s*exec\b", i, re.M) for i in items) or name == "serve_endpoint",
              f"{tag}/{name}: no exec in the collected benchmark task")
        ck.ok(not any(FABRIC_VARS.search(i) for i in items), f"{tag}/{name}: no UCX/OMPI/IB/MNNVL env")
    server_script = "\n".join(tasks["serve_endpoint"]["script"])
    client_script = "\n".join(tasks["benchmark"]["script"])
    ck.ok(server_sh_block(p["raw"]["server.sh"]).replace("\n" + " " * 10, "\n").lstrip() in server_script,
          f"{tag}: server.sh runtime lines present verbatim")
    ck.ok(client_sh_block(p["raw"]["client.sh"]).replace("\n" + " " * 10, "\n").lstrip() in client_script,
          f"{tag}: client.sh runtime lines present verbatim")
    ck.ok("--report-dir \"${REPORT_DIR}\"" in client_script, f"{tag}: benchmark writes the durable REPORT_DIR")

    # Jinja hazard: in any string sflow renders, `{#` / `{%` would be eaten.
    rendered = [s for s in walk_strings(doc) if "${{" in s]
    ck.ok(not any("{#" in s or "{%" in s for s in rendered), f"{tag}: no Jinja-hazard {{# / {{% in rendered strings")

    # 6. topology == EXPECTED and Slurm
    exp = EXPECTED[c]
    ck.ok(variables["CONCURRENCY"] == c, f"{tag}: CONCURRENCY")
    ck.ok(variables["DP"] == exp["dp"] == p["dp"], f"{tag}: DP")
    ck.ok(variables["SERVER_NODES"] == exp["slurm_nodes"] == p["slurm_nodes"], f"{tag}: SERVER_NODES == SLURM_NODES")
    ck.ok(variables["GPUS_PER_NODE"] == exp["gpus_per_node"], f"{tag}: GPUS_PER_NODE")
    ck.ok(variables["CLIENT_RESERVE_NODES"] == 0, f"{tag}: client co-located by default")
    ck.ok(variables["PORT"] == exp["port"] and variables["MAX_NUM_BATCHED_TOKENS"] == exp["max_num_batched_tokens"],
          f"{tag}: PORT / MAX_NUM_BATCHED_TOKENS")
    ck.ok(gc["datasets"][0]["agentic_inference"]["num_trajectories_to_issue"] == exp["trajectories"],
          f"{tag}: trajectories")
    ck.ok(variables["MODE"] == "perf", f"{tag}: MODE default perf")
    for key in ("MODEL_REPOSITORY", "MODEL_REVISION", "DATASET_SHA256", "DYNAMO_VERSION",
                "VLLM_BRANCH", "VLLM_COMMIT", "ENDPOINTS_REVISION", "CLIENT_IMAGE_DIGEST"):
        ck.ok(str(variables[key]) == deps[key], f"{tag}: pin {key}")
    ck.ok(variables["SERVER_IMAGE"] == pyxis_to_ref(deps["GB300_SERVER_IMAGE"]), f"{tag}: SERVER_IMAGE digest")
    ck.ok(variables["CLIENT_IMAGE"].endswith("@" + deps["CLIENT_IMAGE_DIGEST"]), f"{tag}: CLIENT_IMAGE digest")
    srv = tasks["serve_endpoint"]["resources"]
    ck.ok(srv["nodes"]["indices"] == [0] and srv["gpus"]["count"] == "${{ variables.GPUS_PER_NODE }}",
          f"{tag}: serve_endpoint = node 0, all GPUs")
    backend = doc["backends"][0]
    ck.ok(backend["name"] == "cluster" and "compute_domain" not in backend and backend["rdma"] == "disable"
          and backend["scheduling"] == "device_plugin" and backend["collect_max_file_size"] == "1Gi"
          and backend["collect_grace_seconds"] == 900, f"{tag}: backend block")
    launcher = launcher_path.read_text() if launcher_path.exists() else ""
    ck.ok(f"\nREQUIRED_NODES={p['server_nodes']}\n" in launcher and p["server_nodes"] == 1,
          f"{tag}: launch_test.sh REQUIRED_NODES=1")

    # 7. probes verbatim
    sp = tasks["serve_endpoint"]["probes"]
    ck.ok(sp["failure"]["log_watch"]["regex_pattern"] == p["server_failure"], f"{tag}: server failure probe")
    ck.ok(sp["readiness"]["log_watch"]["regex_pattern"] == p["server_ready"]
          and sp["readiness"]["timeout"] == p["server_ready_timeout"], f"{tag}: server readiness probe")
    ck.ok(tasks["benchmark"]["probes"]["failure"]["log_watch"]["regex_pattern"] == p["client_failure"],
          f"{tag}: client failure probe")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify instead of writing (SLURM_TO_GKE.md 6.2)")
    parser.add_argument("--slurm-root", type=pathlib.Path, default=DEFAULT_DROP,
                        help="the drop's NVIDIA/src/qwen3.6-35b-a3b directory")
    parser.add_argument("--dest", type=pathlib.Path, default=HERE, help="output root (default: next to this script)")
    args = parser.parse_args()

    drop = load_drop(args.slurm_root)
    template = TEMPLATE.read_text()

    if args.check:
        ck = Checker()
        for c in sorted(drop["points"]):
            print(f"point_{c}:")
            check_point(ck, c, drop, args.dest, template)
        if ck.failures:
            print(f"\n{len(ck.failures)} of {ck.n} checks FAILED", file=sys.stderr)
            return 1
        print(f"\n{ck.n} checks passed for {len(drop['points'])} points")
        return 0

    for c in sorted(drop["points"]):
        p = drop["points"][c]
        d = recipe_dir(args.dest, p["server_nodes"], c)
        d.mkdir(parents=True, exist_ok=True)
        (d / recipe_name(c)).write_text(render(template, c, drop))
        launcher = d / "launch_test.sh"
        launcher.write_text(render_launcher(c, drop))
        launcher.chmod(0o755)
        print(f"wrote {(d / recipe_name(c)).relative_to(args.dest)}  +  launch_test.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
