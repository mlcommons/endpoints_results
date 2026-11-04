#!/usr/bin/env python3
"""Generate the nine DeepSeek-V4.1-Flash GKE sflow recipes from the NVIDIA
v1.0.3 Slurm code drop and ``recipe_base.yaml``.

    ./generate_point_recipes.py            # (re)generate every point
    ./generate_point_recipes.py --table    # print the per-point table and exit
    ./generate_point_recipes.py --check    # verify everything on disk against
                                           # the drop; non-zero exit on drift

Inputs
------
  recipe_base.yaml   the GKE layer (backend, operators, fabric preflight,
                     benchmark-task pattern).  Hand maintained; a fix there
                     propagates to all nine points.
  the NVIDIA drop    nv-mlpinf-partner-partner-endpoints-v1.0.3/NVIDIA/src/
                       deepseek-v4.1-flash/configs/GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo/
                         point_<C>/{dsv4.1_flash_config,client,server-env,dynamo_env,endpoint_env}.yaml
                         vllm_acc_perf_acc.yaml, slurm_env.yaml
                       deepseek-v4.1-flash/tools/{run_with_env,cpuset}.py
                       deepseek-v4.1-flash/setup_swebench_service.sbatch
                       deepseek-v4.1-flash/submission/GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo/system_desc.json
                       sflow/tools/{run_with_env,generate_endpoint_yaml}.py
                     EVERY point-specific value is read from here at run time.

What is copied verbatim (and asserted by --check)
-------------------------------------------------
  * the point's variable blocks (TOTAL_GPUS, NUM_WORKERS, VLLM_ARGS, VLLM_PORT,
    CONCURRENCY, DATASET_REPEATS, NUM_TRAJECTORIES) -- raw bytes;
  * server-env.yaml, dynamo_env.yaml, endpoint_env.yaml, dsv41 tools/{run_with_env,cpuset}.py,
    and sflow tools/{run_with_env,generate_endpoint_yaml}.py -- as artifacts;
  * client.yaml -- as an artifact, differing from the drop ONLY by the documented
    rewrite(s) (guide section 5.1): report_dir -> ${REPORT_DIR}; and
    settings.metrics_tokenizer_workers: 32 only if the drop does not set it
    (this drop sets 16, which is kept and recorded);
  * the dynamo_frontend / vllm_worker / endpoint_client scripts -- the drop's
    text with ANCHORED substitutions (each must match an exact number of times,
    and --check proves reversing them restores the drop byte-for-byte);
  * every probe (structurally equal to the drop).

Output: 1nodes/point_<C>/deepseek_v41_flash_gke_vllm_point<C>.yaml + launch_test.sh.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml (or use the sflow venv's python3)")

HERE = pathlib.Path(__file__).resolve().parent

# deepseek-v4.1-flash -> v1.0.3 -> kubernetes -> self_contained -> examples -> nv-sflow -> <root>
DEFAULT_DROP_ROOT = (HERE.parents[5] / "code-drop-nvidia"
                     / "nv-mlpinf-partner-partner-endpoints-v1.0.3")
MODEL_REL = pathlib.Path("NVIDIA/src/deepseek-v4.1-flash")
CONFIG_REL = MODEL_REL / "configs/GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo"
SUBMISSION_REL = MODEL_REL / "submission/GB300-NVL72_GB300-288GB_aarch64x4_vLLM-Dynamo"
TOOLS_REL = pathlib.Path("NVIDIA/src/sflow/tools")

GPUS_PER_NODE = 4
DROP_POINTS = [16, 24, 32, 64, 128, 256, 512, 1024, 2048]
POINTS = [16, 24, 32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_IMAGE_PULL_SECRET = "nv-gitlab-pull"

# Guide section 3.2: an ASSERTION, not a data source.  Every value is read from
# the drop independently and compared; a mismatch means the drop changed.
EXPECTED = {
    #      workers TP  DP  EP    repeats trajectories mode    batched cudagraph max_new tok_workers
    16:   (4,      1,  4,  True, 1,      613,         "both", 10240,  10230,    8192,   16),
    24:   (4,      1,  4,  True, 1,      613,         "both", 10240,  10230,    8192,   16),
    32:   (4,      1,  4,  True, 1,      613,         "perf", 10240,  10230,    8192,   16),
    64:   (4,      1,  4,  True, 2,      1226,        "perf", 10240,  10230,    8192,   16),
    128:  (4,      1,  4,  True, 2,      1226,        "both", 10240,  10230,    8192,   16),
    256:  (4,      1,  4,  True, 4,      2452,        "perf", 10240,  10230,    8192,   16),
    512:  (4,      1,  4,  True, 8,      4904,        "perf", 3072,   3072,     8192,   16),
    1024: (4,      1,  4,  True, 8,      4904,        "perf", 3072,   3072,     8192,   16),
    2048: (4,      1,  4,  True, 12,     7356,        "both", 3600,   3600,     8192,   16),
}

# Shared drop files: any change must force a re-review of the port.
DROP_SHA256 = {
    CONFIG_REL / "vllm_acc_perf_acc.yaml":
        "f4536dfd48b06488d00281296330eef78cad22f670cbf88021de01bd85dd722d",
    CONFIG_REL / "slurm_env.yaml":
        "e53e562983de9cf0a7e8f6df96c3b8046e53ad2b89ba489f1a40378697ef8870",
    MODEL_REL / "tools/cpuset.py":
        "37f0157119a060df72440d04c09d4c09a2e98752a37aa2955c04c4fe0bd2a36d",
    MODEL_REL / "tools/run_with_env.py":
        "5239eadf537b71b7daf81440dad0585db0cb0330ff6fe225e375914481841ae7",
    MODEL_REL / "setup_swebench_service.sbatch":
        "16c45d33370c0dd85c4fd72271662bad5972dd972ed7320b61ee9c7d725ebd90",
    TOOLS_REL / "run_with_env.py":
        "21e6bc4beba621c6ed5e0df483b89245eb6e7727ea4cc2ef6bd4a3755b418a77",
    TOOLS_REL / "generate_endpoint_yaml.py":
        "965e9b08a42c563aade161041652e27c49924fac35c9b879ad8c2883b99ce5c2",
    MODEL_REL / "README.md":
        "11f404caaf0aeab59d8ac0aab86af2a340ae52e67bce5f47605b469606f8458c",
    SUBMISSION_REL / "system_desc.json":
        "c6a0b6f6c0955be7f3e67e225d27d06eed85ed91596ab748003f593f13a011bc",
}

# Point variables copied as raw bytes, and the ones the port replaces.
POINT_VARS = ["TOTAL_GPUS", "NUM_WORKERS", "VLLM_ARGS", "VLLM_PORT", "CONCURRENCY",
              "DATASET_REPEATS", "NUM_TRAJECTORIES"]
REPLACED_VARS = {
    "MODEL_PATH": "node-local SSD path (guide 5.4 edit 2)",
    "DATA_PATH": "node-local SSD path (guide 5.4 edit 2)",
    "MODE": "value both/perf -> perf default (no in-cluster SWE-bench runtime)",
    "ENDPOINT_CONFIG": "CLIENT_CONFIG artifact",
    "ENDPOINT_ENV_YAML": "ENDPOINT_ENV_YAML artifact",
    "SERVER_ENV_YAML": "SERVER_ENV_YAML artifact",
    "DYNAMO_ENV_YAML": "DYNAMO_ENV_YAML artifact",
}

CLIENT_TOKENIZER_DEFAULT = 32   # guide 5.1 rewrite 2, only when the drop is silent

# ---------------------------------------------------------------------------
# Anchored substitutions (guide 6.1).  (old, new, expected_count)
# ---------------------------------------------------------------------------
FRONTEND_SUBS = [
    ('python3 ${{ variables.WORK_DIR }}/NVIDIA/src/deepseek-v4.1-flash/tools/run_with_env.py frontend ${{ variables.DYNAMO_ENV_YAML }}',
     'python3 ${{ artifacts.DSV41_RUN_WITH_ENV.path }} frontend ${{ artifacts.DYNAMO_ENV_YAML.path }}', 1),
    ('echo "${{ variables.HEAD_NODE_IP }}:${{ variables.VLLM_PORT }}" > ${{ artifacts.VLLM_URLS.path }}',
     'cat ${{ artifacts.VLLM_URLS.path }}  # GKE: static artifact (ConfigMaps are read-only)', 1),
    ('python3 ${{ variables.WORK_DIR }}/NVIDIA/src/deepseek-v4.1-flash/tools/cpuset.py',
     'python3 ${{ artifacts.CPUSET_PY.path }}', 1),
    ('"$SFLOW_WORKFLOW_OUTPUT_DIR/host_cpus.txt"',
     '"$RUN_ROOT/host_cpus.txt"', 1),
]
WORKER_SUBS = [
    ('python3 ${{ variables.WORK_DIR }}/NVIDIA/src/deepseek-v4.1-flash/tools/run_with_env.py worker ${{ variables.DYNAMO_ENV_YAML }} ${{ variables.SERVER_ENV_YAML }}',
     'python3 ${{ artifacts.DSV41_RUN_WITH_ENV.path }} worker ${{ artifacts.DYNAMO_ENV_YAML.path }} ${{ artifacts.SERVER_ENV_YAML.path }}', 1),
    ('python3 ${{ variables.WORK_DIR }}/NVIDIA/src/deepseek-v4.1-flash/tools/cpuset.py',
     'python3 ${{ artifacts.CPUSET_PY.path }}', 1),
]
CLIENT_SUBS = [
    ('${{ variables.WORK_DIR }}/NVIDIA/src/sflow/tools/run_with_env.py ${{ variables.ENDPOINT_ENV_YAML }}',
     '${{ artifacts.RUN_WITH_ENV.path }} ${{ artifacts.ENDPOINT_ENV_YAML.path }}', 1),
    ('${{ variables.WORK_DIR }}/NVIDIA/src/sflow/tools/generate_endpoint_yaml.py',
     '${{ artifacts.GENERATE_ENDPOINT_YAML.path }}', 1),
    ('--input ${{ variables.ENDPOINT_CONFIG }}',
     '--input ${{ artifacts.CLIENT_CONFIG.path }}', 1),
    ('"$SFLOW_WORKFLOW_OUTPUT_DIR/host_cpus.txt"',
     '"$RUN_ROOT/host_cpus.txt"', 1),
]
FORBIDDEN_AFTER_PORT = ("WORK_DIR", "SCRATCH_DIR", "SLURM_JOB_ID", "SLURM_NODES",
                        "SLURMD_NODENAME", "slurm_cluster")


def fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    raise SystemExit(f"ERROR: {msg}")


def apply_subs(text: str, subs, what: str) -> str:
    for old, new, count in subs:
        n = text.count(old)
        if n != count:
            fail(f"{what}: anchor {old!r} matched {n}x, expected {count}x -- the drop changed; "
                 f"re-read it before updating the substitution table")
        if text.count(new):
            fail(f"{what}: replacement {new!r} already present before substitution")
        text = text.replace(old, new)
    for word in FORBIDDEN_AFTER_PORT:
        if word in text:
            fail(f"{what}: {word!r} survived the port -- add a documented substitution")
    return text


def reverse_subs(text: str, subs) -> str:
    for old, new, _count in reversed(subs):
        text = text.replace(new, old)
    return text


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_var_block(text: str, key: str) -> str:
    """Raw lines of a 2-space-indented `  KEY:` block, verbatim (with newline)."""
    lines = text.splitlines(keepends=True)
    start = [i for i, ln in enumerate(lines) if ln.rstrip("\n") == f"  {key}:"]
    if len(start) != 1:
        fail(f"variable block {key!r}: found {len(start)} definitions")
    out = [lines[start[0]]]
    for ln in lines[start[0] + 1:]:
        if ln.strip() and not ln.startswith("    "):
            break
        out.append(ln)
    return "".join(out)


def indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join((pad + ln) if ln.strip() else "" for ln in text.rstrip("\n").split("\n"))


# ---------------------------------------------------------------------------
# Reading the drop
# ---------------------------------------------------------------------------
def read_shared(drop: pathlib.Path) -> dict:
    for rel, want in DROP_SHA256.items():
        p = drop / rel
        if not p.is_file():
            fail(f"missing drop file {p}")
        got = sha256(p)
        if got != want:
            fail(f"{rel} sha256 {got} != pinned {want} -- the drop changed; re-review the port "
                 f"and update DROP_SHA256")
    cfg = drop / CONFIG_REL
    slurm_env = yaml.safe_load((cfg / "slurm_env.yaml").read_text())["variables"]
    wf_text = (cfg / "vllm_acc_perf_acc.yaml").read_text()
    wf = yaml.safe_load(wf_text)
    tasks = {t["name"]: t for t in wf["workflow"]["tasks"]}
    if sorted(tasks) != ["dynamo_frontend", "endpoint_client", "host_prepare", "swebench_service", "vllm_worker"]:
        fail(f"workflow tasks changed: {sorted(tasks)}")
    for name in ("dynamo_frontend", "vllm_worker", "endpoint_client"):
        if len(tasks[name]["script"]) != 1:
            fail(f"{name}: expected a single `- |` script item")

    vllm_image = str(slurm_env["VLLM_CONTAINER_IMAGE"]["value"]).removeprefix("docker://")
    client_image = str(slurm_env["ENDPOINT_CONTAINER_IMAGE"]["value"]).removeprefix("docker://")

    sys_desc = json.loads((drop / SUBMISSION_REL / "system_desc.json").read_text())
    vllm_sw_version = (sys_desc.get("serving_framework")
                       or fail("serving_framework missing from system_desc.json"))

    readme = (drop / MODEL_REL / "README.md").read_text()
    m = re.search(r"hf download (\S+) \\\n\s+--revision ([0-9a-f]{40})", readme)
    if not m:
        fail("model repo/revision not found in the model README")
    model_repo, model_rev = m.group(1), m.group(2)
    ds_readme = (drop / "dataset" / "README.md").read_text()
    sec = ds_readme.split("## DeepSeek-V4.1-Flash", 1)[1].split("\n## ", 1)[0]
    m = re.search(r"`([0-9a-f]{64})`", sec)
    dataset_sha = m.group(1) if m else fail("dataset sha256 not found in dataset/README.md")

    swe_sbatch = (drop / MODEL_REL / "setup_swebench_service.sbatch").read_text()
    m = re.search(r"(?m)^ENDPOINTS_COMMIT=([0-9a-f]{40})$", swe_sbatch)
    swe_commit = m.group(1) if m else fail("ENDPOINTS_COMMIT not found")
    swe_script = "\n".join(tasks["swebench_service"]["script"])
    m = re.search(r"--auth-token (\S+)", swe_script)
    swe_token = m.group(1) if m else fail("swebench --auth-token not found")
    m = re.search(r"--image-registry (\S+)", swe_script)
    swe_registry = m.group(1) if m else fail("swebench --image-registry not found")
    if "--runtime pyxis" not in swe_script:
        fail("swebench_service no longer uses --runtime pyxis; re-evaluate GKE feasibility")

    for key, want in (("SLURM_NODES", 1), ("GPUS_PER_NODE", GPUS_PER_NODE)):
        if slurm_env[key]["value"] != want:
            fail(f"slurm_env.yaml {key} = {slurm_env[key]['value']}, expected {want}")

    return dict(
        wf_text=wf_text, wf=wf, tasks=tasks,
        slurm_nodes=slurm_env["SLURM_NODES"]["value"],
        vllm_image=vllm_image, client_image=client_image,
        vllm_sw_version=vllm_sw_version,
        dsv41_run_with_env=(drop / MODEL_REL / "tools/run_with_env.py").read_text(),
        cpuset_py=(drop / MODEL_REL / "tools/cpuset.py").read_text(),
        run_with_env=(drop / TOOLS_REL / "run_with_env.py").read_text(),
        gen_endpoint=(drop / TOOLS_REL / "generate_endpoint_yaml.py").read_text(),
        model_repo=model_repo, model_rev=model_rev, dataset_sha=dataset_sha,
        swe_commit=swe_commit, swe_token=swe_token, swe_registry=swe_registry,
        gpus_per_worker_block=raw_var_block(wf_text.split("\nartifacts:", 1)[0] + "\n",
                                            "GPUS_PER_WORKER"),
    )


def read_point(drop: pathlib.Path, point: int) -> dict:
    d = drop / CONFIG_REL / f"point_{point}"
    cfg_text = (d / "dsv4.1_flash_config.yaml").read_text()
    client_text = (d / "client.yaml").read_text()
    cfg = yaml.safe_load(cfg_text)["variables"]
    extra = set(cfg) - set(POINT_VARS) - set(REPLACED_VARS)
    missing = (set(POINT_VARS) | set(REPLACED_VARS)) - set(cfg)
    if extra or missing:
        fail(f"point_{point}: variable set changed (new {sorted(extra)}, gone {sorted(missing)})")
    v = {k: cfg[k]["value"] for k in cfg}
    args = str(v["VLLM_ARGS"])
    m = re.search(r"--tensor-parallel-size (\d+)", args)
    tp = int(m.group(1)) if m else fail(f"point_{point}: no --tensor-parallel-size in VLLM_ARGS")
    m = re.search(r"--data-parallel-size (\d+)", args)
    dp = int(m.group(1)) if m else 1
    m = re.search(r"--max-num-seqs (\d+)", args)
    max_num_seqs = int(m.group(1)) if m else None
    m = re.search(r"--max-num-batched-tokens (\d+)", args)
    max_batched_tokens = int(m.group(1)) if m else None
    m = re.search(r"--max-cudagraph-capture-size (\d+)", args)
    max_cudagraph = int(m.group(1)) if m else None
    m = re.search(r"--served-model-name (\S+)", args)
    served = m.group(1) if m else fail(f"point_{point}: no --served-model-name in VLLM_ARGS")
    m = re.search(r"--speculative-config '([^']+)'", args)
    spec = m.group(1) if m else fail(f"point_{point}: no --speculative-config in VLLM_ARGS")
    m = re.search(r"\*\s*(\d+)\s*\}\}", str(v["NUM_TRAJECTORIES"]))
    per_pass = int(m.group(1)) if m else fail(f"point_{point}: NUM_TRAJECTORIES formula changed")
    client = yaml.safe_load(client_text)
    info = dict(
        dir=d, cfg_text=cfg_text, vars=v, tp=tp, dp=dp,
        ep="--enable-expert-parallel" in args,
        max_num_seqs=max_num_seqs,
        max_batched_tokens=max_batched_tokens,
        max_cudagraph=max_cudagraph,
        served=served, spec=spec,
        per_pass=per_pass,
        trajectories=v["DATASET_REPEATS"] * per_pass,
        client_text=client_text, client=client,
        server_env=(d / "server-env.yaml").read_text(),
        dynamo_env=(d / "dynamo_env.yaml").read_text(),
        endpoint_env=(d / "endpoint_env.yaml").read_text(),
        raw_blocks={k: raw_var_block(cfg_text, k) for k in POINT_VARS},
    )
    info["nodes"] = -(-v["TOTAL_GPUS"] // GPUS_PER_NODE)
    return info


def assert_point(point: int, info: dict, shared: dict) -> None:
    v, c = info["vars"], info["client"]
    got = (v["NUM_WORKERS"], info["tp"], info["dp"], info["ep"],
           v["DATASET_REPEATS"], info["trajectories"], v["MODE"],
           info["max_batched_tokens"], info["max_cudagraph"],
           c["model_params"]["max_new_tokens"], c["settings"].get("metrics_tokenizer_workers"))
    if got != EXPECTED[point]:
        fail(f"point_{point}: drop {got} != EXPECTED {EXPECTED[point]} -- the drop changed")
    if v["CONCURRENCY"] != point:
        fail(f"point_{point}: CONCURRENCY {v['CONCURRENCY']}")
    if v["NUM_WORKERS"] * info["tp"] != v["TOTAL_GPUS"] or info["dp"] != v["NUM_WORKERS"]:
        fail(f"point_{point}: {v['NUM_WORKERS']} x TP{info['tp']} (DP{info['dp']}) != TOTAL_GPUS {v['TOTAL_GPUS']}")
    if info["nodes"] != shared["slurm_nodes"]:
        fail(f"point_{point}: node math {info['nodes']} != SLURM_NODES {shared['slurm_nodes']}")
    if info["per_pass"] != 613:
        fail(f"point_{point}: trajectories per pass {info['per_pass']} != 613 (dataset size)")
    if c["name"] != f"deepseek-v4.1-flash-c{point}":
        fail(f"point_{point}: client name {c['name']}")
    if c["model_params"]["name"] != info["served"]:
        fail(f"point_{point}: client model {c['model_params']['name']} != served {info['served']}")
    if "seed" in c["model_params"]:
        fail(f"point_{point}: unexpected seed in client.yaml model_params: {c['model_params']['seed']}")
    if str(c["settings"]["load_pattern"]["target_concurrency"]) != "${CONCURRENCY}":
        fail(f"point_{point}: target_concurrency is not ${{CONCURRENCY}}")
    if v["VLLM_PORT"] != 8000:
        fail(f"point_{point}: VLLM_PORT {v['VLLM_PORT']}")
    if "metrics_tokenizer_workers" in (c["settings"].get("client") or {}):
        fail(f"point_{point}: metrics_tokenizer_workers under settings.client (F14)")
    swe = [d for d in c["datasets"] if d.get("type") == "accuracy"]
    if [d["name"] for d in swe] != ["swe_bench"]:
        fail(f"point_{point}: accuracy datasets changed: {[d['name'] for d in swe]}")
    ex = swe[0]["accuracy_config"]["extras"]
    if ex["swebench_service_auth_token"] != shared["swe_token"]:
        fail(f"point_{point}: client SWE token != service --auth-token")
    if not re.fullmatch(r"http://127\.0\.0\.1:\d+", ex["swebench_service_url"]):
        fail(f"point_{point}: unexpected swebench_service_url {ex['swebench_service_url']}")


# ---------------------------------------------------------------------------
# client.yaml rewrite (guide 5.1)
# ---------------------------------------------------------------------------
def rewrite_client(text: str, point: int) -> tuple[str, list[str]]:
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("report_dir:")]
    if len(idx) != 1:
        fail(f"point_{point}: expected one report_dir line, found {len(idx)}")
    notes = [f"report_dir: {lines[idx[0]][len('report_dir:'):].strip()} -> ${{REPORT_DIR}}"]
    lines[idx[0]] = "report_dir: ${REPORT_DIR}"
    if not any(ln.startswith("  metrics_tokenizer_workers:") for ln in lines):
        cl = [i for i, ln in enumerate(lines) if ln == "  client:"]
        if len(cl) != 1:
            fail(f"point_{point}: expected one settings.client block")
        lines.insert(cl[0], f"  metrics_tokenizer_workers: {CLIENT_TOKENIZER_DEFAULT}")
        notes.append(f"added settings.metrics_tokenizer_workers: {CLIENT_TOKENIZER_DEFAULT}")
    return "\n".join(lines), notes


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def port_scripts(shared: dict) -> dict:
    t = shared["tasks"]
    client_ported = apply_subs(t["endpoint_client"]["script"][0], CLIENT_SUBS, "endpoint_client")
    if "benchmark from-config" not in client_ported:
        fail("endpoint_client: expected 'benchmark from-config' in script item")
    if any(ln.lstrip().startswith("exec ") for ln in client_ported.splitlines()):
        fail("endpoint_client: script item uses exec (guide F29)")
    return dict(
        frontend=apply_subs(t["dynamo_frontend"]["script"][0], FRONTEND_SUBS, "dynamo_frontend"),
        worker=apply_subs(t["vllm_worker"]["script"][0], WORKER_SUBS, "vllm_worker"),
        client=client_ported,
    )


def fmt_table(points: dict[int, dict]) -> list[str]:
    rows = ["| C | workers x TP (DP) | EP | GPUs | nodes | drop MODE | DATASET_REPEATS | trajectories | "
            "max-num-seqs/worker | batched / cudagraph |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for p, i in points.items():
        v = i["vars"]
        rows.append(f"| {p} | {v['NUM_WORKERS']} x TP{i['tp']} (DP{i['dp']}) | {'on' if i['ep'] else 'off'} | "
                    f"{v['TOTAL_GPUS']} | {i['nodes']} | {v['MODE']} | {v['DATASET_REPEATS']} | "
                    f"{i['trajectories']} | {i['max_num_seqs'] or 'default'} | "
                    f"{i['max_batched_tokens']} / {i['max_cudagraph']} |")
    return rows


def header(point: int, info: dict, shared: dict, notes: list[str], points: dict) -> str:
    v, c = info["vars"], info["client"]
    t = c["settings"]["timeouts"]
    lines = [
        f"# THIS POINT: C{point} = {v['NUM_WORKERS']} x dynamo.vllm TP{info['tp']} DP{info['dp']}"
        f"{' + expert parallel (DEP4)' if info['ep'] else ''} behind one Dynamo frontend",
        f"#   = {v['TOTAL_GPUS']} GB300 GPUs = {info['nodes']} node (slurm_env.yaml SLURM_NODES "
        f"{shared['slurm_nodes']}); REQUIRED_NODES={info['nodes']}.",
        f"#   Client: target_concurrency {point}, num_trajectories_to_issue "
        f"{info['trajectories']} ({v['DATASET_REPEATS']} x 613), max_new_tokens "
        f"{c['model_params']['max_new_tokens']},",
        f"#   metrics_drain_timeout_s {t.get('metrics_drain_timeout_s')}, "
        f"metrics_tokenizer_workers {c['settings'].get('metrics_tokenizer_workers')} (drop value, kept).",
        f"#   Speculative decoding: {info['spec']}",
        "#",
        "# All points (1nodes/):",
    ] + ["#   " + r for r in fmt_table(points)] + [
        "#",
        "# PROVENANCE (record with every result):",
        f"#   model   : {shared['model_repo']} @ {shared['model_rev']}",
        f"#   server  : {shared['vllm_image']} ({shared['vllm_sw_version']})",
        f"#   client  : {shared['client_image']}",
        f"#   dataset : perf/mlperf_agentic_inference_dataset.jsonl sha256 {shared['dataset_sha']}",
        f"#   SWE-bench service (acc/both, not in-cluster): endpoints {shared['swe_commit']}",
        "#",
        "# client.yaml rewrites applied (everything else byte-identical to the drop):",
    ] + [f"#   - {n}" for n in notes]
    return "\n".join(lines)


def render(point: int, info: dict, shared: dict, ported: dict, points: dict, base: str) -> str:
    client_text, notes = rewrite_client(info["client_text"], point)
    swe_url = [d for d in info["client"]["datasets"] if d.get("type") == "accuracy"][0][
        "accuracy_config"]["extras"]["swebench_service_url"]
    rewrite_note = "\n".join(["  # GKE REWRITE (guide 5.1):"] + [f"  #   {n}" for n in notes] + [
        "  # ${REPORT_DIR} is a durable hostPath on the node's SSD, exported by",
        "  # endpoint_client (the drop's ${SFLOW_TASK_OUTPUT_DIR} is a pod emptyDir).",
    ])
    subs = {
        "__HEADER__": header(point, info, shared, notes, points),
        "__POINT_VARS__": "".join(info["raw_blocks"][k] for k in POINT_VARS).rstrip("\n"),
        "__GPUS_PER_WORKER_VAR__": shared["gpus_per_worker_block"].rstrip("\n"),
        "__DROP_MODE__": str(info["vars"]["MODE"]),
        "__SERVER_ENV_YAML__": indent(info["server_env"], 6),
        "__DYNAMO_ENV_YAML__": indent(info["dynamo_env"], 6),
        "__ENDPOINT_ENV_YAML__": indent(info["endpoint_env"], 6),
        "__DSV41_RUN_WITH_ENV_PY__": indent(shared["dsv41_run_with_env"], 6),
        "__CPUSET_PY__": indent(shared["cpuset_py"], 6),
        "__RUN_WITH_ENV_PY__": indent(shared["run_with_env"], 6),
        "__GENERATE_ENDPOINT_YAML_PY__": indent(shared["gen_endpoint"], 6),
        "__CLIENT_REWRITE_NOTE__": rewrite_note,
        "__CLIENT_CONFIG_YAML__": indent(client_text, 6),
        "__FRONTEND_SCRIPT__": indent(ported["frontend"], 10),
        "__WORKER_SCRIPT__": indent(ported["worker"], 10),
        "__CLIENT_SCRIPT__": indent(ported["client"], 10),
        "__SWEBENCH_LOCAL_PORT__": swe_url.rsplit(":", 1)[1],
        "__SWEBENCH_LOCAL__": swe_url,
        "__SWEBENCH_COMMIT__": shared["swe_commit"],
        "__SWEBENCH_TOKEN__": shared["swe_token"],
        "__SERVED_MODEL__": info["served"],
        "__MODEL_REPO__": shared["model_repo"],
        "__MODEL_REVISION__": shared["model_rev"],
        "__DATASET_SHA256__": shared["dataset_sha"],
        "__VLLM_SW_VERSION__": shared["vllm_sw_version"],
        "__VLLM_IMAGE__": shared["vllm_image"],
        "__CLIENT_IMAGE__": shared["client_image"],
        "__CLIENT_INDEX_DIGEST__": CLIENT_INDEX_DIGEST,
        "__POINT__": str(point),
    }
    s = base
    for tok, val in subs.items():
        if tok not in s:
            fail(f"template token {tok} missing from recipe_base.yaml")
        s = s.replace(tok, val)
    left = sorted(set(re.findall(r"__[A-Z0-9_]+__", s)))
    if left:
        fail(f"point_{point}: unsubstituted tokens {left}")
    return s


# Observed via the ghcr.io registry API (multi-arch index incl. linux/arm64).
CLIENT_INDEX_DIGEST = "sha256:6cf516d0932d6790590308e7d04e5a866fffbab3a9001d65222b639ee8350240"

LAUNCH_TEMPLATE = r"""#!/usr/bin/env bash
# Launch DeepSeek-V4.1-Flash (MLPerf Endpoints v1.0.3) point C@POINT@ on a GB300
# (A4X-max) GKE cluster: @TOPO@ = @GPUS@ GPUs = @NODES@ node --
# NVIDIA's native Slurm topology (SLURM_NODES=1).
# GENERATED by ../../generate_point_recipes.py -- do not hand-edit.
#
# PREREQUISITES (../../README.md, and ../../../../SLURM_TO_GKE.md section 7):
#   1. @NODES@ schedulable A4X-max node (counted below before launching).
#   2. kubectl apply -f ../../stage-assets.yaml     # model + dataset, EVERY node
#   3. kubectl apply -f ../../mk-report-dir.yaml    # report + runtime hostPaths
#   4. Secret `nv-gitlab-pull` created in target namespace for registry.gitlab.com
#   5. kubectl apply -f ../../prepull-images.yaml   # recommended
#   6. kubectl context pointing at the target cluster.
#
# Site overrides:
#   NAMESPACE=<ns>  NODEPOOL=<a4x-max nodepool>   (env vars)
#   SKIP_NODE_CHECK=1                             skip the node count below
#   anything else via sflow flags, e.g.
#     ./launch_test.sh -s VLLM_IMAGE=<image> -s CLIENT_IMAGE=<mirror>
#     ./launch_test.sh -s MODE=both -s SWEBENCH_SERVICE_ADDR=<host:port>
#     ./launch_test.sh --dry-run
set -euo pipefail

cd "$(dirname "$0")"

# sflow is normally installed in a venv rather than on PATH.
SFLOW_VENV="${SFLOW_VENV:<workdir>/sflow_workspace/.venv}"
if [ -x "${SFLOW_VENV}/bin/sflow" ]; then
  PATH="${SFLOW_VENV}/bin:${PATH}"
fi
if ! command -v sflow >/dev/null 2>&1; then
  echo "sflow not found. Set SFLOW_VENV=/path/to/venv or install sflow." >&2
  exit 1
fi

RECIPE="@RECIPE@"
NAMESPACE="${NAMESPACE:-default}"
NODEPOOL="${NODEPOOL:<nodepool>}"
REQUIRED_NODES=@NODES@

# Fail fast instead of letting the reservation sit for its 1800 s timeout:
# count Ready, uncordoned nodes in the pool that advertise 4 GPUs.
if [ "${SKIP_NODE_CHECK:-0}" != "1" ]; then
  if ! command -v kubectl >/dev/null 2>&1; then
    echo "kubectl not found; install it or set SKIP_NODE_CHECK=1." >&2
    exit 1
  fi
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

NODE_SELECTOR="${NODE_SELECTOR:-cloud.google.com/gke-nodepool=${NODEPOOL}}"
sflow run -f "${RECIPE}" \
  --kube-namespace "${NAMESPACE}" \
  --kube-node-selector "${NODE_SELECTOR}" \
  "$@"
"""


def recipe_name(point: int) -> str:
    return f"deepseek_v41_flash_gke_vllm_point{point}.yaml"


def point_dir(dest: pathlib.Path, point: int, info: dict) -> pathlib.Path:
    return dest / f"{info['nodes']}nodes" / f"point_{point}"


def render_launch(point: int, info: dict) -> str:
    v = info["vars"]
    return (LAUNCH_TEMPLATE
            .replace("@POINT@", str(point))
            .replace("@TOPO@", f"{v['NUM_WORKERS']} x TP{info['tp']} (DEP4)")
            .replace("@GPUS@", str(v["TOTAL_GPUS"]))
            .replace("@NODES@", str(info["nodes"]))
            .replace("@RECIPE@", recipe_name(point)))


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
class Checker:
    def __init__(self) -> None:
        self.failures = 0
        self.passed = 0

    def ok(self, cond: bool, label: str, detail: str = "") -> bool:
        if cond:
            self.passed += 1
            print(f"[ ok ] {label}")
        else:
            self.failures += 1
            print(f"[FAIL] {label}" + (f"\n        {detail}" if detail else ""))
        return cond

    def same(self, label: str, got: str, want: str) -> None:
        if not self.ok(got == want, label):
            for ln in list(difflib.unified_diff(want.splitlines(), got.splitlines(),
                                                "drop", "gke", lineterm=""))[:30]:
                print("        " + ln)


def bash_syntax_ok(script: str) -> tuple[bool, str]:
    s = re.sub(r"\$\{\{.*?\}\}", "SFLOWEXPR", script)
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(s)
        name = f.name
    r = subprocess.run(["bash", "-n", name], capture_output=True, text=True)
    pathlib.Path(name).unlink()
    return r.returncode == 0, r.stderr.strip()


def check(dest: pathlib.Path, points: dict, shared: dict, ported: dict, base: str) -> int:
    ck = Checker()
    t = shared["tasks"]
    for point, info in points.items():
        L = f"point_{point}"
        pdir = point_dir(dest, point, info)
        path = pdir / recipe_name(point)
        if not ck.ok(path.is_file(), f"{L}: {path.relative_to(dest)} exists (else MISSING)"):
            continue
        text = path.read_text()
        want_text = render(point, info, shared, ported, points, base)
        ck.ok(text == want_text, f"{L}: recipe matches a fresh render (else DIFFERS: hand-edited or stale)")
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            ck.ok(False, f"{L}: recipe parses as YAML", str(exc).replace("\n", " "))
            continue
        arts = {a["name"]: a for a in doc["artifacts"]}
        varz = doc["variables"]
        # verbatim artifacts
        ck.same(f"{L}: server-env.yaml byte-identical", arts["SERVER_ENV_YAML"]["content"], info["server_env"])
        ck.same(f"{L}: dynamo_env.yaml byte-identical", arts["DYNAMO_ENV_YAML"]["content"], info["dynamo_env"])
        ck.same(f"{L}: endpoint_env.yaml byte-identical", arts["ENDPOINT_ENV_YAML"]["content"], info["endpoint_env"])
        ck.same(f"{L}: dsv41 tools/run_with_env.py byte-identical",
                arts["DSV41_RUN_WITH_ENV"]["content"], shared["dsv41_run_with_env"])
        ck.same(f"{L}: dsv41 tools/cpuset.py byte-identical",
                arts["CPUSET_PY"]["content"], shared["cpuset_py"])
        ck.same(f"{L}: sflow tools/run_with_env.py byte-identical",
                arts["RUN_WITH_ENV"]["content"], shared["run_with_env"])
        ck.same(f"{L}: generate_endpoint_yaml.py byte-identical",
                arts["GENERATE_ENDPOINT_YAML"]["content"], shared["gen_endpoint"])
        # client.yaml: exactly the documented rewrites
        got_client = arts["CLIENT_CONFIG"]["content"]
        want_client, notes = rewrite_client(info["client_text"], point)
        ck.same(f"{L}: client.yaml = drop + documented rewrites only ({'; '.join(notes)})",
                got_client, want_client)
        diff = [d for d in difflib.ndiff(info["client_text"].splitlines(), got_client.splitlines())
                if d[:1] in "+-"]
        # report_dir is one replaced line (one '-', one '+'); an added key is one '+'.
        ck.ok(len(diff) == 2 + (len(notes) - 1),
              f"{L}: client.yaml differs from the drop in exactly {len(notes)} place(s)", repr(diff))
        cc = yaml.safe_load(got_client)
        ck.ok(cc["report_dir"] == "${REPORT_DIR}", f"{L}: client report_dir is ${{REPORT_DIR}}")
        ck.ok("seed" not in cc["model_params"], f"{L}: client model_params has no seed (matches v1.0.3 drop)")
        ck.ok(cc["settings"].get("metrics_tokenizer_workers") ==
              (info["client"]["settings"].get("metrics_tokenizer_workers") or CLIENT_TOKENIZER_DEFAULT)
              and "metrics_tokenizer_workers" not in (cc["settings"].get("client") or {}),
              f"{L}: settings.metrics_tokenizer_workers is a settings key with the recorded value")
        ck.ok(cc["endpoint_config"]["endpoints"] == [], f"{L}: endpoints [] (resolved at run time)")
        for k in ("submission_ref", "datasets", "settings", "model_params", "endpoint_config"):
            ck.ok(k in cc, f"{L}: client keeps '{k}'")
        # point variables: raw bytes and values
        for k in POINT_VARS:
            ck.ok(info["raw_blocks"][k] in text, f"{L}: variable block {k} byte-identical to the drop")
            ck.ok(varz[k]["value"] == info["vars"][k], f"{L}: variable {k} value == drop")
        ck.ok(shared["gpus_per_worker_block"] in text, f"{L}: GPUS_PER_WORKER block byte-identical")
        ck.ok(varz["MODE"]["value"] == "perf", f"{L}: MODE default perf (drop: {info['vars']['MODE']})")
        # topology
        topo = (info["vars"]["NUM_WORKERS"], info["tp"], info["dp"], info["ep"],
                info["vars"]["DATASET_REPEATS"], info["trajectories"], info["vars"]["MODE"],
                info["max_batched_tokens"], info["max_cudagraph"],
                info["client"]["model_params"]["max_new_tokens"],
                info["client"]["settings"].get("metrics_tokenizer_workers"))
        ck.ok(topo == EXPECTED[point], f"{L}: topology == EXPECTED {EXPECTED[point]}")
        ck.ok(info["nodes"] == shared["slurm_nodes"] == 1 and pdir.parent.name == "1nodes",
              f"{L}: nodes = ceil({info['vars']['TOTAL_GPUS']}/4) = {info['nodes']} == SLURM_NODES, in 1nodes/")
        ck.ok(doc["backends"][0]["nodes"] == "${{ variables.TOTAL_NODES }}", f"{L}: backend nodes = TOTAL_NODES")
        be = doc["backends"][0]
        ck.ok(be.get("rdma") == "disable" and be.get("compute_domain") == {"create": True}
              and be.get("scheduling") == "device_plugin" and be.get("host_network") is True
              and be.get("host_ipc") is True,
              f"{L}: backend rdma=disable, compute_domain {{create: true}}, device_plugin, host_network, host_ipc")
        # operators
        ops = {o["name"]: o for o in doc["operators"]}
        ck.ok(ops["vllm_container"].get("image_pull_secrets") == ["${{ variables.IMAGE_PULL_SECRET }}"]
              and ops["vllm_container"].get("security_context") == {"privileged": True}
              and varz["IMAGE_PULL_SECRET"]["value"] == DEFAULT_IMAGE_PULL_SECRET,
              f"{L}: vllm_container uses IMAGE_PULL_SECRET ({DEFAULT_IMAGE_PULL_SECRET}) and privileged: true")
        # scripts
        tasks = {x["name"]: x for x in doc["workflow"]["tasks"]}
        ck.ok(sorted(tasks) == ["dynamo_frontend", "endpoint_client", "vllm_worker"],
              f"{L}: tasks = frontend/worker/client (host_prepare and swebench_service not ported)")
        for name, key, subs in (("dynamo_frontend", "frontend", FRONTEND_SUBS),
                                ("vllm_worker", "worker", WORKER_SUBS)):
            got = tasks[name]["script"][-1]
            ck.same(f"{L}: {name} script = drop with anchored substitutions", got, ported[key])
            ck.same(f"{L}: {name} script reverse-substituted == drop byte-for-byte",
                    reverse_subs(got, subs), t[name]["script"][0])
        items = [str(s) for s in tasks["endpoint_client"]["script"]]
        ck.ok(items.count(ported["client"]) == 1 and items[-2] == ported["client"],
              f"{L}: endpoint_client carries the drop's script item once before report mirroring (no exec)")
        ck.same(f"{L}: endpoint_client script reverse-substituted == drop byte-for-byte",
                reverse_subs(items[-2], CLIENT_SUBS), t["endpoint_client"]["script"][0])
        ck.ok(not any(s.lstrip().startswith("exec ") for s in items), f"{L}: no exec in endpoint_client (F29)")
        for name in ("dynamo_frontend", "vllm_worker", "endpoint_client"):
            ck.ok(tasks[name].get("probes") == t[name].get("probes"), f"{L}: {name} probes == drop")
        # every script item is a string and parses with bash -n
        all_ok, errs = True, []
        for name, task in tasks.items():
            for j, s in enumerate(task["script"]):
                if not isinstance(s, str):
                    all_ok = False
                    errs.append(f"{name}[{j}] is {type(s).__name__} (F17)")
        for name, task in tasks.items():
            good, err = bash_syntax_ok("\n".join(s for s in task["script"] if isinstance(s, str)))
            if not good:
                all_ok = False
                errs.append(f"{name}: bash -n: {err}")
        ck.ok(all_ok, f"{L}: every script item is a string and passes bash -n", "; ".join(errs))
        # images / ports
        ck.ok(varz["VLLM_IMAGE"]["value"] == shared["vllm_image"], f"{L}: VLLM_IMAGE = drop VLLM_CONTAINER_IMAGE")
        ck.ok(varz["CLIENT_IMAGE"]["value"] == shared["client_image"], f"{L}: CLIENT_IMAGE = drop ENDPOINT_CONTAINER_IMAGE")
        # launch_test.sh
        launch = pdir / "launch_test.sh"
        ck.ok(launch.is_file() and launch.read_text() == render_launch(point, info),
              f"{L}: launch_test.sh matches the generator (REQUIRED_NODES={info['nodes']})")
        ck.ok(launch.is_file() and bool(launch.stat().st_mode & 0o111), f"{L}: launch_test.sh is executable")

    # --- cross-file consistency ---------------------------------------------
    stage = (dest / "stage-assets.yaml")
    st = stage.read_text() if stage.is_file() else ""
    ck.ok(f"hf://{shared['model_repo']}@{shared['model_rev']}" in st,
          "stage-assets.yaml: SRC_MODEL is the README's repo@revision")
    ck.ok(shared["dataset_sha"] in st, "stage-assets.yaml: DATASET_SHA256 matches dataset/README.md")
    ck.ok("STAGED: node ready" in st and ".sflow-staged" in st and "*.incomplete" in st
          and "swe_bench_verified_test.parquet" in st,
          "stage-assets.yaml: marker, *.incomplete cleanup, swe_bench_verified_test.parquet, final 'STAGED: node ready'")
    ck.ok("<node-local-ssd>/model-cache" in st
          and "agentic_dataset/perf/mlperf_agentic_inference_dataset.jsonl" in st,
          "stage-assets.yaml: stages to the recipe's MODEL_PATH / DATA_PATH layout")
    mk = (dest / "mk-report-dir.yaml")
    mkt = mk.read_text() if mk.is_file() else ""
    ck.ok("dsv41-flash-reports" in mkt and "dsv41-flash-runtime" in mkt and "1777" in mkt,
          "mk-report-dir.yaml: creates REPORTS_ROOT and RUNTIME_ROOT_HOST (1777)")
    pp = (dest / "prepull-images.yaml")
    ppt = pp.read_text() if pp.is_file() else ""
    ck.ok(shared["vllm_image"] in ppt and shared["client_image"] in ppt
          and DEFAULT_IMAGE_PULL_SECRET in ppt,
          f"prepull-images.yaml: pulls VLLM_IMAGE and CLIENT_IMAGE with {DEFAULT_IMAGE_PULL_SECRET}")
    for f in ("stage-assets.yaml", "mk-report-dir.yaml", "prepull-images.yaml"):
        p = dest / f
        if p.is_file():
            docs = list(yaml.safe_load_all(p.read_text()))
            specs = [d["spec"]["template"]["spec"] for d in docs if d and d.get("kind") == "DaemonSet"]
            ck.ok(bool(specs) and all(any(tl.get("value") == "arm64" for tl in s.get("tolerations", []))
                                      for s in specs), f"{f}: DaemonSet tolerates arm64 and parses")
    rd = dest / "README.md"
    ck.ok(rd.is_file() and "../../SLURM_TO_GKE.md" in rd.read_text(), "README.md links ../../SLURM_TO_GKE.md")
    print()
    print(f"{ck.passed} checks passed, {ck.failures} failed")
    return ck.failures


def print_table(points: dict) -> None:
    print("\n".join(fmt_table(points)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slurm-root", type=pathlib.Path, default=DEFAULT_DROP_ROOT,
                    help="root of the v1.0.3 drop (default: %(default)s)")
    ap.add_argument("--dest", type=pathlib.Path, default=HERE,
                    help="model directory to write/check (default: this directory)")
    ap.add_argument("--check", action="store_true", help="verify files on disk against the drop")
    ap.add_argument("--table", action="store_true", help="print the per-point table and exit")
    args = ap.parse_args()

    drop = args.slurm_root
    if not (drop / CONFIG_REL).is_dir():
        fail(f"drop not found at {drop / CONFIG_REL}; pass --slurm-root")
    shared = read_shared(drop)
    found = sorted(int(d.name.split("_")[1]) for d in (drop / CONFIG_REL).glob("point_*"))
    if found != DROP_POINTS:
        fail(f"drop points {found} != {DROP_POINTS}")
    points = {}
    for p in POINTS:
        info = read_point(drop, p)
        assert_point(p, info, shared)
        points[p] = info
    print(f"read {len(DROP_POINTS)} drop points -> {len(points)} points total")
    print(f"EXPECTED table: OK ({len(points)}/{len(POINTS)} points match; every point = "
          f"{shared['slurm_nodes']} node = SLURM_NODES)")
    ported = port_scripts(shared)
    base = (HERE / "recipe_base.yaml").read_text()

    if args.table:
        print_table(points)
        return 0
    if args.check:
        failures = check(args.dest.resolve(), points, shared, ported, base)
        if failures:
            print(f"CHECK FAILED: {failures} problem(s)")
            return 1
        print(f"CHECK PASSED: all {len(points)} recipes match the drop "
              f"(verbatim artifacts, anchored scripts, documented client rewrites)")
        return 0

    for p, info in points.items():
        pdir = point_dir(args.dest, p, info)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / recipe_name(p)).write_text(render(p, info, shared, ported, points, base))
        launch = pdir / "launch_test.sh"
        launch.write_text(render_launch(p, info))
        launch.chmod(0o755)
        print(f"wrote {(pdir / recipe_name(p)).relative_to(args.dest)}  and  "
              f"{launch.relative_to(args.dest)}")
    print()
    print_table(points)
    return 0


if __name__ == "__main__":
    sys.exit(main())
