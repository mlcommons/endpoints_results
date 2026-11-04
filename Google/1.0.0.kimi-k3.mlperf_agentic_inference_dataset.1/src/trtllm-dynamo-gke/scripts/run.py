#!/usr/bin/env python3
"""Launch any Kimi point through one SFlow workflow with measured fabric placement."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import yaml
from collections import defaultdict
from itertools import combinations
import copy
import shlex


def plan_placement(
    nodes, reports, num_ctx, ctx_gpus, num_gen, gen_gpus, segment_nodes, dwdp=False
):
    if len(nodes) != len(set(nodes)) or set(nodes) != set(reports):
        raise ValueError("Require unique nodes and exactly matching fabric reports")
    if segment_nodes < 1 or len(nodes) % segment_nodes:
        raise ValueError("Reserved node count must be a multiple of segment size")
    if min(num_ctx, ctx_gpus, num_gen, gen_gpus) < 1:
        raise ValueError("Invalid serving topology")
    total = num_ctx * ctx_gpus + num_gen * gen_gpus
    if total > 48 or total > len(nodes) * 4:
        raise ValueError("Serving GPU budget exceeded")
    if dwdp and ctx_gpus != 1:
        raise ValueError("DWDP grouping expects one GPU per rank")
    pools = defaultdict(list)
    for i, node in enumerate(nodes):
        report = reports[node]
        identity = report.get("fabric_identity")
        if report.get("gpus") != 4 or report.get("healthy") is not True or not identity:
            raise ValueError("Missing or unhealthy measured fabric: " + node)
        pools[tuple(identity)].extend((i, gpu) for gpu in range(4))
    groups = (
        [("dwdp", num_ctx)]
        if dwdp
        else [(f"prefill_{i}", ctx_gpus) for i in range(num_ctx)]
    )
    groups += [(f"decode_{i}", gen_gpus) for i in range(num_gen)]
    # Largest units first prevents small workers consuming the only large rack.
    groups.sort(key=lambda pair: (-pair[1], pair[0]))

    def place(index, available, assigned):
        if index == len(groups):
            return assigned
        name, count = groups[index]
        for domain, slots in sorted(available.items()):
            if len(slots) < count:
                continue
            by_node = defaultdict(list)
            for slot in slots:
                by_node[slot[0]].append(slot)
            # Multi-node ranks use full nodes; single-node workers may share.
            if count > 4:
                if count % 4:
                    raise ValueError("Multi-node workers require whole nodes")
                candidates = [n for n, s in by_node.items() if len(s) == 4]
                selections = (
                    sum((by_node[n] for n in ns), [])
                    for ns in combinations(candidates, count // 4)
                )
            else:
                selections = (s[:count] for s in by_node.values() if len(s) >= count)
            for chosen in selections:
                remaining = dict(available)
                used = set(chosen)
                remaining[domain] = [s for s in slots if s not in used]
                result = place(
                    index + 1, remaining, {**assigned, name: (domain, chosen)}
                )
                if result is not None:
                    return result
        return None

    assigned = place(0, dict(pools), {})
    if assigned is None:
        raise ValueError(
            "Allocation cannot keep every communicating worker on one fabric"
        )
    if dwdp:
        domain, slots = assigned.pop("dwdp")
        for rank, slot in enumerate(slots):
            assigned[f"prefill_{rank}"] = (domain, [slot])
    workers = {}
    used = set()
    for name, (domain, slots) in sorted(assigned.items()):
        used.update(slots)
        workers[name] = {
            "fabric_identity": list(domain),
            "slots": [{"node_index": n, "node": nodes[n], "gpu": g} for n, g in slots],
        }
    return {
        "segmented": True,
        "segment_nodes": segment_nodes,
        "active_gpus": total,
        "reserved_gpus": len(nodes) * 4,
        "idle_gpus": len(nodes) * 4 - total,
        "workers": workers,
        "idle_nodes": [
            node
            for i, node in enumerate(nodes)
            if all((i, gpu) not in used for gpu in range(4))
        ],
    }


def build(config, nodes, reports, segment_nodes, validator, placement_path):
    cfg = copy.deepcopy(config)
    variables = cfg["variables"]
    if isinstance(variables, list):
        variables = {v["name"]: v for v in variables}
    workflow_variables = cfg["workflow"]["variables"]
    if isinstance(workflow_variables, list):
        workflow_variables = {v["name"]: v for v in workflow_variables}

    def value(name):
        return variables[name]["value"]

    placement = plan_placement(
        nodes,
        reports,
        int(value("NUM_CTX")),
        int(value("CTX_GPUS")),
        int(value("NUM_GEN")),
        int(value("GEN_GPUS")),
        segment_nodes,
        dwdp=int(value("CTX_GPUS")) == 1,
    )
    assert placement["active_gpus"] == int(value("TOTAL_GPUS"))
    variables["TOTAL_NODES"]["value"] = len(nodes)
    placement["nodes"] = nodes
    placement["reports"] = reports
    placement["dwdp"] = int(value("CTX_GPUS")) == 1
    # The DWDP rendezvous must follow its actual rank-zero placement.
    master = placement["workers"]["prefill_0"]["slots"][0]["node_index"]
    workflow_variables["HEAD_NODE_IP"]["value"] = (
        "${{ backends.cluster.nodes[" + str(master) + "].ip_address }}"
    )
    # Opt in to colocating the HTTP frontend and client with an actual decode
    # worker; allocation indices alone do not identify the serving role.
    frontend_placement = variables.get("DYNAMO_FRONTEND_PLACEMENT", {}).get("value")
    frontend_node = None
    if frontend_placement is not None:
        if frontend_placement != "decode":
            raise ValueError("DYNAMO_FRONTEND_PLACEMENT must be decode when set")
        decode_nodes = {
            slot["node_index"]
            for name, worker in placement["workers"].items()
            if name.startswith("decode_")
            for slot in worker["slots"]
        }
        prefill_nodes = {
            slot["node_index"]
            for name, worker in placement["workers"].items()
            if name.startswith("prefill_")
            for slot in worker["slots"]
        }
        if not decode_nodes or decode_nodes & prefill_nodes:
            raise ValueError("Decode frontend placement requires separate worker hosts")
        frontend_node = min(decode_nodes)
        for name in ("FRONTEND_NODE", "CLIENT_NODE"):
            variables[name]["value"] = frontend_node
        workflow_variables["FRONTEND_NODE_IP"]["value"] = (
            "${{ backends.cluster.nodes[" + str(frontend_node) + "].ip_address }}"
        )
        placement["frontend_client_placement"] = {
            "node_index": frontend_node,
            "node": nodes[frontend_node],
            "basis": "First actual decode-worker node after measured fabric placement",
            "prefill_colocation": False,
        }
    command = (
        "python3 " + shlex.quote(validator) + " check " + shlex.quote(placement_path)
    )
    # SFlow mounts declared artifacts; arbitrary host paths are not visible in
    # server containers merely because the host-side preflight can read them.
    cfg["artifacts"].extend(
        [
            {"name": "SEGMENTED_TOOLS", "uri": "fs://" + str(Path(validator).parent)},
            {
                "name": "SEGMENTED_PLAN",
                "uri": "fs://" + str(Path(placement_path).parent),
            },
        ]
    )
    tasks = []
    names = list(placement["workers"])
    for original in cfg["workflow"]["tasks"]:
        task = copy.deepcopy(original)
        if frontend_node is not None and task["name"] in (
            "frontend",
            "benchmark",
            "client_gate",
        ):
            task["resources"]["nodes"] = {"indices": [frontend_node]}
        if task["name"] in ("prefill", "decode"):
            role = task["name"]
            for name in sorted(n for n in names if n.startswith(role + "_")):
                worker = copy.deepcopy(task)
                worker.pop("replicas", None)
                worker["name"] = name
                rank = int(name.rsplit("_", 1)[1])
                slots = placement["workers"][name]["slots"]
                selected_nodes = sorted({s["node_index"] for s in slots})
                gpu_lists = [
                    {s["gpu"] for s in slots if s["node_index"] == n}
                    for n in selected_nodes
                ]
                assert all(gpus == gpu_lists[0] for gpus in gpu_lists)
                worker["resources"] = {
                    "nodes": {"indices": selected_nodes},
                    "gpus": {"count": len(slots), "indices": sorted(gpu_lists[0])},
                }
                worker["script"] = [
                    worker["script"][0],
                    "export SFLOW_REPLICA_INDEX=" + str(rank),
                    command + " worker " + shlex.quote(name),
                    *worker["script"][1:],
                ]
                tasks.append(worker)
        elif task["name"] == "preflight":
            task["script"] = [
                line.replace("-eq", "-le") if "TOTAL_NODES * 4" in line else line
                for line in task["script"]
            ]
            tasks.append(task)
        else:
            if task["name"] == "control_plane":
                task["resources"] = {"nodes": {"indices": [master]}}
            if task["name"] == "frontend":
                task["depends_on"] = names
            tasks.append(task)
    cfg["workflow"]["tasks"] = tasks
    return cfg, placement


def compose(point, env_file, overrides):
    from sflow.cli.compose import _compose_files

    recipe = Path(__file__).resolve().parents[1]
    files = [
        recipe / "slurm_env.yaml",
        recipe / "setup.yaml",
        recipe / "disagg.yaml",
        point,
    ]
    if env_file.resolve() != files[0].resolve():
        files.append(env_file)
    return yaml.safe_load(
        _compose_files(
            files, ["REPO_ROOT=" + str(recipe.parents[2]), *overrides], None, "ERROR"
        )
    )


def values(config):
    variables = config["variables"]
    return (
        {v["name"]: v["value"] for v in variables}
        if isinstance(variables, list)
        else {k: v["value"] for k, v in variables.items()}
    )


def validate_required(config):
    if any(task["name"] == "swe_service" for task in config["workflow"]["tasks"]):
        registry = values(config).get("SWE_IMAGE_REGISTRY")
        if not registry or not str(registry).strip():
            raise ValueError(
                "SWE_IMAGE_REGISTRY is required for SWE-bench; specify it through --env or --set"
            )


def submit(config, args, output):
    v = values(config)
    # Reserve a unique attempt directory before submitting anything.
    output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.point.resolve()),
        "--env",
        str(args.env.resolve()),
        "--output",
        str(output),
    ]
    for override in args.set:
        command += ["--set", override]
    directives = [
        f"--account={v['SLURM_ACCOUNT']}",
        f"--partition={v['SLURM_PARTITION']}",
        f"--time={v['WALL_TIME']}",
        f"--nodes={v['TOTAL_NODES']}",
        "--exclusive",
        f"--segment={v['SEGMENT_NODES']}",
    ]
    directives += list(v.get("SLURM_EXTRA_ARGS", []))
    if v.get("SLURM_QOS"):
        directives.append("--qos=" + str(v["SLURM_QOS"]))
    if any("\n" in str(value) for value in directives):
        raise ValueError("Slurm options must not contain newlines")
    script = output / "run.sbatch"
    script.write_text(
        "#!/bin/bash\n"
        + "\n".join("#SBATCH " + option for option in directives)
        + "\n#SBATCH --output="
        + shlex.quote(str(output / "slurm-%j.log"))
        + "\nset -euo pipefail\nexec "
        + shlex.join(command)
        + "\n"
    )
    subprocess.run(["sbatch", "--parsable", str(script)], check=True)


def run_workflow(command, srun, output, memory_clock, restore_range):
    """Monitor optional clocks and always stop children before restoring them."""
    node = str(Path(__file__).with_name("node.py"))
    clock_command = srun + [sys.executable, node]
    options = ["--memory-clock", str(memory_clock), "--restore-range", restore_range]
    children = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")

    for sig in previous:
        signal.signal(sig, interrupted)
    try:
        monitor = None
        if memory_clock:
            subprocess.run(
                clock_command + ["pin", str(output / "clocks"), *options],
                check=True,
                timeout=120,
            )
            monitor = subprocess.Popen(
                clock_command + ["monitor", str(output / "clocks"), *options],
                start_new_session=True,
            )
            children.append(monitor)
        run = subprocess.Popen(command, start_new_session=True)
        children.append(run)
        while run.poll() is None:
            if monitor is not None and monitor.poll() is not None:
                raise RuntimeError("Clock monitoring failed; inspect the saved samples")
            try:
                run.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        (output / "exit_code").write_text(str(run.returncode) + "\n")
        if run.returncode:
            raise subprocess.CalledProcessError(run.returncode, command)
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            for child in reversed(children):
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=10)
        finally:
            try:
                if memory_clock:
                    subprocess.run(
                        clock_command + ["restore", str(output / "clocks"), *options],
                        check=True,
                        timeout=120,
                    )
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)


def main():
    recipe = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("point", type=Path)
    parser.add_argument("--env", type=Path, default=recipe / "slurm_env.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit a batch job; otherwise use the current allocation",
    )
    args = parser.parse_args()
    cfg = compose(args.point.resolve(), args.env.resolve(), args.set)
    validate_required(cfg)
    output = args.output.resolve()
    if args.submit:
        submit(cfg, args, output)
        return
    job = os.environ["SLURM_JOB_ID"]
    raw = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
    fields = dict(v.split("=", 1) for v in raw.split() if "=" in v)
    nodes = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True
    ).splitlines()
    v = values(cfg)
    if (
        len(nodes) != int(v["TOTAL_NODES"])
        or fields["Partition"] != v["SLURM_PARTITION"]
    ):
        raise ValueError(
            "Allocation must match the configured node count and partition"
        )
    if int(fields["SegmentSize"]) != int(v["SEGMENT_NODES"]):
        raise ValueError("Allocation segment size differs from the point configuration")
    output.mkdir(parents=True, exist_ok=True)
    (output / "workspaces").mkdir()  # Exclusive: never reuse an earlier attempt.
    (output / "client-results").mkdir()
    cfg = compose(
        args.point.resolve(),
        args.env.resolve(),
        [
            *args.set,
            "RESULTS_PATH=" + str(output / "client-results"),
            "SLURM_ACCOUNT=" + fields["Account"],
        ],
    )
    v = values(cfg)
    srun = [
        "srun",
        "--jobid=" + job,
        "--overlap",
        "-N",
        str(len(nodes)),
        "-n",
        str(len(nodes)),
        "--ntasks-per-node=1",
        "--cpus-per-task=1",
    ]
    node = str(recipe / "scripts/node.py")
    setup = v["ALLOCATION_SETUP"]
    cleanup = v["ALLOCATION_CLEANUP"]
    if set(setup) - {"pin-clocks"} or set(cleanup) - {"restore-clocks"}:
        raise ValueError("Unknown allocation setup/cleanup step")
    if ("pin-clocks" in setup) != bool(int(v["MEMORY_CLOCK_MHZ"])) or (
        "restore-clocks" in cleanup
    ) != ("pin-clocks" in setup):
        raise ValueError("Clock pinning must include monitoring and restoration")
    user = subprocess.check_output(["id", "-un"], text=True).strip()
    cache = Path(v["CONTAINER_SCRATCH"]) / f"{user}-k3-dynamo-{job}"
    os.environ["ENROOT_DATA_PATH"] = str(cache / "data")
    os.environ["ENROOT_TEMP_PATH"] = str(cache / "tmp")
    cache.joinpath("data").mkdir(parents=True, exist_ok=True)
    cache.joinpath("tmp").mkdir(parents=True, exist_ok=True)
    subprocess.run(
        srun + [sys.executable, node, "probe", str(output / "fabric")],
        check=True,
        timeout=120,
    )
    reports = {
        n: json.loads((output / "fabric" / (n + ".json")).read_text()) for n in nodes
    }
    workflow, placement = build(
        cfg,
        nodes,
        reports,
        int(v["SEGMENT_NODES"]),
        node,
        str(output / "placement.json"),
    )
    (output / "placement.json").write_text(json.dumps(placement, indent=2) + "\n")
    flow = output / "workflow.yaml"
    flow.write_text(yaml.safe_dump(workflow, sort_keys=False))
    command = [
        "sflow",
        "run",
        "-f",
        str(flow),
        "--output-dir",
        str(output / "output"),
        "--workspace-dir",
        str(output / "workspaces"),
    ]
    run_workflow(
        command,
        srun,
        output,
        int(v["MEMORY_CLOCK_MHZ"]),
        str(v["MEMORY_CLOCK_RESTORE"]),
    )


if __name__ == "__main__":
    main()
