#!/usr/bin/env python3
"""Pin four Dynamo/vLLM workers and CPU roles to disjoint core sets.

Inspect the CLI without changing any process affinity:
    python3 ci/engine/vllm/dynamo_cpu_affinity.py --help

Within a four-GPU Slurm allocation, pin an existing frontend process:
    python3 ci/engine/vllm/dynamo_cpu_affinity.py --role frontend --pid <PID>
"""
import argparse
import csv
import glob
import json
import os
import subprocess
from pathlib import Path


def die(message):
    raise SystemExit(f"dynamo CPU affinity: {message}")


def cpus(value):
    result = set()
    value = value.strip()
    if not value:
        return result
    for part in value.split(","):
        first, separator, last = part.partition("-")
        if not first or (separator and not last):
            die(f"invalid CPU list {value!r}")
        try:
            start, end = int(first), int(last or first)
        except ValueError:
            die(f"invalid CPU list {value!r}")
        if end < start:
            die(f"invalid CPU list {value!r}")
        result.update(range(start, end + 1))
    return result


def rendered(value):
    return ",".join(map(str, sorted(value)))


def bdf(value):
    domain, separator, address = value.strip().lower().partition(":")
    if not separator:
        die(f"invalid PCI address {value!r}")
    return f"{int(domain, 16):04x}:{address}"


def gpu_devices():
    result = []
    for name in glob.glob("/sys/bus/pci/devices/*"):
        device = Path(name)
        if (device / "vendor").read_text().strip() != "0x10de" or not (device / "class").read_text().startswith("0x03"):
            continue
        numa = int((device / "numa_node").read_text())
        result.append({"pci": device.name.lower(), "numa": numa})
    return sorted(result, key=lambda gpu: gpu["pci"])


def selected_devices(devices, pci_list):
    if pci_list:
        selected_pci = [bdf(value) for value in pci_list.split(",") if value.strip()]
        if len(selected_pci) != 4 or len(set(selected_pci)) != 4:
            die("--worker-gpu-pci must contain four distinct PCI addresses")
        selected = [gpu for gpu in devices if gpu["pci"] in selected_pci]
        if len(selected) != 4:
            die("--worker-gpu-pci includes a GPU not visible in sysfs")
    elif len(devices) == 4:
        selected = devices
    else:
        die("host has more or fewer than four GPUs; set --worker-gpu-pci explicitly")
    if any(gpu["numa"] < 0 for gpu in selected):
        die("a selected GPU has no NUMA node")
    return sorted(selected, key=lambda gpu: gpu["pci"])


def physical_cores(allowed):
    groups = {}
    for cpu in allowed:
        siblings = frozenset(cpus(Path(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        ).read_text()))
        groups[siblings] = frozenset(siblings & allowed)
    return sorted(groups.values(), key=lambda group: min(group))


def cores_by_numa(cores, allowed):
    cpu_nodes = {}
    for path in glob.glob("/sys/devices/system/node/node*/cpulist"):
        node = int(Path(path).parent.name.removeprefix("node"))
        for cpu in cpus(Path(path).read_text()) & allowed:
            cpu_nodes[cpu] = node
    if set(cpu_nodes) != allowed:
        die("the Slurm cpuset includes CPUs without a NUMA node")
    result = {}
    for core in cores:
        nodes = {cpu_nodes[cpu] for cpu in core}
        if len(nodes) != 1:
            die(f"hardware thread siblings cross NUMA nodes: {rendered(core)}")
        result.setdefault(nodes.pop(), []).append(core)
    return result


def reserve(free, local_gpu_counts, count, role):
    candidates = [node for node in free
                  if len(free[node]) - count >= local_gpu_counts[node]]
    if not candidates:
        die(f"no GPU-local NUMA node has {count} physical cores for {role}")
    node = max(candidates, key=lambda item: (len(free[item]) / local_gpu_counts[item], -item))
    return frozenset().union(*[free[node].pop() for _ in range(count)])


def allocation(devices, node_cores, frontend_cores, client_cores):
    free = {node: list(cores) for node, cores in node_cores.items()}
    local_gpu_counts = {node: sum(gpu["numa"] == node for gpu in devices) for node in free}
    free = {node: cores for node, cores in free.items() if local_gpu_counts[node]}
    plan, worker_counts = {"worker": {}}, {}
    plan["frontend"] = reserve(free, local_gpu_counts, frontend_cores, "frontend")
    plan["client"] = reserve(free, local_gpu_counts, client_cores, "client")
    for node in sorted({gpu["numa"] for gpu in devices}):
        local = [gpu for gpu in devices if gpu["numa"] == node]
        count, extra = divmod(len(free.get(node, [])), len(local))
        if count < 1:
            die(f"NUMA node {node} has fewer physical cores than local GPUs")
        for position, gpu in enumerate(local):
            worker_count = count + (position < extra)
            plan["worker"][gpu["pci"]] = frozenset().union(
                *[free[node].pop(0) for _ in range(worker_count)]
            )
            worker_counts[gpu["pci"]] = worker_count
    masks = list(plan["worker"].values()) + [plan["frontend"], plan["client"]]
    if any(not mask for mask in masks) or sum(map(len, masks)) != len(set().union(*masks)):
        die("worker, frontend, and client CPU sets overlap")
    return plan, worker_counts


def worker_gpu(devices, cuda_visible):
    values = [value.strip() for value in cuda_visible.split(",") if value.strip()]
    if len(values) != 1:
        die(f"worker requires one CUDA_VISIBLE_DEVICES entry, got {cuda_visible!r}")
    rows = csv.reader(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"], text=True).splitlines())
    visible = []
    for index, uuid, pci in rows:
        visible.append({"index": int(index), "uuid": uuid.strip(), "pci": bdf(pci)})
    if len(visible) == 1:
        candidate = visible[0]
    else:
        matches = [gpu for gpu in visible if str(gpu["index"]) == values[0] or gpu["uuid"].startswith(values[0])]
        if len(matches) != 1:
            die(f"CUDA_VISIBLE_DEVICES={values[0]!r} does not identify one visible GPU")
        candidate = matches[0]
    matches = [gpu for gpu in devices if gpu["pci"] == candidate["pci"]]
    if len(matches) != 1:
        die(f"CUDA-visible GPU {candidate['pci']} is not selected for this workflow")
    return {**matches[0], "index": candidate["index"], "uuid": candidate["uuid"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("worker", "frontend", "client"), required=True)
    parser.add_argument("--worker-gpu-pci", default=os.environ.get("WORKER_GPU_PCI", ""))
    parser.add_argument("--frontend-cores", type=int, default=8)
    parser.add_argument("--client-cores", type=int, default=8)
    parser.add_argument("--gpu-visible")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--report-dir", default=os.environ.get("SFLOW_WORKFLOW_OUTPUT_DIR", ""))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if bool(args.pid) == bool(args.command) or min(args.frontend_cores, args.client_cores) < 1:
        die("provide a command or --pid, and positive physical-core counts")
    allowed = os.sched_getaffinity(0)
    devices = selected_devices(gpu_devices(), args.worker_gpu_pci)
    plan, worker_counts = allocation(devices, cores_by_numa(physical_cores(allowed), allowed),
                                     args.frontend_cores, args.client_cores)
    gpu = worker_gpu(devices, args.gpu_visible or os.environ.get("CUDA_VISIBLE_DEVICES", "")) if args.role == "worker" else None
    selected, pid = plan["worker"][gpu["pci"]] if gpu else plan[args.role], args.pid or os.getpid()
    os.sched_setaffinity(pid, selected)
    if os.sched_getaffinity(pid) != selected:
        die("kernel did not apply the requested CPU set")
    report = {"pid": pid, "role": args.role, "cpus": rendered(selected), "gpu": gpu,
              "workers": devices, "physical_core_counts": {"workers": worker_counts,
              "frontend": args.frontend_cores,
              "client": args.client_cores},
              "plan": {"workers": {pci: rendered(mask) for pci, mask in plan["worker"].items()},
              "frontend": rendered(plan["frontend"]), "client": rendered(plan["client"])}}
    print(json.dumps({"dynamo_cpu_affinity": report}, sort_keys=True), flush=True)
    if args.report_dir:
        directory = Path(args.report_dir) / "cpu-affinity"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{args.role}-{gpu['pci'] if gpu else args.role}.json").write_text(json.dumps(report, sort_keys=True) + "\n")
    if args.command:
        os.execvpe(args.command[0], args.command, os.environ)


if __name__ == "__main__":
    main()
