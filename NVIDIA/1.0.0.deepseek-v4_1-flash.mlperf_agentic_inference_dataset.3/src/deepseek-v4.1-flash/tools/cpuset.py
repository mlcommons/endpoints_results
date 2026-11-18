#!/usr/bin/env python3
"""CPU sets that keep the engine processes off the frontend's cores.

usage: cpuset.py worker <rank> <cores_per_rank> <num_ranks>
       cpuset.py frontend <cores_per_rank> <num_ranks>
Rank r drives GPU r (one node, GPUs 0..num_ranks-1). Each rank gets <cores_per_rank> physical cores on its GPU's
NUMA node, both SMT siblings included, taken in order after the cores of lower ranks on the same node. Core 0 of
each node is skipped (housekeeping/interrupts). `frontend` prints every online CPU that no rank owns.
Output is a taskset/numactl CPU list, e.g. "1-12,177-188".
"""
import re
import subprocess
import sys


def read(path):
    with open(path) as f:
        return f.read().strip()


def parse_list(s):
    out = []
    for part in s.split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def fmt_list(cpus):
    cpus = sorted(set(cpus))
    runs, start = [], None
    for i, c in enumerate(cpus):
        if start is None:
            start = c
        if i + 1 == len(cpus) or cpus[i + 1] != c + 1:
            runs.append(f"{start}-{c}" if c != start else f"{c}")
            start = None
    return ",".join(runs)


def gpu_numa_nodes(n):
    topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=True).stdout
    topo = re.sub(r"\x1b\[[0-9;]*m", "", topo).splitlines()
    nodes = {}
    for line in topo[1:]:
        f = line.split()
        if f and re.fullmatch(r"GPU\d+", f[0]) and int(f[0][3:]) < n:
            nodes[int(f[0][3:])] = int(f[-2])
    return [nodes[g] for g in range(n)]


def physical_cores(node):
    # First thread of each core on the node, in CPU order.
    seen, cores = set(), []
    for c in parse_list(read(f"/sys/devices/system/node/node{node}/cpulist")):
        sib = tuple(parse_list(read(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")))
        if sib not in seen:
            seen.add(sib)
            cores.append(sib)
    return cores


def rank_sets(k, n):
    nodes = gpu_numa_nodes(n)
    sets = []
    for r in range(n):
        slot = sum(1 for q in range(r) if nodes[q] == nodes[r])
        cores = physical_cores(nodes[r])[1:]
        chosen = cores[slot * k:(slot + 1) * k]
        if len(chosen) < k:
            sys.exit(f"node {nodes[r]} has too few cores for rank {r}")
        sets.append([c for sib in chosen for c in sib])
    return sets


if sys.argv[1] == "worker":
    r, k, n = map(int, sys.argv[2:5])
    print(fmt_list(rank_sets(k, n)[r]))
elif sys.argv[1] == "frontend":
    k, n = map(int, sys.argv[2:4])
    owned = {c for s in rank_sets(k, n) for c in s}
    print(fmt_list(c for c in parse_list(read("/sys/devices/system/cpu/online")) if c not in owned))
else:
    sys.exit(__doc__)
