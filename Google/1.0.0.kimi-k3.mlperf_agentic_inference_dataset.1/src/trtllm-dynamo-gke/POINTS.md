> Site-specific names, paths and addresses are replaced with `<placeholders>`.

# Kimi K3 v1.0.3 on GKE: point costs, checker bands, candidate sets

This file summarizes the 16 official GB300 points in the v1.0.3 drop and their coverage properties. Inputs:

- The NVIDIA v1.0.3 GB300 configs: `GB300-NVL72_GB300-288GB_aarch64x48_TRTLLM-Dynamo/point_*.yaml`. These give topology, node placement, and trajectory counts.
- `v1.0.3` changes vs `v1.0.2`:
  - `C8` (12 nodes / 48 GPUs) is replaced by `C13` (8 nodes / 32 GPUs, `1 ctx×8 + 3 gen×8`, 613 trajectories).
  - `C32` issues `613` trajectories (1 pass) instead of `1226` (2 passes).
  - Serving architecture is TensorRT-LLM (`1.3.0rc25.post1`) + Dynamo (`1.6.0.dev20260918`) disaggregated with an `etcd` + `nats-server` control plane on node 0 and a Dynamo frontend on `FRONTEND_NODE` (`2` for `C12..C80`, `4` for `C192..C1024`), collocated with `CLIENT_NODE`.
  - Official SWE-bench accuracy points in the drop are `C12, C16, C32, C1024`.

The pool has 18 nodes × 4 GPUs. One pass is the 613-trajectory dataset (20,403 turns).

## Per-point topology and cost (all 16 GB300 points)

Notes on the columns:
- Region and `_mandatory_band` are computed for the full set: `C_min = 12` and declared `C_max = 1024`.
- `client` is the estimated benchmark time and `wall` = central `client` + 25 min (load ~20 min).
- Topology: `ctx×8` means one 8-GPU context worker.

| C | nodes/GPUs | topology | frontend / client node | trajectories (passes) | drop SWE-bench | client h | wall h | region | `_mandatory_band` |
|---:|---|---|---|---:|---|---|---:|---|---|
| 12 | 12/48 | 1 ctx×8 + 5 gen×8 | n2 / n2 | 613 (1) | yes (`both`) | 2.34 | 2.76 | low_latency | ultra_low |
| 13 | 8/32 | 1 ctx×8 + 3 gen×8 | n2 / n2 | 613 (1) | perf only | 2.25 | 2.67 | low | ultra_low |
| 16 | 8/32 | 1 ctx×8 + 3 gen×8 | n2 / n2 | 613 (1) | yes (`both`) | 2.04 | 2.46 | low | ultra_low |
| 24 | 8/32 | 1 ctx×8 + 3 gen×8 | n2 / n2 | 613 (1) | perf only | 1.68 | 2.10 | med | ultra_low |
| 32 | 8/32 | 1 ctx×8 + 3 gen×8 | n2 / n2 | 613 (1) | yes (`both`) | 1.48 | 1.90 | med | ultra_low |
| 48 | 8/32 | 1 ctx×8 + 3 gen×8 | n2 / n2 | 1226 (2) | perf only | 2.47 | 2.89 | med | med |
| 64 | 8/32 | 1 ctx×8 + 3 gen×8 | n2 / n2 | 1839 (3) | perf only | 3.34 | 3.75 | med | med |
| 80 | 8/32 | 1 ctx×8 + 3 gen×8 | n2 / n2 | 1839 (3) | perf only | 3.07 | 3.49 | med | med |
| 192 | 12/48 | 16 ctx×1 + 2 gen×16 | n4 / n4 | 3065 (5) | perf only | 4.06 | 4.48 | high | high |
| 288 | 12/48 | 16 ctx×1 + 2 gen×16 | n4 / n4 | 4904 (8) | perf only | 4.97 | 5.38 | high | high |
| 384 | 12/48 | 16 ctx×1 + 2 gen×16 | n4 / n4 | 4291 (7) | perf only | 3.59–4.35 (3.95) | 4.37 | high | high |
| 448 | 8/32 | 16 ctx×1 + 1 gen×16 | n4 / n4 | 4291 (7) | perf only | 3.24–4.35 (3.75) | 4.17 | high | high |
| 512 | 8/32 | 16 ctx×1 + 1 gen×16 | n4 / n4 | 4904 (8) | perf only | 3.39–4.97 (4.10) | 4.52 | high | high |
| 576 | 8/32 | 16 ctx×1 + 1 gen×16 | n4 / n4 | 4904 (8) | perf only | 3.14–4.97 (3.95) | 4.36 | high | high |
| 640 | 8/32 | 16 ctx×1 + 1 gen×16 | n4 / n4 | 4904 (8) | perf only | 2.93–4.97 (3.81) | 4.23 | high | high |
| 1024 | 8/32 | 16 ctx×1 + 1 gen×16 | n4 / n4 | 4904 (8) | yes (`both`) | 2.14–4.97 (3.26) | 3.68 | high | high |

Full-set regions (`C_min = 12, C_max = 1024`): `low_latency` 1–12, `low` 13–22, `med` 23–113, `high` 114–1024, `margin` 1025–1127.

## Candidate sets (C_max = highest point = 1024)

How to read the table:
- A 12-node point occupies the pool alone. Two 8-node points fit side by side (16 ≤ 18).
- `paired` = makespan with 8-node points run two at a time.

| set | points | wall h sequential | paired | node-h | `_mandatory_band` | checks |
|---|---|---:|---:|---:|---|---|
| core | 16, 24, 48, 1024 | 11.1 | 5.8 | 89 | ULC, med, high | fails point count (4 < 7) |
| A | 13, 16, 24, 32, 48, 80, 1024 | **18.2** | **9.3** | 146 | ULC, med, high | pass; all 8-node |
| B | 12, 16, 24, 32, 48, 80, 1024 | 19.3 | 11.1 | 165 | ULC, med, high | pass; includes all 4 drop SWE-bench points |
| Full | all 16 GB300 points | 56.2 | — | 523 | ULC, med, high | full 16-point NVIDIA submission grid |

Regions for each set (computed by the checker):

| set | C_min | low_latency | low | med | high | where each point lands |
|---|---:|---|---|---|---|---|
| A | 13 | 1–13 | 14–23 | 24–114 | 115–1024 | C13 low_latency, C16 low, C24..C80 med, C1024 high |
| B and Full | 12 | 1–12 | 13–22 | 23–113 | 114–1024 | C12 low_latency, C13/C16 low, C24..C80 med, C192..C1024 high |
