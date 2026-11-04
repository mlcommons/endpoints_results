> Site-specific names, paths and addresses are replaced with `<placeholders>`.

# DeepSeek-R1 — MLPerf Endpoints v1.0.3 on GKE (GB300 NVL72 / A4X-max)

This is the GKE sflow port of the NVIDIA Slurm recipe
`code-drop-nvidia/nv-mlpinf-partner-partner-endpoints-v1.0.3/NVIDIA/src/deepseek-r1`.
It covers all twelve approved concurrency points:
**C32, C48, C64, C128, C256, C512, C1024, C2048, C4096, C8192, C16384, C28080**.

Each point uses 72 GB300 GPUs on **18 nodes** (the Slurm-faithful topology). Prefill
and decode are disaggregated, with MTP speculative decoding.

In v1.0.3, `configs/` is **byte-identical** to v1.0.2 across all twelve points and
`_shared/` (only `submission/` templates and `system_desc.json` / `system_power.json`
were added upstream). The generator is a copy of the v1.0.2 one
([v1.0.2/deepseek-r1/](../../v1.0.2/deepseek-r1/README.md)) pointing `DEFAULT_SLURM_ROOT`
and `DROP_VERSION` at `v1.0.3`. The GKE layer is **unchanged**: `k8s_mpi` per role, ComputeDomain, the
UCX/NCCL MNNVL block, the atomic hang-detector patch with
`TRTLLM_HANG_DETECTOR_TIMEOUT=3600`, frontend `--request_timeout 14400`, `client.yaml`
verbatim plus two rewrites, `--longer-runs` with the 85 % guard, and the prepull and
report-dir DaemonSets. Follow [SLURM_TO_GKE.md](../../SLURM_TO_GKE.md) for the full
procedure. The v1.0.0 runbook ([v1.0.0/deepseek-r1/README.md](../../v1.0.0/deepseek-r1/README.md))
still applies section by section.

> [!IMPORTANT]
> v1.0.2 serves a **different checkpoint** from v1.0.1
> (`centml/DeepSeek-R1-NVFP4-v2-mlpinf`, not `nvidia/DeepSeek-R1-NVFP4`). Do not
> compare v1.0.2 numbers with the v1.0.1 results, and do not run these recipes
> until `stage-assets.yaml` has staged the v2 checkpoint on every node (§3).

---

## 1. What changed from v1.0.1

The drop was compared file by file against v1.0.1. For all 12 points, `server-ctx.yaml`,
`server-gen.yaml` and `client.yaml` are **byte-identical**. Topology, concurrency,
`n_samples_to_issue` and the images are also unchanged. The changes are:

| Drop change | Points | v1.0.1 → v1.0.2 | GKE port |
|---|---|---|---|
| Checkpoint (`MODEL_PATH`) | all | `${SCRATCH_DIR}/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4` → `${SCRATCH_DIR}/models/DeepSeek-R1-NVFP4-v2`. The README pins `centml/DeepSeek-R1-NVFP4-v2-mlpinf@93947a0d…` | `MODEL_PATH` → `…/model-cache/DeepSeek-R1-NVFP4-v2`, staged by `stage-assets.yaml` (DaemonSet `stage-dsr1-v2-assets`). The v1.0.1 checkpoint stays next to it, so v1.0.1 recipes keep working. |
| `DATA_PATH` default | all | `${SCRATCH_DIR}/data/deepseek-r1-endpoints/deepseek_r1_eval.parquet` → `${SCRATCH_DIR}/data/deepseek_r1_eval.parquet` | None: same file (same sha256), same node-local path. Description updated. |
| `CLIENT_RESERVE_GPUS` | C8192, C16384, C28080 | `8` → **removed** (also from the `TOTAL_GPUS` expression), so `SLURM_NODES` goes 20 → **18** | No functional change. GKE already kept the client on node 0 (`CLIENT_RESERVE_NODES=0`), so Slurm now matches GKE. `EXPECTED` asserts `SLURM_NODES=18` on every point. The generator refuses the key if it comes back. The `CLIENT_RESERVE_NODES` text is rewritten. |
| `server-env.yaml` | all | `HF_HOME` removed. The file is now identical on all 12 points. | Mapped key by key (below). The server HF caches stay node-local (`/tmp/hf`, `/tmp/tiktoken`), as proven on GKE. |
| `endpoint_env.yaml` | all | `HF_HOME` → `/tmp/hf_cache`, `TIKTOKEN_RS_CACHE_DIR` → `/tmp/hf_cache/tiktoken_rs` | The benchmark task already exported exactly these values. They are now verbatim and asserted. |
| `_shared/slurm_env.yaml` + template | shared | New `SITE_ENV_YAML: ''`. Server tasks run `run_with_env.py <server-env> ${SITE_ENV_YAML}`, and the snapshot copies `site-env.yaml` when it is set. | Not ported. It defaults to empty (asserted), and the GKE compatibility layer is this port's site overlay. |

### New env/path plumbing → GKE ([SLURM_TO_GKE.md §5.1/§5.6](../../SLURM_TO_GKE.md))

| Drop key | Value | GKE |
|---|---|---|
| `TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER`, `TRTLLM_SERVER_DISABLE_GC`, `TRTLLM_WORKER_DISABLE_GC`, `TRTLLM_ENABLE_PDL`, `NCCL_GRAPH_MIXING_SUPPORT`, `TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED`, `TLLM_NUMA_AWARE_WORKER_AFFINITY` | `1`/`0` | **Kept** (model/runtime). Exported in the compat block and forwarded with `mpirun -x K=V`. Both are asserted. |
| `UCX_CUDA_IPC_ENABLE_MNNVL` | `n` | **Dropped.** GKE needs `y` (KV over MNNVL, §5.6). |
| `UCX_TLS` | `^posix` | **Dropped.** GKE pins `cuda_copy,cuda_ipc,sm,self,tcp`. |
| `UCX_RNDV_SCHEME` | `put_zcopy` | **Dropped.** The GKE UCX block sets the same value itself. |
| `endpoint_env.yaml` `HF_HOME`, `TIKTOKEN_RS_CACHE_DIR` | `/tmp/hf_cache`, `/tmp/hf_cache/tiktoken_rs` | Exported verbatim by the benchmark task (plus the GKE `HF_HUB_CACHE` and `mkdir`). |
| `endpoint_env.yaml` `PATH` | `/opt/venv/bin:…` | Handled by the benchmark task's `CLIENT_VENV` resolution (FIX 1). |
| `CONTAINER_MOUNTS` `…/hf_cache:/hf_cache`, `SITE_ENV_YAML` | — | Not ported. There is no shared FS, and the site overlay is the GKE layer. |

A key that is in neither list makes the generator stop and ask for classification.

### Generator changes (see the "v1.0.2 port" section of its docstring)

- `DEFAULT_SLURM_ROOT` now points at the v1.0.2 drop. `EXPECTED` gained `SLURM_NODES` (18 everywhere).
- New asserted pins, all fail-closed. They run before both generate and `--check`:
  - `read_point` checks `MODEL_PATH`, `DATA_PATH`, `CONTAINER_IMAGE`, `server-env.yaml` and `endpoint_env.yaml`, and refuses `CLIENT_RESERVE_GPUS`.
  - `assert_drop_shared` checks the client and LCB images, and that `SITE_ENV_YAML` is `''`.
  - `assert_gke_consistency` checks that the template's `MODEL_PATH`/`DATA_PATH`/image defaults match. It also checks that every kept server-env var is exported and forwarded, and that `stage-assets.yaml` stages `CHECKPOINT` into `model-cache/DeepSeek-R1-NVFP4-v2`.
- Template:
  - `MODEL_PATH` value changed.
  - The checkpoint pin was added to every recipe header.
  - Comments and descriptions were updated.
  - The workflow is renamed `dsr1_endpoints_v1_0_2_disagg_gke` (variants: `…_v1_0_2_disagg_gke_c<C>_16nodes`), so run ids and report dirs from the two checkpoints never mix.
- `--check` bug fix: an inner loop variable shadowed the point label, so v1.0.1 printed every problem as `server-gen:`. The checks and the problem count are unchanged; only the labels are now correct.
- New opt-in modes `--diag` / `--check-diag` (`DIAGNOSTICS`, `diag_info`, `apply_diag_text`, `check_diag`) write and verify the DIAGNOSTIC-only recipes under `diag/` (see §2). `--diag` writes nothing else, and the twelve 18-node renders are byte-identical with or without it.

**Recipe diff vs v1.0.1** (all 12 points): after dropping comments and descriptions, the
**only** functional differences are `variables.MODEL_PATH.value` and `workflow.name`.
`launch_test.sh` changed only in comments.

## 2. What is here

```
deepseek-r1/
  README.md                     this file
  generate_point_recipes.py     the ONLY place the GKE layer lives
  stage-assets.yaml             DaemonSet stage-dsr1-v2-assets: v2 checkpoint + eval parquet onto every node
  mk-report-dir.yaml            DaemonSet: pre-create the durable report hostPath (same as v1.0.1)
  prepull-images.yaml           DaemonSet: warm the three run images (same as v1.0.1)
  18nodes/
    point_<C>/
      deepseek_r1_gke_disagg_point<C>.yaml    generated -- do not hand-edit
      launch_test.sh                          generated -- do not hand-edit (REQUIRED_NODES=18)
      sflow_output/                           run output for this point (gitignored)
  diag/                         DIAGNOSTIC ONLY -- NOT submission topologies (see below)
    point_1024_gen2xdep32/
      deepseek_r1_gke_disagg_point1024_diag_gen2xdep32.yaml   generated by --diag
      launch_test.sh                                          generated by --diag (REQUIRED_NODES=18)
```

```bash
./generate_point_recipes.py                         # rewrite all twelve 18-node points (default mode)
./generate_point_recipes.py --check                 # verify against the v1.0.2 Slurm drop (+ any diag/ on disk)
./generate_point_recipes.py --longer-runs           # RERUN recipes: C32/C48/C64 -> 17,552 samples
./generate_point_recipes.py --check --longer-runs   # verify those (fails on default-mode recipes, by design)
./generate_point_recipes.py --variants 16           # also write 16nodes/ variants
./generate_point_recipes.py --variants 16 --layout  # print the 16-node splits only
./generate_point_recipes.py --diag                  # write ONLY diag/ (never touches 18nodes/)
./generate_point_recipes.py --check-diag            # verify diag/ (every diagnostic must exist)
```

The committed recipes are in **default mode** (the drop's `n_samples_to_issue`).
`--check` asserts the same things as v1.0.1, plus the v1.0.2 pins above:
- Server heredocs equal the drop, byte for byte, apart from the three documented edits.
- `CLIENT_CONFIG` equals the drop apart from the two documented rewrites.
- Topology matches `dsr1_config.yaml`.
- Decode has CUDA graphs up to `max_batch_size`, and prefill runs eager.
- Every script item is a string and passes `bash -n`.
- The recipe and `launch_test.sh` match a fresh render.

### Topology per point

| point | ctx | gen | C/replica | n_samples | nodes | `SLURM_NODES` |
|---|---|---|---|---|---|---|
| 32    | 4 × DEP2 | 1 × DEP64 | 32   | 8 776   | 18 | 18 |
| 48    | 4 × DEP2 | 1 × DEP64 | 48   | 8 776   | 18 | 18 |
| 64    | 4 × DEP2 | 1 × DEP64 | 64   | 13 164  | 18 | 18 |
| 128   | 4 × DEP2 | 1 × DEP64 | 128  | 21 940  | 18 | 18 |
| 256   | 2 × DEP4 | 2 × DEP32 | 128  | 21 940  | 18 | 18 |
| 512   | 4 × DEP2 | 1 × DEP64 | 512  | 74 596  | 18 | 18 |
| 1024  | 4 × DEP2 | 1 × DEP64 | 1024 | 105 312 | 18 | 18 |
| 2048  | 4 × DEP2 | 2 × DEP32 | 1024 | 193 072 | 18 | 18 |
| 4096  | 2 × DEP4 | 4 × DEP16 | 1024 | 285 220 | 18 | 18 |
| 8192  | 4 × DEP2 | 4 × DEP16 | 2048 | 351 040 | 18 | 18 (v1.0.1: 20) |
| 16384 | 4 × DEP2 | 4 × DEP16 | 4096 | 364 204 | 18 | 18 (v1.0.1: 20) |
| 28080 | 2 × DEP4 | 4 × DEP16 | 7020 | 658 200 | 18 | 18 (v1.0.1: 20) |

On every point, ctx sits on nodes 0–1, gen starts at node 2, and the CPU-only client and
LCB pods share node 0. In v1.0.2 this is also what Slurm does.

### Diagnostic recipes (`diag/`): DIAGNOSTIC ONLY, NOT SUBMISSION TOPOLOGIES

`diag/` holds experiments derived from an 18-node point. They are **never reported as
that point**. Each recipe is a fresh render of its point in which ONLY the decode split
changes. Its header banner lists the purpose and every deviation, and every changed line
carries an inline `# DIAG <name>:` marker. Only `--diag` writes them.

**`point_1024_gen2xdep32` (V1c).** This is C1024 with decode **2 × DEP32** instead of
1 × DEP64. It asks whether splitting the 64-rank decode server changes decode throughput or
TPOT when the 64 decode GPUs, the per-rank `max_batch_size` (16), the 1,024 decode slots,
prefill, client and GKE layer are all held fixed.

| Deviation from `18nodes/point_1024` | C1024 | V1c |
|---|---|---|
| `NUM_GEN_SERVERS` | 1 | 2 |
| `GPUS_PER_GEN_RANK` | 64 | 32 |
| server-gen `tensor_parallel_size` / `moe_expert_parallel_size` (EOF_GEN + snapshot) | 64 | 32 |
| `workflow.name` (tags the run dir and the report dir) | `dsr1_endpoints_v1_0_2_disagg_gke` | `…_c1024_diag_gen2xdep32` |

The `gke-topology.txt` snapshot also gains a `DIAG = …` line.

Node math is still 72 GPUs on 18 nodes:
- ctx 4 × DEP2 sits on nodes 0–1.
- Gen server 0 uses nodes 2–9 (`GEN_URLS` node 2, port 30004).
- Gen server 1 uses nodes 10–17 (`GEN_URLS` node 10, port 30005).
- Decode slots are 2 × 32 × 16 = 1,024 ≥ C.

The server side is NVIDIA's C2048 topology (4 × DEP2 + 2 × DEP32, identical
`server-ctx.yaml`) with C1024's per-rank decode config. C2048 has `max_batch_size` 32,
`max_num_tokens` 128 and a graph ladder up to 32. V1c keeps C1024's 16 / 64 / ladder up to 16.

`--check-diag` fails closed. So does a plain `--check` whenever `diag/` is on disk. Either one
fails if:
- the diag differs, once parsed, from a fresh render of C1024 in anything except the keys
  above, its names and comments;
- a changed non-comment line lacks its marker;
- the banner is gone;
- the decode GPUs, server nodes or `max_batch_size` change;
- the recipe or `launch_test.sh` differ from a fresh render.

Launch it like a point, with `cd diag/point_1024_gen2xdep32 && ./launch_test.sh`. The script
prints a DIAGNOSTIC warning before `sflow run`.

## 3. Before the first run

This is the same procedure as v1.0.1/v1.0.0 §3, except that the **checkpoint DaemonSet is new**:

```bash
kubectl apply -f stage-assets.yaml      && kubectl rollout status ds/stage-dsr1-v2-assets --timeout=90m
kubectl logs -l app=stage-dsr1-v2-assets --tail=3 --prefix | sort  # every node: STAGED: node ready
kubectl delete -f stage-assets.yaml
kubectl apply -f mk-report-dir.yaml     && kubectl rollout status ds/dsr1-mk-report-dir --timeout=300s
kubectl delete -f mk-report-dir.yaml
kubectl apply -f prepull-images.yaml    && kubectl rollout status ds/prepull-dsr1-images --timeout=45m
kubectl delete -f prepull-images.yaml
```

| What | Host path | Source |
|---|---|---|
| Checkpoint | `<node-local-ssd>/model-cache/DeepSeek-R1-NVFP4-v2` | `hf://centml/DeepSeek-R1-NVFP4-v2-mlpinf@93947a0d7bd04f73ff98636f6b18ff5839e7aaf9`; the revision is written to `.sflow-staged` |
| Eval parquet | `<node-local-ssd>/mlperf-data/deepseek_r1_eval.parquet` | endpoints `e9f427c`, sha256 `36facfae…6f02` (same as v1.0.1) |
| Reports | `<node-local-ssd>/dsr1-reports` | `mk-report-dir.yaml` |

| Role | Image |
|---|---|
| Server / frontend | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14` |
| Client | `ghcr.io/mlcommons/endpoints:e9f427c` (mirror to AR if ghcr is unreachable) |
| LiveCodeBench | `ghcr.io/mlcommons/endpoints:9d749906-livecodebench` |

## 4. Run a point

```bash
cd 18nodes/point_1024
SKIP_NODE_CHECK=1 ./launch_test.sh --dry-run      # ✓ Configuration is valid
mkdir -p sflow_output
setsid nohup ./launch_test.sh > sflow_output/launch-$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
./launch_test.sh -s MODE=both                     # perf + accuracy (Slurm sets MODE=both)
```

Sweep, from `kubernetes/`:

```bash
G=v1.0.2/deepseek-r1/18nodes; mkdir -p $G/sweep
setsid nohup tools/run_sweep.sh $G > $G/sweep/sweep-$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
```

Validate every run with
[SLURM_TO_GKE.md §10](../../SLURM_TO_GKE.md#10-phase-8-validate-every-run). Also confirm
from the server log that the weights were loaded from `…/DeepSeek-R1-NVFP4-v2`, and on one
node check that `cat …/model-cache/DeepSeek-R1-NVFP4-v2/.sflow-staged` prints `93947a0d…`.

## 5. Provenance to record with every result

| Item | Value |
|---|---|
| Drop | `nv-mlpinf-partner-partner-endpoints-v1.0.3` |
| Checkpoint | `centml/DeepSeek-R1-NVFP4-v2-mlpinf@93947a0d7bd04f73ff98636f6b18ff5839e7aaf9`: 413,340,550,290 B, 173 files, including 163 safetensors shards (per `stage-assets.yaml`) |
| Server image | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14`, arm64 manifest `sha256:e71eb8056ac1c717e6b727f9d66ebba9442e95f7ebc3514007b541046e84b088` (drop README) |
| Client | `ghcr.io/mlcommons/endpoints:e9f427c` = endpoints commit `e9f427ce74fd90e7fafdbc77ee7746cc00245692`. The image digest is **not pinned by the drop**: record the `imageID` of the benchmark pod at run time. |
| LiveCodeBench | `ghcr.io/mlcommons/endpoints:9d749906-livecodebench`. Digest not pinned: record the `imageID` at run time. |
| Dataset | `deepseek_r1_eval.parquet`, sha256 `36facfae18f42587b7e4863e16a0e96446470761017d207791e609537d0f6f02` |
| sflow | The drop README pins nv-sflow `b8afb5b86b3f46d9fa771531990fc7afaefc69b9`. Record the sflow commit actually used. |
| GKE deviations | `metrics_tokenizer_workers: 32`, `report_dir: ${REPORT_DIR}`, `model_params.seed` set to Seed Set A (`9_315_206_023_656_308_754`), iter-stats on, KV transfer timeouts, the GKE UCX/NCCL block in place of the `UCX_*` keys in `server-env.yaml`, per-pod disjoint `/proc/sys/kernel/ns_last_pid` offset in `dsr1-rank-env.sh` (prevents cross-pod `(system_uuid, pid)` peer-cache collisions in `ucp_cuda_ipc` MNNVL when pods run in isolated PID namespaces), node-local server HF caches |

## 6. Upstream anomalies (carried, flagged)

- The NVIDIA README still says `hf_cache` is mounted at `/hf_cache` "for the server and
  client environment files". However, v1.0.2's `endpoint_env.yaml` uses `/tmp/hf_cache`, and
  `server-env.yaml` no longer sets `HF_HOME`. The port follows the env files.
- Warmup is disabled on all points, as in v1.0.1, so high-C numbers may read slightly lower
  than a warmed run would. This is NVIDIA's choice and is not overridden.
- The `--longer-runs` projections (`LONGER_RUNS`) were measured on the v1.0.1 checkpoint.
  Re-check them against the first v2 runs before relying on them.

## 7. Status

| Step | State |
|---|---|
| Generator `--check` | ✅ 12/12 points (2026-10-11, offline) |
| `--check --longer-runs` | Behaves as in v1.0.1: RC 1 on the default-mode recipes, with 3 problems each on C32/C48/C64 |
| `--dry-run` | ✅ 12/12 `✓ Configuration is valid` (sandboxed: kubectl and cloud CLI blocked, no cluster contact) |
| `diag/point_1024_gen2xdep32` (V1c, DIAGNOSTIC ONLY) | ✅ `--check-diag` and `--check` pass, and 16 tamper cases each fail closed. The sandboxed `--dry-run` passes, and its allocation map shows `decode_server_0` on nodes 2–9 and `decode_server_1` on nodes 10–17 (2026-10-07, offline) |
| Cluster runs | In progress on v1.0.2 (`centml/DeepSeek-R1-NVFP4-v2-mlpinf@93947a0d`) with `ns_last_pid` MNNVL fix and Seed Set A (`9_315_206_023_656_308_754`). |
