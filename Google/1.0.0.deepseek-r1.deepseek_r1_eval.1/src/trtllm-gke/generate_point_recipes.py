#!/usr/bin/env python3
"""Generate the GKE sflow recipes for every approved DeepSeek-R1 concurrency point.

Source of truth
---------------
Topology, server engine config and client workload are READ OUT OF the NVIDIA
Slurm drop every time this script runs -- nothing about the workload is typed in
here:

    nv-mlpinf-partner-partner-endpoints-v1.0.3/NVIDIA/src/deepseek-r1/
      configs/GB300-NVL72_GB300-288GB_aarch64x72_TRTLLM/point_<C>/
        dsr1_config.yaml    -> NUM_{CTX,GEN}_SERVERS, GPUS_PER_{CTX,GEN}_RANK,
                               NUM_FRONTEND_SERVERS, BASE_PORT, FRONTEND_PORT;
                               MODEL_PATH / DATA_PATH / CONTAINER_IMAGE and the
                               absence of CLIENT_RESERVE_GPUS are ASSERTED
        server-ctx.yaml     -> pasted VERBATIM into the EOF_CTX heredoc
        server-gen.yaml     -> pasted VERBATIM into the EOF_GEN heredoc
        client.yaml         -> emitted VERBATIM as the CLIENT_CONFIG artifact
        server-env.yaml     -> ASSERTED: its model/runtime keys are the ones the
                               GKE layer exports; its Slurm-only keys are known
        endpoint_env.yaml   -> ASSERTED: the client cache paths the benchmark
                               task exports
      configs/_shared/slurm_env.yaml
                            -> client/LCB image tags (ASSERTED)

The GKE half -- UCX/NCCL/OMPI transport, MPI operator split, hostPath staging,
the four Kimi-K3 result-collection fixes -- lives in exactly ONE place: the
RECIPE_TEMPLATE string below. Fix the GKE layer once here and re-run; every
point picks the fix up.

The §1 table from the conversion spec is asserted against the drop (EXPECTED
below). A mismatch means the drop changed and the spec needs revisiting -- the
script refuses to emit rather than silently shipping different numbers.

Usage
-----
    ./generate_point_recipes.py               # (re)write every 18-node point
    ./generate_point_recipes.py --check       # verify what is on disk
    ./generate_point_recipes.py --points 256 32
    ./generate_point_recipes.py --variants 16 # ALSO emit 16nodes/ variants
    ./generate_point_recipes.py --variants 16 --layout  # print the splits only
    ./generate_point_recipes.py --longer-runs # the RERUN recipes: more samples
                                              # on the LONGER_RUNS points
    ./generate_point_recipes.py --check --longer-runs   # check those instead
    ./generate_point_recipes.py --diag        # write ONLY the DIAGNOSTIC recipes
                                              # (DIAGNOSTICS -> diag/); never
                                              # touches 18nodes/
    ./generate_point_recipes.py --check-diag  # verify them (a plain --check
                                              # also verifies any on disk)

v1.0.3 port (2026-10-14)
------------------------
Copied from the v1.0.2 generator. In v1.0.3, configs/ is byte-identical to
v1.0.2 across all twelve points and _shared/ (only submission/ templates and
system_desc/system_power metadata were added upstream). DEFAULT_SLURM_ROOT and
DROP_VERSION now point at nv-mlpinf-partner-partner-endpoints-v1.0.3.

v1.0.2 port (2026-10-07)
------------------------
Copied from the v1.0.1 generator. The drop's server-ctx.yaml, server-gen.yaml
and client.yaml are byte-identical to v1.0.1 on all twelve points, as are the
topologies, images and client commit. What v1.0.2 changed, and what it means
here:
  * CHECKPOINT. Every point now serves centml/DeepSeek-R1-NVFP4-v2-mlpinf at
    93947a0d7bd04f73ff98636f6b18ff5839e7aaf9 (dsr1_config.yaml MODEL_PATH
    default $SCRATCH_DIR/models/DeepSeek-R1-NVFP4-v2). The NVIDIA README says
    other NVFP4 builds, including nvidia/DeepSeek-R1-NVFP4 (what the v1.0.1 GKE
    runs served), are different checkpoints. MODEL_PATH here ->
    model-cache/DeepSeek-R1-NVFP4-v2, staged by ./stage-assets.yaml. Results
    are NOT comparable with the v1.0.1 GKE runs.
  * CLIENT_RESERVE_GPUS: 8 removed from C8192/C16384/C28080, so SLURM_NODES is
    18 on every point. GKE already kept the client on node 0
    (CLIENT_RESERVE_NODES=0), so no functional change: EXPECTED now asserts
    SLURM_NODES 18 and read_point() refuses a drop that re-adds the key.
  * Slurm-side plumbing, mapped per SLURM_TO_GKE.md §5.1 (SERVER_ENV_*,
    ENDPOINT_ENV assert the drop still says what the mapping assumes):
      - server-env.yaml lost HF_HOME. Model/runtime keys are unchanged and
        still exported by the GKE layer; UCX_* stay replaced. The GKE server
        cache paths (/tmp/hf, rank-local HF_MODULES_CACHE) are unchanged.
      - endpoint_env.yaml now sets HF_HOME=/tmp/hf_cache and
        TIKTOKEN_RS_CACHE_DIR=/tmp/hf_cache/tiktoken_rs: exactly what the
        GKE benchmark task already exported, so only its comment changes.
      - SITE_ENV_YAML (new, empty by default) and the multi-file
        run_with_env.py are a Slurm site overlay for transport overrides. Not
        ported: on GKE the site transport layer IS gke_compat_block().
      - Default DATA_PATH moved to $SCRATCH_DIR/data/deepseek_r1_eval.parquet:
        same file and sha256; the GKE hostPath is unchanged.
  * Workflow name dsr1_endpoints_v1_0_2_disagg_gke, so v1.0.2 run dirs and
    durable report dirs cannot be confused with v1.0.1 (other checkpoint).
  * New fail-closed pins, run before generate AND --check: read_point()
    (paths, image, env files, no CLIENT_RESERVE_GPUS), assert_drop_shared()
    (client/LCB images, SITE_ENV_YAML == ''), assert_gke_consistency()
    (template MODEL_PATH/DATA_PATH/image defaults, kept server-env vars
    exported + forwarded, ./stage-assets.yaml stages CHECKPOINT).
  * --check fix: an inner loop variable shadowed the point label, so v1.0.1
    reported every problem as "server-gen:". Same checks, correct labels.
Everything else in the GKE layer is byte-for-byte the v1.0.1 generator's.

v1.0.1 port (2026-09-30)
------------------------
Carried over unchanged from the v1.0.0 generator (the drop's server YAMLs,
dsr1_config.yaml topologies, _shared/ template and slurm_env.yaml are
byte-identical to v1.0.0 apart from the directory rename
deepseek_r1/..._TRT -> deepseek-r1/..._TRTLLM). What v1.0.1 changed, all in
client.yaml and all carried verbatim:
  * n_samples_to_issue: C32 6582->8776, C48 6582->8776, C64 10200->13164,
    C16384 360000->364204 (EXPECTED below).
  * warmup: C2048/C8192/C16384/C28080 went from {enabled: true, salt: true,
    drain: true} to {enabled: false, salt: false}. No v1.0.1 point warms up,
    so the runtime `GKE EDIT: warmup.salt` flip below is now a no-op guard.
Default output is the Slurm-faithful 18nodes/ tree only: the pool has 18
healthy nodes. The reduced-node machinery is kept behind --variants.

Longer runs (opt-in: --longer-runs, 2026-10-06)
-----------------------------------------------
The reruns raise n_samples_to_issue on the LONGER_RUNS points to 4 dataset
passes. Without the flag every output, and what --check accepts, is exactly as
before. With it, CLIENT_CONFIG of those points carries a third documented edit.

Reduced-node variants (opt-in: --variants 16)
---------------------------------------------
Every Slurm point is 72 GPUs = 18 nodes (18nodes/point_<C>/). When the <nodepool>
pool is short, 16nodes/point_<C>/ runs the SAME point (same concurrency,
samples, client.yaml, server-ctx.yaml) on 16 nodes. Unlike GPT-OSS, the decode
server cannot keep
its shape -- six points are 1 x DEP64, which cannot sit next to any prefill
pool in 64 GPUs -- so a variant also reshapes server-gen.yaml (TP/EP,
max_batch_size, max_num_tokens, CUDA-graph ladder; nothing else, --check
enforces it). plan_variant() is the rule; EXPECTED_VARIANT asserts its output.
The variants are NOT submission topologies.

Diagnostic recipes (opt-in: --diag, 2026-10-07)
-----------------------------------------------
diag/point_<C>_<name>/ holds DIAGNOSTIC ONLY recipes: NOT submission
topologies, never to be reported as point C. Each is a fresh render of an
18-node point with ONLY its decode split changed (DIAGNOSTICS): the same 72
GPUs / 18 nodes, prefill, client.yaml, per-rank decode config and GKE layer.
diag_info() fails closed on the GPU / node / slot math, check_diag() on any
difference from the point's render other than the DIAG keys, names and
comments; every changed line carries an inline "# DIAG <name>:" marker.
  * gen2xdep32 (V1c): C1024 with decode 2 x DEP32 instead of 1 x DEP64 --
    NUM_GEN_SERVERS 1->2, GPUS_PER_GEN_RANK 64->32, server-gen
    tensor_parallel_size / moe_expert_parallel_size 64->32. Workflow name
    dsr1_endpoints_v1_0_2_disagg_gke_c1024_diag_gen2xdep32 tags the run and
    report dirs.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap

import yaml

# --------------------------------------------------------------------------
# Locations. The NVIDIA drop lives under <workdir>/code-drop-nvidia/<drop>/,
# next to the nv-sflow checkout; override with --slurm-root / --dest if yours
# differs.
# --------------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_SLURM_ROOT = (
    HERE.parents[5]  # <workdir>  (HERE = .../kubernetes/v1.0.3/deepseek-r1)
    / "code-drop-nvidia"  # all NVIDIA code drops live here, one folder per drop
    / "nv-mlpinf-partner-partner-endpoints-v1.0.3/NVIDIA/src/deepseek-r1"
)
ARCH_DIR = "configs/GB300-NVL72_GB300-288GB_aarch64x72_TRTLLM"
DROP_VERSION = "v1.0.3"

GPUS_PER_NODE = 4  # A4X-max / GB300 NVL72: 4 GPUs per host.

POINTS = [32, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 28080]

# Conversion spec §1, asserted against the drop. ctxN, ctxTP, genN, genTP,
# frontends, concurrency, n_samples, max_new_tokens, SLURM_NODES.
# v1.0.2: SLURM_NODES is 18 everywhere. v1.0.1 had 20 on C8192/C16384/C28080
# (CLIENT_RESERVE_GPUS: 8); v1.0.2 removed that key, nothing else moved.
EXPECTED = {
    32:    (4, 2, 1, 64, 4, 32,    8776,   20000, 18),   # v1.0.0: 6582
    48:    (4, 2, 1, 64, 4, 48,    8776,   20000, 18),   # v1.0.0: 6582
    64:    (4, 2, 1, 64, 4, 64,    13164,  20000, 18),   # v1.0.0: 10200
    128:   (4, 2, 1, 64, 4, 128,   21940,  20000, 18),
    256:   (2, 4, 2, 32, 4, 256,   21940,  20000, 18),
    512:   (4, 2, 1, 64, 4, 512,   74596,  20000, 18),
    1024:  (4, 2, 1, 64, 4, 1024,  105312, 20000, 18),
    2048:  (4, 2, 2, 32, 4, 2048,  193072, 20000, 18),
    4096:  (2, 4, 4, 16, 4, 4096,  285220, 20000, 18),
    8192:  (4, 2, 4, 16, 4, 8192,  351040, 20000, 18),   # v1.0.1: 20 nodes
    16384: (4, 2, 4, 16, 4, 16384, 364204, 20000, 18),   # v1.0.0: 360000; v1.0.1: 20 nodes
    28080: (2, 4, 4, 16, 4, 28080, 658200, 20000, 18),   # v1.0.1: 20 nodes
}

# --------------------------------------------------------------------------
# v1.0.2 checkpoint, paths, env and images -- ASSERTED, not data sources.
# read_point() / assert_drop_shared() refuse a drop that no longer says this,
# and assert_gke_consistency() refuses a template or ./stage-assets.yaml that
# drifted from it. Each one is a place where the drop and the GKE port must
# agree and where a silent mismatch would not fail at run time.
# --------------------------------------------------------------------------
# The checkpoint the v1.0.2 README pins for every point. ./stage-assets.yaml
# (written separately, never by this script) must stage exactly this.
CHECKPOINT = "centml/DeepSeek-R1-NVFP4-v2-mlpinf@93947a0d7bd04f73ff98636f6b18ff5839e7aaf9"
# dsr1_config.yaml defaults (v1.0.1: .../models/deepseek-r1/fp4-quantized-
# modelopt/deepseek_r1-torch-fp4 and .../data/deepseek-r1-endpoints/...).
DROP_MODEL_PATH = "${{ variables.SCRATCH_DIR }}/models/DeepSeek-R1-NVFP4-v2"
DROP_DATA_PATH = "${{ variables.SCRATCH_DIR }}/data/deepseek_r1_eval.parquet"
# Their node-local GKE equivalents (RECIPE_TEMPLATE MODEL_PATH / DATA_PATH).
# The model dir keeps the drop's basename; it sits next to the v1.0.1
# model-cache/DeepSeek-R1-NVFP4 (other checkpoint) on the same hostPath.
GKE_MODEL_CACHE = "<node-local-ssd>/model-cache"
GKE_MODEL_PATH = GKE_MODEL_CACHE + "/DeepSeek-R1-NVFP4-v2"
GKE_DATA_PATH = "<node-local-ssd>/mlperf-data/deepseek_r1_eval.parquet"

# server-env.yaml (SLURM_TO_GKE.md §5.1: "model/runtime vars ONLY; discard
# every UCX_*/OMPI_*/IB/RoCE var"). All twelve v1.0.2 files are identical.
# SERVER_ENV_KEEP is exported by gke_compat_block() and forwarded by MPIRUN_X,
# verbatim. SERVER_ENV_DROP is discarded, each for the reason given. A key in
# neither (v1.0.1's HF_HOME, or anything new) stops the generator.
SERVER_ENV_KEEP = {
    "TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER": "1",
    "TRTLLM_SERVER_DISABLE_GC": "1",
    "TRTLLM_WORKER_DISABLE_GC": "1",
    "TRTLLM_ENABLE_PDL": "1",
    "NCCL_GRAPH_MIXING_SUPPORT": "0",
    "TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED": "1",
    "TLLM_NUMA_AWARE_WORKER_AFFINITY": "1",
}
SERVER_ENV_DROP = {
    "UCX_CUDA_IPC_ENABLE_MNNVL": ("n", "Slurm moves KV over RoCE; GKE needs y (KV over MNNVL), §5.6"),
    "UCX_TLS": ("^posix", "GKE pins cuda_copy,cuda_ipc,sm,self,tcp, §5.6"),
    "UCX_RNDV_SCHEME": ("put_zcopy", "same value, set by the GKE UCX block itself"),
}
# endpoint_env.yaml (client env). v1.0.2 moved the caches from /hf_cache or
# /work/.hf_cache to container-local /tmp/hf_cache -- the values the GKE
# benchmark task already exported -- so these two are now exported verbatim.
# PATH is handled by the benchmark task's CLIENT_VENV resolution (FIX 1).
ENDPOINT_ENV = {
    "PATH": "/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HF_HOME": "/tmp/hf_cache",
    "TIKTOKEN_RS_CACHE_DIR": "/tmp/hf_cache/tiktoken_rs",
}
ENDPOINT_ENV_EXPORTED = ("HF_HOME", "TIKTOKEN_RS_CACHE_DIR")
# Images the drop names -> recipe variable that must default to it.
DROP_IMAGES = {
    "CONTAINER_IMAGE": ("TRTLLM_IMAGE", "docker://nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14"),
    "ENDPOINT_CONTAINER_IMAGE": ("CLIENT_IMAGE", "ghcr.io/mlcommons/endpoints:e9f427c"),
    "LCB_CONTAINER_IMAGE": ("LCB_IMAGE", "ghcr.io/mlcommons/endpoints:9d749906-livecodebench"),
}

# --------------------------------------------------------------------------
# Longer runs for the reruns. OFF by default; `--longer-runs` turns them on.
#
# The MLPerf Endpoints First-Time Submitter's Guide wants about 4 dataset
# passes in the steady-state window, plus the region's minimum duration
# (checker <ip>: 600 s low_latency, 1,200 s otherwise). NVIDIA's
# n_samples_to_issue gives C32 and C48 2 passes and C64 3, so the reruns raise
# it to 4 passes of the 4,388-prompt dataset. Nothing else in client.yaml
# changes. Approved 2026-10-06 (Y5 decision 13) unless NVIDIA or MLCommons say
# the drop's lengths are fine.
#
# Each entry carries two projections from the measured Oct runs (scratch
# run_length_plan.py / run_length_watchdog.py): perf_s, the predicted
# performance-phase seconds, and client_s, the measured client run plus the
# added performance seconds. NVIDIA's DSR1 client.yaml sets neither
# max_issue_duration_ms nor the whole-run watchdog settings.timeouts.
# run_timeout_s, but main() checks both against 85 % if a drop ever adds them.
#
# v1.0.2: client.yaml is byte-identical to v1.0.1, so the drop's n and the
# guard's inputs are unchanged. The perf_s / client_s projections were
# measured on the v1.0.1 checkpoint (nvidia/DeepSeek-R1-NVFP4); re-check them
# against the first v2-checkpoint runs before relying on them.
# --------------------------------------------------------------------------
LONGER_RUNS = {
    #     n_samples_to_issue  perf_s  client_s   drop n (passes) -> new (passes)
    32: (17552,              10353,  13142),    # 8,776 (2) -> 4
    48: (17552,              6772,   8751),     # 8,776 (2) -> 4
    64: (17552,              5164,   6871),     # 13,164 (3) -> 4
    128: (43880,             6400,   8254),     # 21,940 (5) -> 10 (halves -2.23% ramp-down drain loss)
}
DATASET_PROMPTS = 4388  # the DSR1 performance dataset: one pass
LONGER_RUNS_HEADROOM = 0.85

# --------------------------------------------------------------------------
# Reduced-node variants, written as <N>nodes/point_<C>/. OFF by default for
# v1.0.1 and v1.0.2 (the pool has 18 healthy nodes); `--variants 16` turns
# them on. plan_variant() explains the rule; the knobs it uses are these three.
# EXPECTED_VARIANT was derived for v1.0.0; plan_variant() depends only on
# topology, concurrency and max_batch_size, none of which v1.0.1 or v1.0.2
# changed.
# --------------------------------------------------------------------------
VARIANT_NODES = (16,)  # the node counts `--variants` accepts

# Decode EP sizes a variant may use, largest first. Both are shapes NVIDIA
# certifies for DSR1 decode in this very drop (DEP32: point_256/2048; DEP16:
# point_4096..28080). DEP64 cannot sit next to ANY prefill pool in 64 GPUs.
# DEP8 is excluded: the drop never uses it, and it doubles the expert bytes
# every GPU streams per decode step versus DEP16.
VARIANT_GEN_TPS = (32, 16)

# Per-rank decode batch up to which ONE largest-EP decode server is preferred
# over more, smaller ones. It is the largest per-rank batch at which NVIDIA
# itself still runs its single largest-EP server (point_1024: 1 x DEP64,
# max_batch_size 16); above it the drop switches to more, smaller servers.
MAX_EP_RANK_BATCH = 16

# The split plan_variant() picks. An ASSERTION, not a data source (like
# EXPECTED): the generator recomputes every row from the drop and refuses to
# write anything if a row no longer matches.
EXPECTED_VARIANT = {
    16: {
        #       ctxN genN genTP max_batch_size   Slurm 72-GPU topology
        32:    (16,  1,  32,   1),    # 4 x DEP2 + 1 x DEP64 (mbs 1)
        48:    (16,  1,  32,   2),    # 4 x DEP2 + 1 x DEP64 (mbs 1)
        64:    (16,  1,  32,   2),    # 4 x DEP2 + 1 x DEP64 (mbs 1)
        128:   (16,  1,  32,   4),    # 4 x DEP2 + 1 x DEP64 (mbs 2)
        256:   (8,   1,  32,   8),    # 2 x DEP4 + 2 x DEP32 (mbs 4)
        512:   (16,  1,  32,  16),    # 4 x DEP2 + 1 x DEP64 (mbs 8)
        1024:  (8,   3,  16,  22),    # 4 x DEP2 + 1 x DEP64 (mbs 16)
        2048:  (8,   3,  16,  43),    # 4 x DEP2 + 2 x DEP32 (mbs 32)
        4096:  (4,   3,  16,  86),    # 2 x DEP4 + 4 x DEP16 (mbs 64)
        8192:  (8,   3,  16, 171),    # 4 x DEP2 + 4 x DEP16 (mbs 128)
        16384: (8,   3,  16, 342),    # 4 x DEP2 + 4 x DEP16 (mbs 256)
        28080: (4,   3,  16, 585),    # 2 x DEP4 + 4 x DEP16 (mbs 512)
    },
}

# The only server-gen.yaml keys a variant may change (plus the GKE edits every
# point gets). --check diffs the parsed YAML against the drop and fails on any
# other key.
RESHAPE_KEYS = (
    "tensor_parallel_size",
    "moe_expert_parallel_size",
    "max_batch_size",
    "max_num_tokens",
    "cuda_graph_config",
)

# --------------------------------------------------------------------------
# DIAGNOSTIC recipes, written as diag/point_<C>_<name>/. Opt-in: `--diag`
# writes them (and ONLY them -- it never touches 18nodes/), `--check-diag`
# verifies them, and a plain `--check` also verifies every one present on
# disk. NOT submission topologies and NOT reduced-node variants: the same 72
# GPUs on the same 18 nodes, the same client.yaml, the same prefill and the
# same per-rank decode config as the 18-node point -- ONLY the decode split
# differs. Each is derived from a fresh render of its point, and
# check_diag() fails closed if it differs from that render in anything but
# the keys below, its names and comments.
# --------------------------------------------------------------------------
DIAG_DIR = "diag"
DIAGNOSTICS = {
    "gen2xdep32": dict(
        point=1024, gen_n=2, gen_tp=32,
        purpose=(
            "V1c: does splitting C1024's single 64-rank decode server (1 x "
            "DEP64 across 16 nodes) into two independent 32-rank servers (2 x "
            "DEP32, 8 nodes each) change decode throughput / TPOT, with the "
            "decode GPUs (64), per-rank max_batch_size (16), decode slots "
            "(1024), prefill, client and GKE layer all held fixed? Isolates "
            "the decode EP / all-to-all span. Same decode shape as NVIDIA's "
            "C2048 (4 x DEP2 + 2 x DEP32), with C1024's per-rank config."
        ),
    ),
}
# The only server-gen.yaml keys a diagnostic changes (both = the new decode
# TP/EP size); everything else in EOF_GEN stays NVIDIA's for that point.
DIAG_GEN_KEYS = ("tensor_parallel_size", "moe_expert_parallel_size")


# ==========================================================================
# Node math -- conversion spec §2, verified against sflow/app/resource_planner.py
# ==========================================================================
#
# Two different code paths in the planner, and they pack differently:
#
#   gpus.count <= gpus_per_node   (resource_planner.py:1144-1178)
#       Scans the allocation nodes in order and takes the FIRST contiguous free
#       GPU run of the requested size, reserving [start, start+count). Sub-node
#       servers therefore tightly BACKFILL -- including across the ctx->gen
#       boundary. A TP2 ctx server pair shares one node.
#
#   gpus.count >  gpus_per_node   (resource_planner.py:1180-1246)
#       Requires count % gpus_per_node == 0, then collects nodes whose free-GPU
#       cursor is identical (i.e. WHOLE free nodes) and takes count/gpus_per_node
#       of them.
#
# Hence server i's node index:
def ctx_node(i: int, ctx_tp: int) -> int:
    """Node index of ctx server *i*. Always tightly packed from GPU 0."""
    return (i * ctx_tp) // GPUS_PER_NODE


def gen_base_gpu(ctx_gpus: int, gen_tp: int) -> int:
    """First GPU offset the gen servers may use.

    A sub-node gen server backfills the ctx servers' leftover GPUs; a
    multi-node one must start on a whole-node boundary.
    """
    if gen_tp <= GPUS_PER_NODE:
        return ctx_gpus
    return ceil_div(ctx_gpus, GPUS_PER_NODE) * GPUS_PER_NODE


def gen_node(i: int, ctx_gpus: int, gen_tp: int) -> int:
    """Node index of gen server *i*."""
    return (gen_base_gpu(ctx_gpus, gen_tp) + i * gen_tp) // GPUS_PER_NODE


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# ==========================================================================
# Mandatory edits to the Slurm server YAML (conversion spec §3)
# ==========================================================================

CTX_HEADER = """\
# ----------------------------------------------------------------------------
# server-ctx.yaml -- VERBATIM from the NVIDIA Slurm drop
#   .../deepseek-r1/{arch}/point_{point}/server-ctx.yaml
# apart from the two mandatory GKE edits marked "GKE EDIT" below. Do not
# re-tune anything here: these are NVIDIA's certified numbers for this point.
#
# ctx runs DEP{ctx_tp} (tensor_parallel_size = moe_expert_parallel_size = {ctx_tp}).
# cuda_graph_config stays null and that is CORRECT for prefill: it batches
# ~16k tokens per step, so kernel-launch overhead is noise and eager is right.
# Graphs are only mandatory on the decode path (see server-gen.yaml).
# ----------------------------------------------------------------------------
"""

GEN_HEADER = """\
# ----------------------------------------------------------------------------
# server-gen.yaml -- VERBATIM from the NVIDIA Slurm drop
#   .../deepseek-r1/{arch}/point_{point}/server-gen.yaml
# apart from the two mandatory GKE edits marked "GKE EDIT" below.
#
# gen runs DEP{gen_tp} at C={per_replica} per replica.
#
# CUDA GRAPHS: checked, and NOT touched. Conversion spec §3.1 says to replace
# `cuda_graph_config: null` with an explicit batch_sizes ladder -- that edit is
# NOT needed in the endpoints v1.0.2 drop. Every one of the twelve
# server-gen.yaml files already ships an explicit
#   cuda_graph_config: {{enable_padding: true, batch_sizes: [...]}}
# ladder covering 1..max_batch_size, so the batch_sizes list below is NVIDIA's,
# byte-for-byte, not ours. (This is the single biggest trap in the port: the
# older DSR1 drop had null here on point_256 and running eager measured 177 ms
# per decode step vs 13.6 ms with graphs -- ~3.3k tok/s instead of ~45.8k, with
# no error and a perfectly healthy-looking server.) If a future drop
# reintroduces null, `--check` will still pass but the run will be 13x slow:
# grep the log for "Creating CUDA graph instances" (README check 2).
# ----------------------------------------------------------------------------
"""

# The two KV-transfer timeouts the proven point_256 GKE recipe adds to BOTH
# cache_transceiver_config blocks.
KV_TIMEOUTS = (
    "  # GKE EDIT (conversion spec §3.3): KV transfer timeouts. The Slurm drop\n"
    "  # leaves these at the TRT-LLM defaults, which assume a RoCE fabric that\n"
    "  # hands off in single-digit ms. On GKE the ctx->gen hop is UCX cuda_ipc\n"
    "  # over MNNVL and the first transfer of a run also pays ComputeDomain\n"
    "  # channel setup; the default window expires mid-handshake and the\n"
    "  # request fails with a KV transfer timeout while both servers stay\n"
    "  # 'healthy'. Present in the proven 45,760 tok/s point_256 recipe.\n"
    "  kv_transfer_timeout_ms: 120000\n"
    "  kv_transfer_sender_future_timeout_ms: 10000\n"
)

ITER_STATS_NOTE = (
    "# GKE EDIT: disable per-iteration CUDA event sync and stdout log formatting\n"
    "# in PyExecutor.profile_step() (matching gpt-oss-120b). enable_iter_perf_stats\n"
    "# and enable_iter_req_stats stay false as in the Slurm drop (when true, they\n"
    "# invoke torch.cuda.mem_get_info() and _populate_req_stats() on every decode\n"
    "# step across all 64 DEP64 ranks).\n"
)


def apply_server_edits(text: str) -> str:
    """Apply the conversion-spec §3 mandatory edits to a Slurm server YAML.

    Deliberately narrow and assertive: every substitution must hit exactly
    once, so a changed drop fails loudly here instead of silently shipping an
    unedited config.
    """
    out = []
    stats_note_done = False
    for line in text.splitlines(keepends=True):
        if line.rstrip("\n") == "print_iter_log: true":
            if not stats_note_done:
                out.append(ITER_STATS_NOTE)
                stats_note_done = True
            out.append("print_iter_log: false\n")
            continue
        out.append(line)
        # cache_transceiver_config's `backend: UCX` is indented two spaces;
        # moe_config's `backend: TRTLLM` is not UCX, so this is unambiguous.
        if line.rstrip("\n") == "  backend: UCX":
            out.append(KV_TIMEOUTS)
    result = "".join(out)
    assert "print_iter_log: false" in result, "print_iter_log edit missed"
    assert "enable_iter_perf_stats: false" in result, "iter perf stats should stay false"
    assert "enable_iter_req_stats: false" in result, "iter req stats should stay false"
    assert result.count("kv_transfer_timeout_ms") == 1, "kv timeout edit missed"
    assert "cuda_graph_config: null" not in result or "tensor_parallel_size" in result
    return result


# ==========================================================================
# Mandatory edits to the Slurm client YAML (conversion spec §5)
# ==========================================================================

TOKENIZER_WORKERS_NOTE = """\
  # ---------------------------------------------------------------------
  # GKE ADDITION -- not a Slurm value.
  #
  # After the load phase the client tokenizes every response to compute
  # OSL / TPS / TPOT. That stage is SILENT: it prints nothing until it
  # finishes. On Kimi-K3 C192 the default worker count turned this into
  # 2 h 05 m of log silence, `kubectl logs` stalled for 7,474 s, sflow's
  # collect sentinel could never be delivered, and the ENTIRE report was
  # lost after a flawless benchmark. 32 cuts it roughly 8x.
  #
  # The benchmark pod has 144 CPUs and is otherwise idle here, so a low
  # value is simply leaving the machine unused.
  #
  # NVIDIA's own value for this point: {slurm_value}. Restore it with
  # `-s METRICS_TOKENIZER_WORKERS=<n>` if you need byte-parity with Slurm;
  # it changes only post-processing speed, never the measured tokens.
  # ---------------------------------------------------------------------
"""

REPORT_DIR_NOTE = """\
# GKE EDIT (conversion spec §5, fix 3): report_dir moved off the pod.
# Slurm writes to ${SFLOW_TASK_OUTPUT_DIR}, which on Kubernetes is an emptyDir
# that dies with the pod and is only recovered through sflow's EXIT-trap +
# log-sentinel channel. That channel has failed twice in this cluster (once to
# `exec` destroying the trap, once to a stalled log stream) AFTER a perfect
# benchmark. ${REPORT_DIR} is a per-run directory on the node's local SSD,
# exported by the benchmark task; the bytes survive regardless.
"""


MODEL_SEED = 9315206023656308754

MODEL_SEED_NOTE = f"""\
  # GKE ADDITION (seed-runtime-match, §4.6 / §2.1.1): MLCommons seed set A
  # (cohort 2026-10-C1) requires model_seed = {MODEL_SEED}.
"""


def longer_run_note(pad: str, drop_n: int, new_n: int) -> str:
    return "".join(f"{pad}{ln}\n" for ln in (
        "# GKE EDIT, opt-in (--longer-runs) -- NOT NVIDIA's value. NVIDIA's",
        f"# n_samples_to_issue is {drop_n} ({drop_n / DATASET_PROMPTS:g} passes of the "
        f"{DATASET_PROMPTS:,}-prompt",
        f"# dataset). The rerun issues {new_n} ({new_n / DATASET_PROMPTS:g} passes) so it "
        "reaches about",
        "# 4 passes and its region's minimum duration (the First-Time Submitter's",
        "# Guide steady-state window). Nothing else in this file changes for it.",
    ))


def longer_n(info: dict) -> int | None:
    """This point's --longer-runs n_samples_to_issue, or None for NVIDIA's."""
    return info["n_samples"] if "drop_n_samples" in info else None


def apply_client_edits(text: str, workers_expr: str, n_samples: int | None = None) -> str:
    """Rewrite the Slurm client.yaml for GKE.

    Only three textual changes (conversion spec §5 + §4.6 seed set A):
      1. report_dir  -> the durable hostPath ${REPORT_DIR}
      2. settings.metrics_tokenizer_workers -> 32 (GKE addition)
      3. model_params.seed -> 9315206023656308754 (MLCommons seed set A)
    and, with --longer-runs only (`n_samples`), a fourth:
      4. settings.runtime.n_samples_to_issue -> LONGER_RUNS' value

    ${MODEL_PATH} / ${DATA_PATH} are intentionally left alone -- the benchmark
    task exports them to the node-local staging paths, so the config stays
    byte-identical to NVIDIA's. endpoint_config.endpoints stays `[]` and is
    filled from the FRONTEND_URLS artifact at runtime, exactly as Slurm's
    generate_endpoint_yaml.py does.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    slurm_workers = "unset (client default)"
    replaced_workers = False
    replaced_seed = False
    replaced_n = 0

    for line in lines:
        stripped = line.rstrip("\n")
        if stripped == "  top_p: 1.0" and not replaced_seed:
            out.append(line)
            out.append(MODEL_SEED_NOTE)
            out.append(f"  seed: {MODEL_SEED}\n")
            replaced_seed = True
            continue
        if stripped.startswith("  metrics_tokenizer_workers:"):
            slurm_workers = stripped.split(":", 1)[1].strip()
            out.append(TOKENIZER_WORKERS_NOTE.format(slurm_value=slurm_workers))
            out.append(f"  metrics_tokenizer_workers: {workers_expr}\n")
            replaced_workers = True
            continue
        if stripped == "endpoint_config:" and not replaced_workers:
            # settings: is the block immediately above endpoint_config:, so
            # appending here lands the key at the end of settings:.
            out.append(TOKENIZER_WORKERS_NOTE.format(slurm_value=slurm_workers))
            out.append(f"  metrics_tokenizer_workers: {workers_expr}\n")
            replaced_workers = True
            out.append(line)
            continue
        if stripped == "report_dir: ${SFLOW_TASK_OUTPUT_DIR}":
            out.append(REPORT_DIR_NOTE)
            out.append("report_dir: ${REPORT_DIR}\n")
            continue
        m = re.match(r"^(\s+)n_samples_to_issue: (\d+)$", stripped)
        if n_samples is not None and m:
            out.append(longer_run_note(m.group(1), int(m.group(2)), n_samples))
            out.append(f"{m.group(1)}n_samples_to_issue: {n_samples}\n")
            replaced_n += 1
            continue
        out.append(line)

    result = "".join(out)
    assert replaced_seed, "model_params.seed not placed"
    assert replaced_workers, "metrics_tokenizer_workers not placed"
    assert "report_dir: ${REPORT_DIR}" in result, "report_dir edit missed"
    assert n_samples is None or replaced_n == 1, (
        f"--longer-runs: expected one n_samples_to_issue line, found {replaced_n}"
    )
    # The explanatory comment above the edit legitimately names the old value,
    # so only inspect live YAML lines.
    live = [l for l in result.split("\n") if not l.strip().startswith("#")]
    assert not any("${SFLOW_TASK_OUTPUT_DIR}" in l for l in live), (
        "stale report_dir left behind"
    )
    return result


# ==========================================================================
# Reading the Slurm drop
# ==========================================================================


def read_flat_env(path: pathlib.Path, label: str) -> dict[str, str]:
    """A drop env file (server-env.yaml / endpoint_env.yaml) as {str: str}.
    Fails closed on anything that is not a flat mapping of scalars."""
    if not path.is_file():
        sys.exit(f"{label}: missing {path}")
    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict) or not all(
        isinstance(v, (str, int, float)) and not isinstance(v, bool)
        for v in doc.values()
    ):
        sys.exit(f"{label}: {path.name} is not a flat mapping of string/number "
                 "values -- refusing to guess how run_with_env.py would read it")
    return {str(k): str(v) for k, v in doc.items()}


def read_point(slurm_root: pathlib.Path, point: int) -> dict:
    d = slurm_root / ARCH_DIR / f"point_{point}"
    if not d.is_dir():
        sys.exit(f"Slurm point directory not found: {d}")

    cfg = yaml.safe_load((d / "dsr1_config.yaml").read_text())["variables"]

    def iv(name: str, default=None):
        if name not in cfg:
            if default is None:
                sys.exit(f"point_{point}: dsr1_config.yaml has no {name}")
            return default
        return int(cfg[name]["value"])

    def sv(name: str) -> str:
        if not isinstance(cfg.get(name), dict) or "value" not in cfg[name]:
            sys.exit(f"point_{point}: dsr1_config.yaml has no {name}")
        return str(cfg[name]["value"])

    # v1.0.2 removed CLIENT_RESERVE_GPUS (v1.0.1: 8 on C8192/C16384/C28080,
    # added into TOTAL_GPUS so the Slurm client got its own node). The GKE
    # recipes only ever quoted it in a comment -- the client shares a node
    # here -- but a drop that brings it back changes the Slurm allocation and
    # SLURM_NODES, so stop and re-review rather than silently carry it.
    if "CLIENT_RESERVE_GPUS" in cfg or "CLIENT_RESERVE_GPUS" in sv("TOTAL_GPUS"):
        sys.exit(f"point_{point}: dsr1_config.yaml uses CLIENT_RESERVE_GPUS again "
                 "(the v1.0.2 drop removed it) -- re-review CLIENT_RESERVE_NODES "
                 "and EXPECTED SLURM_NODES before generating.")
    # Checkpoint / dataset / image pins: the GKE recipe hardcodes node-local
    # equivalents (assert_gke_consistency), so a drop that moves any of these
    # must be re-ported by hand, never followed silently.
    for name, want in (
        ("MODEL_PATH", DROP_MODEL_PATH),
        ("DATA_PATH", DROP_DATA_PATH),
        ("CONTAINER_IMAGE", DROP_IMAGES["CONTAINER_IMAGE"][1]),
    ):
        if sv(name) != want:
            sys.exit(f"point_{point}: dsr1_config.yaml {name} is {sv(name)!r}, "
                     f"expected {want!r} ({DROP_VERSION}). Refusing to generate -- "
                     "re-check the checkpoint / dataset / image pins first.")
    server_env = read_flat_env(d / "server-env.yaml", f"point_{point}")
    want_env = dict(SERVER_ENV_KEEP)
    want_env.update({k: v for k, (v, _why) in SERVER_ENV_DROP.items()})
    if server_env != want_env:
        delta = sorted(set(server_env.items()) ^ set(want_env.items()))
        sys.exit(f"point_{point}: server-env.yaml no longer matches SERVER_ENV_KEEP"
                 f" + SERVER_ENV_DROP; differing (key, value) pairs: {delta}. "
                 "Classify each per SLURM_TO_GKE.md §5.1 (and update "
                 "gke_compat_block + MPIRUN_X) before generating.")
    endpoint_env = read_flat_env(d / "endpoint_env.yaml", f"point_{point}")
    if endpoint_env != ENDPOINT_ENV:
        delta = sorted(set(endpoint_env.items()) ^ set(ENDPOINT_ENV.items()))
        sys.exit(f"point_{point}: endpoint_env.yaml no longer matches ENDPOINT_ENV;"
                 f" differing (key, value) pairs: {delta}. Re-port the benchmark "
                 "task's exports before generating.")

    client_text = (d / "client.yaml").read_text()
    client = yaml.safe_load(client_text)

    info = dict(
        point=point,
        ctx_n=iv("NUM_CTX_SERVERS"),
        ctx_tp=iv("GPUS_PER_CTX_RANK"),
        gen_n=iv("NUM_GEN_SERVERS"),
        gen_tp=iv("GPUS_PER_GEN_RANK"),
        frontends=iv("NUM_FRONTEND_SERVERS"),
        base_port=iv("BASE_PORT"),
        frontend_port=iv("FRONTEND_PORT"),
        # v1.0.1 read CLIENT_RESERVE_GPUS here; v1.0.2 has none (refused
        # above), so it is 0 on every point. Kept as a field because the
        # variant comments still branch on it.
        client_reserve_gpus=0,
        slurm_mode=cfg["MODE"]["value"],
        concurrency=int(client["settings"]["load_pattern"]["target_concurrency"]),
        n_samples=int(client["settings"]["runtime"]["n_samples_to_issue"]),
        max_new_tokens=int(client["model_params"]["max_new_tokens"]),
        ctx_yaml=(d / "server-ctx.yaml").read_text(),
        gen_yaml=(d / "server-gen.yaml").read_text(),
        client_yaml=client_text,
    )
    info["ctx_gpus"] = info["ctx_n"] * info["ctx_tp"]
    info["gen_gpus"] = info["gen_n"] * info["gen_tp"]
    info["total_gpus"] = info["ctx_gpus"] + info["gen_gpus"]
    info["server_nodes"] = ceil_div(info["total_gpus"], GPUS_PER_NODE)
    info["slurm_nodes"] = ceil_div(
        info["total_gpus"] + info["client_reserve_gpus"], GPUS_PER_NODE
    )
    return info


def assert_drop_shared(slurm_root: pathlib.Path) -> None:
    """The drop-wide file the recipes depend on (configs/_shared/
    slurm_env.yaml). Images must still be the ones DROP_IMAGES maps onto
    recipe variables, and SITE_ENV_YAML (new in v1.0.2: an optional extra env
    file the Slurm template passes to run_with_env.py after server-env.yaml)
    must still default to empty -- the GKE compat layer is this port's site
    overlay."""
    path = slurm_root / "configs" / "_shared" / "slurm_env.yaml"
    if not path.is_file():
        sys.exit(f"drop file not found: {path}")
    v = yaml.safe_load(path.read_text()).get("variables", {})
    for key, (_recipe_var, want) in DROP_IMAGES.items():
        if key == "CONTAINER_IMAGE":
            continue  # per point, in dsr1_config.yaml (read_point)
        got = str(v.get(key, {}).get("value"))
        if got != want:
            sys.exit(f"{path}: {key} is {got!r}, expected {want!r} "
                     f"({DROP_VERSION}). Update DROP_IMAGES and the recipe "
                     "image defaults together.")
    if "SITE_ENV_YAML" not in v or str(v["SITE_ENV_YAML"].get("value") or "") != "":
        sys.exit(f"{path}: SITE_ENV_YAML is missing or no longer defaults to '' "
                 "-- decide whether the GKE port needs that overlay first.")


def _yaml_var_default(name: str) -> str:
    """The `value:` the RECIPE_TEMPLATE gives variable `name` (unrendered).
    Only lines indented >= 4 (or blank, inside a folded description) belong
    to the variable, so the match cannot run into the next variable."""
    m = re.search(rf"^  {re.escape(name)}:\n(?:(?:    .*)?\n)*?    value: (.*)$",
                  RECIPE_TEMPLATE, re.M)
    if not m:
        sys.exit(f"RECIPE_TEMPLATE has no variable {name}")
    return m.group(1).strip().strip('"')


def assert_gke_consistency() -> None:
    """The GKE side of the v1.0.2 pins: what the recipes will actually use
    must equal what the drop and ./stage-assets.yaml say. Fails closed."""
    problems = []
    for name, want in (("MODEL_PATH", GKE_MODEL_PATH), ("DATA_PATH", GKE_DATA_PATH)):
        if _yaml_var_default(name) != want:
            problems.append(f"RECIPE_TEMPLATE {name} is {_yaml_var_default(name)!r}, "
                            f"expected {want!r}")
    for key, (recipe_var, drop_ref) in DROP_IMAGES.items():
        want = drop_ref.removeprefix("docker://")
        if _yaml_var_default(recipe_var) != want:
            problems.append(f"RECIPE_TEMPLATE {recipe_var} default is "
                            f"{_yaml_var_default(recipe_var)!r}; the drop's {key} "
                            f"is {want!r}")
    compat = gke_compat_block("ctx") + "\n" + gke_compat_block("gen")
    for k, val in SERVER_ENV_KEEP.items():
        if compat.count(f"        - export {k}={val}\n") != 2:
            problems.append(f"gke_compat_block does not export {k}={val} "
                            "(server-env.yaml keeps it) on both roles")
        if f"          -x {k}={val}\n" not in MPIRUN_X + "\n":
            problems.append(f"MPIRUN_X does not forward -x {k}={val}")
    for k in ENDPOINT_ENV_EXPORTED:
        if f"- export {k}={ENDPOINT_ENV[k]}\n" not in RECIPE_TEMPLATE:
            problems.append(f"benchmark task does not export {k}={ENDPOINT_ENV[k]}"
                            " (endpoint_env.yaml)")
    stage = HERE / "stage-assets.yaml"  # read only; written by hand, not here
    if not stage.is_file():
        problems.append(f"{stage} not found (it stages the v1.0.2 checkpoint)")
    else:
        st = stage.read_text()
        model_dir = "/model-cache/" + GKE_MODEL_PATH.rsplit("/", 1)[1]
        for needle in (f'SRC_MODEL: "hf://{CHECKPOINT}"', f"MODEL_DIR={model_dir}",
                       GKE_MODEL_CACHE):
            if needle not in st:
                problems.append(f"{stage.name} does not contain {needle!r}")
    if problems:
        sys.exit("GKE side drifted from the v1.0.2 pins:\n  " + "\n  ".join(problems))


def assert_spec_table(info: dict) -> None:
    point = info["point"]
    want = EXPECTED[point]
    got = (
        info["ctx_n"], info["ctx_tp"], info["gen_n"], info["gen_tp"],
        info["frontends"], info["concurrency"], info["n_samples"],
        info["max_new_tokens"], info["slurm_nodes"],
    )
    if got != want:
        sys.exit(
            f"point_{point}: the drop no longer matches conversion spec §1.\n"
            f"  spec: {want}\n  drop: {got}\n"
            "Refusing to generate -- reconcile the spec with the drop first."
        )
    if info["total_gpus"] != 72:
        sys.exit(f"point_{point}: expected 72 server GPUs, drop has {info['total_gpus']}")


# ==========================================================================
# Reduced-node variants: choosing the prefill/decode split
# ==========================================================================


def extend_ladder(ladder: list[int], mbs: int) -> list[int]:
    """NVIDIA's CUDA-graph ladder, extended so its top equals *mbs*.

    Unchanged if it already reaches mbs. Otherwise new rungs are added in the
    same spacing NVIDIA uses at the top of its own ladders -- 1/8 of the
    largest power of two <= mbs (…, 40, 48, 56, 64 / …, 80, 96, 112, 128 /
    …, 320, 384, 448, 512) -- and mbs itself is the last rung. With
    enable_padding a batch is padded up to the next rung, so a gap right below
    the top would waste up to that much decode compute every step.
    """
    if mbs <= max(ladder):
        return list(ladder)
    step = max(1, (1 << (mbs.bit_length() - 1)) // 8)
    out = list(ladder)
    rung = (max(ladder) // step + 1) * step
    while rung < mbs:
        out.append(rung)
        rung += step
    out.append(mbs)
    return out


def plan_variant(info: dict, nodes: int) -> dict:
    """Pick the prefill/decode split that should perform best on *nodes* nodes.

    Every Slurm point is 8 prefill GPUs + 64 decode GPUs. In 64 GPUs that
    cannot be kept, so the decode shape has to change -- for the six DEP64
    points there is no way around it at all. The rule, in order:

    1. PREFILL SERVERS KEEP NVIDIA'S SHAPE AND CONFIG, byte for byte, and the
       prefill pool never shrinks below the drop's 8 GPUs.

    2. DECODE uses a shape NVIDIA certifies in this drop (VARIANT_GEN_TPS) and
       as many servers of it as fit next to that prefill pool:
         1 x DEP32 (32 GPUs)  or  3 x DEP16 (48 GPUs).

    3. Which one depends on the per-rank decode batch b = ceil(C / ranks) --
       each attention-DP rank schedules its own batch, so b is what one GPU
       actually runs.
         b <= MAX_EP_RANK_BATCH on the largest EP: take the largest EP.
           Decode here is weight-streaming bound: the point_256 GKE run
           measured the SAME ~13.6 ms step at batch 1, 2, 3 and 4
           (deepseek-r1/point_256/TUNING_NOTES.md). Extra ranks add nothing
           while a larger EP streams fewer expert bytes per GPU per step.
           NVIDIA makes the same choice: one DEP64 up to b = 16 (point_1024).
         otherwise: take the most decode ranks. Step time now grows with b,
           so spreading C over 48 ranks instead of 32 buys throughput -- the
           same reason NVIDIA leaves max-EP for 2 x DEP32 / 4 x DEP16 above
           point_1024.

    4. max_batch_size = max(drop value, ceil(C / ranks)) so every one of the
       C in-flight requests has a decode slot, as in every NVIDIA point
       (gen_n x TP x max_batch_size >= C). max_num_tokens scales with it and
       the CUDA-graph ladder is extended to the new top (extend_ladder).

    5. Every GPU left over goes to MORE PREFILL servers of the drop's shape.
       In a closed-loop concurrency run each request's TTFT is part of its
       lifetime, so lower prefill queueing is throughput, not only latency.
    """
    budget = nodes * GPUS_PER_NODE
    conc = info["concurrency"]
    min_ctx_gpus = info["ctx_gpus"]

    cands = []
    for tp in VARIANT_GEN_TPS:
        gen_n = (budget - min_ctx_gpus) // tp
        if gen_n < 1:
            continue
        cands.append(dict(gen_tp=tp, gen_n=gen_n, ranks=gen_n * tp,
                          rank_batch=ceil_div(conc, gen_n * tp)))
    if not cands:
        sys.exit(f"point_{info['point']}: no decode shape fits {nodes} nodes")

    largest_ep = max(cands, key=lambda c: c["gen_tp"])
    if largest_ep["rank_batch"] <= MAX_EP_RANK_BATCH:
        best = largest_ep
        regime = (f"per-rank batch {best['rank_batch']} <= {MAX_EP_RANK_BATCH}: "
                  f"weight-streaming bound, largest EP wins")
    else:
        best = max(cands, key=lambda c: (c["ranks"], c["gen_tp"]))
        regime = (f"per-rank batch on DEP{largest_ep['gen_tp']} would be "
                  f"{largest_ep['rank_batch']} > {MAX_EP_RANK_BATCH}: "
                  f"most decode ranks wins")

    gen_gpus = best["gen_n"] * best["gen_tp"]
    ctx_n = (budget - gen_gpus) // info["ctx_tp"]

    gen = yaml.safe_load(info["gen_yaml"])
    old_mbs, old_mnt = gen["max_batch_size"], gen["max_num_tokens"]
    mbs = max(old_mbs, best["rank_batch"])
    mnt = ceil_div(old_mnt * mbs, old_mbs)
    ladder = extend_ladder(gen["cuda_graph_config"]["batch_sizes"], mbs)

    plan = dict(
        nodes=nodes, ctx_n=ctx_n, gen_n=best["gen_n"], gen_tp=best["gen_tp"],
        mbs=mbs, mnt=mnt, ladder=ladder, regime=regime, cands=cands,
        old_tp=gen["tensor_parallel_size"], old_mbs=old_mbs, old_mnt=old_mnt,
        old_ladder=list(gen["cuda_graph_config"]["batch_sizes"]),
    )

    want = EXPECTED_VARIANT[nodes][info["point"]]
    got = (ctx_n, best["gen_n"], best["gen_tp"], mbs)
    if got != want:
        sys.exit(
            f"point_{info['point']}: the {nodes}-node split is now {got} "
            f"(ctxN, genN, genTP, max_batch_size); EXPECTED_VARIANT says {want}.\n"
            "The drop or the rule changed -- re-read plan_variant() before "
            "updating the table."
        )
    if gen_n_slots(plan) < conc:
        sys.exit(f"point_{info['point']}: {nodes}-node plan has fewer decode "
                 f"slots than target_concurrency {conc}")
    if info["frontends"] > nodes:
        sys.exit(f"point_{info['point']}: {info['frontends']} frontends need "
                 f"one node each, only {nodes}")
    return plan


def gen_n_slots(plan: dict) -> int:
    return plan["gen_n"] * plan["gen_tp"] * plan["mbs"]


def reshape_gen(text: str, point: int, plan: dict) -> str:
    """Rewrite the drop's server-gen.yaml to the variant's decode shape.

    Only the RESHAPE_KEYS change, each one preceded by a comment naming the
    value NVIDIA ships. Line-oriented and assertive, like apply_server_edits:
    every key must be found exactly once.
    """
    nodes = plan["nodes"]
    scalars = {
        "tensor_parallel_size": plan["gen_tp"],
        "moe_expert_parallel_size": plan["gen_tp"],
        "max_batch_size": plan["mbs"],
        "max_num_tokens": plan["mnt"],
    }
    seen = {k: 0 for k in (*scalars, "batch_sizes")}
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip("\n")
        m = re.match(r"^([a-z_]+): (\d+)$", stripped)
        if m and m.group(1) in scalars:
            key, old = m.group(1), int(m.group(2))
            seen[key] += 1
            new = scalars[key]
            if new != old:
                out.append(f"# {nodes}-NODE RESHAPE: point_{point} ships {key}: {old}.\n")
                out.append(f"{key}: {new}\n")
            else:
                out.append(line)
            i += 1
            continue
        if stripped == "  batch_sizes:":
            seen["batch_sizes"] += 1
            j = i + 1
            old_ladder = []
            while j < len(lines) and re.match(r"^  - \d+$", lines[j].rstrip("\n")):
                old_ladder.append(int(lines[j].split("-", 1)[1]))
                j += 1
            if old_ladder != plan["ladder"]:
                out.append(
                    f"  # {nodes}-NODE RESHAPE: point_{point} ships {old_ladder}; "
                    f"extended to cover max_batch_size {plan['mbs']}.\n"
                )
            out.append(line)
            out.extend(f"  - {b}\n" for b in plan["ladder"])
            i = j
            continue
        out.append(line)
        i += 1

    bad = {k: n for k, n in seen.items() if n != 1}
    assert not bad, f"point_{point}: reshape keys not found exactly once: {bad}"
    result = "".join(out)
    cfg = yaml.safe_load(result)
    assert cfg["tensor_parallel_size"] == plan["gen_tp"]
    assert cfg["moe_expert_parallel_size"] == plan["gen_tp"]
    assert cfg["max_batch_size"] == plan["mbs"]
    assert cfg["max_num_tokens"] == plan["mnt"]
    assert cfg["cuda_graph_config"]["batch_sizes"] == plan["ladder"]
    return result


def variant_info(info: dict, plan: dict) -> dict:
    """The Slurm point with the variant's topology and reshaped decode config.

    Everything downstream (render_recipe, check_point) works on this dict
    exactly as on a Slurm point; `variant` / `slurm` mark it as a variant.
    """
    v = dict(info)
    v.update(
        ctx_n=plan["ctx_n"], gen_n=plan["gen_n"], gen_tp=plan["gen_tp"],
        gen_yaml=reshape_gen(info["gen_yaml"], info["point"], plan),
        variant=plan, slurm=info,
    )
    v["ctx_gpus"] = v["ctx_n"] * v["ctx_tp"]
    v["gen_gpus"] = v["gen_n"] * v["gen_tp"]
    v["total_gpus"] = v["ctx_gpus"] + v["gen_gpus"]
    v["server_nodes"] = ceil_div(v["total_gpus"], GPUS_PER_NODE)
    if v["total_gpus"] != plan["nodes"] * GPUS_PER_NODE or v["server_nodes"] != plan["nodes"]:
        sys.exit(f"point_{info['point']}: {plan['nodes']}-node plan uses "
                 f"{v['total_gpus']} GPUs / {v['server_nodes']} nodes")
    ctx_nodes = {ctx_node(i, v["ctx_tp"]) for i in range(v["ctx_n"])}
    gen_nodes = set()
    for i in range(v["gen_n"]):
        start = gen_base_gpu(v["ctx_gpus"], v["gen_tp"]) + i * v["gen_tp"]
        gen_nodes |= {g // GPUS_PER_NODE for g in range(start, start + v["gen_tp"])}
    if ctx_nodes & gen_nodes or (ctx_nodes | gen_nodes) != set(range(plan["nodes"])):
        sys.exit(f"point_{info['point']}: {plan['nodes']}-node plan does not "
                 f"tile nodes 0..{plan['nodes'] - 1} (ctx {sorted(ctx_nodes)}, "
                 f"gen {sorted(gen_nodes)})")
    return v


def diag_gen_yaml(info: dict, name: str, new_tp: int) -> str:
    """The point's server-gen.yaml with ONLY DIAG_GEN_KEYS set to new_tp, each
    changed line marked inline. Fails closed if the keys are not exactly
    where and what the point says."""
    out, hit = [], []
    for line in info["gen_yaml"].splitlines(keepends=True):
        m = re.match(r"^([a-z_]+): (\d+)\n?$", line)
        if m and m.group(1) in DIAG_GEN_KEYS:
            if int(m.group(2)) != info["gen_tp"]:
                sys.exit(f"diag {name}: {m.group(1)} is {m.group(2)} in point_"
                         f"{info['point']}, expected {info['gen_tp']}")
            out.append(f"{m.group(1)}: {new_tp}  # DIAG {name}: NVIDIA point_"
                       f"{info['point']} has {m.group(2)} "
                       f"({info['gen_n']} x DEP{info['gen_tp']})\n")
            hit.append(m.group(1))
        else:
            out.append(line)
    if sorted(hit) != sorted(DIAG_GEN_KEYS):
        sys.exit(f"diag {name}: expected exactly one top-level line each for "
                 f"{DIAG_GEN_KEYS} in point_{info['point']} server-gen.yaml, "
                 f"found {hit}")
    return "".join(out)


def diag_info(info: dict, name: str) -> dict:
    """An 18-node point with a DIAGNOSTICS decode split. Same 72 GPUs / 18
    nodes, same per-rank decode config; render_recipe() works on it as on a
    point, apply_diag_text() then marks it. Every invariant fails closed."""
    spec = DIAGNOSTICS[name]
    point, gen_n, gen_tp = spec.get("point"), spec.get("gen_n"), spec.get("gen_tp")
    if not all(type(x) is int and x > 0 for x in (point, gen_n, gen_tp)) or \
            not isinstance(spec.get("purpose"), str) or not spec["purpose"].strip() or \
            not re.fullmatch(r"[a-z0-9]+", name):
        sys.exit(f"diag {name!r}: DIAGNOSTICS needs a [a-z0-9]+ name, positive int "
                 "point / gen_n / gen_tp and a purpose")
    if info["point"] != point or info.get("variant") or "drop_n_samples" in info:
        sys.exit(f"diag {name}: needs the default-mode 18-node point_{point}")
    if (gen_n, gen_tp) == (info["gen_n"], info["gen_tp"]):
        sys.exit(f"diag {name}: {gen_n} x DEP{gen_tp} IS point_{point}'s decode split")
    d = dict(info)
    d.update(gen_n=gen_n, gen_tp=gen_tp,
             gen_yaml=diag_gen_yaml(info, name, gen_tp), base=info)
    d["gen_gpus"] = gen_n * gen_tp
    d["total_gpus"] = d["ctx_gpus"] + d["gen_gpus"]
    d["server_nodes"] = ceil_div(d["total_gpus"], GPUS_PER_NODE)
    gen_cfg = yaml.safe_load(d["gen_yaml"])
    slots = gen_n * gen_tp * gen_cfg["max_batch_size"]
    # Node math, exactly as the GEN_URLS artifact computes it (gen_base is a
    # whole-node boundary for gen TP > 4): server i -> node
    # (gen_base + i * GPUS_PER_GEN_RANK) // 4, port BASE_PORT + NUM_CTX + i.
    gen_base = gen_base_gpu(d["ctx_gpus"], gen_tp)
    url_nodes = [gen_node(i, d["ctx_gpus"], gen_tp) for i in range(gen_n)]
    url_ports = [d["base_port"] + d["ctx_n"] + i for i in range(gen_n)]
    # Nodes each server's GPUs land on (never empty; sub-node servers overlap,
    # which the tiling check below rejects).
    ctx_nodes = {gpu // GPUS_PER_NODE for gpu in range(d["ctx_gpus"])}
    gen_nodes = [{gpu // GPUS_PER_NODE for gpu in range(gen_base + i * gen_tp,
                                                        gen_base + (i + 1) * gen_tp)}
                 for i in range(gen_n)]
    problems = []
    if d["gen_gpus"] != info["gen_gpus"] or d["total_gpus"] != info["total_gpus"]:
        problems.append(f"{gen_n} x DEP{gen_tp} = {d['gen_gpus']} decode GPUs, "
                        f"point has {info['gen_gpus']}")
    if d["server_nodes"] != info["server_nodes"]:
        problems.append(f"{d['server_nodes']} server nodes, point has {info['server_nodes']}")
    if gen_tp % GPUS_PER_NODE:
        problems.append(f"DEP{gen_tp} is not whole nodes")
    if slots < d["concurrency"]:
        problems.append(f"{slots} decode slots < target_concurrency {d['concurrency']}")
    if gen_cfg["tensor_parallel_size"] != gen_tp or gen_cfg["moe_expert_parallel_size"] != gen_tp:
        problems.append("server-gen TP/EP != GPUS_PER_GEN_RANK")
    if any(ctx_nodes & g for g in gen_nodes) or \
            set().union(ctx_nodes, *gen_nodes) != set(range(d["server_nodes"])) or \
            sum(len(g) for g in gen_nodes) != d["gen_gpus"] // GPUS_PER_NODE:
        problems.append(f"nodes do not tile 0..{d['server_nodes'] - 1} "
                        f"(ctx {sorted(ctx_nodes)}, gen {[sorted(g) for g in gen_nodes]})")
    if url_nodes != [min(g) for g in gen_nodes]:
        problems.append(f"gen URL nodes {url_nodes} are not the first node of "
                        f"each gen server {[min(g) for g in gen_nodes]}")
    if problems:
        sys.exit(f"diag {name}: " + "; ".join(problems))
    d["diag"] = dict(
        name=name, purpose=spec["purpose"], slots=slots, gen_base=gen_base,
        url_nodes=url_nodes, url_ports=url_ports,
        gen_nodes=[(min(g), max(g)) for g in gen_nodes],
        mbs=gen_cfg["max_batch_size"],
    )
    return d


def label_of(info: dict) -> str:
    if info.get("diag"):
        return f"point_{info['point']}_diag_{info['diag']['name']}"
    if info.get("variant"):
        return f"point_{info['point']}_{info['variant']['nodes']}nodes"
    return f"point_{info['point']}"


def nodes_of(info: dict) -> int:
    if info.get("variant"):
        return info["variant"]["nodes"]
    return info["server_nodes"]


def recipe_dir(info: dict, dest: pathlib.Path) -> pathlib.Path:
    # Grouped by total node count: 18nodes/point_<C>/, 16nodes/point_<C>/;
    # diagnostics live apart, in diag/point_<C>_<name>/.
    if info.get("diag"):
        return dest / DIAG_DIR / f"point_{info['point']}_{info['diag']['name']}"
    return dest / f"{nodes_of(info)}nodes" / f"point_{info['point']}"


def recipe_name(info: dict) -> str:
    if info.get("diag"):
        return (f"deepseek_r1_gke_disagg_point{info['point']}_diag_"
                f"{info['diag']['name']}.yaml")
    if info.get("variant"):
        return (f"deepseek_r1_gke_disagg_point{info['point']}_"
                f"{info['variant']['nodes']}nodes.yaml")
    return f"deepseek_r1_gke_disagg_point{info['point']}.yaml"


# ==========================================================================
# The ONE base recipe. Placeholders are @@NAME@@ (str.replace, not .format,
# because the body is full of Jinja braces).
# ==========================================================================

RECIPE_TEMPLATE = r'''version: "0.1"

# ============================================================================
# DeepSeek-R1 -- MLPerf Endpoints v1.0.2, point C@@POINT@@ (disaggregated) on
# GKE A4X-max (GB300 NVL72).
# ============================================================================
#
# GENERATED FILE -- edit ../../generate_point_recipes.py and re-run. The GKE layer
# lives in exactly one place (that script's RECIPE_TEMPLATE) so a fix here
# propagates to all twelve points instead of being hand-copied eleven times.
#
# GKE port of the NVIDIA Slurm recipe:
#   nv-mlpinf-partner-partner-endpoints-v1.0.2/NVIDIA/src/deepseek-r1/
#     @@ARCH@@/point_@@POINT@@/
#       dsr1_config.yaml   topology (below)
#       server-ctx.yaml    pasted verbatim into EOF_CTX
#       server-gen.yaml    pasted verbatim into EOF_GEN
#       client.yaml        emitted verbatim as the CLIENT_CONFIG artifact
#       server-env.yaml    runtime flags kept in the GKE layer; UCX_* replaced
#       endpoint_env.yaml  client cache dirs, exported verbatim (benchmark)
#     configs/_shared/templates/trtllm_disagg_endpoints_dsr1_perf_acc.yaml
#                          task graph (snapshot_configs / servers / frontends /
#                          lcb_service / endpoint_benchmark)
#     configs/_shared/slurm_env.yaml   site paths + images -- DISCARDED
#
# CHECKPOINT (v1.0.2 README pin, the same for every point):
#   @@CHECKPOINT@@
# staged per node into MODEL_PATH by ../../stage-assets.yaml. It is NOT the
# v1.0.1 checkpoint (nvidia/DeepSeek-R1-NVFP4); results from the two are not
# comparable.
#
# Ported per ../../../../SLURM_TO_GKE.md. The split that matters:
#
#   MODEL / WORKLOAD half -- COPIED VERBATIM from the drop:
#     every line inside EOF_CTX / EOF_GEN, the whole CLIENT_CONFIG artifact,
#     the topology variables, the concurrency and sample counts. These are
#     NVIDIA's certified numbers. Do not re-tune them.
#
#   FABRIC / LAUNCHER half -- REPLACED with the GKE compatibility layer:
#     srun + Pyxis/Enroot        -> kubernetes backend + MPI operator
#     --segment=N NVLink place.  -> compute_domain: {create: true}   (MNNVL)
#     RoCE/IB NCCL + UCX rc_x    -> rdma: disable + UCX cuda_ipc/MNNVL
#     shared Lustre/NFS          -> node-local SSD hostPath (see Staging)
#     shared-FS URL handshake    -> CTX_URLS / GEN_URLS / FRONTEND_URLS
#     ${SFLOW_TASK_OUTPUT_DIR}   -> durable hostPath ${REPORT_DIR}
#
# Porting the fabric half FROM Slurm is the single biggest time sink: those
# settings are correct on a RoCE cluster, wrong here, and they fail SILENTLY --
# a healthy server, zero errors, and up to 10x less throughput.
#
# TOPOLOGY (identical to Slurm point_@@POINT@@):
#   @@CTX_N@@ ctx x DEP@@CTX_TP@@ + @@GEN_N@@ gen x DEP@@GEN_TP@@
#     = @@CTX_GPUS@@ + @@GEN_GPUS@@ = @@TOTAL_GPUS@@ GPUs = @@SERVER_NODES@@ nodes (a full NVL72).
#   @@PLACEMENT_COMMENT@@
#
# BEFORE YOU RUN -- on EVERY node of the pool (see ../../README.md §3):
#   kubectl apply -f ../../stage-assets.yaml      # model + eval parquet
#   kubectl apply -f ../../mk-report-dir.yaml     # durable report hostPath
#   kubectl apply -f ../../prepull-images.yaml    # optional but recommended
# On Kubernetes a missing fs:// path is only a WARNING at plan time -- the run
# starts anyway and the benchmark silently falls back to a synthetic dataset.
# The benchmark task below hard-fails instead; confirm from the log regardless.
# ============================================================================

variables:
  # ---------------------------------------------------------------------
  # POINT TOPOLOGY -- verbatim from point_@@POINT@@/dsr1_config.yaml.
  # ---------------------------------------------------------------------
  NUM_CTX_SERVERS:
    description: "Prefill (context) server replicas."
    type: integer
    value: @@CTX_N@@
  GPUS_PER_CTX_RANK:
    description: "GPUs per prefill replica == its TP/EP size."
    type: integer
    value: @@CTX_TP@@
  NUM_GEN_SERVERS:
    description: "Decode (generation) server replicas."
    type: integer
    value: @@GEN_N@@
  GPUS_PER_GEN_RANK:
    description: "GPUs per decode replica == its TP/EP size."
    type: integer
    value: @@GEN_TP@@
  NUM_FRONTEND_SERVERS:
    description: "trtllm-serve disaggregated frontends (Slurm: @@FRONTENDS@@)."
    type: integer
    value: @@FRONTENDS@@
  BASE_PORT:
    description: "ctx server i listens on BASE_PORT+i; gen server i on BASE_PORT+NUM_CTX_SERVERS+i."
    type: integer
    value: @@BASE_PORT@@
  FRONTEND_PORT:
    description: "Frontend replica i listens on FRONTEND_PORT+i."
    type: integer
    value: @@FRONTEND_PORT@@

  # ---------------------------------------------------------------------
  # NODE MATH -- conversion spec §2, verified against
  # sflow/app/resource_planner.py.
  #
  #   gpus.count <= gpus_per_node  -> the planner takes the first contiguous
  #       free GPU run on the first node that has one, so sub-node servers
  #       tightly BACKFILL, including across the ctx->gen boundary.
  #   gpus.count >  gpus_per_node  -> requires count % gpus_per_node == 0 and
  #       picks whole free nodes.
  #
  # For DeepSeek-R1 ctx is always exactly 8 GPUs = 2 whole nodes and gen TP is
  # always >= 16, so ctx lands on nodes 0-1 and gen server i starts at
  # node 2 + i*(genTP/4). The CTX_URLS / GEN_URLS artifacts below still
  # implement the GENERAL packing-aware formula, because the naive
  # `ceil(ctx_gpus / gpus_per_node)` form is wrong the moment a gen server
  # becomes smaller than a node (it is wrong today for GPT-OSS point_35/70/140)
  # and a wrong gen node index does not fail -- it just points every frontend
  # at a host that is not serving.
  # ---------------------------------------------------------------------
  GPUS_PER_NODE:
    description: "A4X-max / GB300 NVL72: 4 GPUs per host."
    type: integer
    value: 4
  SERVER_NODES:
    description: "ceil(@@TOTAL_GPUS@@ GPUs / 4) -- the nodes the servers occupy."
    type: integer
    value: @@SERVER_NODES@@
  CLIENT_RESERVE_NODES:
    description: >-
      Extra nodes reserved for the client, held out beyond the server nodes.

      The v1.0.2 drop removed CLIENT_RESERVE_GPUS (v1.0.1 set 8 on
      C8192/C16384/C28080 to give the Slurm client its own node), so every
      Slurm point now allocates exactly ceil(TOTAL_GPUS / 4) = 18 nodes and its
      endpoints client shares a host with a server rank -- the same placement
      GKE always used: the client is a CPU-only pod under host_network and the
      proven point_256 run co-scheduled it on node 0 with ctx server 0 at
      45,760 tok/s. So the default is 0 and the allocation stays at
      @@SERVER_NODES@@ nodes, i.e. exactly the 18-node pool.

      `-s CLIENT_RESERVE_NODES=N` gives the client N node(s) of its own (this
      is NOT the Slurm allocation, and needs 18+N nodes); the benchmark and
      lcb_service tasks then move to node @@SERVER_NODES@@.
    type: integer
    value: 0
  TOTAL_NODES:
    type: integer
    value: ${{ variables.SERVER_NODES + variables.CLIENT_RESERVE_NODES }}
  BENCH_NODE_INDEX:
    description: >-
      Node index for the CPU-only client + LCB + snapshot_configs pods: 0 when
      CLIENT_RESERVE_NODES == 0 (shares node 0 with prefill server 0, leaving
      the synchronous multi-node decode collective on nodes 2..17 completely
      undisturbed), or SERVER_NODES (the first reserved client node) when
      CLIENT_RESERVE_NODES > 0.
    type: integer
    value: ${{ variables.SERVER_NODES if variables.CLIENT_RESERVE_NODES > 0 else 0 }}

  # ---------------------------------------------------------------------
  # GKE SITE -- replaces configs/_shared/slurm_env.yaml.
  # ---------------------------------------------------------------------
  K8S_NAMESPACE:
    value: default
  GKE_NODEPOOL:
    description: "cloud.google.com/gke-nodepool selector for the A4X-max pool."
    value: <nodepool>
  COMPUTE_DOMAIN_CHANNEL:
    value: auto

  MODEL_PATH:
    description: >-
      DeepSeek-R1 NVFP4 checkpoint + tokenizer, on each node's local SSD. v1.0.2
      pins centml/DeepSeek-R1-NVFP4-v2-mlpinf@93947a0d7bd04f73ff98636f6b18ff5839e7aaf9
      (Slurm MODEL_PATH ${SCRATCH_DIR}/models/DeepSeek-R1-NVFP4-v2; v1.0.1 used
      .../deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4, a different
      checkpoint). GKE has no shared FS, so it is staged per node by
      ../../stage-assets.yaml (DaemonSet stage-dsr1-v2-assets), which writes the
      revision into MODEL_PATH/.sflow-staged. The v1.0.1 checkpoint stays in
      model-cache/DeepSeek-R1-NVFP4 next to it; do not point v1.0.2 runs at it.
    value: <node-local-ssd>/model-cache/DeepSeek-R1-NVFP4-v2
  DATA_PATH:
    description: >-
      Official MLPerf DSR1 eval parquet (Slurm DATA_PATH; v1.0.2 moved the Slurm
      default to ${SCRATCH_DIR}/data/deepseek_r1_eval.parquet, same file, same
      node-local path here). Used for BOTH the
      `dsr1-perf` performance dataset and the `deepseek_r1_accuracy` dataset in
      client.yaml. NOT cosmetic: on the synthetic fallback, MTP acceptance
      collapsed from ~2.58 to 2.03 of 4 tokens per step and throughput read
      38.2k instead of 45.8k with no config change. A run on the wrong dataset
      cannot be compared to a Slurm number in either direction.
    value: <node-local-ssd>/mlperf-data/deepseek_r1_eval.parquet

  MODE:
    description: >-
      perf | acc | both -- passed to `inference-endpoint benchmark --mode`.

      The Slurm point sets MODE=@@SLURM_MODE@@. Default is `both` so every
      submission run executes both performance and accuracy unless explicitly
      overridden with `-s MODE=perf`.
    value: both
  WITH_LCB:
    description: >-
      Start the LiveCodeBench service (true|false).

      Both Slurm templates launch lcb_service in EVERY mode, including perf, and
      the DSR1 README says to provide all three images. It is ported here as a
      CPU-only task pinned to the client's node; with host_network: true the
      client reaches it on localhost:13835, matching Slurm exactly.

      Set false only if the LCB image is unavailable at your site. The task then
      runs a placeholder on an image that is already present (see the `lcb`
      operator below) so the dependency graph is unchanged. Do not set false for
      an accuracy submission run.
    value: "true"

  TRTLLM_IMAGE:
    description: >-
      Server / frontend image. This is the tag the endpoints v1.0.2 drop
      certifies (point_@@POINT@@/dsr1_config.yaml CONTAINER_IMAGE, unchanged from
      v1.0.1); its arm64
      manifest is sha256:e71eb8056ac1c717e6b727f9d66ebba9442e95f7ebc3514007b541046e84b088.

      FALLBACK, known to work with the runtime patches below:
        nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1
      That is the image the 45,760 tok/s point_256 GKE run actually used. If
      1.3.0rc14 misbehaves, try it before debugging anything else.

      IMPORTANT: the hang_detector patch and the four frontend `sed` patches
      below are pinned to this image's python layout
      (/usr/local/lib/python3.12/dist-packages/tensorrt_llm/...). They are all
      `|| true`, so a wrong path is SILENT -- each one echoes a
      "GKE_PATCH missing" line instead, grep for it after an image change.
    value: nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14
  CLIENT_IMAGE:
    description: >-
      MLCommons Endpoints client, ENDPOINT_CONTAINER_IMAGE from the drop. This
      is Endpoints commit e9f427ce74fd90e7fafdbc77ee7746cc00245692 -- RECORD IT
      WITH THE RESULT, MLPerf requires the client revision.

      ghcr.io is often unreachable from GKE nodes; mirror it and override:
        crane copy ghcr.io/mlcommons/endpoints:e9f427c \
          <registry>/<proj>/<repo>/endpoints-client:e9f427c
        ./launch_test.sh -s CLIENT_IMAGE=<that image>
      It MUST be the arm64 manifest; an amd64-only image crash-loops on these
      nodes with "exec format error".
    value: ghcr.io/mlcommons/endpoints:e9f427c
  LCB_IMAGE:
    description: "LCB_CONTAINER_IMAGE from the drop. Mirror it like CLIENT_IMAGE."
    value: ghcr.io/mlcommons/endpoints:9d749906-livecodebench
  LCB_PORT:
    description: "uvicorn port from the Slurm template. The client expects 13835 on localhost."
    type: integer
    value: 13835

  METRICS_TOKENIZER_WORKERS:
    description: >-
      GKE addition -- see the long comment in the CLIENT_CONFIG artifact.
      NVIDIA's value for this point is @@SLURM_TOKENIZER_WORKERS@@.
    type: integer
    value: 32

artifacts:
  # ---------------------------------------------------------------------
  # Server URL handshake.
  #
  # Slurm has each server's rank 0 append its host:port to a file on shared
  # storage and the frontends read it back. GKE has no shared filesystem, so
  # the addresses are computed up front from the reserved node IPs using the
  # packing-aware node math (conversion spec §2).
  # ---------------------------------------------------------------------
  - name: CTX_URLS
    uri: <workdir>
    content: |
      {# ctx server i is always tightly packed from GPU 0. The `i *` is
         load-bearing: GPUS_PER_CTX_RANK // GPUS_PER_NODE alone is 0 for a
         TP2 server and collapses every ctx replica onto node 0. #}
      {% for i in range(variables.NUM_CTX_SERVERS) %}
      ${{ backends.k8s_cluster.nodes[(i * variables.GPUS_PER_CTX_RANK) // variables.GPUS_PER_NODE].ip_address }}:${{ variables.BASE_PORT + i }}
      {% endfor %}
  - name: GEN_URLS
    uri: <workdir>
    content: |
      {# gen_base: a sub-node gen server BACKFILLS the ctx servers' leftover
         GPUs (planner takes the first contiguous free run); a multi-node one
         must start on a whole-node boundary. Jinja has no ceil, hence the
         (x + n - 1) // n form. For DSR1 gen TP is always >= 16 so this always
         evaluates to the ceil branch, but the general form keeps the template
         correct for any future point. #}
      {% set ctx_gpus = variables.NUM_CTX_SERVERS * variables.GPUS_PER_CTX_RANK %}
      {% set gen_base = ctx_gpus if variables.GPUS_PER_GEN_RANK <= variables.GPUS_PER_NODE else ((ctx_gpus + variables.GPUS_PER_NODE - 1) // variables.GPUS_PER_NODE) * variables.GPUS_PER_NODE %}
      {% for i in range(variables.NUM_GEN_SERVERS) %}
      ${{ backends.k8s_cluster.nodes[(gen_base + i * variables.GPUS_PER_GEN_RANK) // variables.GPUS_PER_NODE].ip_address }}:${{ variables.BASE_PORT + variables.NUM_CTX_SERVERS + i }}
      {% endfor %}
  - name: FRONTEND_URLS
    uri: <workdir>
    content: |
      {# Conversion spec §8: place frontend replicas across the prefill nodes
         (0..ctx_full_nodes-1) so all decode_server nodes (nodes 2..17) stay
         100% free of frontend_server CPU/network interrupt overhead. #}
      {% set ctx_full_nodes = (variables.NUM_CTX_SERVERS * variables.GPUS_PER_CTX_RANK) // variables.GPUS_PER_NODE %}
      {% for i in range(variables.NUM_FRONTEND_SERVERS) %}
      ${{ backends.k8s_cluster.nodes[i % ctx_full_nodes].ip_address }}:${{ variables.FRONTEND_PORT + i }}
      {% endfor %}

  # ---------------------------------------------------------------------
  # CLIENT_CONFIG -- point_@@POINT@@/client.yaml, emitted VERBATIM.
  #
  # Deliberately NOT synthesised in Python from a handful of extracted values,
  # which is what the older DSR1 GKE template did. Emitting the file verbatim
  # is what preserves `submission_ref` (the MLPerf ruleset id), the
  # `deepseek_r1_accuracy` dataset and its legacy_mlperf_deepseek_r1 evaluator,
  # and this point's own `timeouts` / `warmup` blocks -- all of which a
  # hand-rolled dict silently drops.
  #
  # ${VAR} placeholders are expanded by the Endpoints client at load time from
  # the task environment, exactly as on Slurm. ${{ }} is resolved by sflow.
  # Only two lines differ from the drop, both marked GKE EDIT/ADDITION.
  # ---------------------------------------------------------------------
  - name: CLIENT_CONFIG
    uri: <workdir>
    content: |
@@CLIENT_YAML@@

  # ---------------------------------------------------------------------
  # Host mounts. fs:// is a hostPath mount, NOT a fetch: nothing is copied,
  # the bytes must already be on the node (../../stage-assets.yaml).
  # ---------------------------------------------------------------------
  - name: MODEL_CACHE
    uri: fs://${{ variables.MODEL_PATH }}
  - name: MLPERF_DATA
    uri: fs://<node-local-ssd>/mlperf-data
  - name: NVIDIA_DRIVER_LIB
    uri: <workdir>/bin/nvidia/lib64
  - name: IMEX_CHANNELS
    uri: fs:///dev/nvidia-caps-imex-channels
  # Durable results (conversion spec §5, fix 3). sflow's normal path --
  # emptyDir + EXIT trap + a sentinel printed to the pod log -- lost a
  # completed report twice in the Kimi-K3 sweep:
  #   C24  `exec` replaced the shell and destroyed the EXIT trap.
  #   C192 the client went silent for 2 h 05 m during its tokenization drain,
  #        `kubectl logs` stalled for 7,474 s, the sentinel was never
  #        delivered, and the emptyDir died with the pod.
  # Both times the benchmark itself was flawless. Writing to the node's local
  # SSD depends on none of that machinery.
  #
  # Must be pre-created on every node (../../mk-report-dir.yaml): sflow mounts
  # fs:// with a strict `Directory` type, so kubelet refuses the pod rather
  # than silently creating an empty dir.
  - name: REPORTS_DIR
    uri: fs://<node-local-ssd>/dsr1-reports

operators:
  # ---------------------------------------------------------------------
  # TWO MPI operators, not one (conversion spec §2).
  #
  # slots_per_worker must match the POD's GPU count, not the node's. A single
  # shared operator hardcoding GPUS_PER_NODE makes a @@CTX_TP@@-GPU ctx pod advertise 4
  # MPI slots, and mpirun then over-subscribes ranks onto GPUs the pod does not
  # own. min(TP, GPUS_PER_NODE) is the correct value for both roles.
  # ---------------------------------------------------------------------
  - name: mpi_trtllm_ctx
    type: k8s_mpi
    image: ${{ variables.TRTLLM_IMAGE }}
    shm_size: 64Gi
    run_as_root: true
    security_context:
      privileged: true
    mpi:
      route: auto
      slots_per_worker: ${{ [variables.GPUS_PER_CTX_RANK, variables.GPUS_PER_NODE] | min }}
      run_launcher_as_worker: true
      launcher_creation_policy: WaitForWorkersReady
      ssh_port: 2222
      ensure_sshd: true
  - name: mpi_trtllm_gen
    type: k8s_mpi
    image: ${{ variables.TRTLLM_IMAGE }}
    shm_size: 64Gi
    run_as_root: true
    security_context:
      privileged: true
    mpi:
      route: auto
      slots_per_worker: ${{ [variables.GPUS_PER_GEN_RANK, variables.GPUS_PER_NODE] | min }}
      run_launcher_as_worker: true
      launcher_creation_policy: WaitForWorkersReady
      ssh_port: 2222
      ensure_sshd: true
  # CPU-only trtllm pod: the disaggregated frontends and snapshot_configs.
  - name: trtllm_k8s
    type: k8s
    image: ${{ variables.TRTLLM_IMAGE }}
    shm_size: 16Gi
    run_as_root: true
    security_context:
      privileged: true
  - name: client
    type: k8s
    image: ${{ variables.CLIENT_IMAGE }}
    shm_size: 16Gi
    run_as_root: true
  - name: lcb
    type: k8s
    # WITH_LCB=false falls back to the client image rather than skipping the
    # task: sflow has no conditional-task construct, so the task always exists
    # and the DAG never changes shape. Reusing an image the run already needs
    # means a disabled LCB costs no extra pull and cannot ImagePullBackOff on a
    # site that has no access to the LCB registry -- which is the entire reason
    # the switch exists.
    image: ${{ variables.LCB_IMAGE if variables.WITH_LCB == 'true' else variables.CLIENT_IMAGE }}
    shm_size: 16Gi
    run_as_root: true

backends:
  - name: k8s_cluster
    type: kubernetes
    default: true
    namespace: ${{ variables.K8S_NAMESPACE }}
    # @@NODES_COMMENT@@
    nodes: ${{ variables.TOTAL_NODES }}
    gpus_per_node: ${{ variables.GPUS_PER_NODE }}
    # pod IP == node IP, so CTX_URLS / GEN_URLS / FRONTEND_URLS (computed from
    # the reserved node InternalIPs) are reachable, and the client can talk to
    # lcb_service on localhost:13835 exactly as it does under Slurm.
    host_network: true
    host_ipc: true
    scheduling: device_plugin
    node_selector:
      cloud.google.com/gke-nodepool: ${{ variables.GKE_NODEPOOL }}
    # A4X-max on GKE publishes no usable in-pod RDMA NIC: nvidia_peermem is not
    # visible (no GPUDirect RDMA) and this image's HPC-X nccl_rdma_sharp_plugin
    # fails to dlopen ("libmlx5.so.1: version MLX5_1.25 not found"), so NCCL was
    # falling back to NET/IB over RoCE with no rail pinning. `disable` sets
    # NCCL_IB_DISABLE=1 / NCCL_IBEXT_DISABLE=1 / NCCL_NET_PLUGIN=none and stops
    # the probe. Cross-node NVLink (MNNVL) is P2P, NOT a NET transport, so it is
    # unaffected -- on the proven point_256 run 1792 of 1792 channels stayed on
    # P2P/MNNVL.
    #
    # Do NOT add an mrdma.google.com ResourceClaim or the gIB NCCL plugin: gIB
    # accelerates NCCL's NETWORK path and inside one NVL72 domain there is no
    # NCCL network traffic to accelerate. The KV path is UCX, which cannot load
    # a NCCL plugin at all.
    rdma: disable
    # The NVLink ComputeDomain -- the GKE replacement for Slurm's --segment=N,
    # and what actually makes MNNVL work across the @@SERVER_NODES@@ nodes.
    compute_domain:
      create: true
    tolerations:
      - key: "kubernetes.io/arch"
        operator: "Equal"
        value: "arm64"
        effect: "NoSchedule"
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
    reservation:
      timeout: 1800
    # Result collection sizing (conversion spec §5, fix 4). sflow's default
    # per-file cap is 10 MiB and anything above it is skipped with only a
    # "sflow: NOT syncing large output file back to driver" line -- the run
    # still reports success. Measured on the Kimi-K3 C96 run, the client's own
    # report/scores.json tripped exactly this and the collected report had no
    # per-sample detail at all.
    #
    # 1Gi is chosen, not arbitrary: far above scores.json (tens of MB) and far
    # below events.jsonl (multi-GB), so the useful file is kept and the giant
    # one is still excluded -- dragging a multi-GB events.jsonl through
    # `kubectl cp` inside the grace window risks a mid-copy teardown, which is
    # a worse failure than skipping it.
    collect_max_file_size: 1Gi
    # Default 120 s bounds BOTH the pod's post-run hold and the driver-side
    # copy. A larger cap means more to move; the pod exits as soon as the copy
    # finishes, so a large value costs nothing on a fast copy.
    collect_grace_seconds: 900

workflow:
  # v1.0.2: renamed from dsr1_endpoints_v1_0_1_disagg_gke so run ids and
  # report dirs of the two drops (different checkpoints) never mix.
  name: dsr1_endpoints_v1_0_2_disagg_gke
  tasks:
    # =====================================================================
    # snapshot_configs -- ported from the Slurm template's task of the same
    # name. The DSR1 README requires keeping snapshot_configs/ with the run
    # output, so the resolved server and client configs travel with the result.
    #
    # On Slurm these are files on shared storage and it just `cp`s them. Here
    # the server configs are heredocs inside the server tasks, so the same
    # bytes are written again below. They cannot drift: both copies are emitted
    # from the same Slurm file by generate_point_recipes.py, and `--check`
    # diffs BOTH against the drop.
    #
    # v1.0.2's Slurm snapshot also copies site-env.yaml when SITE_ENV_YAML is
    # set. It defaults to '' (asserted by the generator) and the GKE
    # compatibility layer in the server tasks is this port's site overlay, so
    # there is no extra file to snapshot.
    # =====================================================================
    - name: snapshot_configs
      operator: trtllm_k8s
      resources:
        nodes:
          indices: ["${{ variables.BENCH_NODE_INDEX }}"]
      script:
        - set -euo pipefail
        - mkdir -p "${SFLOW_TASK_OUTPUT_DIR}"
        - |
          cat << 'EOF_SNAP_CTX' > "${SFLOW_TASK_OUTPUT_DIR}/server-ctx.yaml"
@@CTX_YAML_SNAP@@
        - EOF_SNAP_CTX
        - |
          cat << 'EOF_SNAP_GEN' > "${SFLOW_TASK_OUTPUT_DIR}/server-gen.yaml"
@@GEN_YAML_SNAP@@
        - EOF_SNAP_GEN
        - cp "${{ artifacts.CLIENT_CONFIG.path }}" "${SFLOW_TASK_OUTPUT_DIR}/client.yaml"
        # The GKE recipe itself is the third config that defines this run.
        # Record the resolved topology so the snapshot is self-describing.
        - |
          cat << 'EOF_SNAP_ENV' > "${SFLOW_TASK_OUTPUT_DIR}/gke-topology.txt"
          point                = @@POINT@@
          NUM_CTX_SERVERS      = ${{ variables.NUM_CTX_SERVERS }}
          GPUS_PER_CTX_RANK    = ${{ variables.GPUS_PER_CTX_RANK }}
          NUM_GEN_SERVERS      = ${{ variables.NUM_GEN_SERVERS }}
          GPUS_PER_GEN_RANK    = ${{ variables.GPUS_PER_GEN_RANK }}
          NUM_FRONTEND_SERVERS = ${{ variables.NUM_FRONTEND_SERVERS }}
          TOTAL_NODES          = ${{ variables.TOTAL_NODES }}
          MODE                 = ${{ variables.MODE }}
          WITH_LCB             = ${{ variables.WITH_LCB }}
          TRTLLM_IMAGE         = ${{ variables.TRTLLM_IMAGE }}
          CLIENT_IMAGE         = ${{ variables.CLIENT_IMAGE }}
          MODEL_PATH           = ${{ variables.MODEL_PATH }}
          DATA_PATH            = ${{ variables.DATA_PATH }}
          EOF_SNAP_ENV
        - ls -la "${SFLOW_TASK_OUTPUT_DIR}"

    # =====================================================================
    # PREFILL (context) servers -- @@CTX_N@@ x DEP@@CTX_TP@@.
    # =====================================================================
    - name: prefill_server
      replicas:
        count: ${{ variables.NUM_CTX_SERVERS }}
        policy: parallel
      resources:
        gpus:
          count: ${{ variables.GPUS_PER_CTX_RANK }}
      operator: mpi_trtllm_ctx
      script:
        - |
          cat << 'EOF_CTX' > server-ctx.yaml.$$
@@CTX_YAML@@
        - EOF_CTX
        # Atomic rename: co-located servers share one merged-pod filesystem
        # (GPT-OSS C140 died on the same race with `cp`, 2026-10-01).
        - mv -f server-ctx.yaml.$$ server-ctx.yaml
@@GKE_COMPAT_CTX@@
        - export PORT=$(( ${{ variables.BASE_PORT }} + ${SFLOW_REPLICA_INDEX} ))
        # `exec` is CORRECT here (and wrong in the benchmark task): the servers
        # are signal-terminated and produce no artifacts sflow has to collect.
        - >
          exec mpirun -np ${{ variables.GPUS_PER_CTX_RANK }} --allow-run-as-root
@@MPIRUN_X@@
          /tmp/dsr1-rank-env.sh prefill
          trtllm-llmapi-launch
          trtllm-serve ${{ variables.MODEL_PATH }}
          --host 0.0.0.0 --port ${PORT}
          --extra_llm_api_options server-ctx.yaml
      probes:
        readiness:
          log_watch:
            match_pattern: "Application startup complete."
            match_count: 1
          # Slurm allows 1800 s. Doubled: on GKE the weights come off the node's
          # local SSD on first touch with nothing warmed, and the ComputeDomain
          # channel has to come up before NCCL init completes.
          timeout: 3600
          interval: 20
        failure:
          log_watch:
            match_pattern: "Traceback (most recent call last)"
            match_count: 5
          interval: 30

    # =====================================================================
    # DECODE (generation) servers -- @@GEN_N@@ x DEP@@GEN_TP@@.
    # =====================================================================
    - name: decode_server
      replicas:
        count: ${{ variables.NUM_GEN_SERVERS }}
        policy: parallel
      resources:
        gpus:
          count: ${{ variables.GPUS_PER_GEN_RANK }}
      operator: mpi_trtllm_gen
      script:
        - |
          cat << 'EOF_GEN' > server-gen.yaml.$$
@@GEN_YAML@@
        - EOF_GEN
        - mv -f server-gen.yaml.$$ server-gen.yaml
@@GKE_COMPAT_GEN@@
        - export PORT=$(( ${{ variables.BASE_PORT }} + ${{ variables.NUM_CTX_SERVERS }} + ${SFLOW_REPLICA_INDEX} ))
        - >
          exec mpirun -np ${{ variables.GPUS_PER_GEN_RANK }} --allow-run-as-root
@@MPIRUN_X@@
          /tmp/dsr1-rank-env.sh decode
          trtllm-llmapi-launch
          trtllm-serve ${{ variables.MODEL_PATH }}
          --host 0.0.0.0 --port ${PORT}
          --extra_llm_api_options server-gen.yaml
      probes:
        readiness:
          log_watch:
            match_pattern: "Application startup complete."
            match_count: 1
          timeout: 3600
          interval: 20
        failure:
          log_watch:
            match_pattern: "Traceback (most recent call last)"
            match_count: 5
          interval: 30

    # =====================================================================
    # Disaggregated frontends -- @@FRONTENDS@@ replicas pinned across the
    # prefill nodes (0..ctx_full_nodes-1), keeping all decode_server nodes
    # (nodes 2..17) 100% free of frontend_server CPU/network load.
    # CPU-only pods that fan requests out to the ctx/gen servers.
    # =====================================================================
    - name: frontend_server
      replicas:
        count: ${{ variables.NUM_FRONTEND_SERVERS }}
        policy: parallel
      resources:
        nodes:
          indices: >-
            {% set ctx_full_nodes = (variables.NUM_CTX_SERVERS * variables.GPUS_PER_CTX_RANK) // variables.GPUS_PER_NODE %}[{% for i in range(variables.NUM_FRONTEND_SERVERS) %}${{ i % ctx_full_nodes }}{% if not loop.last %}, {% endif %}{% endfor %}]
          count: 1
      operator: trtllm_k8s
      script:
        - set -uo pipefail
        - export LD_LIBRARY_PATH=<workdir>/bin/nvidia/lib64:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
        # `trtllm-serve disaggregated` imports the full TRT-LLM stack, which
        # probes the GPU at import time. These four seds stub those probes so it
        # runs on a pod with no GPU allocated. All are `|| true` -- a failed
        # patch must not kill the frontend -- but a SILENT failed patch is how
        # you get an import crash 40 minutes into a run, so each path is checked
        # and a missing one is reported loudly. They are pinned to the
        # TRTLLM_IMAGE python layout; re-check them when that image changes.
        - |
          TLLM_PKG=/usr/local/lib/python3.12/dist-packages/tensorrt_llm
          for f in profiler.py _torch/cuda_tile_utils.py \
                   _torch/custom_ops/torch_custom_ops.py \
                   _torch/attention_backend/flashinfer.py \
                   serve/openai_disagg_server.py; do
            [ -f "${TLLM_PKG}/${f}" ] || echo "GKE_PATCH missing: ${TLLM_PKG}/${f} -- frontend GPU stubs may not apply (image layout changed?)"
          done
        - sed -i 's/with PyNVMLContext():/if False:/' /usr/local/lib/python3.12/dist-packages/tensorrt_llm/profiler.py 2>/dev/null || true
        - sed -i 's/if (cc := torch.cuda.get_device_properties()) and (cc.major, cc.minor) < (10, 0):/if torch.cuda.is_available() and (cc := torch.cuda.get_device_properties()) and (cc.major, cc.minor) < (10, 0):/' /usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/cuda_tile_utils.py 2>/dev/null || true
        - sed -i 's/deep_gemm.set_pdl(get_env_enable_pdl())/pass/' /usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py 2>/dev/null || true
        - sed -i 's/capability = torch.cuda.get_device_capability()/capability = (10, 0) if torch.cuda.is_available() else (10, 0)/' /usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/attention_backend/flashinfer.py 2>/dev/null || true
        - sed -i 's/config = uvicorn.Config(self.app,/config = uvicorn.Config(self.app, access_log=False,/' /usr/local/lib/python3.12/dist-packages/tensorrt_llm/serve/openai_disagg_server.py 2>/dev/null || true
        - export PORT=$(( ${{ variables.FRONTEND_PORT }} + ${SFLOW_REPLICA_INDEX} ))
        - MY_IP=$(echo ${SFLOW_TASK_ASSIGNED_NODE_IPS} | cut -d',' -f1)
        - MY_CONFIG=server_config_${SFLOW_REPLICA_INDEX}.yaml
        # GKE replacement for the Slurm template's generate_master_yaml.py.
        - |
          python3 -c "
          import sys, yaml
          ctx_urls = [u.strip().replace('http://', '').replace('https://', '') for u in open('${{ artifacts.CTX_URLS.path }}').read().splitlines() if u.strip()]
          gen_urls = [u.strip().replace('http://', '').replace('https://', '') for u in open('${{ artifacts.GEN_URLS.path }}').read().splitlines() if u.strip()]
          assert len(ctx_urls) == ${{ variables.NUM_CTX_SERVERS }}, f'expected ${{ variables.NUM_CTX_SERVERS }} ctx urls, got {ctx_urls}'
          assert len(gen_urls) == ${{ variables.NUM_GEN_SERVERS }}, f'expected ${{ variables.NUM_GEN_SERVERS }} gen urls, got {gen_urls}'
          config = {
              'hostname': '0.0.0.0',
              'port': int('${PORT}'),
              'backend': 'pytorch',
              'context_servers': {'num_instances': len(ctx_urls), 'urls': ctx_urls, 'router': {'type': 'load_balancing'}},
              'generation_servers': {'num_instances': len(gen_urls), 'urls': gen_urls, 'router': {'type': 'load_balancing'}}
          }
          with open('${MY_CONFIG}', 'w') as f:
              yaml.dump(config, f, default_flow_style=False, sort_keys=False)
          "
        - cat ${MY_CONFIG}
        # Slurm uses --request_timeout 7200 --server_start_timeout 7200. The
        # request timeout is doubled to 14400 because max_new_tokens is 20000
        # and the highest-concurrency points queue far deeper than Slurm's
        # dedicated-client allocation does; a frontend-side timeout shows up as
        # client-side failures that look like a server fault.
        - exec trtllm-serve disaggregated -c "${MY_CONFIG}" --request_timeout 14400 --server_start_timeout 7200
      probes:
        readiness:
          log_watch:
            match_pattern: "Application startup complete."
            match_count: 1
          timeout: 3600
          interval: 10
      depends_on:
        - prefill_server
        - decode_server

    # =====================================================================
    # LiveCodeBench service -- ported from the Slurm template's lcb_service.
    # CPU-only, pinned to the client's node. Slurm starts it in EVERY mode
    # (including perf) and the DSR1 README says to supply all three images, so
    # the default here is on. With host_network the client reaches it at
    # localhost:13835, which is the address Slurm's client uses too.
    # =====================================================================
    - name: lcb_service
      operator: lcb
      resources:
        nodes:
          # Quoted: a bare ${{ }} inside a YAML flow sequence is not valid
          # YAML. sflow resolves the string and coerces it to int.
          indices: ["${{ variables.BENCH_NODE_INDEX }}"]
      script:
        - set -uo pipefail
        - |
          if [ "${{ variables.WITH_LCB }}" != "true" ]; then
            echo "LCB_SERVICE disabled (WITH_LCB=false): LiveCodeBench is NOT running."
            echo "  MODE=${{ variables.MODE }}. This is only safe for MODE=perf."
            echo "  An accuracy run that needs LCB will fail against localhost:${{ variables.LCB_PORT }}."
            echo "LCB_SERVICE_READY"
            exec sleep infinity
          fi
        - echo "lcb_service starting on port ${{ variables.LCB_PORT }} (localhost)"
        - cd /app
        - >
          exec /app/venv/bin/uvicorn server:app --host 0.0.0.0
          --port ${{ variables.LCB_PORT }} --timeout-keep-alive 7200
          --ws-ping-interval 30 --ws-ping-timeout 10
      probes:
        readiness:
          log_watch:
            match_pattern: "re:Uvicorn running|LCB_SERVICE_READY"
            match_count: 1
          timeout: 600
          interval: 5

    # =====================================================================
    # MLCommons Endpoints client. The Slurm template's endpoint_benchmark.
    #
    # This task carries all four Kimi-K3 result-collection fixes (conversion
    # spec §5). Each of them cost a completed run.
    # =====================================================================
    - name: benchmark
      operator: client
      resources:
        nodes:
          # Quoted: a bare ${{ }} inside a YAML flow sequence is not valid
          # YAML. sflow resolves the string and coerces it to int.
          indices: ["${{ variables.BENCH_NODE_INDEX }}"]
      depends_on:
        - frontend_server
        - lcb_service
      script:
        - set -euo pipefail
        # -------------------------------------------------------------
        # FIX 1: resolve the client interpreter BEFORE scrubbing the env.
        #
        # sflow starts task containers with `bash -l` (plugins/k8s/render.py:
        # command: ["bash", "-l", <entrypoint>]). A login shell re-runs
        # /etc/profile, which hard-assigns
        #   PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games
        # and so DISCARDS the image's ENV PATH. The endpoints client lives in a
        # uv venv at $UV_PROJECT_ENVIRONMENT (/opt/venv) -- exactly what gets
        # dropped. `python3` then becomes the bare system interpreter and the
        # benchmark dies with
        #   ModuleNotFoundError: No module named 'inference_endpoint'
        # ...after the @@SERVER_NODES@@-node reservation and the full weight load.
        #
        # Ordinary env vars DO survive the login shell (only PATH is rewritten),
        # so UV_PROJECT_ENVIRONMENT is a reliable anchor. The drop's own
        # endpoint_env.yaml confirms the location: PATH=/opt/venv/bin:...
        # -------------------------------------------------------------
        - |
          CLIENT_VENV="${UV_PROJECT_ENVIRONMENT:-/opt/venv}"
          if [ -x "${CLIENT_VENV}/bin/python3" ]; then
            export PATH="${CLIENT_VENV}/bin:${PATH}"
          fi
        - unset PYTHONPATH PYTHONHOME UV_PROJECT_ENVIRONMENT
        - export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
        # Prove the client is runnable BEFORE anything expensive. By the time
        # this task starts, @@SERVER_NODES@@ nodes are reserved and DeepSeek-R1 is resident
        # on @@TOTAL_GPUS@@ GPUs. Discovering a bad CLIENT_IMAGE at the last line of the
        # script wastes all of it; discovering it here costs seconds.
        - |
          echo "client interpreter: $(command -v python3)"
          if ! python3 -c 'import inference_endpoint' 2>/dev/null; then
            echo "FATAL: 'inference_endpoint' is not importable with $(command -v python3)." >&2
            echo "  CLIENT_IMAGE=${{ variables.CLIENT_IMAGE }}" >&2
            echo "  Either that image is not an MLCommons endpoints build, or its" >&2
            echo "  virtualenv is not at \${UV_PROJECT_ENVIRONMENT:-/opt/venv}." >&2
            echo "  Mirror ghcr.io/mlcommons/endpoints:e9f427c (arm64) and pass" >&2
            echo "  -s CLIENT_IMAGE=<your registry copy>." >&2
            exit 1
          fi
        # On Kubernetes a missing fs:// path is only a plan-time WARNING, and
        # the Endpoints client falls back to a synthetic dataset with a message
        # that is easy to miss in a 100k-line log. That silently invalidated
        # three consecutive runs on this cluster. Refuse instead.
        - |
          if [ ! -f "${{ variables.DATA_PATH }}" ]; then
            echo "FATAL: eval dataset not found at ${{ variables.DATA_PATH }}" >&2
            echo "  Stage it on EVERY node: kubectl apply -f ../../stage-assets.yaml" >&2
            echo "  A synthetic-fallback run is NOT comparable to the Slurm number" >&2
            echo "  in either direction -- refusing to run." >&2
            exit 1
          fi
        - |
          if [ ! -d "${{ variables.MODEL_PATH }}" ]; then
            echo "FATAL: tokenizer/model dir not found at ${{ variables.MODEL_PATH }}" >&2
            echo "  The client tokenizes with this checkpoint; stage it on every node." >&2
            exit 1
          fi
        - 'echo "dataset: $(ls -l ${{ variables.DATA_PATH }})"'
        - sha256sum ${{ variables.DATA_PATH }} 2>/dev/null || echo "(sha256sum unavailable in this image)"
        # client.yaml refers to ${MODEL_PATH} / ${DATA_PATH} exactly as the
        # Slurm copy does; only the values differ (node-local, not shared FS).
        - export MODEL_PATH=${{ variables.MODEL_PATH }}
        - export DATA_PATH=${{ variables.DATA_PATH }}
        # v1.0.2's endpoint_env.yaml sets exactly HF_HOME=/tmp/hf_cache and
        # TIKTOKEN_RS_CACHE_DIR=/tmp/hf_cache/tiktoken_rs (v1.0.1: the shared
        # /hf_cache mount), so these are now verbatim; the generator refuses a
        # drop that changes them. HF_HUB_CACHE + mkdir are GKE additions.
        - export HF_HOME=/tmp/hf_cache
        - export HF_HUB_CACHE=/tmp/hf_cache/hub
        - export TIKTOKEN_RS_CACHE_DIR=/tmp/hf_cache/tiktoken_rs
        - mkdir -p /tmp/hf_cache/hub /tmp/hf_cache/tiktoken_rs
        # -------------------------------------------------------------
        # FIX 3: durable report_dir on a hostPath.
        # Print the node and path LOUDLY and EARLY -- once the pod is gone this
        # line is how the results are found. If the log stream stalls later
        # (Kimi C192 stalled for 7,474 s), this has already been delivered.
        # -------------------------------------------------------------
        - |
          export REPORT_DIR="${{ artifacts.REPORTS_DIR.path }}/$(basename "${SFLOW_WORKFLOW_OUTPUT_DIR}")"
          mkdir -p "${REPORT_DIR}"
          echo "DSR1_REPORT_DIR node=${SFLOW_TASK_NODE_NAME:-$(hostname)} path=${REPORT_DIR}"
        # GKE replacement for the Slurm template's generate_endpoint_yaml.py:
        # fill endpoint_config.endpoints from the FRONTEND_URLS artifact. Done
        # as a yaml round-trip so every other key -- submission_ref, the
        # accuracy dataset, this point's timeouts/warmup blocks, and the
        # ${VAR} placeholders the client expands itself -- survives untouched.
        #
        # GKE EDIT (guard; a no-op for every v1.0.1/v1.0.2 point): warmup.salt
        # true -> false in the RESOLVED copy only. The v1.0.1 and v1.0.2 drops
        # (byte-identical client.yaml) disable
        # warmup on every point (enabled: false, salt: false), so this only
        # fires if a future drop re-enables it. In v1.0.0, points
        # 2048/8192/16384/28080 shipped `warmup: {enabled, salt, drain}:
        # true`, and the e9f427c client aborted before issuing anything with
        #   "sample has 'input_tokens'; salt cannot bust a pre-tokenized cache"
        # because the DSR1 parquet is pre-tokenized (C2048 16-node run,
        # 2026-09-30, after the full 10 min server bring-up). Salt only exists
        # to defeat prefix-cache reuse between warmup and the timed phase, and
        # every server here runs kv_cache_config.enable_block_reuse: false --
        # there is no prefix cache, so this is measurement-neutral. Warmup
        # itself (enabled/drain) is kept.
        - |
          python3 -c "
          import sys, yaml
          raw = [u.strip() for u in open('${{ artifacts.FRONTEND_URLS.path }}').read().splitlines() if u.strip()]
          if not raw:
              sys.exit('FATAL: FRONTEND_URLS artifact is empty -- no frontend registered.')
          urls = [u if u.startswith('http') else 'http://' + u for u in raw]
          cfg = yaml.safe_load(open('${{ artifacts.CLIENT_CONFIG.path }}'))
          assert cfg['endpoint_config']['endpoints'] == [], 'client.yaml endpoints are not empty; refusing to overwrite'
          cfg['endpoint_config']['endpoints'] = urls
          warm = (cfg.get('settings') or dict()).get('warmup')
          if isinstance(warm, dict) and warm.get('salt'):
              warm['salt'] = False
              print('GKE EDIT: warmup.salt true -> false (pre-tokenized dataset; enable_block_reuse is false)')
          yaml.dump(cfg, open('/tmp/endpoint_resolved.yaml', 'w'), default_flow_style=False, sort_keys=False)
          print('frontend endpoints:', urls)
          "
        - cat /tmp/endpoint_resolved.yaml
        # -------------------------------------------------------------
        # FIX 2: NO `exec` HERE -- LOAD-BEARING.
        #
        # sflow installs `trap _sflow_collect EXIT` in the entrypoint shell
        # (k8s_operator.py:376). That trap tars up $SFLOW_WORKFLOW_OUTPUT_DIR,
        # prints `[[sflow-collect-ready]]`, and holds the container open for
        # collect_grace_seconds so the driver can copy results back. `exec`
        # REPLACES the shell process and destroys the trap: the marker is never
        # printed and the driver gives up with
        #   "pod reached a terminal phase without either collect sentinel
        #    appearing ...: node-local output was NOT collected"
        # Observed for real on the Kimi-K3 C24 run -- 39869/39869 samples
        # completed and the entire report/ died with the pod. `exec` is correct
        # for the server tasks above; it is wrong here, because this is the only
        # task that produces artifacts.
        # -------------------------------------------------------------
        - >
          inference-endpoint benchmark from-config
          -c /tmp/endpoint_resolved.yaml --mode ${{ variables.MODE }}
        # Belt and braces: also place the SMALL report files where sflow's
        # normal collect can pick them up, so a healthy run needs no manual
        # fetch. events.jsonl is excluded deliberately -- it is far over
        # collect_max_file_size and the durable copy on the node has it.
        # `|| true` because a failure here must never mask the benchmark's own
        # exit code; the node-local copy already exists either way.
        - |
          DEST="${SFLOW_WORKFLOW_OUTPUT_DIR}/report"
          mkdir -p "${DEST}"
          ( cd "${REPORT_DIR}" \
            && find . -type f ! -name 'events.jsonl' -print0 \
               | tar cf - --null -T - ) 2>/dev/null \
            | ( cd "${DEST}" && tar xf - ) 2>/dev/null || true
          echo "mirrored report -> ${DEST}:"
          ls -la "${DEST}" || true
          echo "durable copy retained on node at ${REPORT_DIR}"
'''


# --------------------------------------------------------------------------
# The GKE compatibility layer (conversion spec §4) -- copied verbatim from the
# proven deepseek-r1/point_256/deepseek_r1_gke_disagg_point256.yaml server-task
# preamble. Identical on ctx and gen; ROLE only changes one comment.
# --------------------------------------------------------------------------

def gke_compat_block(role: str) -> str:
    ucx_comment = (
        """        # --- GKE canonical transport block -----------------------------------
        # COPY VERBATIM. Do NOT port the Slurm equivalents from server-env.yaml
        # (which sets UCX_CUDA_IPC_ENABLE_MNNVL=n, UCX_TLS=^posix); those are
        # correct on a RoCE cluster and catastrophic here, and they fail without
        # an error message.
        #
        # KV cache moves ctx->gen over cross-node NVLink via UCX cuda_ipc+MNNVL.
        #
        # UCX_CUDA_IPC_ENABLE_MNNVL MUST be 'y'. Slurm says 'n' because RoCE RDMA
        # carries the transfer there. With 'n' on GKE every inter-node lane falls
        # back to tcp/eth0 (confirmed in the UCX_PROTO_INFO tables) and TRT-LLM
        # even logs its own "consider setting UCX_CUDA_IPC_ENABLE_MNNVL=y" hint:
        # TTFT went to ~20 s and throughput was capped at roughly a tenth.
        #
        # `tcp` stays in UCX_TLS as the active-message/control channel -- a
        # cuda-only list fails with "no active messages transport". Bulk KV still
        # goes over cuda_ipc; seeing tcp/eth0 on the TAGGED MESSAGE rows in the
        # log is normal and expected.
        #
        # UCX_NET_DEVICES=eth0 is REQUIRED and must be set explicitly. sflow used
        # to inject it as a side effect of `rdma: auto`; with `rdma: disable` it
        # no longer does, and UCX then enumerates every host interface and picks
        # GKE's Cilium interface, publishing a link-local 169.254.x.x address no
        # peer can route to. Run 20260829-001919 died on exactly that:
        #   sock.c:348 UCX ERROR connect(dest_addr=<ip>:33475): Connection refused
        #   what(): Error in addConnection(connRequest) for rank 0: Destination is unreachable
        # This pins only NET-type devices; cuda_ipc/cuda_copy are ACC-type
        # (UCX_ACC_DEVICES) and are unaffected.
        #
        # Do NOT add rc_x / UCX_IB_GID_INDEX / UCX_IB_ROCE_* and do NOT point
        # UCX_NET_DEVICES at mlx5_*: those NICs are not inside the pod, so UCX
        # silently drops them from the TLS list -- they look like tuned RDMA
        # while doing nothing."""
        if role == "ctx"
        else "        # --- GKE canonical transport block (see prefill_server for the full rationale) ---"
    )
    return f"""        # ---------------------------------------------------------------------
        # GKE compatibility layer -- conversion spec §4. Copied verbatim from the
        # proven point_256 recipe (45,760 tok/s vs a 44,858 Slurm reference).
        # Identical on prefill and decode. None of it comes from the Slurm
        # recipe's server-env.yaml; see the UCX comment below for why.
        # ---------------------------------------------------------------------
        # TRT-LLM's hang detector aborts a rank after 300 s without progress. At
        # these concurrencies a legitimate MoE/MTP step plus a cold KV transfer
        # can exceed that during warmup, and the abort looks like a crash rather
        # than a timeout. Raise it to an hour. `|| true` in spirit -- the python
        # is wrapped in try/except and prints a GKE_PATCH missing line naming
        # the file if the image layout moved, which it will on a TRTLLM_IMAGE
        # change.
        #
        # IDEMPOTENT + ATOMIC. sflow merges co-located GPU tasks into ONE pod
        # (e.g. several DEP2 prefill servers per node), so up to four servers
        # run this at once against the same container filesystem. An in-place
        # open(path, "w") truncates the file; a sibling importing tensorrt_llm
        # at that instant reads a half-written module and dies with
        # `ImportError: cannot import name HangDetector` (hit on the GPT-OSS
        # 17-node validation run, 2026-09-26). Skip if already patched,
        # otherwise write a temp file in the same directory and os.replace()
        # it, so readers only ever see the old or the new file.
        - |
          python3 -c '
          import os, tempfile
          path = "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/pyexecutor/hang_detector.py"
          old = "self.timeout = timeout if timeout is not None else 300"
          new = "self.timeout = int(os.environ.get(\\"TRTLLM_HANG_DETECTOR_TIMEOUT\\", \\"3600\\")) if timeout is None else timeout"
          try:
              with open(path, "r") as f:
                  s = f.read()
              if new in s:
                  print("hang_detector: already patched")
              elif old not in s:
                  print(f"GKE_PATCH missing: hang_detector pattern not found in {{path}}; NOT patched")
              else:
                  s = s.replace(old, new)
                  if "import os" not in s:
                      s = "import os\\n" + s
                  fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".sflow-tmp")
                  with os.fdopen(fd, "w") as f:
                      f.write(s)
                  os.chmod(tmp, os.stat(path).st_mode)
                  os.replace(tmp, path)
                  print("hang_detector: patched (atomic)")
          except Exception as e:
              print(f"GKE_PATCH missing: could not patch hang_detector at {{path}}: {{e}}")

          # Backport NVIDIA TensorRT-LLM 1.3.0rc25 fix for get_numa_aware_cpu_affinity:
          # in 1.3.0rc14, nvmlDeviceGetHandleByIndex(device_id) was called with the
          # local CUDA ordinal (0..1 for DEP2 prefill) without mapping through
          # CUDA_VISIBLE_DEVICES. When sflow packs two DEP2 prefill servers onto one
          # 4-GPU GB300 node via CUDA_VISIBLE_DEVICES=0,1,2,3 and 2,3,0,1, the second
          # server on physical GPUs 2 and 3 (Grace Socket 1, CPUs 72..143) queried
          # physical GPUs 0 and 1 and bound itself to Grace Socket 0 (CPUs 0..71).
          u_path = "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/llmapi/utils.py"
          u_old = "        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)"
          u_new = "        _cvd = [x.strip() for x in (os.environ.get(\\"CUDA_VISIBLE_DEVICES\\") or \\"\\").split(\\",\\") if x.strip()]\\n        handle = pynvml.nvmlDeviceGetHandleByIndex(int(_cvd[device_id]) if (0 <= device_id < len(_cvd) and _cvd[device_id].isdigit()) else device_id)"
          try:
              with open(u_path, "r") as f:
                  us = f.read()
              if u_new in us:
                  print("numa_affinity: already patched")
              elif u_old not in us:
                  print(f"GKE_PATCH missing: numa_affinity pattern not found in {{u_path}}; NOT patched")
              else:
                  us = us.replace(u_old, u_new, 1)
                  fd, tmp = tempfile.mkstemp(dir=os.path.dirname(u_path), suffix=".sflow-tmp")
                  with os.fdopen(fd, "w") as f:
                      f.write(us)
                  os.chmod(tmp, os.stat(u_path).st_mode)
                  os.replace(tmp, u_path)
                  print("numa_affinity: patched CUDA_VISIBLE_DEVICES resolution (atomic)")
          except Exception as e:
              print(f"GKE_PATCH missing: could not patch numa_affinity at {{u_path}}: {{e}}")

          # Disable uvicorn per-request HTTP access logging (access_log=False) in
          # trtllm-serve so the main asyncio event loop does not format and
          # synchronously write HTTP 200 OK lines during high-concurrency runs.
          o_path = "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/serve/openai_server.py"
          o_old = "config = uvicorn.Config(self.app,"
          o_new = "config = uvicorn.Config(self.app, access_log=False,"
          try:
              with open(o_path, "r") as f:
                  os_txt = f.read()
              if o_new in os_txt:
                  print("openai_server access_log: already patched")
              elif o_old not in os_txt:
                  print(f"GKE_PATCH missing: openai_server uvicorn.Config pattern not found in {{o_path}}; NOT patched")
              else:
                  os_txt = os_txt.replace(o_old, o_new, 1)
                  fd, tmp = tempfile.mkstemp(dir=os.path.dirname(o_path), suffix=".sflow-tmp")
                  with os.fdopen(fd, "w") as f:
                      f.write(os_txt)
                  os.chmod(tmp, os.stat(o_path).st_mode)
                  os.replace(tmp, o_path)
                  print("openai_server access_log: disabled (atomic)")
          except Exception as e:
              print(f"GKE_PATCH missing: could not patch openai_server access_log at {{o_path}}: {{e}}")
          '
        # Bigger socket buffers for the UCX/NCCL control channels. Best-effort:
        # a non-privileged pod cannot set these and it is not fatal.
        - sysctl -w net.core.rmem_max=134217728 2>/dev/null || true
        - sysctl -w net.core.wmem_max=134217728 2>/dev/null || true
        - sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728" 2>/dev/null || true
        - sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728" 2>/dev/null || true
        - export LD_LIBRARY_PATH="/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:/usr/local/nvidia/lib64:${{LD_LIBRARY_PATH:-}}"
        # Slurm mounts a shared /hf_cache; node-local /tmp is the GKE equivalent.
        # (v1.0.2's server-env.yaml no longer sets HF_HOME at all; these
        # node-local caches are GKE-layer values and stay as proven.)
        - export HF_HOME=/tmp/hf
        - export TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken
        # Rank-local HF_MODULES_CACHE, as the Slurm template does, to avoid a
        # shared-lock stall when every rank imports trust_remote_code modules.
        - export HF_MODULES_CACHE=/tmp/trtllm-hf-modules-${{SFLOW_REPLICA_INDEX:-0}}
        - mkdir -p "${{HF_MODULES_CACHE}}"
        # Per-rank wrapper (same as Kimi-K3 G_RANK_ENV): sets a disjoint per-pod
        # /proc/sys/kernel/ns_last_pid base so UCX's (pid, pid_ns, cu_dev)
        # remote MNNVL VMM handle cache (libuct_cuda.so uct_cuda_ipc_map_memhandle)
        # never collides across containers on identically booted GKE nodes,
        # restores full-node CPU affinity so TLLM_NUMA_AWARE_WORKER_AFFINITY=1
        # can bind each GPU worker to its NUMA node, and isolates per-rank
        # HF_MODULES_CACHE.
        - |
          cat > "/tmp/dsr1-rank-env.sh.$$" <<'EOF_RANK_ENV'
          #!/usr/bin/env bash
          set -euo pipefail
          unset BASH_ENV ENV NCCL_ENV_PLUGIN NCCL_TUNER_PLUGIN NCCL_PROFILER_PLUGIN NCCL_TUNER_CONFIG_PATH
          export NCCL_NET_PLUGIN=none
          taskset -pc "0-$(($(nproc --all 2>/dev/null || echo 144) - 1))" "$$" >/dev/null 2>&1 || true
          role="$1"; shift
          export DSR1_RANK="${{OMPI_COMM_WORLD_RANK:?dsr1-rank-env.sh must be started by mpirun}}"
          if [ "$role" = "prefill" ]; then
            pod_slot=$(( ${{SFLOW_REPLICA_INDEX:-0}} * 2 + DSR1_RANK / 4 ))
          else
            pod_slot=$(( 16 + ${{SFLOW_REPLICA_INDEX:-0}} * 16 + DSR1_RANK / 4 ))
          fi
          pod_pid_base=$(( 100000 + pod_slot * 30000 ))
          (
            flock -x 9
            cur_pid=$(cat /proc/sys/kernel/ns_last_pid)
            if [ "$cur_pid" -lt "$pod_pid_base" ]; then
              echo "$pod_pid_base" > /proc/sys/kernel/ns_last_pid
            fi
          ) 9>/tmp/dsr1-pid-ns.lock
          export HF_MODULES_CACHE="/tmp/trtllm-hf-modules-${{role}}-${{SFLOW_REPLICA_INDEX:-0}}-${{DSR1_RANK}}"
          mkdir -p "$HF_MODULES_CACHE"
          bash -c 'exec "$@"' _ "$@"
          EOF_RANK_ENV
          chmod 0755 "/tmp/dsr1-rank-env.sh.$$"
          mv -f "/tmp/dsr1-rank-env.sh.$$" /tmp/dsr1-rank-env.sh
        # Open MPI over the pod network. ob1/tcp on eth0 only: there is no
        # verbs BTL in the pod, and letting OMPI autodetect picks cilium_host.
        - export OMPI_MCA_pml=ob1
        - export OMPI_MCA_btl=tcp,self
        - export OMPI_MCA_btl_tcp_if_include=eth0
        - export OMPI_MCA_oob_tcp_if_include=eth0
        - export OMPI_MCA_hwloc_base_binding_policy=none
        - export OMPI_MCA_rmaps_base_mapping_policy=node
        - export OMPI_MCA_coll='^ucc'
        - export TLLM_LOG_LEVEL=INFO
        - export TRITON_PTXAS_PATH=/usr/local/cuda-13.1/bin/ptxas
        # --- kept from the Slurm server-env.yaml (model/runtime flags only) ---
        - export TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER=1
        - export TRTLLM_SERVER_DISABLE_GC=1
        - export TRTLLM_WORKER_DISABLE_GC=1
        - export TRTLLM_ENABLE_PDL=1
        - export TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED=1
        - export NCCL_GRAPH_MIXING_SUPPORT=0
        - export TLLM_NUMA_AWARE_WORKER_AFFINITY=1
        # --- NCCL on MNNVL ----------------------------------------------------
        # With rdma: disable there is no NET transport left, so cross-node NVLink
        # is the only correct path: NCCL_MNNVL_ENABLE=1 turns it on and
        # NCCL_CUMEM_ENABLE=1 gives the cuMem/VMM allocator the fabric handles it
        # needs. NCCL_DEBUG=INFO stays on because README check 4 greps it for
        # "via P2P/MNNVL".
        - export NCCL_MNNVL_ENABLE=1
        - export NCCL_CUMEM_ENABLE=1
        - export MC_FORCE_MNNVL=1
        - export NCCL_DEBUG=INFO
        # Pin every socket bootstrap to the routable interface, same reason as
        # UCX_NET_DEVICES below.
        - export GLOO_SOCKET_IFNAME=eth0
        - export TP_SOCKET_IFNAME=eth0
        - export NCCL_SOCKET_IFNAME=eth0
{ucx_comment}
        - export UCX_TLS=cuda_copy,cuda_ipc,sm,self,tcp
        - export UCX_NET_DEVICES=eth0
        - export UCX_CUDA_IPC_ENABLE_MNNVL=y
        - export UCX_RNDV_SCHEME=put_zcopy
        - export UCX_MEMTYPE_CACHE=n
        - export UCX_MEMTYPE_REG_WHOLE=n
        - export UCX_PROTO_INFO=y
        - export TRTLLM_HANG_DETECTOR_TIMEOUT=3600"""


# mpirun only forwards variables named with -x; a launcher-shell `export` never
# reaches the remote ranks. This list must stay in sync with the exports above.
MPIRUN_X = """          -x HF_HOME=/tmp/hf
          -x TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken
          -x HF_MODULES_CACHE
          -x LD_LIBRARY_PATH
          -x NCCL_MNNVL_ENABLE=1
          -x NCCL_CUMEM_ENABLE=1
          -x MC_FORCE_MNNVL=1
          -x NCCL_DEBUG=INFO
          -x GLOO_SOCKET_IFNAME=eth0
          -x TP_SOCKET_IFNAME=eth0
          -x NCCL_SOCKET_IFNAME=eth0
          -x TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED=1
          -x TRITON_PTXAS_PATH
          -x TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER=1
          -x TRTLLM_SERVER_DISABLE_GC=1
          -x TRTLLM_WORKER_DISABLE_GC=1
          -x TRTLLM_ENABLE_PDL=1
          -x NCCL_GRAPH_MIXING_SUPPORT=0
          -x UCX_TLS=cuda_copy,cuda_ipc,sm,self,tcp
          -x UCX_NET_DEVICES=eth0
          -x UCX_CUDA_IPC_ENABLE_MNNVL=y
          -x UCX_RNDV_SCHEME=put_zcopy
          -x UCX_MEMTYPE_CACHE=n
          -x UCX_MEMTYPE_REG_WHOLE=n
          -x UCX_PROTO_INFO=y
          -x TLLM_NUMA_AWARE_WORKER_AFFINITY=1
          -x TRTLLM_HANG_DETECTOR_TIMEOUT"""


LAUNCH_SCRIPT = r'''#!/usr/bin/env bash
# Launch DeepSeek-R1 MLPerf-endpoints v1.0.2 point C@@POINT@@ on a GB300 (A4X-max)
# GKE cluster. Slurm-faithful topology:
#   @@CTX_N@@ ctx x DEP@@CTX_TP@@ + @@GEN_N@@ gen x DEP@@GEN_TP@@ = @@TOTAL_GPUS@@ GPUs = @@SERVER_NODES@@ nodes.
# GENERATED by ../../generate_point_recipes.py -- do not hand-edit.
#
# PREREQUISITES (../../README.md §3):
#   1. @@SERVER_NODES@@ schedulable A4X-max nodes (counted below before launching;
#      add N if you pass -s CLIENT_RESERVE_NODES=N).
#   2. kubectl apply -f ../../stage-assets.yaml    # v2 checkpoint + eval parquet, every node
#   3. kubectl apply -f ../../mk-report-dir.yaml   # durable report hostPath
#   4. kubectl apply -f ../../prepull-images.yaml  # recommended: avoid a cold pull
#      burning the reservation window
#   5. CLIENT_IMAGE / LCB_IMAGE reachable from the nodes (mirror ghcr.io into
#      Artifact Registry if not) -- pass with -s.
#   6. kubectl context pointing at the target cluster.
#
# Site overrides:
#   NAMESPACE=<ns>  NODEPOOL=<a4x-max nodepool>   (env vars)
#   REQUIRED_NODES=<n>                            node-gate threshold (default @@SERVER_NODES@@)
#   SKIP_NODE_CHECK=1                             skip the node count below
#   anything else via sflow flags, e.g.
#     ./launch_test.sh -s MODE=both -s CLIENT_IMAGE=<registry>/endpoints:e9f427c
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

RECIPE="deepseek_r1_gke_disagg_point@@POINT@@.yaml"
NAMESPACE="${NAMESPACE:-default}"
NODEPOOL="${NODEPOOL:<nodepool>}"
REQUIRED_NODES="${REQUIRED_NODES:-@@SERVER_NODES@@}"

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
    echo "Not enough healthy nodes; refusing to launch. Set SKIP_NODE_CHECK=1 to override," >&2
    echo "or regenerate reduced-node variants: ../../generate_point_recipes.py --variants 16" >&2
    exit 1
  fi
fi

sflow run -f "${RECIPE}" \
  --kube-namespace "${NAMESPACE}" \
  --kube-node-selector "cloud.google.com/gke-nodepool=${NODEPOOL}" \
  "$@"
'''


# ==========================================================================
# Rendering
# ==========================================================================


def indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else pad for line in text.rstrip("\n").split("\n"))


def placement_comment(info: dict) -> str:
    ctx_gpus = info["ctx_gpus"]
    ctx_nodes = sorted({ctx_node(i, info["ctx_tp"]) for i in range(info["ctx_n"])})
    gen_starts = [gen_node(i, ctx_gpus, info["gen_tp"]) for i in range(info["gen_n"])]
    gen_span = info["gen_tp"] // GPUS_PER_NODE
    parts = [
        f"ctx servers -> node(s) {ctx_nodes[0]}-{ctx_nodes[-1]}; "
        f"gen server i -> nodes " +
        ", ".join(
            f"{s}-{s + gen_span - 1}" if gen_span > 1 else f"{s}"
            for s in gen_starts
        )
    ]
    return "".join(parts)


GEN_VARIANT_HEADER = """\
# ----------------------------------------------------------------------------
# server-gen.yaml -- from the NVIDIA Slurm drop
#   .../deepseek-r1/{arch}/point_{point}/server-gen.yaml
# RESHAPED for the {nodes}-node variant, plus the two mandatory GKE edits
# marked "GKE EDIT" below. NOT NVIDIA's certified decode config.
#
# NVIDIA's point_{point} decode is {slurm_gen_n} x DEP{slurm_tp} (max_batch_size {old_mbs}
# per rank); that cannot sit next to a prefill pool in {nodes} nodes, so this
# variant runs {gen_n} x DEP{gen_tp} at C={per_replica} per replica (see the recipe header
# and generate_point_recipes.py plan_variant()). ONLY these keys differ from
# the drop, each marked "{nodes}-NODE RESHAPE":
{changes}
# Everything else -- MTP, MoE backend, KV cache, scheduler, the CUDA-graph
# padding -- is NVIDIA's point_{point} file. Decode keeps real CUDA graphs whose
# ladder tops out at max_batch_size (--check asserts it).
# ----------------------------------------------------------------------------
"""


def reshape_changes(plan: dict) -> list[str]:
    """Human-readable list of the server-gen.yaml keys a variant changed."""
    out = [f"#   tensor_parallel_size / moe_expert_parallel_size : "
           f"{plan['old_tp']} -> {plan['gen_tp']}"]
    if plan["mbs"] != plan["old_mbs"]:
        out.append(f"#   max_batch_size : {plan['old_mbs']} -> {plan['mbs']}   "
                   f"(ceil(C / decode ranks): every request gets a slot)")
    if plan["mnt"] != plan["old_mnt"]:
        out.append(f"#   max_num_tokens : {plan['old_mnt']} -> {plan['mnt']}   "
                   f"(scaled with max_batch_size, as in the drop)")
    if plan["ladder"] != plan["old_ladder"]:
        added = plan["ladder"][len(plan["old_ladder"]):]
        out.append(f"#   cuda_graph_config.batch_sizes : + {added}")
    return out


def topo_desc(ctx_n: int, ctx_tp: int, gen_n: int, gen_tp: int) -> str:
    return f"{ctx_n} x DEP{ctx_tp} + {gen_n} x DEP{gen_tp}"


def variant_banner(info: dict) -> str:
    plan, slurm = info["variant"], info["slurm"]
    nodes, point, conc = plan["nodes"], info["point"], info["concurrency"]
    slurm_ranks = slurm["gen_n"] * slurm["gen_tp"]
    ranks = info["gen_n"] * info["gen_tp"]
    lines = [
        "#",
        f"# *** {nodes}-NODE VARIANT OF point_{point} -- NOT A SUBMISSION TOPOLOGY ***",
        "# " + "=" * 76,
        "#",
        "# WHY THIS FILE EXISTS. Every approved DSR1 point is 72 GPUs = 18 nodes (a",
        f"# full NVL72 domain) and the <nodepool> pool has {nodes} usable nodes (README §1).",
        "# This variant runs the SAME point -- same concurrency, same sample count,",
        "# client.yaml and server-ctx.yaml byte-identical to the drop -- on",
        f"# {nodes} nodes ({nodes * GPUS_PER_NODE} GPUs).",
        "#",
        f"# THE ONLY FUNCTIONAL CHANGES vs ../../18nodes/point_{point}:",
        f"#   prefill : {slurm['ctx_n']:>2} x DEP{slurm['ctx_tp']}  -> {info['ctx_n']:>2} x DEP{info['ctx_tp']}"
        f"   (server-ctx.yaml unchanged)",
        f"#   decode  : {slurm['gen_n']:>2} x DEP{slurm['gen_tp']} -> {info['gen_n']:>2} x DEP{info['gen_tp']}"
        f"   (server-gen.yaml reshaped:)",
        *["#       " + c[1:].lstrip() for c in reshape_changes(plan)],
        f"#   => 72 -> {info['total_gpus']} GPUs, 18 -> {nodes} nodes. "
        f"NUM_FRONTEND_SERVERS stays {info['frontends']}.",
        "#",
        "# HOW THE SPLIT WAS CHOSEN (../../generate_point_recipes.py, plan_variant()).",
        "# Prefill keeps NVIDIA's server shape and never drops below the drop's",
        f"# {slurm['ctx_gpus']} GPUs. Decode uses an EP size NVIDIA certifies in this drop, as",
        "# many servers of it as fit next to that prefill pool. Candidates:",
    ]
    for c in plan["cands"]:
        mark = "  <- chosen" if (c["gen_tp"], c["gen_n"]) == (info["gen_tp"], info["gen_n"]) else ""
        lines.append(
            f"#     {c['gen_n']} x DEP{c['gen_tp']:<2} ({c['ranks']} decode ranks): "
            f"per-rank batch {c['rank_batch']}{mark}"
        )
    lines += [
        f"#   rule: {plan['regime']}.",
        f"#   Leftover GPUs become more prefill servers: {info['ctx_n']} x DEP{info['ctx_tp']}"
        f" ({info['ctx_gpus']} GPUs).",
        "#",
        f"# EXPECTED EFFECT -- not like-for-like with an 18-node point_{point} result:",
        f"#   * decode ranks {slurm_ranks} -> {ranks}; per-rank batch at C={conc}: "
        f"{ceil_div(conc, slurm_ranks)} -> {ceil_div(conc, ranks)}.",
        f"#     decode slots {gen_n_slots(plan)} >= target_concurrency {conc}: "
        "no request waits for one.",
        f"#   * prefill GPUs {slurm['ctx_gpus']} -> {info['ctx_gpus']}: lower TTFT / prefill queueing.",
        f"#   * the NVLink ComputeDomain spans {nodes} nodes, not a full 18-node domain.",
        "#",
        "# Regenerate with ../../generate_point_recipes.py; never edit this file in place.",
    ]
    return "\n".join(lines) + "\n"


def sub_once(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        sys.exit(f"variant rewrite {what!r}: expected exactly 1 match in the "
                 f"rendered recipe, found {n} -- RECIPE_TEMPLATE changed; update "
                 "apply_variant_text()")
    return text.replace(old, new)


def variant_workflow_name(info: dict) -> str:
    return (f"dsr1_endpoints_v1_0_2_disagg_gke_c{info['point']}_"
            f"{info['variant']['nodes']}nodes")


def apply_variant_text(out: str, info: dict) -> str:
    """Comment / label rewrites for a variant. Functional deltas come from the
    topology variables and the reshaped EOF_GEN; the workflow name is the only
    functional change made here (so a run directory cannot pass for 18-node)."""
    plan, slurm, point = info["variant"], info["slurm"], info["point"]
    nodes = plan["nodes"]
    head = "# GKE A4X-max (GB300 NVL72).\n# " + "=" * 76 + "\n"
    out = sub_once(out, head, head + variant_banner(info), "header banner")
    out = sub_once(
        out,
        "#   MODEL / WORKLOAD half -- COPIED VERBATIM from the drop:\n"
        "#     every line inside EOF_CTX / EOF_GEN, the whole CLIENT_CONFIG artifact,\n"
        "#     the topology variables, the concurrency and sample counts. These are\n"
        "#     NVIDIA's certified numbers. Do not re-tune them.\n",
        "#   MODEL / WORKLOAD half -- COPIED VERBATIM from the drop:\n"
        "#     every line inside EOF_CTX, the whole CLIENT_CONFIG artifact, the\n"
        "#     concurrency and sample counts. These are NVIDIA's certified numbers.\n"
        f"#     EXCEPT, for this {nodes}-node variant, the server counts, the decode\n"
        f"#     TP/EP and the \"{nodes}-NODE RESHAPE\" lines in EOF_GEN (see above).\n",
        "workload paragraph",
    )
    out = sub_once(
        out,
        f"# TOPOLOGY (identical to Slurm point_{point}):\n",
        f"# TOPOLOGY ({nodes}-node variant; Slurm point_{point} is "
        f"{topo_desc(slurm['ctx_n'], slurm['ctx_tp'], slurm['gen_n'], slurm['gen_tp'])}"
        " = 72 GPUs = 18 nodes):\n",
        "topology heading",
    )
    out = sub_once(out, f"GPUs = {nodes} nodes (a full NVL72).\n",
                   f"GPUs = {nodes} nodes.\n", "topology scope")
    out = sub_once(
        out,
        f"  # POINT TOPOLOGY -- verbatim from point_{point}/dsr1_config.yaml.\n",
        f"  # POINT TOPOLOGY -- point_{point}/dsr1_config.yaml, EXCEPT NUM_CTX_SERVERS,\n"
        f"  # NUM_GEN_SERVERS and GPUS_PER_GEN_RANK: re-split for {nodes} nodes (header).\n",
        "topology variables comment",
    )
    ctx_nodes = info["ctx_gpus"] // GPUS_PER_NODE
    out = sub_once(
        out,
        "  # For DeepSeek-R1 ctx is always exactly 8 GPUs = 2 whole nodes and gen TP is\n"
        "  # always >= 16, so ctx lands on nodes 0-1 and gen server i starts at\n"
        "  # node 2 + i*(genTP/4).",
        f"  # In this {nodes}-node variant ctx is {info['ctx_gpus']} GPUs = {ctx_nodes} whole nodes and\n"
        f"  # gen TP is {info['gen_tp']}, so ctx lands on nodes 0-{ctx_nodes - 1} and gen server i\n"
        f"  # starts at node {ctx_nodes} + i*{info['gen_tp'] // GPUS_PER_NODE}.",
        "node math comment",
    )
    # v1.0.2: the template no longer quotes a Slurm CLIENT_RESERVE_GPUS (the
    # drop removed it); only the node arithmetic in that paragraph changes.
    out = sub_once(
        out,
        "      is NOT the Slurm allocation, and needs 18+N nodes);",
        f"      is NOT the Slurm allocation, and needs {nodes}+N nodes);",
        "client reserve comment",
    )
    out = sub_once(out, "  name: dsr1_endpoints_v1_0_2_disagg_gke\n",
                   f"  name: {variant_workflow_name(info)}\n", "workflow name")
    out = sub_once(
        out,
        f"          point                = {point}\n",
        f"          point                = {point}\n"
        f"          VARIANT              = {nodes}-node "
        f"({topo_desc(info['ctx_n'], info['ctx_tp'], info['gen_n'], info['gen_tp'])}; "
        f"Slurm {topo_desc(slurm['ctx_n'], slurm['ctx_tp'], slurm['gen_n'], slurm['gen_tp'])})"
        " -- NOT a submission topology\n",
        "gke-topology.txt VARIANT line",
    )
    return out


def diag_workflow_name(d: dict) -> str:
    return f"dsr1_endpoints_v1_0_2_disagg_gke_c{d['point']}_diag_{d['diag']['name']}"


def diag_sub(text: str, old: str, new: str, what: str, count: int = 1) -> str:
    n = text.count(old)
    if n != count:
        sys.exit(f"diag rewrite {what!r}: expected {count} match(es) in the rendered "
                 f"recipe, found {n} -- RECIPE_TEMPLATE changed; update apply_diag_text()")
    return text.replace(old, new)


def diag_banner(d: dict) -> str:
    g, b, point = d["diag"], d["base"], d["point"]
    name = g["name"]
    cfg = yaml.safe_load(d["gen_yaml"])
    client = yaml.safe_load(d["client_yaml"])
    bar = "# " + "#" * 76 + "\n"

    def boxed(s: str) -> str:
        return f"# ###   {s:<66} ###\n"

    purpose = textwrap.fill(g["purpose"], width=78, initial_indent="#   ",
                            subsequent_indent="#   ")
    nodes = "\n".join(
        f"#   gen server {i} -> nodes {lo}-{hi}, GEN_URLS entry = node {n}, port {p}"
        for i, ((lo, hi), n, p) in enumerate(zip(g["gen_nodes"], g["url_nodes"], g["url_ports"]))
    )
    return (
        bar + boxed("DIAGNOSTIC ONLY, NOT A SUBMISSION TOPOLOGY")
        + boxed(f"diag {name} of point C{point} -- never report it as C{point}")
        + bar
        + "# PURPOSE\n" + purpose + "\n#\n"
        + f"# EXACT DEVIATIONS from a fresh render of 18nodes/point_{point} (the\n"
        + f"# committed ../../18nodes/point_{point}/{recipe_name(b)}).\n"
        + f"# Every changed non-comment line carries an inline \"# DIAG {name}:\" marker:\n"
        + f"#   variables NUM_GEN_SERVERS             {b['gen_n']:>3} -> {d['gen_n']}\n"
        + f"#   variables GPUS_PER_GEN_RANK           {b['gen_tp']:>3} -> {d['gen_tp']}\n"
        + f"#   server-gen tensor_parallel_size       {b['gen_tp']:>3} -> {d['gen_tp']}   (EOF_GEN + EOF_SNAP_GEN)\n"
        + f"#   server-gen moe_expert_parallel_size   {b['gen_tp']:>3} -> {d['gen_tp']}   (EOF_GEN + EOF_SNAP_GEN)\n"
        + f"#   workflow.name -> {diag_workflow_name(d)}\n"
        + f"#     (the run dir and the durable report dir carry c{point}_diag_{name})\n"
        + "#   gke-topology.txt gains a \"DIAG = ...\" line\n"
        + "# Comments restating the topology (gen x DEP, C per replica, placement)\n"
        + "# follow from the two variables.\n"
        + "#\n"
        + "# UNCHANGED (--check / --check-diag fail closed otherwise): prefill\n"
        + f"#   {d['ctx_n']} x DEP{d['ctx_tp']} and server-ctx.yaml; every other server-gen key --\n"
        + f"#   max_batch_size {cfg['max_batch_size']}, max_num_tokens {cfg['max_num_tokens']}, "
        + f"cuda_graph batch_sizes {cfg['cuda_graph_config']['batch_sizes']},\n"
        + f"#   MTP num_nextn_predict_layers {cfg['speculative_config']['num_nextn_predict_layers']}, "
        + f"use_low_precision_moe_combine {str(cfg['moe_config']['use_low_precision_moe_combine']).lower()},\n"
        + "#   kv_cache / cache_transceiver config; client.yaml (target_concurrency\n"
        + f"#   {client['settings']['load_pattern']['target_concurrency']}, n_samples_to_issue "
        + f"{client['settings']['runtime']['n_samples_to_issue']}); frontends; env / GKE layer;\n"
        + "#   images; MODEL_PATH / DATA_PATH.\n"
        + "#\n"
        + f"# NODE MATH: {d['total_gpus']} GPUs = {d['server_nodes']} nodes. ctx {d['ctx_n']} x DEP{d['ctx_tp']} = "
        + f"{d['ctx_gpus']} GPUs -> nodes 0-{d['ctx_gpus'] // GPUS_PER_NODE - 1}.\n"
        + f"#   gen_base = GPU {g['gen_base']} (whole-node boundary: gen TP {d['gen_tp']} > {GPUS_PER_NODE}).\n"
        + nodes + "\n"
        + f"# DECODE SLOTS: {d['gen_n']} x {d['gen_tp']} ranks x max_batch_size {g['mbs']} = "
        + f"{g['slots']} >= C {d['concurrency']}\n"
        + f"#   (C {d['concurrency'] // d['gen_n']} per replica / {d['gen_tp']} ranks = "
        + f"{d['concurrency'] // d['gen_n'] // d['gen_tp']} per rank; C{point} has "
        + f"{d['concurrency'] // b['gen_n']} / {b['gen_tp']} = "
        + f"{d['concurrency'] // b['gen_n'] // b['gen_tp']}).\n"
        + bar
    )


def apply_diag_text(out: str, d: dict) -> str:
    """Banner + inline DIAG markers. The functional deltas come from the
    topology variables and diag_gen_yaml(); the workflow name is the only
    functional change made here (so the run and report dirs carry the tag)."""
    g, b, point = d["diag"], d["base"], d["point"]
    name = g["name"]
    rule = "# " + "=" * 76 + "\n"
    out = diag_sub(
        out,
        f"# DeepSeek-R1 -- MLPerf Endpoints v1.0.2, point C{point} (disaggregated) on\n"
        "# GKE A4X-max (GB300 NVL72).\n" + rule,
        f"# DeepSeek-R1 -- MLPerf Endpoints v1.0.2, DIAGNOSTIC {name} of point C{point}\n"
        "# (disaggregated) on GKE A4X-max (GB300 NVL72). NOT a submission topology.\n"
        + rule + diag_banner(d),
        "title + banner",
    )
    out = diag_sub(
        out,
        "#     NVIDIA's certified numbers. Do not re-tune them.\n",
        "#     NVIDIA's certified numbers. Do not re-tune them.\n"
        f"#     EXCEPT, in this DIAGNOSTIC, the decode split and the server-gen\n"
        f"#     TP/EP lines marked \"DIAG {name}\" (see the banner above).\n",
        "workload paragraph",
    )
    out = diag_sub(
        out,
        f"# TOPOLOGY (identical to Slurm point_{point}):\n",
        f"# TOPOLOGY (DIAGNOSTIC {name}; Slurm point_{point} is "
        f"{topo_desc(b['ctx_n'], b['ctx_tp'], b['gen_n'], b['gen_tp'])} = "
        f"{b['total_gpus']} GPUs = {b['server_nodes']} nodes):\n",
        "topology heading",
    )
    out = diag_sub(
        out,
        f"  # POINT TOPOLOGY -- verbatim from point_{point}/dsr1_config.yaml.\n",
        f"  # POINT TOPOLOGY -- point_{point}/dsr1_config.yaml, EXCEPT NUM_GEN_SERVERS and\n"
        f"  # GPUS_PER_GEN_RANK (DIAG {name}, marked inline; see the banner).\n",
        "topology variables comment",
    )
    for var, desc, new, old in (
        ("NUM_GEN_SERVERS", "Decode (generation) server replicas.", d["gen_n"], b["gen_n"]),
        ("GPUS_PER_GEN_RANK", "GPUs per decode replica == its TP/EP size.", d["gen_tp"], b["gen_tp"]),
    ):
        stem = f"  {var}:\n    description: \"{desc}\"\n    type: integer\n    value: {new}"
        out = diag_sub(out, stem + "\n",
                       stem + f"  # DIAG {name}: NVIDIA point_{point} has {old}\n", var)
    gen_src = (
        "          # server-gen.yaml -- VERBATIM from the NVIDIA Slurm drop\n"
        f"          #   .../deepseek-r1/{ARCH_DIR}/point_{point}/server-gen.yaml\n"
        "          # apart from the two mandatory GKE edits marked \"GKE EDIT\" below\n"
    )
    out = diag_sub(
        out,
        gen_src.rstrip("\n") + ".\n",
        gen_src
        + f"          # AND the two \"DIAG {name}\" lines (decode TP/EP {b['gen_tp']} -> {d['gen_tp']},\n"
        f"          # marked inline). DIAGNOSTIC ONLY: NOT NVIDIA's point_{point} decode shape.\n",
        "server-gen header (EOF_GEN + EOF_SNAP_GEN)", count=2,
    )
    out = diag_sub(out, "  name: dsr1_endpoints_v1_0_2_disagg_gke\n",
                   f"  name: {diag_workflow_name(d)}  # DIAG {name}: run / report dir tag\n",
                   "workflow name")
    out = diag_sub(
        out,
        f"          point                = {point}\n",
        f"          point                = {point}\n"
        f"          DIAG                 = {name}: "
        f"{topo_desc(d['ctx_n'], d['ctx_tp'], d['gen_n'], d['gen_tp'])} (NVIDIA point_{point}: "
        f"{topo_desc(b['ctx_n'], b['ctx_tp'], b['gen_n'], b['gen_tp'])})"
        " -- DIAGNOSTIC ONLY, NOT a submission topology\n",
        "gke-topology.txt DIAG line",
    )
    return out


def render_recipe(info: dict) -> str:
    point = info["point"]
    plan = info.get("variant")
    ctx_yaml = apply_server_edits(info["ctx_yaml"])
    gen_yaml = apply_server_edits(info["gen_yaml"])
    client_yaml = apply_client_edits(
        info["client_yaml"], "${{ variables.METRICS_TOKENIZER_WORKERS }}", longer_n(info)
    )

    slurm_workers = "unset (client default)"
    for line in info["client_yaml"].splitlines():
        if line.startswith("  metrics_tokenizer_workers:"):
            slurm_workers = line.split(":", 1)[1].strip()

    ctx_header = CTX_HEADER.format(arch=ARCH_DIR, point=point, ctx_tp=info["ctx_tp"])
    if plan:
        gen_header = GEN_VARIANT_HEADER.format(
            arch=ARCH_DIR,
            point=point,
            nodes=plan["nodes"],
            slurm_gen_n=info["slurm"]["gen_n"],
            slurm_tp=plan["old_tp"],
            old_mbs=plan["old_mbs"],
            gen_n=info["gen_n"],
            gen_tp=info["gen_tp"],
            per_replica=info["concurrency"] // info["gen_n"],
            changes="\n".join(reshape_changes(plan)),
        )
    else:
        gen_header = GEN_HEADER.format(
            arch=ARCH_DIR,
            point=point,
            gen_tp=info["gen_tp"],
            per_replica=info["concurrency"] // info["gen_n"],
        )

    if plan:
        nodes_comment = (
            f"{info['server_nodes']} node(s) of servers "
            f"({info['ctx_gpus']} ctx + {info['gen_gpus']} gen GPUs) -- "
            f"{plan['nodes']}-node variant; Slurm point_{point} uses "
            f"SLURM_NODES = {info['slurm_nodes']}."
        )
    else:
        nodes_comment = (
            f"{info['server_nodes']} node(s) of servers "
            f"({info['ctx_gpus']} ctx + {info['gen_gpus']} gen GPUs). "
            f"Slurm SLURM_NODES = {info['slurm_nodes']}"
            + (
                f" (= {info['server_nodes']} + "
                f"{info['client_reserve_gpus'] // GPUS_PER_NODE} client-reserve "
                "node(s); see CLIENT_RESERVE_NODES)."
                if info["client_reserve_gpus"]
                else "."
            )
        )

    subs = {
        "@@POINT@@": str(point),
        "@@ARCH@@": ARCH_DIR,
        "@@CTX_N@@": str(info["ctx_n"]),
        "@@CTX_TP@@": str(info["ctx_tp"]),
        "@@GEN_N@@": str(info["gen_n"]),
        "@@GEN_TP@@": str(info["gen_tp"]),
        "@@FRONTENDS@@": str(info["frontends"]),
        "@@BASE_PORT@@": str(info["base_port"]),
        "@@FRONTEND_PORT@@": str(info["frontend_port"]),
        "@@CTX_GPUS@@": str(info["ctx_gpus"]),
        "@@GEN_GPUS@@": str(info["gen_gpus"]),
        "@@TOTAL_GPUS@@": str(info["total_gpus"]),
        "@@SERVER_NODES@@": str(info["server_nodes"]),
        "@@CHECKPOINT@@": CHECKPOINT,
        "@@SLURM_MODE@@": str(info["slurm_mode"]),
        "@@SLURM_TOKENIZER_WORKERS@@": slurm_workers,
        "@@PLACEMENT_COMMENT@@": placement_comment(info),
        "@@NODES_COMMENT@@": nodes_comment,
        "@@CTX_YAML@@": indent(ctx_header + ctx_yaml, 10),
        "@@GEN_YAML@@": indent(gen_header + gen_yaml, 10),
        "@@CTX_YAML_SNAP@@": indent(ctx_header + ctx_yaml, 10),
        "@@GEN_YAML_SNAP@@": indent(gen_header + gen_yaml, 10),
        "@@CLIENT_YAML@@": indent(client_yaml, 6),
        "@@GKE_COMPAT_CTX@@": gke_compat_block("ctx"),
        "@@GKE_COMPAT_GEN@@": gke_compat_block("gen"),
        "@@MPIRUN_X@@": MPIRUN_X,
    }

    out = RECIPE_TEMPLATE
    for k, v in subs.items():
        out = out.replace(k, v)
    leftover = [tok for tok in ("@@",) if tok in out]
    if leftover:
        bad = [l for l in out.splitlines() if "@@" in l][:5]
        sys.exit(f"{label_of(info)}: unsubstituted placeholder(s):\n  " + "\n  ".join(bad))
    if plan:
        out = apply_variant_text(out, info)
    if info.get("diag"):
        out = apply_diag_text(out, info)
    return out


LAUNCH_VARIANT_SCRIPT = r'''#!/usr/bin/env bash
# Launch the @@NODES@@-NODE VARIANT of DeepSeek-R1 MLPerf-endpoints v1.0.2 point
# C@@POINT@@ on a GB300 (A4X-max) GKE cluster. NOT a submission topology:
#   @@TOPO@@ = @@TOTAL_GPUS@@ GPUs = @@NODES@@ nodes
#   (Slurm point_@@POINT@@: @@SLURM_TOPO@@ = 72 GPUs = 18 nodes).
# The header of @@RECIPE@@ says what changed and why.
# GENERATED by ../../generate_point_recipes.py -- do not hand-edit.
#
# PREREQUISITES (../../README.md §3):
#   1. @@NODES@@ schedulable A4X-max nodes (counted below before launching).
#   2. kubectl apply -f ../../stage-assets.yaml    # v2 checkpoint + eval parquet, every node
#   3. kubectl apply -f ../../mk-report-dir.yaml   # durable report hostPath
#   4. kubectl apply -f ../../prepull-images.yaml  # recommended: avoid a cold pull
#      burning the reservation window
#   5. CLIENT_IMAGE / LCB_IMAGE reachable from the nodes (mirror ghcr.io into
#      Artifact Registry if not) -- pass with -s.
#   6. kubectl context pointing at the target cluster.
#
# Site overrides:
#   NAMESPACE=<ns>  NODEPOOL=<a4x-max nodepool>   (env vars)
#   SKIP_NODE_CHECK=1                             skip the node count below
#   anything else via sflow flags, e.g.
#     ./launch_test.sh -s MODE=both -s CLIENT_IMAGE=<registry>/endpoints:e9f427c
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

RECIPE="@@RECIPE@@"
NAMESPACE="${NAMESPACE:-default}"
NODEPOOL="${NODEPOOL:<nodepool>}"
REQUIRED_NODES=@@NODES@@

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
'''


def render_diag_launch(d: dict) -> str:
    """The 18-node point's launch_test.sh, re-pointed at the diag recipe and
    marked DIAGNOSTIC (same node gate: REQUIRED_NODES 18)."""
    b, g, point = d["base"], d["diag"], d["point"]
    out = render_launch(b)
    out = diag_sub(
        out,
        f"# Launch DeepSeek-R1 MLPerf-endpoints v1.0.2 point C{point} on a GB300 (A4X-max)\n"
        "# GKE cluster. Slurm-faithful topology:\n"
        f"#   {b['ctx_n']} ctx x DEP{b['ctx_tp']} + {b['gen_n']} gen x DEP{b['gen_tp']} = "
        f"{b['total_gpus']} GPUs = {b['server_nodes']} nodes.\n"
        "# GENERATED by ../../generate_point_recipes.py -- do not hand-edit.\n",
        f"# DIAGNOSTIC ONLY, NOT A SUBMISSION TOPOLOGY -- never report it as C{point}.\n"
        f"# Launch diag {g['name']} of DeepSeek-R1 MLPerf-endpoints v1.0.2 point C{point}\n"
        "# on a GB300 (A4X-max) GKE cluster:\n"
        f"#   {d['ctx_n']} ctx x DEP{d['ctx_tp']} + {d['gen_n']} gen x DEP{d['gen_tp']} = "
        f"{d['total_gpus']} GPUs = {d['server_nodes']} nodes\n"
        f"#   (NVIDIA point_{point}: {b['gen_n']} gen x DEP{b['gen_tp']}). The banner of\n"
        f"#   {recipe_name(d)} lists every deviation.\n"
        "# GENERATED by ../../generate_point_recipes.py --diag -- do not hand-edit.\n",
        "launch header",
    )
    out = diag_sub(out, f'RECIPE="{recipe_name(b)}"\n', f'RECIPE="{recipe_name(d)}"\n',
                   "launch RECIPE")
    out = diag_sub(
        out,
        'sflow run -f "${RECIPE}" \\\n',
        f'echo "DIAGNOSTIC ONLY, NOT A SUBMISSION TOPOLOGY: diag {g["name"]} of C{point} '
        f'({d["gen_n"]} x DEP{d["gen_tp"]} decode)" >&2\n'
        'sflow run -f "${RECIPE}" \\\n',
        "launch warning line",
    )
    return out


def render_launch(info: dict) -> str:
    if info.get("diag"):
        return render_diag_launch(info)
    plan = info.get("variant")
    if plan:
        slurm = info["slurm"]
        out = LAUNCH_VARIANT_SCRIPT
        subs = {
            "@@NODES@@": str(plan["nodes"]),
            "@@POINT@@": str(info["point"]),
            "@@TOPO@@": topo_desc(info["ctx_n"], info["ctx_tp"], info["gen_n"], info["gen_tp"]),
            "@@SLURM_TOPO@@": topo_desc(slurm["ctx_n"], slurm["ctx_tp"],
                                        slurm["gen_n"], slurm["gen_tp"]),
            "@@TOTAL_GPUS@@": str(info["total_gpus"]),
            "@@RECIPE@@": recipe_name(info),
        }
    else:
        out = LAUNCH_SCRIPT
        subs = {
            "@@POINT@@": str(info["point"]),
            "@@CTX_N@@": str(info["ctx_n"]),
            "@@CTX_TP@@": str(info["ctx_tp"]),
            "@@GEN_N@@": str(info["gen_n"]),
            "@@GEN_TP@@": str(info["gen_tp"]),
            "@@TOTAL_GPUS@@": str(info["total_gpus"]),
            "@@SERVER_NODES@@": str(info["server_nodes"]),
        }
    for k, v in subs.items():
        out = out.replace(k, v)
    if "@@" in out:
        sys.exit(f"{label_of(info)}: unsubstituted launch placeholder(s)")
    return out


# ==========================================================================
# --check: diff the generated recipes back against the Slurm drop
# ==========================================================================


def extract_heredoc(text: str, marker: str) -> str:
    """Pull the body of a `cat << 'MARKER'` heredoc back out of a recipe."""
    start = text.index(f"cat << '{marker}'")
    start = text.index("\n", start) + 1
    end = text.index(f"- {marker}\n", start)
    body = text[start:end]
    lines = []
    for line in body.split("\n"):
        if not line.strip():
            continue
        assert line.startswith(" " * 10), f"heredoc line not indented 10: {line!r}"
        lines.append(line[10:])
    return "\n".join(lines) + "\n"


def extract_artifact_content(text: str, name: str) -> str:
    marker = f"  - name: {name}\n"
    start = text.index(marker)
    start = text.index("    content: |\n", start) + len("    content: |\n")
    lines = []
    for line in text[start:].split("\n"):
        if line.strip() and not line.startswith(" " * 6):
            break
        if not line.strip():
            continue
        lines.append(line[6:])
    return "\n".join(lines) + "\n"


def strip_comments(text: str) -> str:
    return "\n".join(
        l for l in text.split("\n") if l.strip() and not l.strip().startswith("#")
    )


def check_point(info: dict, dest: pathlib.Path) -> list[str]:
    point = info["point"]
    label = label_of(info)
    path = recipe_dir(info, dest) / recipe_name(info)
    problems: list[str] = []
    if not path.exists():
        return [f"{label}: MISSING {path}"]
    text = path.read_text()

    # 0. It must parse.
    try:
        yaml.safe_load(text)
    except Exception as e:
        problems.append(f"{label}: recipe is not valid YAML: {e}")

    # 1. Server YAML byte-equality modulo the mandatory edits.
    #    (v1.0.2 fix: the loop variable used to be `label`, which shadowed the
    #    point label, so every later problem was reported as "server-gen:".)
    for marker, slurm_text, what in (
        ("EOF_CTX", info["ctx_yaml"], "server-ctx"),
        ("EOF_GEN", info["gen_yaml"], "server-gen"),
        ("EOF_SNAP_CTX", info["ctx_yaml"], "snapshot server-ctx"),
        ("EOF_SNAP_GEN", info["gen_yaml"], "snapshot server-gen"),
    ):
        got = strip_comments(extract_heredoc(text, marker))
        want = strip_comments(apply_server_edits(slurm_text))
        if got != want:
            problems.append(f"{label}: {what} differs from the Slurm drop")
            problems += _diff_lines(want, got, indent="    ")

    # 2. The server body must also be byte-equal to Slurm once the edits are
    #    undone, so a stray hand-edit to a generated file is caught.
    for marker, slurm_text, what in (
        ("EOF_CTX", info["ctx_yaml"], "server-ctx"),
        ("EOF_GEN", info["gen_yaml"], "server-gen"),
    ):
        got = strip_comments(extract_heredoc(text, marker))
        undone = (
            got.replace("print_iter_log: false", "print_iter_log: true")
            .replace("  kv_transfer_timeout_ms: 120000\n", "")
            .replace("  kv_transfer_sender_future_timeout_ms: 10000\n", "")
        )
        if undone.strip() != strip_comments(slurm_text).strip():
            problems.append(
                f"{label}: {what} is NOT the Slurm file plus the three "
                "documented edits (someone hand-edited the generated recipe?)"
            )

    # 3. Client YAML (one more documented edit for a --longer-runs point).
    got = strip_comments(extract_artifact_content(text, "CLIENT_CONFIG"))
    want = strip_comments(
        apply_client_edits(info["client_yaml"], "${{ variables.METRICS_TOKENIZER_WORKERS }}",
                           longer_n(info))
    )
    if got != want:
        problems.append(f"{label}: CLIENT_CONFIG differs from the Slurm drop")
        problems += _diff_lines(want, got, indent="    ")

    # 4. The client config must still carry the fields that make it a submission.
    cfg = yaml.safe_load(extract_artifact_content(text, "CLIENT_CONFIG"))
    for key in ("submission_ref", "datasets", "settings", "endpoint_config"):
        if key not in cfg:
            problems.append(f"{label}: CLIENT_CONFIG lost '{key}'")
    n_issue = cfg.get("settings", {}).get("runtime", {}).get("n_samples_to_issue")
    if n_issue != info["n_samples"]:
        problems.append(f"{label}: n_samples_to_issue is {n_issue}, expected "
                        f"{info['n_samples']}")
    ds_names = {d.get("name") for d in cfg.get("datasets", [])}
    if "deepseek_r1_accuracy" not in ds_names:
        problems.append(f"{label}: CLIENT_CONFIG lost the accuracy dataset")
    if cfg.get("report_dir") != "${REPORT_DIR}":
        problems.append(f"{label}: report_dir is {cfg.get('report_dir')!r}")
    if cfg.get("endpoint_config", {}).get("endpoints") != []:
        problems.append(f"{label}: endpoints should stay [] (filled at runtime)")
    warmup = cfg.get("settings", {}).get("warmup")
    if warmup and warmup.get("enabled") and not warmup.get("salt"):
        problems.append(
            f"{label}: warmup.enabled without warmup.salt -- MLPerf rule violation"
        )

    # 5. Topology written into the recipe must match the drop.
    v = yaml.safe_load(text)["variables"]
    for name, want_val in (
        ("NUM_CTX_SERVERS", info["ctx_n"]),
        ("GPUS_PER_CTX_RANK", info["ctx_tp"]),
        ("NUM_GEN_SERVERS", info["gen_n"]),
        ("GPUS_PER_GEN_RANK", info["gen_tp"]),
        ("NUM_FRONTEND_SERVERS", info["frontends"]),
        ("BASE_PORT", info["base_port"]),
        ("FRONTEND_PORT", info["frontend_port"]),
        ("SERVER_NODES", info["server_nodes"]),
    ):
        if int(v[name]["value"]) != want_val:
            problems.append(
                f"{label}: {name} is {v[name]['value']}, "
                f"{'the diag topology' if info.get('diag') else 'drop'} says {want_val}"
            )

    # 6. Every gen config must still have real CUDA graphs. This is the one
    #    thing that fails silently at 13x the cost.
    gen_cfg = yaml.safe_load(extract_heredoc(text, "EOF_GEN"))
    cg = gen_cfg.get("cuda_graph_config")
    if not cg or not cg.get("batch_sizes"):
        problems.append(
            f"{label}: decode cuda_graph_config is null/empty -- "
            "this costs ~13x throughput and produces no error"
        )
    elif max(cg["batch_sizes"]) != gen_cfg["max_batch_size"]:
        problems.append(
            f"{label}: cuda graph batch_sizes top out at "
            f"{max(cg['batch_sizes'])} but max_batch_size is {gen_cfg['max_batch_size']}"
        )
    ctx_cfg = yaml.safe_load(extract_heredoc(text, "EOF_CTX"))
    if ctx_cfg.get("cuda_graph_config") is not None:
        problems.append(f"{label}: prefill should stay eager (cuda_graph_config: null)")

    # 7. Every task script must be a LIST OF STRINGS that parses as bash.
    #
    #    Both halves of this caught a real bug. A plain YAML scalar containing
    #    ": " (e.g. `- echo "dataset: $(ls -l ...)"`) parses as a MAPPING, not a
    #    string, and sflow silently never runs that line. And a mis-quoted
    #    embedded python heredoc is invisible to the YAML parser and to
    #    `sflow --dry-run` alike -- it only surfaces as a shell error inside the
    #    pod, i.e. after the 18-node reservation.
    doc = yaml.safe_load(text)
    for task in doc["workflow"]["tasks"]:
        lines = []
        for i, item in enumerate(task.get("script", [])):
            if not isinstance(item, str):
                problems.append(
                    f"{label}: task '{task['name']}' script[{i}] parsed as "
                    f"{type(item).__name__}, not str -- an unquoted ': ' in the "
                    f"line? sflow will not run it: {item!r}"
                )
            else:
                lines.append(item)
        script = re.sub(r"\$\{\{[^}]*\}\}", "SFLOWVAL", "\n".join(lines))
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
            fh.write(script)
            tmp = fh.name
        r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
        os.unlink(tmp)
        if r.returncode:
            problems.append(
                f"{label}: task '{task['name']}' script is not valid bash:\n"
                + "\n".join("      " + l for l in r.stderr.strip().split("\n")[:8])
            )

    # 8. launch_test.sh present, executable, and valid bash.
    ls = recipe_dir(info, dest) / "launch_test.sh"
    if not ls.exists():
        problems.append(f"{label}: MISSING launch_test.sh")
    elif not ls.stat().st_mode & 0o111:
        problems.append(f"{label}: launch_test.sh is not executable")
    else:
        r = subprocess.run(["bash", "-n", str(ls)], capture_output=True, text=True)
        if r.returncode:
            problems.append(f"{label}: launch_test.sh is not valid bash: {r.stderr.strip()}")
        elif ls.read_text() != render_launch(info):
            problems.append(f"{label}: launch_test.sh differs from a fresh render "
                            "-- hand-edited or not regenerated")

    # 9. The recipe must be exactly what the generator renders now: catches a
    #    hand edit AND a template change that was never regenerated.
    if text != render_recipe(info):
        problems.append(f"{label}: differs from a fresh render of RECIPE_TEMPLATE "
                        "-- hand-edited or not regenerated; run the generator")

    # 10. Variants only: the decode config may differ from NVIDIA's ONLY in the
    #     RESHAPE_KEYS, prefill not at all, and every request needs a slot.
    plan = info.get("variant")
    if plan:
        slurm = info["slurm"]
        edited_keys = {"print_iter_log", "cache_transceiver_config"}
        for marker, drop_text, allowed, what in (
            ("EOF_GEN", slurm["gen_yaml"], set(RESHAPE_KEYS) | edited_keys, "server-gen"),
            ("EOF_CTX", slurm["ctx_yaml"], edited_keys, "server-ctx"),
        ):
            got_cfg = yaml.safe_load(extract_heredoc(text, marker))
            drop_cfg = yaml.safe_load(drop_text)
            diff = {k for k in set(got_cfg) | set(drop_cfg)
                    if got_cfg.get(k) != drop_cfg.get(k)}
            if diff - allowed:
                problems.append(f"{label}: {what} differs from the drop in "
                                f"{sorted(diff - allowed)} -- only {sorted(allowed)} may")
        if gen_cfg["tensor_parallel_size"] != info["gen_tp"] or \
                gen_cfg["moe_expert_parallel_size"] != info["gen_tp"]:
            problems.append(f"{label}: server-gen TP/EP is not {info['gen_tp']} "
                            "(must equal GPUS_PER_GEN_RANK)")
        slots = info["gen_n"] * gen_cfg["tensor_parallel_size"] * gen_cfg["max_batch_size"]
        if slots < info["concurrency"]:
            problems.append(f"{label}: {slots} decode slots < target_concurrency "
                            f"{info['concurrency']}")
        if doc["workflow"]["name"] != variant_workflow_name(info):
            problems.append(f"{label}: workflow.name is {doc['workflow']['name']!r}")
        if "VARIANT              = " not in text:
            problems.append(f"{label}: gke-topology.txt has no VARIANT line")

    return problems


# --------------------------------------------------------------------------
# --check-diag (and a plain --check, for every diag on disk). A diagnostic may
# differ from a FRESH RENDER of its 18-node point only in the DIAG keys, its
# names and comments; anything else fails closed.
# --------------------------------------------------------------------------
HEREDOC_RE = re.compile(r"cat << '(EOF_(?:SNAP_)?(?:CTX|GEN))' [^\n]*\n")
DIAG_TOPO_LINE = "DIAG                 = "


def _diag_norm(obj):
    """A parsed recipe in which every EOF_{SNAP_,}{CTX,GEN} heredoc script item
    is replaced by its parsed YAML (so only comments inside it are ignored)
    and the gke-topology.txt heredoc loses its DIAG line. Nothing else is
    normalized: every other key, value and string must match exactly."""
    if isinstance(obj, dict):
        return {k: _diag_norm(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_diag_norm(v) for v in obj]
    if isinstance(obj, str):
        m = HEREDOC_RE.match(obj)
        if m:
            try:
                body = yaml.safe_load(obj[m.end():])
            except yaml.YAMLError:
                body = "<unparsable YAML>"
            return {"<heredoc>": m.group(1), f"<yaml {m.group(1)}>": body}
        if obj.startswith("cat << 'EOF_SNAP_ENV'"):
            return "\n".join(l for l in obj.split("\n") if not l.startswith(DIAG_TOPO_LINE))
    return obj


def _tree_diff(a, b, path: str, out: list) -> None:
    """Append (path, a, b) for every leaf where two parsed recipes differ.
    List items that are named (tasks, artifacts) are addressed by name."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(a) + [k for k in b if k not in a]:
            p = f"{path}.{k}" if path else str(k)
            if k not in b:
                out.append((p, a[k], "<absent>"))
            elif k not in a:
                out.append((p, "<absent>", b[k]))
            else:
                _tree_diff(a[k], b[k], p, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"<{len(a)} items>", f"<{len(b)} items>"))
        for i, (x, y) in enumerate(zip(a, b)):
            tag = x["name"] if isinstance(x, dict) and isinstance(x.get("name"), str) else i
            _tree_diff(x, y, f"{path}[{tag}]", out)
    elif type(a) is not type(b) or a != b:
        out.append((path, a, b))


def _short(v) -> str:
    r = repr(v)
    return r if len(r) <= 100 else r[:97] + "..."


def _changed_lines(old: str, new: str) -> tuple[list[str], list[str]]:
    """(removed, added) NON-COMMENT, non-blank lines between two texts.
    autojunk=False: difflib's popular-line heuristic would otherwise mis-align
    a ~2,000-line recipe and report unchanged lines as changed."""
    import difflib

    a, b = old.split("\n"), new.split("\n")
    removed, added = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            removed += a[i1:i2]
        if tag in ("replace", "insert"):
            added += b[j1:j2]

    def code(lines: list[str]) -> list[str]:
        return [l for l in lines if l.strip() and not l.strip().startswith("#")]

    return code(removed), code(added)


def check_diag(d: dict, dest: pathlib.Path) -> list[str]:
    g, b, point = d["diag"], d["base"], d["point"]
    name, label = g["name"], label_of(d)
    marker = f"# DIAG {name}:"
    # The non-comment lines a diag changes: NUM_GEN_SERVERS, GPUS_PER_GEN_RANK,
    # workflow.name, and DIAG_GEN_KEYS in EOF_GEN + EOF_SNAP_GEN. Each carries
    # the inline marker; the gke-topology.txt DIAG line is the only addition.
    n_marked = 3 + 2 * len(DIAG_GEN_KEYS)

    # (a) Everything --check verifies for a point -- YAML, server/client
    #     bodies, topology variables vs the diag's OWN topology, CUDA graphs,
    #     task scripts are lists of strings that parse as bash, launch_test.sh
    #     executable / bash / fresh render -- and byte-equality with a fresh
    #     render (catches a hand edit and a diag not regenerated).
    problems = check_point(d, dest)
    path = recipe_dir(d, dest) / recipe_name(d)
    if not path.exists():
        return problems
    text = path.read_text()
    base_text = render_recipe(b)
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return problems  # check_point() reported it
    base_doc = yaml.safe_load(base_text)

    # (b) Semantic diff against a fresh render of 18nodes/point_<C>: only the
    #     expected deviations, each exactly once, with exactly these values.
    diffs: list = []
    _tree_diff(_diag_norm(base_doc), _diag_norm(doc), "", diffs)
    want = {
        "variables.NUM_GEN_SERVERS.value": (b["gen_n"], d["gen_n"]),
        "variables.GPUS_PER_GEN_RANK.value": (b["gen_tp"], d["gen_tp"]),
        "workflow.name": (base_doc["workflow"]["name"], diag_workflow_name(d)),
    }
    heredoc_task = {"EOF_GEN": "decode_server", "EOF_SNAP_GEN": "snapshot_configs"}
    gen_re = re.compile(r"^workflow\.tasks\[([a-z_]+)\]\.script\[\d+\]\.<yaml (EOF_GEN|EOF_SNAP_GEN)>\.("
                        + "|".join(DIAG_GEN_KEYS) + r")$")
    expected = set(want) | {f"{mk}.{k}" for mk in heredoc_task for k in DIAG_GEN_KEYS}
    seen: list[str] = []
    for p, x, y in diffs:
        m = gen_re.match(p)
        if p in want and (x, y) == want[p] and type(x) is type(want[p][0]) \
                and type(y) is type(want[p][1]):
            seen.append(p)
        elif m and heredoc_task[m.group(2)] == m.group(1) and type(x) is int and \
                type(y) is int and (x, y) == (b["gen_tp"], d["gen_tp"]):
            seen.append(f"{m.group(2)}.{m.group(3)}")
        else:
            problems.append(f"{label}: differs from the generated point_{point} at {p}: "
                            f"{_short(x)} -> {_short(y)} -- not a DIAG {name} deviation")
    for e in sorted(expected - set(seen)):
        problems.append(f"{label}: expected DIAG {name} deviation {e} is missing")
    for e in sorted({s for s in seen if seen.count(s) > 1}):
        problems.append(f"{label}: DIAG {name} deviation {e} appears {seen.count(e)} times")

    # (c) Line level: every changed non-comment line carries the inline marker
    #     (the one added gke-topology.txt DIAG line excepted), and no
    #     non-comment line was added or dropped beyond those.
    removed, added = _changed_lines(base_text, text)
    for l in added:
        s = l.strip()
        if marker not in l and not s.startswith(DIAG_TOPO_LINE):
            problems.append(f"{label}: changed line has no inline \"{marker}\" marker: {_short(s)}")
    if len(removed) != n_marked or len(added) != n_marked + 1:
        problems.append(f"{label}: {len(removed)} non-comment line(s) changed and "
                        f"{len(added)} written vs the generated point_{point}; expected "
                        f"{n_marked} and {n_marked} + the DIAG line")

    # (d) Invariants, read back from the file.
    if "DIAGNOSTIC ONLY, NOT A SUBMISSION TOPOLOGY" not in text.split("\nvariables:\n", 1)[0]:
        problems.append(f"{label}: the header lost the DIAGNOSTIC ONLY banner")
    lines = text.split("\n")
    n_topo = sum(1 for l in lines if l.strip().startswith(DIAG_TOPO_LINE))
    if n_topo != 1:
        problems.append(f"{label}: {n_topo} gke-topology.txt DIAG line(s), expected 1")
    got_marked = sum(1 for l in lines if marker in l and not l.strip().startswith("#"))
    if got_marked != n_marked:
        problems.append(f"{label}: {got_marked} inline \"{marker}\" markers, expected {n_marked}")
    v = doc["variables"]
    try:
        n_gen, tp = int(v["NUM_GEN_SERVERS"]["value"]), int(v["GPUS_PER_GEN_RANK"]["value"])
        ctx_gpus = int(v["NUM_CTX_SERVERS"]["value"]) * int(v["GPUS_PER_CTX_RANK"]["value"])
        nodes = int(v["SERVER_NODES"]["value"])
    except (KeyError, TypeError, ValueError) as e:
        return problems + [f"{label}: topology variables unreadable: {e}"]
    if n_gen * tp != b["gen_gpus"]:
        problems.append(f"{label}: {n_gen} x DEP{tp} = {n_gen * tp} decode GPUs; point_{point} "
                        f"has {b['gen_gpus']} (a diag re-splits them, never adds or drops)")
    if nodes != b["server_nodes"] or ctx_gpus + n_gen * tp != nodes * GPUS_PER_NODE:
        problems.append(f"{label}: {ctx_gpus} ctx + {n_gen * tp} gen GPUs on {nodes} nodes; "
                        f"point_{point} is {b['total_gpus']} GPUs on {b['server_nodes']}")
    base_mbs = yaml.safe_load(b["gen_yaml"])["max_batch_size"]
    client = yaml.safe_load(extract_artifact_content(text, "CLIENT_CONFIG"))
    conc = client["settings"]["load_pattern"]["target_concurrency"]
    for mk in heredoc_task:
        cfg = yaml.safe_load(extract_heredoc(text, mk))
        if any(cfg.get(k) != tp for k in DIAG_GEN_KEYS):
            problems.append(f"{label}: {mk} {'/'.join(DIAG_GEN_KEYS)} != GPUS_PER_GEN_RANK {tp}")
        mbs = cfg.get("max_batch_size")
        if mbs != base_mbs:
            problems.append(f"{label}: {mk} max_batch_size {mbs}, point_{point} has {base_mbs}")
        elif n_gen * tp * mbs < conc:
            problems.append(f"{label}: {n_gen} x {tp} x {mbs} = {n_gen * tp * mbs} decode slots "
                            f"< target_concurrency {conc}")
    if doc["workflow"]["name"] != diag_workflow_name(d):
        problems.append(f"{label}: workflow.name is {doc['workflow']['name']!r}, expected "
                        f"{diag_workflow_name(d)!r} (the run / report dir tag)")

    # (e) launch_test.sh differs from the point's only in header comments,
    #     RECIPE and the DIAGNOSTIC warning echo (check_point() already made
    #     it equal to a fresh render).
    ls = recipe_dir(d, dest) / "launch_test.sh"
    if ls.exists():
        removed, added = _changed_lines(render_launch(b), ls.read_text())
        ok_added = (f'RECIPE="{recipe_name(d)}"',
                    'echo "DIAGNOSTIC ONLY, NOT A SUBMISSION TOPOLOGY: ')
        if [l.strip() for l in removed] != [f'RECIPE="{recipe_name(b)}"'] or \
                len(added) != 2 or not all(l.strip().startswith(ok_added) for l in added):
            problems.append(f"{label}: launch_test.sh must differ from point_{point}'s in exactly "
                            f"RECIPE + the DIAGNOSTIC warning echo; got -{removed} +{added}")

    # (f) The committed 18-node recipe, if it is under dest, is the render the
    #     diag was compared with.
    committed = recipe_dir(b, dest) / recipe_name(b)
    if committed.exists() and committed.read_text() != base_text:
        problems.append(f"{label}: {committed} differs from a fresh render, so the diag "
                        "was not compared with what is committed -- run --check")
    return problems


def _diff_lines(want: str, got: str, indent: str = "") -> list[str]:
    import difflib

    return [
        indent + l
        for l in list(
            difflib.unified_diff(
                want.split("\n"), got.split("\n"),
                fromfile="slurm(+edits)", tofile="generated", lineterm="",
            )
        )[:40]
    ]


# ==========================================================================
# main
# ==========================================================================


def print_variant_layout(variants: list[dict]) -> None:
    hdr = (f"{'point':>6} {'Slurm (72 GPUs)':<18} {'variant':<18} {'GPUs':>5} "
           f"{'nodes':>5} {'mbs':>9} {'slots/C':>13}  placement / rule")
    print(hdr)
    print("-" * len(hdr))
    for v in variants:
        s, plan = v["slurm"], v["variant"]
        print(
            f"{v['point']:>6} "
            f"{topo_desc(s['ctx_n'], s['ctx_tp'], s['gen_n'], s['gen_tp']):<18} "
            f"{topo_desc(v['ctx_n'], v['ctx_tp'], v['gen_n'], v['gen_tp']):<18} "
            f"{v['total_gpus']:>5} {v['server_nodes']:>5} "
            f"{str(plan['old_mbs']) + '->' + str(plan['mbs']):>9} "
            f"{str(gen_n_slots(plan)) + '/' + str(v['concurrency']):>13}  "
            f"{placement_comment(v)}"
        )
        print(f"{'':>6} {'':<18} {'':<18} {'':>5} {'':>5} {'':>9} {'':>13}  "
              f"{plan['regime']}")


def apply_longer_runs(infos: list[dict]) -> None:
    """--longer-runs: swap in LONGER_RUNS' n_samples_to_issue, after checking it."""
    for i, info in enumerate(infos):
        if info["point"] not in LONGER_RUNS:
            continue
        point = info["point"]
        new_n, perf_s, client_s = LONGER_RUNS[point]
        drop_n = info["n_samples"]
        settings = yaml.safe_load(info["client_yaml"])["settings"]
        cap_ms = settings["runtime"].get("max_issue_duration_ms")
        watchdog = (settings.get("timeouts") or {}).get("run_timeout_s")
        problems = []
        if new_n <= drop_n:
            problems.append(f"{new_n} is not above NVIDIA's {drop_n}")
        if new_n % DATASET_PROMPTS:
            problems.append(f"{new_n} is not a whole number of "
                            f"{DATASET_PROMPTS}-prompt passes")
        if cap_ms is not None and perf_s >= LONGER_RUNS_HEADROOM * cap_ms / 1000:
            problems.append(f"the predicted {perf_s} s performance phase reaches "
                            f"{LONGER_RUNS_HEADROOM:.0%} of max_issue_duration_ms {cap_ms}")
        if watchdog is not None and client_s >= LONGER_RUNS_HEADROOM * watchdog:
            problems.append(f"the projected {client_s} s client run reaches "
                            f"{LONGER_RUNS_HEADROOM:.0%} of the whole-run watchdog "
                            f"run_timeout_s {watchdog:g}")
        if problems:
            sys.exit(f"point_{point}: --longer-runs refused: " + "; ".join(problems))
        infos[i] = dict(info, n_samples=new_n, drop_n_samples=drop_n)
        print(f"longer run: point_{point} n_samples_to_issue {drop_n:,} -> {new_n:,} "
              f"({drop_n / DATASET_PROMPTS:g} -> {new_n / DATASET_PROMPTS:g} passes); "
              f"predicted performance phase {perf_s:,} s, client run {client_s:,} s"
              + (f" of a {watchdog:,.0f} s watchdog" if watchdog else " (no watchdog)"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slurm-root", type=pathlib.Path, default=DEFAULT_SLURM_ROOT)
    ap.add_argument("--dest", type=pathlib.Path, default=HERE)
    ap.add_argument("--points", type=int, nargs="*", default=POINTS)
    ap.add_argument("--check", action="store_true",
                    help="verify the generated recipes against the Slurm drop")
    ap.add_argument("--layout", action="store_true",
                    help="print the reduced-node variant splits and exit "
                         "(needs --variants)")
    ap.add_argument("--variants", type=int, nargs="*", default=[],
                    choices=VARIANT_NODES, metavar="N",
                    help="ALSO emit/check reduced-node <N>nodes/point_<C>/ "
                         f"variants (supported: {VARIANT_NODES}). Off by "
                         "default: v1.0.2 targets the 18-node pool.")
    ap.add_argument("--longer-runs", action="store_true",
                    help="emit/check the RERUN recipes: n_samples_to_issue raised "
                         "on " + ", ".join(f"C{p}" for p in sorted(LONGER_RUNS))
                         + " (LONGER_RUNS). Off by default.")
    ap.add_argument("--diag", action="store_true",
                    help="write ONLY the DIAGNOSTIC recipes (DIAGNOSTICS -> "
                         f"{DIAG_DIR}/point_<C>_<name>/) and exit; never touches "
                         "18nodes/. NOT submission topologies.")
    ap.add_argument("--check-diag", action="store_true",
                    help="verify the DIAGNOSTIC recipes (every one must exist); a "
                         "plain --check also verifies each one present on disk")
    args = ap.parse_args()

    if not args.slurm_root.is_dir():
        sys.exit(
            f"Slurm drop not found at {args.slurm_root}\n"
            "Pass --slurm-root /path/to/NVIDIA/src/deepseek-r1"
        )
    if args.longer_runs and args.variants:
        sys.exit("--longer-runs is for the 18-node submission points only; drop --variants")
    if args.diag and (args.check or args.check_diag or args.variants or args.longer_runs
                      or args.layout):
        sys.exit("--diag writes ONLY the diag/ recipes: run it alone, then --check-diag")
    if args.check_diag and (args.variants or args.longer_runs or args.layout):
        sys.exit("--check-diag verifies the diag/ recipes only; drop --variants / "
                 "--longer-runs / --layout")
    if (args.diag or args.check_diag) and args.points != POINTS:
        sys.exit("--diag / --check-diag take their points from DIAGNOSTICS; drop --points")

    # v1.0.2 pins (checkpoint, paths, env, images): drop side, then GKE side.
    assert_drop_shared(args.slurm_root)
    assert_gke_consistency()
    infos = []
    for point in args.points:
        info = read_point(args.slurm_root, point)
        assert_spec_table(info)
        infos.append(info)
    if args.longer_runs:
        apply_longer_runs(infos)

    variants = [
        variant_info(info, plan_variant(info, nodes))
        for nodes in args.variants for info in infos
    ]

    # Diagnostics: each from a FRESH default-mode read of its point, whatever
    # --points / --longer-runs say (diag_info() refuses anything else).
    diags = []
    if args.diag or args.check_diag or args.check:
        for name, spec in DIAGNOSTICS.items():
            base = read_point(args.slurm_root, spec["point"])
            assert_spec_table(base)
            diags.append(diag_info(base, name))

    if args.layout:
        print_variant_layout(variants)
        return 0

    if args.diag:
        diag_root = (args.dest / DIAG_DIR).resolve()
        print(f"{'diagnostic':>27} {'topology':<24} {'GPUs':>5} {'nodes':>6} "
              f"{'C/replica':>10} {'slots':>6}  gen servers")
        for d in diags:
            out_dir = recipe_dir(d, args.dest)
            if out_dir.resolve().parent != diag_root:
                sys.exit(f"{label_of(d)}: {out_dir} is not directly under {diag_root}")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / recipe_name(d)).write_text(render_recipe(d))
            ls = out_dir / "launch_test.sh"
            ls.write_text(render_launch(d))
            ls.chmod(0o755)
            g = d["diag"]
            topo = f"{d['ctx_n']}xDEP{d['ctx_tp']} + {d['gen_n']}xDEP{d['gen_tp']}"
            gens = "; ".join(f"nodes {lo}-{hi} (URL node {n}, port {p})" for (lo, hi), n, p
                             in zip(g["gen_nodes"], g["url_nodes"], g["url_ports"]))
            print(f"{label_of(d).removeprefix('point_'):>27} {topo:<24} {d['total_gpus']:>5} "
                  f"{d['server_nodes']:>6} {d['concurrency'] // d['gen_n']:>10} "
                  f"{g['slots']:>6}  {gens}")
        print(f"\nwrote {len(diags)} DIAGNOSTIC recipe(s) under {diag_root} and nothing "
              "else.\nDIAGNOSTIC ONLY, NOT SUBMISSION TOPOLOGIES.")
        print("run `./generate_point_recipes.py --check-diag` to verify")
        return 0

    if args.check or args.check_diag:
        all_problems: list[str] = []
        if args.check:
            for info in infos + variants:
                all_problems += check_point(info, args.dest)
        # --check-diag: every DIAGNOSTICS entry must exist (MISSING is a
        # problem). Plain --check: every one that is on disk.
        checked_diags = [d for d in diags
                         if args.check_diag or recipe_dir(d, args.dest).exists()]
        for d in checked_diags:
            all_problems += check_diag(d, args.dest)
        known = {recipe_dir(d, args.dest).name for d in diags}
        diag_root = args.dest / DIAG_DIR
        for p in sorted(diag_root.iterdir()) if diag_root.is_dir() else []:
            if p.name not in known:
                all_problems.append(f"{DIAG_DIR}/{p.name}: not generated by any DIAGNOSTICS "
                                    "entry (stale or hand-made?) -- remove it or add the entry")
        if args.check:
            print(f"checked {len(infos)} point(s) and {len(variants)} reduced-node "
                  f"variant(s) against {args.slurm_root}")
        print(f"checked {len(checked_diags)} diagnostic(s) under {args.dest / DIAG_DIR}"
              + (f": {', '.join(label_of(d) for d in checked_diags)}" if checked_diags else ""))
        if all_problems:
            print("\nPROBLEMS:")
            for p in all_problems:
                print("  " + p)
            return 1
        if not args.check:
            print("OK. Before any diagnostic: the same fail-closed drop / GKE pins as --check.")
        else:
            print("OK. Before any point (fail closed, also when generating):")
            print(f"  - the {DROP_VERSION} drop still pins MODEL_PATH / DATA_PATH /")
            print("    CONTAINER_IMAGE / client + LCB images, has no CLIENT_RESERVE_GPUS,")
            print("    SITE_ENV_YAML defaults to '', and server-env.yaml / endpoint_env.yaml")
            print("    are exactly SERVER_ENV_KEEP + SERVER_ENV_DROP / ENDPOINT_ENV")
            print("  - the recipe template uses the matching node-local MODEL_PATH /")
            print("    DATA_PATH and image defaults, exports + forwards every kept")
            print("    server-env var, and ./stage-assets.yaml stages CHECKPOINT")
            print("For each point:")
            print("  - server-ctx/gen heredocs (and their snapshot copies) are byte-identical")
            print("    to the Slurm drop modulo the three documented edits")
            print("  - CLIENT_CONFIG is byte-identical to client.yaml modulo the two")
            print("    documented edits; submission_ref + the accuracy dataset survived;")
            print("    warmup.enabled implies warmup.salt; report_dir is the durable one;")
            print("    endpoint_config.endpoints is still [] (filled at runtime)")
            print("  - topology variables match dsr1_config.yaml")
            print("  - decode has real CUDA graphs covering max_batch_size; prefill is eager")
            print("  - every task script is a list of STRINGS and parses as bash")
            print("  - launch_test.sh exists, is executable, parses as bash, and matches")
            print("    a fresh render; so does the recipe itself")
            if args.longer_runs:
                longer = [i["point"] for i in infos if "drop_n_samples" in i]
                print("  - --longer-runs: CLIENT_CONFIG of " + ", ".join(f"C{p}" for p in longer)
                      + " has the third documented edit (n_samples_to_issue)")
            if variants:
                print("And for each reduced-node variant, additionally:")
                print("  - server-ctx is the drop's; server-gen differs from the drop only in")
                print(f"    {', '.join(RESHAPE_KEYS)}")
                print("  - TP/EP == GPUS_PER_GEN_RANK; decode slots >= target_concurrency")
                print("  - workflow.name and gke-topology.txt mark it as a variant")
        if checked_diags:
            print("For each DIAGNOSTIC (diag/; NOT a submission topology):")
            print("  - every per-point --check check, against its own topology; the recipe")
            print("    and launch_test.sh equal a fresh render")
            print("  - parsed, it differs from a fresh render of its 18-node point ONLY in")
            print("    NUM_GEN_SERVERS, GPUS_PER_GEN_RANK, server-gen " + " / ".join(DIAG_GEN_KEYS))
            print("    (EOF_GEN + EOF_SNAP_GEN), workflow.name and the gke-topology.txt DIAG")
            print("    line; line by line, every changed non-comment line carries an inline")
            print("    \"# DIAG <name>:\" marker and nothing else was added or dropped")
            print("  - DIAGNOSTIC ONLY banner present; decode GPUs, server nodes and")
            print("    max_batch_size unchanged; TP/EP == GPUS_PER_GEN_RANK; decode slots")
            print("    >= target_concurrency; launch_test.sh differs from the point's only")
            print("    in RECIPE + the DIAGNOSTIC warning")
        return 0

    print(f"{'point':>13} {'topology':<24} {'GPUs':>5} {'nodes':>6} "
          f"{'C/replica':>10} {'samples':>8}  placement")
    for info in infos + variants:
        d = recipe_dir(info, args.dest)
        d.mkdir(parents=True, exist_ok=True)
        (d / recipe_name(info)).write_text(render_recipe(info))
        ls = d / "launch_test.sh"
        ls.write_text(render_launch(info))
        ls.chmod(0o755)
        topo = f"{info['ctx_n']}xDEP{info['ctx_tp']} + {info['gen_n']}xDEP{info['gen_tp']}"
        name = label_of(info).removeprefix("point_")
        print(
            f"{name:>13} {topo:<24} {info['total_gpus']:>5} "
            f"{info['server_nodes']:>6} {info['concurrency'] // info['gen_n']:>10} "
            f"{info['n_samples']:>8}  {placement_comment(info)}"
        )
    print(f"\nwrote {len(infos)} point(s) and {len(variants)} reduced-node "
          f"variant(s) under {args.dest}")
    if variants:
        print()
        print_variant_layout(variants)
    print("\nrun `./generate_point_recipes.py --check` to verify against the drop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
