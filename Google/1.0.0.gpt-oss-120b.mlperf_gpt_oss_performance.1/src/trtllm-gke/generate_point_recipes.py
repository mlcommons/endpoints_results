#!/usr/bin/env python3
"""Generate the eleven GPT-OSS-120B GKE sflow recipes (MLPerf Endpoints
v1.0.3 drop) from one base recipe, for the Slurm-faithful 18-node topology.

    ./generate_point_recipes.py            # (re)generate all 18-node points
    ./generate_point_recipes.py --layout   # print the node/GPU layout tables
    ./generate_point_recipes.py --check    # diff every generated recipe against
                                           # its Slurm source and fail on drift
    ./generate_point_recipes.py --variants 16   # ALSO emit 16nodes/ variants
    ./generate_point_recipes.py --longer-runs   # the RERUN recipes: more samples
                                                # on the LONGER_RUNS points
    ./generate_point_recipes.py --check --longer-runs   # check those instead

v1.0.3 port (2026-10-14)
------------------------
Copied from the v1.0.2 generator; the default drop is now
nv-mlpinf-partner-partner-endpoints-v1.0.3.  What v1.0.3 changed for GPT-OSS:
  * point_17, point_35, point_70, point_140: client.yaml raises
    settings.timeouts.run_timeout_s from 9000.0 to 16200.0 (carried
    automatically).  Because C17's longer run (11,877 s) now uses
    11,877 / 16,200 = 73.3 % < 85 % of NVIDIA's watchdog natively, C17 no
    longer needs a run_timeout_s override in LONGER_RUNS_OVERRIDES (which is
    now empty).
  * submission/ templates and system_desc/system_power metadata added upstream.
  * Server YAMLs, server-env.yaml, gptoss_config.yaml topologies, sample counts
    and images are byte-identical to v1.0.2 across all eleven points.

v1.0.2 port (2026-10-07)
------------------------
Copied from the v1.0.1 generator; the default drop is now
nv-mlpinf-partner-partner-endpoints-v1.0.2.  What v1.0.2 changed for GPT-OSS:
  * point_2048 only: server-env.yaml adds TRTLLM_ENABLE_PDL: '0', and
    stream_interval goes 100 -> 30 in server-ctx AND server-gen (carried
    automatically: the server YAMLs are embedded byte-for-byte).
  * every gptoss_config.yaml: the default MODEL_PATH moves from
    ${SCRATCH_DIR}/models/gpt-oss/gpt-oss-120b to ${SCRATCH_DIR}/models/
    gpt-oss-120b.  Not ported: MODEL_PATH is the node-local SSD on GKE.
  * _shared/: an optional SITE_ENV_YAML overlay (default '') is merged after
    server-env.yaml by run_with_env.py.  read_shared() refuses a non-empty
    default, because the port would then miss env keys.
  * README: checkpoint pinned to openai/gpt-oss-120b@b5c939de (what
    stage-assets.yaml already stages).  Images, client.yaml,
    endpoint_env.yaml, topology and sample counts are unchanged.
The generator used to hardcode the server env in recipe_base.yaml and never
read server-env.yaml, so C2048's new TRTLLM_ENABLE_PDL would have been
dropped silently.  It now reads every point's server-env.yaml and decides
each key through SERVER_ENV_KEYS (read that comment): keys ported per point
are injected into that point's prefill AND decode env (export + mpirun -x);
any key, or value, the table does not name stops the generator.  --check
compares the rendered env with server-env.yaml for every allowlisted key.

v1.0.1 port (2026-09-30)
------------------------
Copied from the v1.0.0 generator.  The drop moved
gpt_oss_120b/configs/..._TRT -> gpt-oss-120b/configs/..._TRTLLM; _shared/ is
byte-identical to v1.0.0.  What v1.0.1 changed, all carried verbatim:
  * n_samples_to_issue raised on every point except C7168 (EXPECTED below).
  * point_35 topology: 2 x TP1 ctx + 35 x TP2 gen  ->  4 x TP1 ctx + 17 x TP4
    gen (server-gen TP/EP 2 -> 4), i.e. the same shape as C17/C272/C544.
  * point_2048: stream_interval 30 -> 100 in server-ctx AND server-gen.
  * point_7168: the v1.0.0 anomalies are fixed upstream -- model_params.name
    is now ${MODEL_PATH} (was openai/gpt-oss-120b) and report_dir is
    ${SFLOW_TASK_OUTPUT_DIR} (was the relative `results`).
Default output is the 18nodes/ tree only (the pool has 18 healthy nodes);
reduced-node variants are opt-in with --variants.

Longer runs (opt-in: --longer-runs, 2026-10-06)
-----------------------------------------------
The reruns raise n_samples_to_issue on the LONGER_RUNS points to about 4
dataset passes and the region's minimum duration.  Without the flag every
output, and what --check accepts, is exactly as before.  Read the LONGER_RUNS
comment before adding a point: the client's run_timeout_s is a whole-run
watchdog.  C17's longer run outgrows NVIDIA's 9,000 s watchdog, so for C17
alone --longer-runs also raises settings.timeouts.run_timeout_s to 14,400 s
(LONGER_RUNS_OVERRIDES, decision 15 (a)).  That table may hold only that one
key for that one point; the generator refuses anything else in it.

Design
------
Two inputs, one output:

  recipe_base.yaml  -- the GKE layer (fabric, MPI, placement, client fixes).
                       Hand maintained.  A fix here propagates to all eleven
                       points, which is the whole point of having a generator.

  the NVIDIA Slurm drop
    .../NVIDIA/src/gpt-oss-120b/configs/GB300-NVL72_GB300-288GB_aarch64x72_TRTLLM/
        point_<C>/{gptoss_config,server-ctx,server-gen,client}.yaml
                    -- the workload layer.  NOTHING here is retyped: the
                       topology scalars are read out of gptoss_config.yaml and
                       the three YAML files are embedded byte-for-byte.
        point_<C>/server-env.yaml
                    -- the server env.  Every key is decided by
                       SERVER_ENV_KEYS; the "port" keys are injected per
                       point, the rest are asserted or dropped (see there).
    .../configs/_shared/slurm_env.yaml
                    -- read only to assert the images and that SITE_ENV_YAML
                       (the v1.0.2 env overlay) is empty by default.

`--check` re-parses each generated recipe with a YAML parser, pulls the three
embedded configs back out, and compares them to the Slurm originals.  The only
tolerated differences are the two documented client rewrites (plus, with
--longer-runs, n_samples_to_issue on the LONGER_RUNS points and C17's
run_timeout_s from LONGER_RUNS_OVERRIDES); anything else is reported as drift
and exits non-zero.  It also rebuilds the prefill and decode env (top-level
exports plus the mpirun -x list) and requires it to equal server-env.yaml for
every allowlisted key (verify_env()).

Why a separate recipe_base.yaml instead of DSR1's "use point_256 as the
template" trick: GPT-OSS has no blessed point to be the template, every point's
server AND client YAML differs, and making one of the eleven secretly
authoritative would mean `--check` could not check it.  A base file with
explicit __TOKENS__ keeps all eleven outputs equal citizens.

Reduced-node variants (opt-in: --variants 16)
---------------------------------------------
Every Slurm point is 72 GPUs = 18 nodes (18nodes/point_<C>/).
16nodes/point_<C>/ runs the SAME point (same concurrency, same samples) on 16 nodes: server-ctx / server-gen /
client stay byte-identical to the drop and only NUM_CTX_SERVERS /
NUM_GEN_SERVERS change.  rank_splits() picks the split (read its docstring);
EXPECTED_SPLIT asserts the result, so a change in the drop or in the rule
cannot silently move a variant.  The variants are NOT submission topologies.
"""

from __future__ import annotations

import argparse
import difflib
import pathlib
import re
import shlex
import sys
import textwrap
from fractions import Fraction

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml "
             "(or run with the sflow venv's python3)")

HERE = pathlib.Path(__file__).resolve().parent
BASE_RECIPE = HERE / "recipe_base.yaml"

DROP_VERSION = "v1.0.3"
DROP_NAME = f"nv-mlpinf-partner-partner-endpoints-{DROP_VERSION}"

# gpt_oss_120b -> v1.0.3 -> kubernetes -> self_contained -> examples -> nv-sflow -> <root>
DEFAULT_SLURM_ROOT = (
    HERE.parents[5]
    / "code-drop-nvidia"  # all NVIDIA code drops live here, one folder per drop
    / DROP_NAME
    / "NVIDIA" / "src" / "gpt-oss-120b" / "configs"
    / "GB300-NVL72_GB300-288GB_aarch64x72_TRTLLM"
)

GPUS_PER_NODE = 4
POINTS = [17, 35, 70, 140, 272, 544, 1024, 2048, 3840, 7168, 22528]

# Conversion spec section 1, GPT-OSS-120B table.  This is an ASSERTION, not a
# data source -- every value below is independently read from the Slurm YAML and
# compared.  A mismatch means the drop changed and the port must be revisited.
EXPECTED = {
    #       ctxN ctxTP genN genTP  fe   conc  n_samples  max_new
    #                                               v1.0.0 value, for reference
    17:    (4,   1,    17,  4,     1,   17,     6396,    10240),  # 4490
    35:    (4,   1,    17,  4,     1,   35,    19188,    10240),  # 2x TP1 + 35x TP2, 8785
    70:    (2,   1,    35,  2,     1,   70,    19188,    10240),  # 15816
    140:   (2,   1,    35,  2,     1,   140,   31980,    10240),  # 29915
    272:   (4,   1,    17,  4,     1,   272,   76752,    10240),  # 52288
    544:   (4,   1,    17,  4,     1,   544,   95940,    10240),  # 91212
    1024:  (8,   1,    16,  4,     16,  1024, 204672,    10240),  # 141169
    2048:  (8,   1,     8,  8,     8,   2048, 319800,    10240),  # 216535
    3840:  (12,  1,    15,  4,     15,  3840, 460512,    10240),  # 321134
    7168:  (16,  1,     7,  8,     7,   7168, 953004,    10240),  # unchanged
    22528: (28,  1,    11,  4,     11, 22528, 652392,    10240),  # 647616
}
# (v1.0.2 changed none of the values above.)

# --------------------------------------------------------------------------
# server-env.yaml: one decision for every key any point sets
# --------------------------------------------------------------------------
# On Slurm the shared template evals `run_with_env.py <env.yaml>
# [SITE_ENV_YAML]` in prefill_server (TRTLLM_ENV_YAML_CTX) AND decode_server
# (TRTLLM_ENV_YAML_GEN); every gptoss_config.yaml points both at the point's
# own server-env.yaml (read_point() asserts that).  SLURM_TO_GKE.md 5.1: port
# the model/runtime keys a point actually sets, discard UCX_* / fabric keys.
# This table makes that rule explicit, one row per key (v1.0.2: 10 keys over
# the eleven points):
#
#   "port"     model/runtime key that recipe_base.yaml does not set.  Injected
#              PER POINT -- only into the recipes of points whose
#              server-env.yaml sets it -- into prefill AND decode, as an
#              `export` line plus `mpirun -x KEY=VALUE`.  The -x is not
#              optional: MPI forwards only -x variables, so without it the
#              ranks of a TP8 decode server on its second node never see the
#              key.  `values` is the allowlist of values the generator emits.
#   "base"     recipe_base.yaml already exports this key with exactly
#              values[0] to prefill and decode at every point (export plus
#              mpirun -x).  The drop must say the same; if NVIDIA changes it,
#              the generator stops instead of keeping the hardcoded value.
#   "gke-ucx"  Slurm fabric key, NOT ported: the GKE UCX layer in
#              recipe_base.yaml sets `gke` instead (SLURM_TO_GKE.md 5.6).
#              The drop's value must still be one of `values`, so a change in
#              the drop is seen and reviewed.
#
# FAIL CLOSED: a key that is not in this table, a value outside `values`, a
# non-string value or a duplicate key stops the generator before anything is
# written.  Every allowed value is a plain token (SAFE_ENV_VALUE): no `$`
# (run_with_env.py would expand it from the Slurm task env), no quoting, no
# shell syntax.  validate_server_env_table() checks the table itself.  Adding
# a row is a decision, not an edit: read SLURM_TO_GKE.md 5.1 / 5.6 first.
SERVER_ENV_PORT = "port"
SERVER_ENV_BASE = "base"
SERVER_ENV_GKE_UCX = "gke-ucx"
SERVER_ENV_KEYS = {
    # Programmatic Dependent Launch, a TRT-LLM runtime switch.  v1.0.2 sets
    # '0' (PDL OFF) at C2048 only, together with stream_interval 100 -> 30 in
    # both server YAMLs -- the point whose GKE run once died with a
    # device-side assert in sampling.  NB '0' is the OPPOSITE of the DSR1 GKE
    # recipes' TRTLLM_ENABLE_PDL=1, which is DSR1 tuning and never copied.
    "TRTLLM_ENABLE_PDL": dict(decision=SERVER_ENV_PORT, values=("0", "1")),
    # MoE / GC runtime switches: '1' at every point in v1.0.1 and v1.0.2.
    "TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER": dict(
        decision=SERVER_ENV_BASE, values=("1",)),
    "TRTLLM_SERVER_DISABLE_GC": dict(decision=SERVER_ENV_BASE, values=("1",)),
    "TRTLLM_WORKER_DISABLE_GC": dict(decision=SERVER_ENV_BASE, values=("1",)),
    # HF caches: the same container-local /tmp paths as on Slurm.
    "HF_HOME": dict(decision=SERVER_ENV_BASE, values=("/tmp/hf_cache_gptoss",)),
    "HF_HUB_CACHE": dict(decision=SERVER_ENV_BASE,
                         values=("/tmp/hf_cache_gptoss/hub",)),
    "TRANSFORMERS_CACHE": dict(decision=SERVER_ENV_BASE,
                               values=("/tmp/hf_cache_gptoss",)),
    "XDG_CACHE_HOME": dict(decision=SERVER_ENV_BASE,
                           values=("/tmp/hf_cache_gptoss",)),
    # 'n' is right on RoCE, where the NICs carry KV.  On GKE it sends every
    # inter-node UCX lane over tcp/eth0 (SLURM_TO_GKE.md F2): GKE sets 'y'.
    "UCX_CUDA_IPC_ENABLE_MNNVL": dict(decision=SERVER_ENV_GKE_UCX,
                                      values=("n",), gke="y"),
    # Same value as the GKE UCX layer -- but governed there, not here.
    "UCX_RNDV_SCHEME": dict(decision=SERVER_ENV_GKE_UCX,
                            values=("put_zcopy",), gke="put_zcopy"),
}
SERVER_ENV_RULE_FIELDS = frozenset({"decision", "values", "gke"})
ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
SAFE_ENV_VALUE = re.compile(r"[A-Za-z0-9_./:-]+")

# The "port" keys each point's server-env.yaml sets.  Like EXPECTED, an
# ASSERTION, not a data source: the injected values are read from the drop,
# and a point that gains or loses a ported key stops the generator so the
# change gets reviewed.  v1.0.1 set none; v1.0.2 sets one, at C2048.
EXPECTED_PORTED_ENV = {
    2048: {"TRTLLM_ENABLE_PDL": "0"},
}

# Env the DSR1 GKE recipes export and GPT-OSS's drop does not (or, for PDL,
# only at C2048 and with the opposite value).  Used for the per-point note in
# the generated prefill_server script only (server_env_note()).
DSR1_ONLY_ENV = {
    "TRTLLM_ENABLE_PDL": "1",
    "TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED": "1",
    "NCCL_GRAPH_MIXING_SUPPORT": "0",
    "TLLM_NUMA_AWARE_WORKER_AFFINITY": "1",
}

# Recipe variable -> the drop variable naming the same image (Slurm adds a
# docker:// prefix to CONTAINER_IMAGE).  CONTAINER_IMAGE is per point
# (gptoss_config.yaml); the client and LCB images are in _shared/slurm_env.yaml.
# The rendered recipe's default must equal the drop's, so a bumped image in a
# new drop stops the generator instead of being missed.
DROP_IMAGE_VARS = {
    "TRTLLM_IMAGE": "CONTAINER_IMAGE",
    "CLIENT_IMAGE": "ENDPOINT_CONTAINER_IMAGE",
    "LCB_IMAGE": "LCB_CONTAINER_IMAGE",
}

# Reduced-node variants written next to the Slurm points (18nodes/point_<C>/),
# as <N>nodes/point_<C>/.  OFF by default since v1.0.1 (the pool has 18
# healthy nodes); `--variants 16` turns them on.
VARIANT_NODES = (16,)  # the node counts `--variants` accepts

# The prefill/decode split rank_splits() picks for each variant.  Like EXPECTED,
# an ASSERTION, not a data source: the generator recomputes the split from the
# drop and refuses to write anything if it no longer matches this table.
EXPECTED_SPLIT = {
    #         ctxN genN      Slurm 72-GPU split, for reference
    16: {
        17:    (4,  15),   # 4 x TP1 + 17 x TP4
        35:    (4,  15),   # 4 x TP1 + 17 x TP4  (v1.0.1; v1.0.0 was 2 x TP1 + 35 x TP2 -> 2 + 31)
        70:    (2,  31),   # 2 x TP1 + 35 x TP2
        140:   (2,  31),   # 2 x TP1 + 35 x TP2
        272:   (4,  15),   # 4 x TP1 + 17 x TP4
        544:   (4,  15),   # 4 x TP1 + 17 x TP4
        1024:  (8,  14),   # 8 x TP1 + 16 x TP4
        2048:  (8,   7),   # 8 x TP1 +  8 x TP8
        3840:  (12, 13),   # 12 x TP1 + 15 x TP4
        7168:  (16,  6),   # 16 x TP1 +  7 x TP8
        22528: (24, 10),   # 28 x TP1 + 11 x TP4  <- the one point that gives up prefill
    },
}

# Longer runs for the reruns.  OFF by default; `--longer-runs` turns them on.
# --------------------------------------------------------------------------
# The MLPerf Endpoints First-Time Submitter's Guide wants about 4 dataset passes
# in the steady-state window, plus the region's minimum duration (checker
# <ip>: 600 s low_latency, 1,200 s otherwise).  NVIDIA's n_samples_to_issue
# leaves the points below short on one or both, so the reruns raise it to a
# whole number of passes of the 6,396-prompt dataset that reaches both with a
# 15 % margin at the throughput measured in the Oct 1 runs.  Nothing else in
# client.yaml changes, except C17's run_timeout_s (below).  Approved 2026-10-06
# (Y5 decision 13) unless NVIDIA or MLCommons say the drop's lengths are fine.
#
# The pinned client's settings.timeouts.run_timeout_s is a WHOLE-RUN watchdog
# (endpoints e9f427c, config/schema.py `Timeouts`): it is armed before setup and
# covers the performance phase AND all three accuracy phases.  So a longer
# performance phase eats its headroom.  Each entry carries two projections from
# the measured run (scratch run_length_plan.py / run_length_watchdog.py):
#   perf_s   -- predicted performance-phase seconds at the new n
#   client_s -- measured client run seconds + the added performance seconds
# and main() refuses to lengthen a point whose perf_s reaches 85 % of the drop's
# max_issue_duration_ms or whose client_s reaches 85 % of its run_timeout_s.
#
# In v1.0.1/v1.0.2, C17's longer run (11,877 s) outgrew NVIDIA's 9,000 s
# watchdog, so LONGER_RUNS_OVERRIDES raised C17's run_timeout_s to 14,400 s.
# In v1.0.3, NVIDIA raised settings.timeouts.run_timeout_s on C17, C35, C70,
# C140 from 9,000.0 s to 16,200.0 s, so C17's longer run now fits NVIDIA's
# watchdog natively (11,877 / 16,200 = 73.3 % < 85 %) and LONGER_RUNS_OVERRIDES
# is empty.
LONGER_RUNS = {
    #      n_samples_to_issue  perf_s  client_s   drop n (passes) -> new (passes)
    17:    (25584,             4727,   11877),    # 6,396 (1) -> 4, see below
    35:    (38376,             3702,   7939),     # 19,188 (3) -> 6
    70:    (31980,             1695,   4333),     # 19,188 (3) -> 5
    140:   (51168,             1531,   3456),     # 31,980 (5) -> 8
    544:   (127920,            1392,   2681),     # 95,940 (15) -> 20
    22528: (1535040,           2073,   5575),     # 652,392 (102) -> 240
}
# C17's numbers (scratch run_length_plan.json / run_length_watchdog.json; Oct 1
# run: 6,396 samples in 1,181.7 s, whole client run 8,332 s):
#   perf_s   = 1,181.7 s x 25,584 / 6,396 = 4 x 1,181.7 s     =  4,727 s
#   client_s = 8,332 s + (4,727 - 1,182) s = 8,332 + 3,545 s  = 11,877 s
#   guard    : 11,877 / 16,200 = 73.3 % < 85 %   (v1.0.3 run_timeout_s = 16,200 s)
#              4,727 / 17,400 (max_issue_duration_ms) = 27 % < 85 %
DATASET_PROMPTS = 6396  # perf_eval_ref.parquet rows: one dataset pass
LONGER_RUNS_HEADROOM = 0.85

# What --longer-runs changes in client.yaml besides n_samples_to_issue, per
# point.  Deliberately narrow: LONGER_RUNS_OVERRIDE_ALLOWED lists every
# (point, key) this table may hold, and apply_longer_runs() refuses anything
# else before a recipe is rendered -- another point or key, a point outside
# LONGER_RUNS, or a value that is not whole seconds above NVIDIA's (and at
# most twice it, so a typo cannot quietly switch the watchdog off).  Widening
# the allow-list is a decision, not an edit.
LONGER_RUNS_OVERRIDES: dict[int, dict[str, int]] = {}
LONGER_RUNS_OVERRIDE_ALLOWED = frozenset({
    (17, "settings.timeouts.run_timeout_s"),
})


# --------------------------------------------------------------------------
# Node / GPU placement
# --------------------------------------------------------------------------
def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def layout(ctx_n: int, ctx_tp: int, gen_n: int, gen_tp: int,
           gpus_per_node: int = GPUS_PER_NODE):
    """Return (ctx_nodes, gen_nodes, total_nodes, total_gpus) for a point.

    This is the packing-aware formula the recipe's CTX_URLS / GEN_URLS Jinja
    implements, and it is derived directly from sflow/app/resource_planner.py:

      * gpus.count <= gpus_per_node  -> the planner walks allocation nodes in
        order and takes the first contiguous free run, so sub-node servers
        TIGHTLY BACKFILL, including across the ctx -> gen boundary.
      * gpus.count >  gpus_per_node  -> whole free nodes only (the planner
        groups candidates by free-GPU cursor and takes from the lowest group).

    The old gke_recipe_template.yaml's unconditional
    `gen_base = ceil(ctx_gpus / gpus_per_node) * gpus_per_node` is correct for
    every DeepSeek-R1 point and WRONG for GPT-OSS point_35/70/140.
    """
    ctx_gpus = ctx_n * ctx_tp
    gen_gpus = gen_n * gen_tp

    ctx_nodes_used = sorted({(i * ctx_tp) // gpus_per_node for i in range(ctx_n)})

    if gen_tp <= gpus_per_node:
        gen_base = ctx_gpus
    else:
        gen_base = ceil_div(ctx_gpus, gpus_per_node) * gpus_per_node

    gen_nodes_used: set[int] = set()
    for i in range(gen_n):
        start = gen_base + i * gen_tp
        for g in range(start, start + gen_tp):
            gen_nodes_used.add(g // gpus_per_node)

    total_gpus = ctx_gpus + gen_gpus
    total_nodes = ceil_div(total_gpus, gpus_per_node)
    return (
        ctx_nodes_used,
        sorted(gen_nodes_used),
        gen_base,
        total_nodes,
        total_gpus,
    )


def ctx_node(i: int, ctx_tp: int) -> int:
    return (i * ctx_tp) // GPUS_PER_NODE


def gen_node(i: int, gen_base: int, gen_tp: int) -> int:
    return (gen_base + i * gen_tp) // GPUS_PER_NODE


# --------------------------------------------------------------------------
# Reduced-node variants: choosing the prefill/decode split
# --------------------------------------------------------------------------
def decode_slots_per_server(info: dict) -> int:
    """Sequences one decode server can hold at once.

    max_batch_size is PER RANK when enable_attention_dp is on (each attention-DP
    rank schedules its own batch), so a TP4 ADP server holds 4 x max_batch_size.
    Every point from C272 up is sized so that NUM_GEN_SERVERS x this is EXACTLY
    target_concurrency -- which is why dropping a decode server at those points
    leaves requests queued for a slot.
    """
    gen = yaml.safe_load(info["gen_text"])
    ranks = gen["tensor_parallel_size"] if gen.get("enable_attention_dp") else 1
    return gen["max_batch_size"] * ranks


def split_score(info: dict, ctx_n: int, gen_n: int) -> Fraction:
    """Fraction of the Slurm point's capacity a split keeps (see rank_splits)."""
    return min(Fraction(ctx_n, info["ctx_n"]), Fraction(gen_n, info["gen_n"]))


def limiting_pool(info: dict, ctx_n: int, gen_n: int) -> str:
    ctx_r, gen_r = Fraction(ctx_n, info["ctx_n"]), Fraction(gen_n, info["gen_n"])
    if ctx_r < gen_r:
        return "prefill-limited"
    if gen_r < ctx_r:
        return "decode-limited"
    return "balanced"


def rank_splits(info: dict, nodes: int) -> list[tuple[Fraction, int, int]]:
    """Every (score, ctx_n, gen_n) that fits on `nodes` nodes, best first.

    The server SHAPES are fixed by the drop -- TP, max_batch_size, CUDA-graph
    lists -- so on a smaller pool the only free choice is how many prefill and
    decode servers to run.  NVIDIA tuned each point's 72-GPU split; that split
    is taken as balanced, i.e. the prefill pool keeps up with the decode pool
    without a large surplus.  A re-split then keeps about

        min(ctx_n' / ctx_n, gen_n' / gen_n)

    of the point's throughput: whichever pool shrank more becomes the
    bottleneck.  Highest score wins; ties go to more decode servers (they set
    the throughput at every point from C272 up, where decode slots are sized to
    exactly target_concurrency), then to more prefill servers (lower TTFT).

    Why prefill is not traded away lightly: the perf dataset averages 5,011
    input vs ~1,216 output tokens, and the 17-node point_1024 run already had
    each of its 8 TP1 prefill servers ingesting ~96k tokens/s.  Halving that
    pool to buy one decode server (4 + 15) would ask ~190k tokens/s of each --
    which is what that split's 0.500 score encodes.

    A candidate must also pack the way the recipe's Jinja URL formula assumes:
    no sub-node gen server straddling a node boundary (ctx GPUs a multiple of
    the gen TP), TP>4 gen servers on whole nodes, and no empty node.
    """
    ctx_tp, gen_tp = info["ctx_tp"], info["gen_tp"]
    if GPUS_PER_NODE % ctx_tp:
        raise SystemExit(f"ctx TP{ctx_tp} does not divide a node; rank_splits "
                         f"assumes ctx servers pack node by node")
    budget = nodes * GPUS_PER_NODE
    ranked = []
    for ctx_n in range(1, budget // ctx_tp + 1):
        ctx_gpus = ctx_n * ctx_tp
        if gen_tp <= GPUS_PER_NODE:
            if GPUS_PER_NODE % gen_tp or ctx_gpus % gen_tp:
                continue
        elif gen_tp % GPUS_PER_NODE:
            continue
        for gen_n in range(1, (budget - ctx_gpus) // gen_tp + 1):
            ctx_nodes, gen_nodes, _, total_nodes, _ = layout(
                ctx_n, ctx_tp, gen_n, gen_tp)
            if (total_nodes > nodes
                    or set(ctx_nodes) | set(gen_nodes) != set(range(total_nodes))):
                continue
            ranked.append((split_score(info, ctx_n, gen_n), gen_n, ctx_n))
    ranked.sort(reverse=True)
    return [(score, ctx_n, gen_n) for score, gen_n, ctx_n in ranked]


def variant_splits(point: int, info: dict, nodes: int) -> list[tuple[Fraction, int, int]]:
    """rank_splits() plus the EXPECTED_SPLIT assertion and sanity limits."""
    ranked = rank_splits(info, nodes)
    if not ranked:
        raise SystemExit(f"point_{point}: no prefill/decode split fits {nodes} nodes")
    _, ctx_n, gen_n = ranked[0]
    want = EXPECTED_SPLIT[nodes][point]
    if (ctx_n, gen_n) != want:
        raise SystemExit(
            f"point_{point}: best {nodes}-node split is now {ctx_n} ctx + {gen_n} gen, "
            f"EXPECTED_SPLIT says {want[0]} + {want[1]}.  The drop or the rule "
            f"changed -- re-read rank_splits() before updating the table.")
    if info["frontends"] > nodes:
        raise SystemExit(f"point_{point}: {info['frontends']} frontends need "
                         f"{info['frontends']} nodes (one each), only {nodes}")
    return ranked


def variant_info(info: dict, ctx_n: int, gen_n: int) -> dict:
    """The Slurm point with only the two server counts replaced."""
    v = dict(info)
    v.update(ctx_n=ctx_n, gen_n=gen_n)
    return v


# --------------------------------------------------------------------------
# Reading the Slurm drop
# --------------------------------------------------------------------------
class _UniqueKeyLoader(yaml.SafeLoader):
    """yaml.SafeLoader that refuses duplicate and non-string mapping keys.

    Plain PyYAML keeps the LAST of two equal keys without a word; in a
    server-env.yaml that would let an unreviewed second value win silently.
    """

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise yaml.constructor.ConstructorError(
                    None, None, f"non-string key {key!r}", key_node.start_mark)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def yaml_problem(exc: yaml.YAMLError) -> str:
    """A YAML error as `problem at line L, column C`.  str(exc) would quote the
    offending line of the file, which in an env file may be a credential."""
    mark = getattr(exc, "problem_mark", None)
    problem = getattr(exc, "problem", None) or type(exc).__name__
    return problem + (f" at line {mark.line + 1}, column {mark.column + 1}"
                      if mark is not None else "")


def validate_server_env_table() -> None:
    """Refuse a SERVER_ENV_KEYS / EXPECTED_PORTED_ENV row the port cannot honour."""
    decisions = (SERVER_ENV_PORT, SERVER_ENV_BASE, SERVER_ENV_GKE_UCX)
    for key, rule in SERVER_ENV_KEYS.items():
        where = f"SERVER_ENV_KEYS[{key!r}]"
        if not isinstance(key, str) or not ENV_NAME.fullmatch(key):
            raise SystemExit(f"{where}: not an environment variable name")
        if not isinstance(rule, dict) or set(rule) - SERVER_ENV_RULE_FIELDS:
            raise SystemExit(f"{where}: a row is a dict with fields from "
                             f"{sorted(SERVER_ENV_RULE_FIELDS)}")
        decision, values = rule.get("decision"), rule.get("values")
        if decision not in decisions:
            raise SystemExit(f"{where}: decision {decision!r} is not one of "
                             f"{list(decisions)}")
        if (not isinstance(values, tuple) or not values
                or not all(isinstance(x, str) and SAFE_ENV_VALUE.fullmatch(x)
                           for x in values)):
            raise SystemExit(f"{where}: values must be a non-empty tuple of plain "
                             f"tokens matching {SAFE_ENV_VALUE.pattern}")
        if decision == SERVER_ENV_BASE and len(values) != 1:
            raise SystemExit(f"{where}: a 'base' row has exactly one value, the "
                             f"one recipe_base.yaml exports")
        if decision == SERVER_ENV_GKE_UCX:
            gke = rule.get("gke")
            if (not key.startswith("UCX_") or not isinstance(gke, str)
                    or not SAFE_ENV_VALUE.fullmatch(gke)):
                raise SystemExit(f"{where}: a 'gke-ucx' row is a UCX_* key with a "
                                 f"plain-token 'gke' value (the GKE UCX layer's)")
        elif "gke" in rule:
            raise SystemExit(f"{where}: only a 'gke-ucx' row has a 'gke' value")
        if key.startswith("UCX_") and decision != SERVER_ENV_GKE_UCX:
            raise SystemExit(f"{where}: UCX_* is Slurm fabric config "
                             f"(SLURM_TO_GKE.md 5.6); it can only be 'gke-ucx'")
    for point, env in EXPECTED_PORTED_ENV.items():
        if point not in POINTS or not isinstance(env, dict) or not env:
            raise SystemExit(f"EXPECTED_PORTED_ENV[{point!r}]: not a point with a "
                             f"non-empty {{key: value}} dict")
        for key, value in env.items():
            rule = SERVER_ENV_KEYS.get(key)
            if (rule is None or rule["decision"] != SERVER_ENV_PORT
                    or value not in rule["values"]):
                raise SystemExit(f"EXPECTED_PORTED_ENV[{point}][{key!r}]: not a "
                                 f"'port' key of SERVER_ENV_KEYS with an allowed "
                                 f"value")


def parse_server_env(point: int, text: str) -> dict[str, str]:
    """point_<C>/server-env.yaml as {key: value}, in file order.

    FAILS CLOSED: every key must be a row of SERVER_ENV_KEYS and every value
    one of that row's `values`; anything else stops the generator before a
    file is written.  Values are strings, as run_with_env.py exports them
    (an integer is accepted as its decimal string; bool, float and null are
    refused).
    """
    where = f"point_{point}/server-env.yaml"
    try:
        doc = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise SystemExit(f"{where}: not a plain KEY: value mapping "
                         f"({yaml_problem(exc)})") from None
    if not isinstance(doc, dict) or not doc:
        raise SystemExit(f"{where}: expected a non-empty flat KEY: value mapping, "
                         f"got {type(doc).__name__}")
    env: dict[str, str] = {}
    for key, value in doc.items():
        if not ENV_NAME.fullmatch(key):
            raise SystemExit(f"{where}: {key!r} is not an environment variable name")
        rule = SERVER_ENV_KEYS.get(key)
        if rule is None:
            # The value is deliberately not printed: an unknown key may be a
            # credential.
            raise SystemExit(
                f"{where}: {key} is not in SERVER_ENV_KEYS.  Every server-env key "
                f"needs a reviewed decision (port / base / gke-ucx; SLURM_TO_GKE.md "
                f"5.1 and 5.6) before the generator writes anything.")
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise SystemExit(f"{where}: {key} is a {type(value).__name__}; "
                             f"expected a string")
        value = str(value)
        if value not in rule["values"]:
            raise SystemExit(
                f"{where}: {key}={value!r} is not an allowed value "
                f"{list(rule['values'])} (SERVER_ENV_KEYS, decision "
                f"{rule['decision']!r}).  The drop changed -- review the value "
                f"before allowing it.")
        env[key] = value
    missing = [k for k, r in SERVER_ENV_KEYS.items()
               if r["decision"] == SERVER_ENV_BASE and k not in env]
    if missing:
        raise SystemExit(f"{where}: no longer sets {missing}, which "
                         f"recipe_base.yaml exports at every point ('base' in "
                         f"SERVER_ENV_KEYS) -- review before regenerating")
    return env


def read_shared(slurm_root: pathlib.Path) -> dict:
    """The drop's configs/_shared/slurm_env.yaml: the client and LCB images, and
    the v1.0.2 SITE_ENV_YAML overlay, whose default must be empty.

    run_with_env.py merges SITE_ENV_YAML AFTER server-env.yaml, so a non-empty
    default would add or override server env keys that never pass through
    SERVER_ENV_KEYS.  A drop without the variable (v1.0.1) is fine.
    """
    path = slurm_root.parent / "_shared" / "slurm_env.yaml"
    if not path.is_file():
        raise SystemExit(f"{path} not found: --slurm-root must be the drop's "
                         f"configs/<system>/ directory, next to configs/_shared/")
    try:
        variables = yaml.safe_load(path.read_text())["variables"]
        images = {var: str(variables[drop_var]["value"])
                  for var, drop_var in DROP_IMAGE_VARS.items()
                  if drop_var != "CONTAINER_IMAGE"}
    except (yaml.YAMLError, KeyError, TypeError) as exc:
        detail = yaml_problem(exc) if isinstance(exc, yaml.YAMLError) else repr(exc)
        raise SystemExit(f"{path}: cannot read the image variables "
                         f"({detail})") from None
    site = variables.get("SITE_ENV_YAML", {"value": ""})
    site = site.get("value") if isinstance(site, dict) else site
    if site not in ("", None):
        raise SystemExit(
            f"{path}: SITE_ENV_YAML defaults to {site!r}.  On Slurm that file is "
            f"merged after server-env.yaml (run_with_env.py), so its keys would "
            f"reach the servers without passing SERVER_ENV_KEYS; port it first.")
    return dict(path=path, images=images)


def read_point(slurm_root: pathlib.Path, point: int, shared: dict) -> dict:
    d = slurm_root / f"point_{point}"
    cfg = yaml.safe_load((d / "gptoss_config.yaml").read_text())["variables"]

    def v(key):
        try:
            return cfg[key]["value"]
        except (KeyError, TypeError):
            raise SystemExit(f"point_{point}/gptoss_config.yaml: no "
                             f"variables.{key}.value") from None

    client_text = (d / "client.yaml").read_text()
    client = yaml.safe_load(client_text)

    # Both server tasks eval run_with_env.py on these (SERVER_ENV_KEYS); the
    # port reads THIS point's server-env.yaml, once, for prefill AND decode.
    env_suffix = f"/point_{point}/server-env.yaml"
    for key in ("TRTLLM_ENV_YAML_CTX", "TRTLLM_ENV_YAML_GEN"):
        if not str(v(key)).endswith(env_suffix):
            raise SystemExit(
                f"point_{point}: gptoss_config.yaml {key} is {v(key)!r}, not "
                f"...{env_suffix}.  The port reads that one file for prefill AND "
                f"decode; a separate env file per role needs a new reader.")
    server_env_text = (d / "server-env.yaml").read_text()

    info = dict(
        dir=d,
        ctx_n=v("NUM_CTX_SERVERS"),
        ctx_tp=v("GPUS_PER_CTX_RANK"),
        gen_n=v("NUM_GEN_SERVERS"),
        gen_tp=v("GPUS_PER_GEN_RANK"),
        frontends=v("NUM_FRONTEND_SERVERS"),
        base_port=v("BASE_PORT"),
        frontend_port=v("FRONTEND_PORT"),
        ctx_text=(d / "server-ctx.yaml").read_text(),
        gen_text=(d / "server-gen.yaml").read_text(),
        client_text=client_text,
        server_env_text=server_env_text,
        server_env=parse_server_env(point, server_env_text),
        # Slurm's CONTAINER_IMAGE carries an enroot docker:// prefix.
        images=dict(shared["images"],
                    TRTLLM_IMAGE=str(v("CONTAINER_IMAGE")).removeprefix("docker://")),
        concurrency=client["settings"]["load_pattern"]["target_concurrency"],
        n_samples=client["settings"]["runtime"]["n_samples_to_issue"],
        max_new_tokens=client["model_params"]["max_new_tokens"],
        max_issue_duration_ms=client["settings"]["runtime"].get("max_issue_duration_ms"),
        run_timeout_s=client["settings"].get("timeouts", {}).get("run_timeout_s"),
        has_drain="drain" in client["settings"],
        has_warmup="warmup" in client["settings"],
    )
    return info


def ported_env(info: dict) -> dict[str, str]:
    """The point's server-env keys SERVER_ENV_KEYS marks "port", in file order."""
    return {k: val for k, val in info["server_env"].items()
            if SERVER_ENV_KEYS[k]["decision"] == SERVER_ENV_PORT}


def assert_table(point: int, info: dict) -> None:
    got = (
        info["ctx_n"], info["ctx_tp"], info["gen_n"], info["gen_tp"],
        info["frontends"], info["concurrency"], info["n_samples"],
        info["max_new_tokens"],
    )
    want = EXPECTED[point]
    if got != want:
        raise SystemExit(
            f"point_{point}: Slurm drop does not match the conversion spec table.\n"
            f"  spec: {want}\n  drop: {got}\n"
            f"The drop changed -- re-read the spec before regenerating."
        )
    if info["base_port"] != 30000 or info["frontend_port"] != 8000:
        raise SystemExit(f"point_{point}: unexpected ports "
                         f"{info['base_port']}/{info['frontend_port']}")
    total = info["ctx_n"] * info["ctx_tp"] + info["gen_n"] * info["gen_tp"]
    if total != 72:
        raise SystemExit(f"point_{point}: {total} GPUs, expected 72")
    want_env = EXPECTED_PORTED_ENV.get(point, {})
    if ported_env(info) != want_env:
        raise SystemExit(
            f"point_{point}: server-env.yaml sets the ported keys "
            f"{ported_env(info)}, EXPECTED_PORTED_ENV says {want_env}.  The drop "
            f"changed -- review the key in SERVER_ENV_KEYS before updating the "
            f"table.")


# --------------------------------------------------------------------------
# The three client.yaml rewrites (conversion spec section 5 + Seed Set A)
# --------------------------------------------------------------------------
MLPERF_OFFICIAL_SEED = 9315206023656308754

REPORT_DIR_COMMENT = [
    "# GKE REWRITE 1/3 (conversion spec section 5, fix 3 of 4).",
    "# Slurm writes the report into ${SFLOW_TASK_OUTPUT_DIR} (v1.0.0 point_7168 had the",
    "# relative 'results'), which on Kubernetes is the pod's emptyDir.  That",
    "# directory only reaches the driver via an EXIT trap plus a sentinel printed",
    "# to the pod log, and that channel failed twice in the Kimi-K3 sweep after a",
    "# flawless benchmark.  ${REPORT_DIR} is exported by the benchmark task and",
    "# points at a durable hostPath on the node's local SSD instead.",
]

TOKENIZER_COMMENT = [
    "  # GKE ADDITION (rewrite 2/3) -- NOT a Slurm value.  The client tokenizes every output",
    "  # to compute OSL / TPS / TPOT: live during the run on this many in-process",
    "  # threads, and whatever they have not finished in an end-of-run drain.  The",
    "  # default is 4 and the drain is SILENT: it prints nothing until it finishes.",
    "  # Measured on Kimi-K3 C192 that drain ran 2 h 05 m, `kubectl logs` stalled",
    "  # for 7,474 s, the collect sentinel could never be delivered and the entire",
    "  # report was lost even though the benchmark itself was perfect.  The client",
    "  # pod has 144 idle CPUs at this point; 32 cuts the drain roughly 8x.",
    "  # GPT-OSS runs up to C22528 with n_samples_to_issue 953,004, so this matters",
    "  # more here than it did on Kimi.",
    "  # It is a `settings` key in the pinned client (endpoints e9f427c), NOT a",
    "  # `settings.client` one: `client` forbids extra keys, and there the client",
    "  # exits at config load with `extra_forbidden`.",
]

SEED_COMMENT = [
    "  # GKE ADDITION (rewrite 3/3) -- MLCommons Endpoints v1.0 Seed Set A",
    "  # (mlperf-endpoints-v1.0-2026-10-C1-A, endpoints-submission-cli",
    "  # constants.py MODEL_SEED = 9315206023656308754, check seed-runtime-match).",
    "  # The NVIDIA v1.0.2 Slurm drop's client.yaml omits model_params.seed.",
]


def longer_n(info: dict) -> int | None:
    """This point's --longer-runs n_samples_to_issue, or None for NVIDIA's."""
    return info["n_samples"] if "drop_n_samples" in info else None


def longer_watchdog(info: dict) -> int | None:
    """This point's --longer-runs run_timeout_s (LONGER_RUNS_OVERRIDES), or None
    for NVIDIA's."""
    return info.get("longer_run_timeout_s")


def passes(n: int) -> str:
    """`n` samples as passes of the DATASET_PROMPTS-prompt dataset."""
    p = n / DATASET_PROMPTS
    return f"{p:g} pass" if p == 1 else f"{p:g} passes"


def longer_run_comment(indent_: str, drop_n: int, new_n: int,
                       watchdog_too: bool = False) -> list[str]:
    tail = (("# Guide steady-state window).  run_timeout_s below is raised for it",
             "# too; nothing else in this file changes for it.")
            if watchdog_too else
            ("# Guide steady-state window).  Nothing else in this file changes for it.",))
    return [indent_ + ln for ln in (
        "# GKE REWRITE, opt-in (--longer-runs) -- NOT NVIDIA's value.  NVIDIA's",
        f"# n_samples_to_issue is {drop_n} ({passes(drop_n)} of the "
        f"{DATASET_PROMPTS:,}-prompt",
        f"# dataset).  The rerun issues {new_n} ({passes(new_n)}) so it "
        f"reaches about",
        "# 4 passes and its region's minimum duration (the First-Time Submitter's",
        *tail,
    )]


def longer_watchdog_comment(indent_: str, drop_s: str, new_s: str,
                            client_s: int) -> list[str]:
    return [indent_ + ln for ln in (
        "# GKE REWRITE, opt-in (--longer-runs) -- NOT NVIDIA's value.  NVIDIA's",
        f"# run_timeout_s is {drop_s} ({float(drop_s) / 3600:g} h); the rerun sets "
        f"{new_s} ({float(new_s) / 3600:g} h).  It is the",
        "# client's WHOLE-RUN watchdog (setup, performance, three accuracy phases,",
        "# metrics drain): it only stops a stuck run and measures nothing.  The",
        f"# longer run above needs about {round(client_s, -2):,} s end to end, which "
        "NVIDIA's value",
        f"# would abort; {new_s} puts it at {client_s / float(new_s):.1%}, under the "
        f"generator's {LONGER_RUNS_HEADROOM:.0%} guard.",
    )]


def rewrite_client(text: str, point: int, n_samples: int | None = None,
                   run_timeout_s: int | None = None) -> str:
    """Apply exactly the three documented rewrites, with commentary.

    `n_samples` is set by --longer-runs only, and adds a fourth: the value of
    settings.runtime.n_samples_to_issue (LONGER_RUNS).  `run_timeout_s` is set
    by --longer-runs only, for a point in LONGER_RUNS_OVERRIDES (C17), and adds
    a fifth: the value of settings.timeouts.run_timeout_s.
    """
    if "metrics_tokenizer_workers" in text:
        raise SystemExit(f"point_{point}: the Slurm client.yaml now sets "
                         f"metrics_tokenizer_workers itself -- re-read the spec "
                         f"before adding the GKE value on top of it")
    if re.search(r"^\s+seed:", text, re.M):
        raise SystemExit(f"point_{point}: the Slurm client.yaml now sets "
                         f"model_params.seed itself -- re-read the spec "
                         f"before adding the GKE value on top of it")
    if run_timeout_s is not None and (n_samples is None or point not in LONGER_RUNS):
        raise SystemExit(f"point_{point}: a run_timeout_s override only comes with "
                         f"a --longer-runs n_samples_to_issue (LONGER_RUNS)")
    out: list[str] = []
    saw_report = 0
    saw_client = 0
    saw_seed = 0
    saw_n = 0
    saw_timeout = 0
    for line in text.splitlines():
        if re.match(r"^report_dir:", line):
            saw_report += 1
            out.extend(REPORT_DIR_COMMENT)
            out.append("report_dir: ${REPORT_DIR}")
            continue
        if line == "  streaming: 'on'":
            saw_seed += 1
            out.append(line)
            out.extend(SEED_COMMENT)
            out.append(f"  seed: {MLPERF_OFFICIAL_SEED}")
            continue
        if line == "  client:":
            # `  client:` only occurs under `settings:`, so inserting just above
            # it at the same indent makes the addition a direct `settings` key.
            saw_client += 1
            out.extend(TOKENIZER_COMMENT)
            out.append("  metrics_tokenizer_workers: 32")
        m = re.match(r"^(\s+)n_samples_to_issue: (\d+)$", line)
        if n_samples is not None and m:
            saw_n += 1
            out.extend(longer_run_comment(m.group(1), int(m.group(2)), n_samples,
                                          watchdog_too=run_timeout_s is not None))
            out.append(f"{m.group(1)}n_samples_to_issue: {n_samples}")
            continue
        m = re.match(r"^(\s+)run_timeout_s: (\d+(?:\.\d+)?)$", line)
        if run_timeout_s is not None and m:
            saw_timeout += 1
            new = repr(float(run_timeout_s))  # a float in the client's schema
            out.extend(longer_watchdog_comment(m.group(1), m.group(2), new,
                                               LONGER_RUNS[point][2]))
            out.append(f"{m.group(1)}run_timeout_s: {new}")
            continue
        out.append(line)
    if saw_report != 1:
        raise SystemExit(f"point_{point}: expected exactly one report_dir line, "
                         f"found {saw_report}")
    if saw_seed != 1:
        raise SystemExit(f"point_{point}: expected exactly one "
                         f"\"  streaming: 'on'\" line under model_params, "
                         f"found {saw_seed}")
    if saw_client != 1:
        raise SystemExit(f"point_{point}: expected exactly one 'settings.client' "
                         f"block, found {saw_client}")
    if n_samples is not None and saw_n != 1:
        raise SystemExit(f"point_{point}: --longer-runs expected exactly one "
                         f"n_samples_to_issue line, found {saw_n}")
    if run_timeout_s is not None and saw_timeout != 1:
        raise SystemExit(f"point_{point}: --longer-runs expected exactly one "
                         f"numeric run_timeout_s line, found {saw_timeout}")
    rewritten = "\n".join(out) + "\n"
    if run_timeout_s is not None:
        # The line match cannot see the parent keys; the parsed file can.
        settings = yaml.safe_load(rewritten)["settings"]
        got = (settings.get("timeouts") or {}).get("run_timeout_s")
        if got != run_timeout_s:
            raise SystemExit(f"point_{point}: after the rewrite "
                             f"settings.timeouts.run_timeout_s is {got!r}, "
                             f"not {run_timeout_s}")
    return rewritten


def strip_comments(text: str) -> str:
    """Drop whole-line comments so --check can compare against the Slurm file."""
    return "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    ) + "\n"


# --------------------------------------------------------------------------
# Emitting
# --------------------------------------------------------------------------
def indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join((pad + ln) if ln.strip() else "" for ln in text.rstrip("\n").splitlines())


def node_layout_comment(point: int, info: dict, slurm: dict | None = None,
                        nodes: int | None = None) -> str:
    """Topology comment block.  `slurm`/`nodes` are set for a variant only."""
    ctx_nodes, gen_nodes, gen_base, total_nodes, total_gpus = layout(
        info["ctx_n"], info["ctx_tp"], info["gen_n"], info["gen_tp"]
    )
    if slurm is None:
        lines = [
            f"# point_{point}:  {info['ctx_n']} x ctx(TP{info['ctx_tp']}) "
            f"+ {info['gen_n']} x gen(TP{info['gen_tp']}) "
            f"= {total_gpus} GPUs = {total_nodes} nodes (a full NVL72 domain).",
            f"# Frontends: {info['frontends']} (one per node, nodes "
            f"0..{info['frontends'] - 1}).",
        ]
    else:
        lines = [
            f"# point_{point}_{nodes}nodes:  {info['ctx_n']} x ctx(TP{info['ctx_tp']}) "
            f"+ {info['gen_n']} x gen(TP{info['gen_tp']}) "
            f"= {total_gpus} GPUs = {total_nodes} nodes.",
            f"#   (Slurm point_{point} is {slurm['ctx_n']} x ctx(TP{slurm['ctx_tp']}) "
            f"+ {slurm['gen_n']} x gen(TP{slurm['gen_tp']}) = 72 GPUs = 18 nodes.)",
            f"# Frontends: {info['frontends']} (one per node, nodes "
            f"0..{info['frontends'] - 1}) -- unchanged from point_{point}.",
        ]
    lines += [
        "#",
        f"# ctx GPUs 0..{info['ctx_n'] * info['ctx_tp'] - 1}  -> nodes "
        f"{ctx_nodes[0]}..{ctx_nodes[-1]}",
        f"# gen GPUs {gen_base}..{gen_base + info['gen_n'] * info['gen_tp'] - 1}"
        f"  -> nodes {gen_nodes[0]}..{gen_nodes[-1]}",
    ]
    if gen_base != info["ctx_n"] * info["ctx_tp"]:
        lines.append(
            "#   (gen is TP>4 so it takes WHOLE nodes; gen_base is rounded up "
            f"from {info['ctx_n'] * info['ctx_tp']} to {gen_base}.)"
        )
    elif gen_base % GPUS_PER_NODE:
        lines.append(
            f"#   *** gen server 0 BACKFILLS GPUs {gen_base}-"
            f"{gen_base + info['gen_tp'] - 1} of node {gen_base // GPUS_PER_NODE}, "
            "the same node ctx is on. ***"
        )
        lines.append(
            "#   The naive ceil(ctx_gpus / 4) formula would put it on node "
            f"{ceil_div(info['ctx_n'] * info['ctx_tp'], GPUS_PER_NODE)} and give "
            f"{ceil_div(info['ctx_n'] * info['ctx_tp'], GPUS_PER_NODE) + ceil_div(info['gen_n'] * info['gen_tp'], GPUS_PER_NODE)}"
            f" nodes for a {total_gpus}-GPU job."
        )
    lines.append("#")
    lines.append("# Per-server placement (node index : port):")
    ctx_map = ", ".join(
        f"ctx{i}=n{ctx_node(i, info['ctx_tp'])}:{info['base_port'] + i}"
        for i in range(min(info["ctx_n"], 6))
    )
    if info["ctx_n"] > 6:
        ctx_map += f", ... ctx{info['ctx_n'] - 1}=n{ctx_node(info['ctx_n'] - 1, info['ctx_tp'])}:{info['base_port'] + info['ctx_n'] - 1}"
    gen_map = ", ".join(
        f"gen{i}=n{gen_node(i, gen_base, info['gen_tp'])}:{info['base_port'] + info['ctx_n'] + i}"
        for i in range(min(info["gen_n"], 6))
    )
    if info["gen_n"] > 6:
        gen_map += (f", ... gen{info['gen_n'] - 1}="
                    f"n{gen_node(info['gen_n'] - 1, gen_base, info['gen_tp'])}:"
                    f"{info['base_port'] + info['ctx_n'] + info['gen_n'] - 1}")
    lines.append(f"#   {ctx_map}")
    lines.append(f"#   {gen_map}")
    lines.append("#")
    lines.append(
        f"# Client: target_concurrency {info['concurrency']}, "
        f"n_samples_to_issue {info['n_samples']:,}, "
        f"max_new_tokens {info['max_new_tokens']}."
    )
    lines.append(
        f"# Per-point client settings (verbatim): "
        f"max_issue_duration_ms {info['max_issue_duration_ms']}, "
        f"run_timeout_s {info['run_timeout_s']}, "
        f"drain {'present' if info['has_drain'] else 'absent'}, "
        f"warmup {'present' if info['has_warmup'] else 'absent'}."
    )
    if "drop_n_samples" in info:
        lines.append(
            f"# LONGER RUN (--longer-runs): n_samples_to_issue {info['n_samples']:,} "
            f"replaces NVIDIA's {info['drop_n_samples']:,}; see the client config."
        )
    if longer_watchdog(info) is not None:
        lines.append(
            f"# LONGER RUN (--longer-runs): run_timeout_s {longer_watchdog(info):,.0f} "
            f"replaces NVIDIA's {info['run_timeout_s']:,.0f}; see the client config."
        )
    return "\n".join(lines)


LAUNCH_TEMPLATE = """#!/usr/bin/env bash
# Launch GPT-OSS-120B (MLPerf Endpoints v1.0.2) point C{point} on a GB300
# (A4X-max) GKE cluster.  Slurm-faithful topology: 72 GPUs = 18 nodes.
# GENERATED by ../../generate_point_recipes.py -- do not hand-edit.
#
# PREREQUISITES -- see ../../README.md.  In short:
#   1. EIGHTEEN schedulable A4X-max nodes (counted below before launching;
#      SKIP_NODE_CHECK=1 skips the count, REQUIRED_NODES=<n> changes it).
#   2. Assets staged on EVERY node:
#        kubectl apply -f ../../stage-assets.yaml
#        kubectl rollout status ds/stage-gptoss-assets --timeout=60m
#   3. Durable report directory on every node:
#        kubectl apply -f ../../mk-report-dir.yaml
#   4. Recommended, so the run does not spend its reservation window on a cold
#      multi-GB image pull:
#        kubectl apply -f ../../prepull-images.yaml
#   5. kubectl context pointing at the target cluster.
#
# Site overrides (env vars, or -s NAME=value passed straight through to sflow):
#   NAMESPACE=<namespace>   NODEPOOL=<your a4x-max nodepool>
#   -s MODE=both            run performance then accuracy (needs gated GPQA)
#   -s WITH_LCB=false       skip the LiveCodeBench service (perf runs only)
#   -s TRTLLM_IMAGE=<img>   e.g. an Artifact Registry mirror
#   -s CLIENT_IMAGE=<img>   e.g. an Artifact Registry mirror
set -euo pipefail

cd "$(dirname "$0")"

# sflow is typically installed in a venv rather than on PATH.
SFLOW_VENV="${{SFLOW_VENV:<workdir>/sflow_workspace/.venv}}"
if [ -x "${{SFLOW_VENV}}/bin/sflow" ]; then
  PATH="${{SFLOW_VENV}}/bin:${{PATH}}"
fi
if ! command -v sflow >/dev/null 2>&1; then
  echo "sflow not found. Set SFLOW_VENV=/path/to/venv or install sflow." >&2
  exit 1
fi

RECIPE="gpt_oss_120b_gke_disagg_point{point}.yaml"
NAMESPACE="${{NAMESPACE:-default}}"
NODEPOOL="${{NODEPOOL:<nodepool>}}"
REQUIRED_NODES="${{REQUIRED_NODES:-18}}"

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
    echo "Not enough healthy nodes; refusing to launch. Set SKIP_NODE_CHECK=1 to override," >&2
    echo "or regenerate reduced-node variants: ../../generate_point_recipes.py --variants 16" >&2
    exit 1
  fi
fi

sflow run -f "${{RECIPE}}" \\
  --kube-namespace "${{NAMESPACE}}" \\
  --kube-node-selector "cloud.google.com/gke-nodepool=${{NODEPOOL}}" \\
  "$@"
"""


def sub_line_token(text: str, token: str, lines: list[str]) -> str:
    """Replace the one line of `text` that holds only `token` with `lines`, each
    at that line's indent; with no lines, delete the line.  recipe_base.yaml
    must hold exactly one such line per token."""
    pat = re.compile(rf"^([ ]*){re.escape(token)}[ ]*\n", re.M)
    found = pat.findall(text)
    if len(found) != 1:
        raise SystemExit(f"recipe_base.yaml: expected exactly one line holding "
                         f"only {token}, found {len(found)}")
    block = "".join(f"{found[0]}{ln}\n" for ln in lines)
    return pat.sub(lambda _m: block, text, count=1)


def _and_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def server_env_note(point: int, info: dict) -> list[str]:
    """Comment lines for prefill_server (__SERVER_ENV_NOTE__): the DSR1 GKE
    recipe's tuning env this point does NOT get -- and, where the drop itself
    sets one of those keys (v1.0.2: TRTLLM_ENABLE_PDL at C2048), that it is
    ported with the DROP's value, for that point only."""
    env = info["server_env"]
    src = f"point_{point}/server-env.yaml"
    present = [k for k in DSR1_ONLY_ENV if k in env]
    absent = [k for k in DSR1_ONLY_ENV if k not in env]
    paras = []
    if present:
        one = len(present) == 1
        paras.append(
            f"NOTE {src} ({DROP_VERSION} drop) sets "
            f"{_and_list([f'{k}={env[k]!r}' for k in present])}, "
            f"{'a key' if one else 'keys'} the DSR1 GKE recipe also sets: "
            f"{'it is' if one else 'they are'} ported with the DROP's value, for "
            f"this point only (the PER-POINT PORT export below and its mpirun -x).")
        for k in present:
            if env[k] != DSR1_ONLY_ENV[k]:
                paras[-1] += (f"  {k}={env[k]} is NOT the DSR1 recipe's "
                              f"{k}={DSR1_ONLY_ENV[k]}: DSR1 tuning is never copied.")
    if absent:
        one = len(absent) == 1
        paras.append(
            "NOTE what is deliberately ABSENT relative to the DSR1 GKE recipe: "
            f"{_and_list([f'{k}={DSR1_ONLY_ENV[k]}' for k in absent])} "
            f"{'is' if one else 'are'} DSR1 tuning; {src} "
            f"{'does not set it' if one else 'sets none of them'}, so neither "
            f"does this port.  Adding {'it' if one else 'them'} would be "
            f"second-guessing NVIDIA's tuning.")
    lines: list[str] = []
    for i, para in enumerate(paras):
        if i:
            lines.append("#")
        lines += ["# " + ln for ln in textwrap.wrap(
            para, width=70, break_long_words=False, break_on_hyphens=False)]
    return lines


def server_env_exports(point: int, info: dict, task: str) -> list[str]:
    """Script items for `task` (prefill_server / decode_server) exporting the
    'port' keys of THIS point's server-env.yaml; none when it has none."""
    env = ported_env(info)
    if not env:
        return []
    role = task.split("_")[0]
    why = (f"PER-POINT PORT ({DROP_VERSION} drop) -- NOT in recipe_base.yaml: the "
           f"'port' keys of point_{point}/server-env.yaml, with the drop's values "
           f"(SERVER_ENV_KEYS in ../../generate_point_recipes.py).  On Slurm, "
           f"run_with_env.py exports them in {task}.  Here the export covers the "
           f"ranks mpirun starts in this pod and the matching mpirun -x below "
           f"the ranks on other nodes.  The echo puts the value in the server log.")
    lines = ["# " + ln for ln in textwrap.wrap(
        why, width=70, break_long_words=False, break_on_hyphens=False)]
    for key, value in env.items():
        lines.append(f"- export {key}={shlex.quote(value)}")
        lines.append(f'- echo "GKE_SERVER_ENV point_{point} {role} '
                     f'{key}=${{{key}:-UNSET}} (ported from server-env.yaml)"')
    return lines


def server_env_mpi_x(info: dict) -> list[str]:
    """`-x KEY=VALUE` mpirun flags for the ported keys.  They land inside the
    folded mpirun scalar, which cannot hold comments: the export above
    carries the explanation."""
    return [f"-x {k}={shlex.quote(val)}" for k, val in ported_env(info).items()]


def render(point: int, info: dict, base: str,
           layout_comment: str | None = None) -> str:
    s = base
    # The per-point server env first, before any NVIDIA text is embedded.
    for token, lines in (
            ("__SERVER_ENV_NOTE__", server_env_note(point, info)),
            ("__CTX_SERVER_ENV_EXPORTS__",
             server_env_exports(point, info, "prefill_server")),
            ("__CTX_SERVER_ENV_MPI_X__", server_env_mpi_x(info)),
            ("__GEN_SERVER_ENV_EXPORTS__",
             server_env_exports(point, info, "decode_server")),
            ("__GEN_SERVER_ENV_MPI_X__", server_env_mpi_x(info))):
        s = sub_line_token(s, token, lines)
    s = s.replace("__NODE_LAYOUT__", layout_comment if layout_comment is not None
                  else node_layout_comment(point, info))
    s = s.replace("__SERVER_CTX_YAML__", indent(info["ctx_text"], 6))
    s = s.replace("__SERVER_GEN_YAML__", indent(info["gen_text"], 6))
    s = s.replace("__CLIENT_CONFIG_YAML__",
                  indent(rewrite_client(info["client_text"], point, longer_n(info),
                                        longer_watchdog(info)), 6))
    s = s.replace("__NUM_CTX_SERVERS__", str(info["ctx_n"]))
    s = s.replace("__GPUS_PER_CTX_RANK__", str(info["ctx_tp"]))
    s = s.replace("__NUM_GEN_SERVERS__", str(info["gen_n"]))
    s = s.replace("__GPUS_PER_GEN_RANK__", str(info["gen_tp"]))
    s = s.replace("__NUM_FRONTEND_SERVERS__", str(info["frontends"]))
    s = s.replace("__CONCURRENCY__", str(info["concurrency"]))
    s = s.replace("__N_SAMPLES__", str(info["n_samples"]))
    s = s.replace("__POINT__", str(point))
    leftover = sorted(set(re.findall(r"__[A-Z0-9_]+__", s)))
    if leftover:
        raise SystemExit(f"point_{point}: unsubstituted tokens {leftover}")
    return s


def out_path(point: int) -> pathlib.Path:
    return HERE / "18nodes" / f"point_{point}" / f"gpt_oss_120b_gke_disagg_point{point}.yaml"


# --------------------------------------------------------------------------
# Emitting the reduced-node variants
# --------------------------------------------------------------------------
NODE_WORDS = {16: "SIXTEEN", 17: "SEVENTEEN"}


def variant_out_path(point: int, nodes: int) -> pathlib.Path:
    return (HERE / f"{nodes}nodes" / f"point_{point}"
            / f"gpt_oss_120b_gke_disagg_point{point}_{nodes}nodes.yaml")


def variant_workflow_name(point: int, nodes: int) -> str:
    return f"trtllm_disagg_gptoss_gke_c{point}_{nodes}nodes"


def split_desc(ctx_n: int, gen_n: int, info: dict) -> str:
    return (f"{ctx_n} x ctx(TP{info['ctx_tp']}) "
            f"+ {gen_n} x gen(TP{info['gen_tp']})")


def variant_banner(point: int, slurm: dict, nodes: int,
                   ranked: list[tuple[Fraction, int, int]]) -> list[str]:
    """Header block explaining what the variant changes and why."""
    score, ctx_n, gen_n = ranked[0]
    gpus = ctx_n * slurm["ctx_tp"] + gen_n * slurm["gen_tp"]
    more_dec = next((r for r in ranked if r[2] > gen_n), None)
    more_pre = next((r for r in ranked if r[2] < gen_n), None)
    slots = decode_slots_per_server(slurm)
    total_slots = gen_n * slots
    conc = slurm["concurrency"]
    gen_ratio = Fraction(gen_n, slurm["gen_n"])
    lines = [
        "#",
        f"# *** {nodes}-NODE VARIANT OF point_{point} -- NOT A SUBMISSION TOPOLOGY ***",
        "# " + "=" * 77,
        "#",
        "# WHY THIS FILE EXISTS.  Every approved GPT-OSS point is 72 GPUs = 18 nodes",
        "# (a full NVL72 domain).  This variant runs the SAME point -- same",
        "# concurrency, same sample count, server-ctx / server-gen / client configs",
        f"# byte-identical to the Slurm drop -- on {nodes} nodes "
        f"({nodes * GPUS_PER_NODE} GPUs), by re-splitting",
        "# the GPUs between prefill and decode servers.",
        "#",
        f"# THE ONLY FUNCTIONAL CHANGES vs ../../18nodes/point_{point}:",
        f"#     NUM_CTX_SERVERS : {slurm['ctx_n']:>2} -> {ctx_n:>2}"
        f"   (TP{slurm['ctx_tp']} each)",
        f"#     NUM_GEN_SERVERS : {slurm['gen_n']:>2} -> {gen_n:>2}"
        f"   (TP{slurm['gen_tp']} each)",
        f"#     => TOTAL_GPUS 72 -> {gpus}, TOTAL_NODES 18 -> {nodes}",
        f"# NUM_FRONTEND_SERVERS stays {slurm['frontends']}: the router's load is "
        "set by the",
        "# concurrency, which does not change.  Ports, URLs and placement all follow",
        "# from the two counts through the same Jinja as every other point.",
        "#",
        "# HOW THE SPLIT WAS CHOSEN (../../generate_point_recipes.py, rank_splits()).",
        "# The server shapes -- TP, max_batch_size, CUDA graphs -- are fixed by the",
        "# drop, so the only free choice is how many prefill and decode servers to",
        f"# fit into {nodes} nodes.  NVIDIA's 72-GPU split is taken as balanced, so a",
        "# re-split keeps about min(ctx'/ctx, gen'/gen) of the point's capacity.",
        f"# Every split that packs into {nodes} nodes is scored; the best wins, ties",
        "# going to more decode servers:",
        f"#    chosen       : {split_desc(ctx_n, gen_n, slurm)}"
        f"  keeps {float(score):.3f} ({limiting_pool(slurm, ctx_n, gen_n)})",
    ]
    for label, alt in (("more decode ", more_dec), ("more prefill", more_pre)):
        if alt is None:
            lines.append(f"#    {label} : none fits in {nodes} nodes")
        else:
            s, c, g = alt
            lines.append(f"#    {label} : {split_desc(c, g, slurm)}"
                         f"  keeps {float(s):.3f} ({limiting_pool(slurm, c, g)})")
    lines += [
        "#",
        "# EXPECTED EFFECT -- do not compare these numbers to an 18-node result:",
    ]
    if total_slots < conc:
        lines += [
            f"#   * decode slots = {gen_n} x {slots} = {total_slots} "
            f"< target_concurrency {conc}: up to",
            f"#     {conc - total_slots} requests that have finished prefill wait "
            "for a decode slot.",
            "#     That wait lands in the client's latency (TTFT and/or TPOT,",
            "#     depending on when the router streams the first token); each",
            f"#     decode server's batch stays at the Slurm maximum of {slots}.",
        ]
        if score == gen_ratio:
            lines += [
                f"#   * throughput should land near {float(gen_ratio):.3f} of the "
                "18-node run (a little",
                "#     above if that run's decode batches were not always full).",
            ]
        else:
            lines += [
                f"#   * throughput should land between {float(score):.3f} (if the "
                "Slurm prefill pool",
                f"#     had no headroom) and {float(gen_ratio):.3f} (if it had "
                "enough) of the 18-node run.",
            ]
    else:
        lines += [
            f"#   * decode slots = {gen_n} x {slots} = {total_slots} "
            f">= target_concurrency {conc}: no",
            "#     request waits for a decode slot.  The mean batch per decode server",
            f"#     only rises from {conc / slurm['gen_n']:.2f} to "
            f"{conc / gen_n:.2f}, so TPOT and throughput should stay",
            "#     close to the 18-node run.",
        ]
    lines += [
        f"#   * the NVLink ComputeDomain spans {nodes} nodes, not a full 18-node domain.",
        "#",
        "# Regenerate with ../../generate_point_recipes.py, which writes the Slurm points",
        "# AND these variants from ../../recipe_base.yaml; never edit this file in place.",
    ]
    return lines


def sub_once(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"variant rewrite {what!r}: expected exactly 1 match in the "
                         f"rendered recipe, found {n} -- recipe_base.yaml changed; "
                         f"update render_variant()")
    return text.replace(old, new)


def render_variant(point: int, slurm: dict, base: str, nodes: int,
                   ranked: list[tuple[Fraction, int, int]]) -> str:
    """The regular render with the two server counts replaced, plus comments.

    Every rewrite below is comment/label text except the workflow name; the
    functional delta vs point_<C> is NUM_CTX_SERVERS / NUM_GEN_SERVERS alone.
    """
    _, ctx_n, gen_n = ranked[0]
    v = variant_info(slurm, ctx_n, gen_n)
    *_, total_nodes, gpus = layout(ctx_n, slurm["ctx_tp"], gen_n, slurm["gen_tp"])
    if total_nodes != nodes:
        raise SystemExit(f"point_{point}: chosen split uses {total_nodes} nodes, "
                         f"not {nodes}; the variant text assumes a full pool")
    s = render(point, v, base, node_layout_comment(point, v, slurm, nodes))

    head = "# (GB300 NVL72).  *** GENERATED FILE -- DO NOT EDIT IN PLACE ***\n"
    s = sub_once(s, head,
                 head + "\n".join(variant_banner(point, slurm, nodes, ranked)) + "\n",
                 "header banner")
    s = sub_once(
        s,
        "#   * 18 schedulable A4X-max nodes are required.  At the time of writing the\n"
        "#     pool has 16.  No point can run until it is grown.\n",
        f"#   * {nodes} schedulable A4X-max nodes are required for THIS variant (18 for\n"
        f"#     the Slurm-faithful ../../18nodes/point_{point}).  launch_test.sh checks first.\n",
        "node requirement")
    s = sub_once(
        s,
        f"  # POINT TOPOLOGY -- verbatim from point_{point}/gptoss_config.yaml.\n"
        "  # The generator asserts these against the Slurm YAML on every run; do not\n"
        "  # hand-edit them.\n",
        f"  # POINT TOPOLOGY -- from point_{point}/gptoss_config.yaml, EXCEPT\n"
        f"  # NUM_CTX_SERVERS / NUM_GEN_SERVERS, which are re-split for {nodes} nodes\n"
        "  # (see the header).  The generator asserts all of them on every run; do\n"
        "  # not hand-edit them.\n",
        "topology comment")
    for key, what, new_n, old_n in (
            ("NUM_CTX_SERVERS", "prefill (context)", ctx_n, slurm["ctx_n"]),
            ("NUM_GEN_SERVERS", "decode (generation)", gen_n, slurm["gen_n"])):
        s = sub_once(
            s,
            f'    description: "Slurm {key}: {what} server replicas."\n',
            "    description: >-\n"
            f"      Slurm {key}: {what} server replicas.\n"
            f"      {nodes}-NODE VARIANT: {new_n} (Slurm point_{point}: {old_n}).  See the\n"
            "      header for how the prefill/decode split was chosen.\n",
            f"{key} description")
    s = sub_once(
        s,
        '    description: "Cross-check: every GPT-OSS point is exactly 72 GPUs."\n',
        '    description: "Cross-check: every Slurm GPT-OSS point is exactly 72 GPUs;'
        f' this {nodes}-node variant is {gpus}."\n',
        "TOTAL_GPUS description")
    s = sub_once(
        s,
        "      ceil(TOTAL_GPUS / GPUS_PER_NODE) == 18 for every point.  Jinja has no\n",
        f"      ceil(TOTAL_GPUS / GPUS_PER_NODE) == 18 for every point ({nodes} for this\n"
        f"      variant: {gpus} GPUs / {GPUS_PER_NODE}).  Jinja has no\n",
        "TOTAL_NODES description")
    s = sub_once(
        s,
        "    # 72 GPUs / 4 per node = 18.  Matches Slurm SLURM_NODES.\n",
        f"    # {nodes}-node variant: {gpus} GPUs / 4 per node = {nodes}.  (Slurm point:\n"
        "    # 72 / 4 = 18, matching SLURM_NODES.)\n",
        "backend node count")
    s = sub_once(
        s,
        "workflow:\n  name: trtllm_disagg_gptoss_gke\n",
        f"workflow:\n  name: {variant_workflow_name(point, nodes)}\n",
        "workflow name")
    s = sub_once(
        s,
        "          point                 : C${{ variables.CONCURRENCY }}\n",
        "          point                 : C${{ variables.CONCURRENCY }}\n"
        f"          VARIANT               : {nodes}-node ({ctx_n} ctx + {gen_n} gen;"
        f" Slurm {slurm['ctx_n']} + {slurm['gen_n']}) -- NOT a submission topology\n",
        "topology.txt VARIANT line")
    return s


LAUNCH_VARIANT_TEMPLATE = r"""#!/usr/bin/env bash
# Launch the @NODES@-NODE VARIANT of GPT-OSS-120B point C@POINT@ on a GB300
# (A4X-max) GKE cluster.  NOT a submission topology:
#   @SPLIT@ = @GPUS@ GPUs = @NODES@ nodes
#   (Slurm point_@POINT@: @SLURM_SPLIT@ = 72 GPUs = 18 nodes).
# See the header of @RECIPE@ for how the split was
# chosen and what it changes.  GENERATED by ../../generate_point_recipes.py.
#
# PREREQUISITES -- see ../../README.md sections 0-3.  In short:
#   1. @NODES_WORD@ schedulable A4X-max nodes (checked below before launching).
#   2. Assets staged on EVERY node:
#        kubectl apply -f ../../stage-assets.yaml
#        kubectl rollout status ds/stage-gptoss-assets --timeout=60m
#   3. Durable report directory on every node:
#        kubectl apply -f ../../mk-report-dir.yaml
#   4. Recommended, so the run does not spend its reservation window on a cold
#      multi-GB image pull:
#        kubectl apply -f ../../prepull-images.yaml
#   5. kubectl context pointing at the target cluster.
#
# Site overrides (env vars, or -s NAME=value passed straight through to sflow):
#   NAMESPACE=<namespace>   NODEPOOL=<your a4x-max nodepool>
#   -s MODE=both            run performance then accuracy (needs gated GPQA)
#   -s WITH_LCB=false       skip the LiveCodeBench service (perf runs only)
#   -s TRTLLM_IMAGE=<img>   e.g. an Artifact Registry mirror
#   -s CLIENT_IMAGE=<img>   e.g. an Artifact Registry mirror
#   SKIP_NODE_CHECK=1       skip the healthy-node pre-check below
set -euo pipefail

cd "$(dirname "$0")"

# sflow is typically installed in a venv rather than on PATH.
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


def render_variant_launch(point: int, slurm: dict, nodes: int,
                          ranked: list[tuple[Fraction, int, int]]) -> str:
    _, ctx_n, gen_n = ranked[0]
    gpus = ctx_n * slurm["ctx_tp"] + gen_n * slurm["gen_tp"]
    s = (LAUNCH_VARIANT_TEMPLATE
         .replace("@NODES_WORD@", NODE_WORDS.get(nodes, str(nodes)))
         .replace("@NODES@", str(nodes))
         .replace("@POINT@", str(point))
         .replace("@SLURM_SPLIT@", split_desc(slurm["ctx_n"], slurm["gen_n"], slurm))
         .replace("@SPLIT@", split_desc(ctx_n, gen_n, slurm))
         .replace("@GPUS@", str(gpus))
         .replace("@RECIPE@", variant_out_path(point, nodes).name))
    leftover = sorted(set(re.findall(r"@[A-Z_]+@", s)))
    if leftover:
        raise SystemExit(f"point_{point}: unsubstituted launch tokens {leftover}")
    return s


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def print_layout(points: dict[int, dict]) -> None:
    print(f"{'point':>6} {'ctx':>12} {'gen':>12} {'fe':>3} "
          f"{'GPUs':>5} {'nodes':>6}  {'ctx nodes':<10} {'gen nodes':<10} "
          f"{'gen_base':>8}  packing")
    print("-" * 104)
    ok = True
    for point, info in points.items():
        ctx_nodes, gen_nodes, gen_base, total_nodes, total_gpus = layout(
            info["ctx_n"], info["ctx_tp"], info["gen_n"], info["gen_tp"]
        )
        all_nodes = set(ctx_nodes) | set(gen_nodes)
        naive_total = (ceil_div(info["ctx_n"] * info["ctx_tp"], GPUS_PER_NODE)
                       + ceil_div(info["gen_n"] * info["gen_tp"], GPUS_PER_NODE))
        if gen_base != info["ctx_n"] * info["ctx_tp"]:
            note = "gen takes whole nodes (TP>4)"
        elif gen_base % GPUS_PER_NODE:
            note = f"gen0 BACKFILLS node {gen_base // GPUS_PER_NODE}"
        else:
            note = "aligned"
        if naive_total != total_nodes:
            note += f"  [naive formula would say {naive_total}!]"
        bad = (total_gpus != 72 or total_nodes != 18
               or all_nodes != set(range(18)))
        ok = ok and not bad
        ctx_desc = "{} x TP{}".format(info["ctx_n"], info["ctx_tp"])
        gen_desc = "{} x TP{}".format(info["gen_n"], info["gen_tp"])
        ctx_span = "{}-{}".format(ctx_nodes[0], ctx_nodes[-1])
        gen_span = "{}-{}".format(gen_nodes[0], gen_nodes[-1])
        print(
            "{:>6} {:>12} {:>12} {:>3} {:>5} {:>6}  {:<10} {:<10} {:>8}  {}{}".format(
                point, ctx_desc, gen_desc, info["frontends"],
                total_gpus, total_nodes, ctx_span, gen_span, gen_base, note,
                "   <-- BROKEN" if bad else "",
            )
        )
    print("-" * 104)
    print("Every row must read 72 GPUs / 18 nodes with ctx+gen covering nodes 0..17.")
    if not ok:
        raise SystemExit("node layout check FAILED")
    print("node layout check: OK (11/11 points cover exactly nodes 0..17)")


def print_variant_layout(points: dict[int, dict], nodes: int,
                         splits: dict[int, list]) -> None:
    print(f"{nodes}-node variants (point_<C>_{nodes}nodes/): Slurm servers, "
          f"re-split by rank_splits()")
    hdr = (f"{'point':>6} {'Slurm':>7} {'ctx':>9} {'gen':>9} {'fe':>3} "
           f"{'GPUs':>5} {'nodes':>6}  {'ctx nodes':<10} {'gen nodes':<10} "
           f"{'slots / C':>13}  {'keeps':>5}  limited by")
    print(hdr)
    print("-" * len(hdr))
    ok = True
    for point, ranked in splits.items():
        info = points[point]
        score, ctx_n, gen_n = ranked[0]
        ctx_nodes, gen_nodes, _, total_nodes, total_gpus = layout(
            ctx_n, info["ctx_tp"], gen_n, info["gen_tp"])
        covered = (set(ctx_nodes) | set(gen_nodes)) == set(range(nodes))
        bad = (total_nodes != nodes or total_gpus > nodes * GPUS_PER_NODE
               or not covered or info["frontends"] > nodes)
        ok = ok and not bad
        slurm_desc = "{}+{}".format(info["ctx_n"], info["gen_n"])
        ctx_desc = "{} x TP{}".format(ctx_n, info["ctx_tp"])
        gen_desc = "{} x TP{}".format(gen_n, info["gen_tp"])
        ctx_span = "{}-{}".format(ctx_nodes[0], ctx_nodes[-1])
        gen_span = "{}-{}".format(gen_nodes[0], gen_nodes[-1])
        slots = "{} / {}".format(gen_n * decode_slots_per_server(info),
                                 info["concurrency"])
        print("{:>6} {:>7} {:>9} {:>9} {:>3} {:>5} {:>6}  {:<10} {:<10} {:>13}  "
              "{:>5.3f}  {}{}".format(
                  point, slurm_desc, ctx_desc, gen_desc, info["frontends"],
                  total_gpus, total_nodes, ctx_span, gen_span, slots,
                  float(score), limiting_pool(info, ctx_n, gen_n),
                  "   <-- BROKEN" if bad else ""))
    print("-" * len(hdr))
    print(f"Every row must read <= {nodes * GPUS_PER_NODE} GPUs / {nodes} nodes with "
          f"ctx+gen covering nodes 0..{nodes - 1}.")
    if not ok:
        raise SystemExit(f"{nodes}-node layout check FAILED")
    print(f"{nodes}-node layout check: OK ({len(splits)}/{len(splits)} variants "
          f"cover exactly nodes 0..{nodes - 1})")


ENV_TASKS = ("prefill_server", "decode_server")


def _env_of_script(script: list[str]) -> tuple[dict, dict, int, list[str]]:
    """(exports, mpi_x, mpirun items, problems) for one server task's script.

    exports: {KEY: [value, ...]} from the one-line `export KEY=VALUE` items;
    mpi_x:   {KEY: [value or None, ...]} from the mpirun item's `-x KEY[=VALUE]`
             (None: forwards the exported value).
    """
    exports: dict[str, list[str]] = {}
    mpi_x: dict[str, list[str | None]] = {}
    n_mpirun = 0
    problems: list[str] = []
    for item in script:
        line = item.strip()
        if "\n" not in line and line.startswith("export "):
            try:
                words = shlex.split(line)
            except ValueError:
                continue  # not a plain assignment (no allowlisted key is)
            if len(words) == 2 and "=" in words[1]:
                key, value = words[1].split("=", 1)
                exports.setdefault(key, []).append(value)
        elif line.startswith("mpirun "):
            n_mpirun += 1
            try:
                words = shlex.split(line)
            except ValueError as exc:
                problems.append(f"cannot tokenize the mpirun command ({exc})")
                continue
            for flag, arg in zip(words, words[1:]):
                if flag == "-x":
                    key, eq, value = arg.partition("=")
                    mpi_x.setdefault(key, []).append(value if eq else None)
    return exports, mpi_x, n_mpirun, problems


def verify_env(point: int, info: dict, doc: dict) -> tuple[list[str], str]:
    """Compare a parsed recipe's server env with point_<C>/server-env.yaml.

    For prefill_server AND decode_server, for every SERVER_ENV_KEYS key:
      * set by this point, "port" or "base": exactly one `export KEY=<drop
        value>`, exactly one mpirun -x that forwards the same value, and no
        other assignment of KEY anywhere in the script;
      * "port" but NOT set by this point: KEY appears nowhere in the script;
      * "gke-ucx": export and -x carry the GKE UCX layer's value, once each.
    Also: TRTLLM_IMAGE / CLIENT_IMAGE / LCB_IMAGE default to the drop's images,
    and every script item of every task is a string (an unquoted `: ` in a
    script line silently turns it into a YAML mapping).
    Returns (problems, one-line summary).
    """
    problems: list[str] = []
    drop = info["server_env"]
    tasks: dict = {}
    for task in (doc.get("workflow") or {}).get("tasks") or []:
        if not isinstance(task, dict):
            problems.append(f"workflow task {task!r} is not a mapping")
            continue
        tasks[task.get("name")] = task
        for i, item in enumerate(task.get("script") or []):
            if not isinstance(item, str):
                problems.append(f"{task.get('name')}: script item {i} is a "
                                f"{type(item).__name__}, not a string")
    for name in ENV_TASKS:
        if name not in tasks:
            problems.append(f"task {name} is missing")
            continue
        script = [x for x in tasks[name].get("script") or [] if isinstance(x, str)]
        exports, mpi_x, n_mpirun, parse_problems = _env_of_script(script)
        problems += [f"{name}: {p}" for p in parse_problems]
        if n_mpirun != 1:
            problems.append(f"{name}: {n_mpirun} mpirun items, expected 1")
        lines = [ln for item in script for ln in item.splitlines()]
        for key, rule in SERVER_ENV_KEYS.items():
            if rule["decision"] == SERVER_ENV_GKE_UCX:
                want, source = rule["gke"], "the GKE UCX layer"
            elif key in drop:
                want, source = drop[key], f"point_{point}/server-env.yaml"
            elif rule["decision"] == SERVER_ENV_PORT:
                if any(re.search(rf"\b{re.escape(key)}\b", item) for item in script):
                    problems.append(f"{name}: {key} is in the script, but "
                                    f"point_{point}/server-env.yaml does not set it")
                continue
            else:  # a "base" key the drop dropped; parse_server_env refuses it
                problems.append(f"{name}: recipe_base.yaml exports {key}, "
                                f"point_{point}/server-env.yaml does not set it")
                continue
            got_exports = exports.get(key, [])
            if got_exports != [want]:
                problems.append(f"{name}: export {key} gives {got_exports}, "
                                f"expected exactly [{want!r}] ({source})")
            got_x = mpi_x.get(key, [])
            if len(got_x) != 1:
                problems.append(f"{name}: mpirun has {len(got_x)} -x {key}, "
                                f"expected 1")
            else:
                forwarded = got_x[0] if got_x[0] is not None else (
                    got_exports[0] if len(got_exports) == 1 else None)
                if forwarded != want:
                    problems.append(f"{name}: mpirun -x forwards {key}="
                                    f"{forwarded!r}, expected {want!r} ({source})")
            assigned = sum(bool(re.match(rf"\s*(?:export\s+)?{re.escape(key)}=", ln))
                           for ln in lines)
            if assigned != 1:
                problems.append(f"{name}: {key} is assigned {assigned} times in "
                                f"the script, expected once")
    varz = doc.get("variables") or {}
    for var, want in info["images"].items():
        got = (varz.get(var) or {}).get("value")
        if got != want:
            problems.append(f"variable {var} defaults to {got!r}, the drop's "
                            f"{DROP_IMAGE_VARS[var]} is {want!r}")
    ported = ported_env(info)
    n_model = sum(SERVER_ENV_KEYS[k]["decision"] != SERVER_ENV_GKE_UCX for k in drop)
    summary = (f"prefill+decode env == server-env.yaml for all {n_model} "
               f"model/runtime keys (ported here: "
               f"{', '.join(f'{k}={v}' for k, v in ported.items()) or 'none'}), "
               f"UCX keys = GKE layer, images == drop")
    return problems, summary


def check_targets(points: dict[int, dict], variants: dict[int, dict],
                  base: str) -> list[dict]:
    """What --check verifies: the 11 Slurm points, then every variant."""
    targets = []
    for point, info in points.items():
        targets.append(dict(
            point=point, info=info, label=f"point_{point}", path=out_path(point),
            topo=info, topo_source="gptoss_config.yaml",
            recipe_text=render(point, info, base),
            launch_text=LAUNCH_TEMPLATE.format(point=point),
            workflow="trtllm_disagg_gptoss_gke"))
    for nodes, splits in variants.items():
        for point, ranked in splits.items():
            info = points[point]
            _, ctx_n, gen_n = ranked[0]
            targets.append(dict(
                point=point, info=info, label=f"point_{point}_{nodes}nodes",
                path=variant_out_path(point, nodes),
                topo=variant_info(info, ctx_n, gen_n),
                topo_source=f"gptoss_config.yaml + the {nodes}-node split",
                recipe_text=render_variant(point, info, base, nodes, ranked),
                launch_text=render_variant_launch(point, info, nodes, ranked),
                workflow=variant_workflow_name(point, nodes)))
    return targets


def check(points: dict[int, dict], variants: dict[int, dict], base: str) -> int:
    failures = 0
    for t in check_targets(points, variants, base):
        point, info, label, path = t["point"], t["info"], t["label"], t["path"]
        topo = t["topo"]
        if not path.exists():
            print(f"[FAIL] {label}: {path} does not exist -- run the generator")
            failures += 1
            continue
        try:
            doc = yaml.safe_load(path.read_text())
        except Exception as exc:
            print(f"[FAIL] {label}: generated recipe is not valid YAML: {exc}")
            failures += 1
            continue

        arts = {a["name"]: a for a in doc["artifacts"]}
        varz = doc["variables"]

        def cmp(name: str, got: str, want: str, transform=None) -> None:
            nonlocal failures
            g = transform(got) if transform else got
            if g.rstrip("\n") != want.rstrip("\n"):
                failures += 1
                print(f"[FAIL] {label}: {name} differs from the Slurm source")
                for line in list(difflib.unified_diff(
                        want.splitlines(), g.splitlines(),
                        fromfile=f"slurm/{name}", tofile=f"gke/{name}",
                        lineterm=""))[:40]:
                    print("        " + line)
            else:
                print(f"[ ok ] {label}: {name} byte-identical to the Slurm source")

        cmp("server-ctx.yaml", arts["SERVER_CTX_YAML"]["content"], info["ctx_text"])
        cmp("server-gen.yaml", arts["SERVER_GEN_YAML"]["content"], info["gen_text"])

        # client.yaml: the ONLY tolerated differences are the three documented
        # rewrites (four for a point --longer-runs lengthens, five for C17,
        # whose run_timeout_s it also raises).  Reconstruct them from the Slurm
        # source and require equality after stripping the explanatory comment
        # lines the generator inserts.
        want_client = strip_comments(
            rewrite_client(info["client_text"], point, longer_n(info),
                           longer_watchdog(info)))
        n_rewrites = (3 + (longer_n(info) is not None)
                      + (longer_watchdog(info) is not None))
        cmp(f"client.yaml ({n_rewrites} documented rewrites only)",
            arts["CLIENT_CONFIG"]["content"], want_client, transform=strip_comments)

        # Topology scalars must match gptoss_config.yaml (for a variant: with
        # NUM_CTX_SERVERS / NUM_GEN_SERVERS replaced by its split).
        for key, want_val in (
            ("NUM_CTX_SERVERS", topo["ctx_n"]),
            ("GPUS_PER_CTX_RANK", topo["ctx_tp"]),
            ("NUM_GEN_SERVERS", topo["gen_n"]),
            ("GPUS_PER_GEN_RANK", topo["gen_tp"]),
            ("NUM_FRONTEND_SERVERS", topo["frontends"]),
            ("BASE_PORT", topo["base_port"]),
            ("FRONTEND_PORT", topo["frontend_port"]),
            ("CONCURRENCY", topo["concurrency"]),
            ("N_SAMPLES", topo["n_samples"]),
        ):
            got_val = varz[key]["value"]
            if got_val != want_val:
                failures += 1
                print(f"[FAIL] {label}: variable {key} = {got_val!r}, "
                      f"{t['topo_source']} says {want_val!r}")
        print(f"[ ok ] {label}: topology variables match {t['topo_source']}")

        # The client config must still parse as YAML on its own, and must still
        # carry everything the old template used to throw away.
        cc = yaml.safe_load(arts["CLIENT_CONFIG"]["content"])
        for required in ("submission_ref", "datasets", "settings",
                         "endpoint_config", "model_params"):
            if required not in cc:
                failures += 1
                print(f"[FAIL] {label}: client config lost '{required}'")
        acc = [d["name"] for d in cc["datasets"] if d.get("type") == "accuracy"]
        if sorted(acc) != ["aime25::gptoss", "gpqa::gptoss", "livecodebench::gptoss"]:
            failures += 1
            print(f"[FAIL] {label}: accuracy datasets are {acc}")
        if cc["model_params"].get("seed") != MLPERF_OFFICIAL_SEED:
            failures += 1
            print(f"[FAIL] {label}: model_params.seed is "
                  f"{cc['model_params'].get('seed')!r}, expected "
                  f"{MLPERF_OFFICIAL_SEED}")
        if cc["settings"].get("metrics_tokenizer_workers") != 32:
            failures += 1
            print(f"[FAIL] {label}: settings.metrics_tokenizer_workers is not 32")
        if "metrics_tokenizer_workers" in (cc["settings"].get("client") or {}):
            failures += 1
            print(f"[FAIL] {label}: metrics_tokenizer_workers is under "
                  f"settings.client -- the pinned client rejects it there "
                  f"(extra_forbidden) and exits at config load")
        if cc["report_dir"] != "${REPORT_DIR}":
            failures += 1
            print(f"[FAIL] {label}: report_dir is {cc['report_dir']!r}")
        if cc["endpoint_config"]["endpoints"] != []:
            failures += 1
            print(f"[FAIL] {label}: endpoints should stay [] (filled at runtime)")
        if cc["settings"]["runtime"]["n_samples_to_issue"] != topo["n_samples"]:
            failures += 1
            print(f"[FAIL] {label}: settings.runtime.n_samples_to_issue is "
                  f"{cc['settings']['runtime']['n_samples_to_issue']}, expected "
                  f"{topo['n_samples']}")
        want_timeout = longer_watchdog(info)
        if want_timeout is None:
            want_timeout = info["run_timeout_s"]
        got_timeout = (cc["settings"].get("timeouts") or {}).get("run_timeout_s")
        if got_timeout != want_timeout:
            failures += 1
            print(f"[FAIL] {label}: settings.timeouts.run_timeout_s is "
                  f"{got_timeout}, expected {want_timeout}")
        print(f"[ ok ] {label}: accuracy datasets, submission_ref, report_dir, "
              f"metrics_tokenizer_workers, model_params.seed all present")

        # MPI slots must match the POD, not the node.
        ops = {o["name"]: o for o in doc["operators"]}
        for op, tp in (("mpi_trtllm_ctx", info["ctx_tp"]),
                       ("mpi_trtllm_gen", info["gen_tp"])):
            expr = str(ops[op]["mpi"]["slots_per_worker"])
            if "| min" not in expr:
                failures += 1
                print(f"[FAIL] {label}: {op}.slots_per_worker is {expr!r}, "
                      f"expected a min(...) expression (pod has "
                      f"{min(tp, GPUS_PER_NODE)} GPUs)")

        # Servers must not have been hand-edited to use UCX timeouts that only
        # apply to DSR1's UCX backend.
        for name, text in (("server-ctx", info["ctx_text"]),
                           ("server-gen", info["gen_text"])):
            if "backend: DEFAULT" not in text:
                failures += 1
                print(f"[FAIL] {label}: {name}.yaml no longer says "
                      f"'backend: DEFAULT' -- re-read spec section 3")
            if "kv_transfer_timeout_ms" in arts[
                    "SERVER_CTX_YAML" if name == "server-ctx" else "SERVER_GEN_YAML"]["content"]:
                failures += 1
                print(f"[FAIL] {label}: {name}.yaml has DSR1's UCX "
                      f"kv_transfer_timeout_ms; GPT-OSS uses backend: DEFAULT")

        if doc["workflow"]["name"] != t["workflow"]:
            failures += 1
            print(f"[FAIL] {label}: workflow.name is {doc['workflow']['name']!r}, "
                  f"expected {t['workflow']!r}")
        else:
            print(f"[ ok ] {label}: workflow.name is {t['workflow']}")

        # The server env, key by key against point_<C>/server-env.yaml.
        env_problems, env_summary = verify_env(point, info, doc)
        for problem in env_problems:
            print(f"[FAIL] {label}: {problem}")
        failures += len(env_problems)
        if not env_problems:
            print(f"[ ok ] {label}: {env_summary}")

        # The file on disk must be exactly what the generator renders now: this
        # catches both a hand edit and a recipe_base.yaml change that was never
        # regenerated.
        if path.read_text() != t["recipe_text"]:
            failures += 1
            print(f"[FAIL] {label}: differs from a fresh render of recipe_base.yaml "
                  f"-- hand-edited, or not regenerated; run the generator")
        else:
            print(f"[ ok ] {label}: identical to a fresh render of recipe_base.yaml")

        launch = path.parent / "launch_test.sh"
        if not launch.exists():
            failures += 1
            print(f"[FAIL] {label}: launch_test.sh missing")
        elif launch.read_text() != t["launch_text"]:
            failures += 1
            print(f"[FAIL] {label}: launch_test.sh differs from the generator's "
                  f"template -- run the generator")
        else:
            print(f"[ ok ] {label}: launch_test.sh matches the generator's template")
    return failures


def validate_longer_runs_overrides() -> None:
    """Refuse anything in LONGER_RUNS_OVERRIDES that the allow-list does not name."""
    if not isinstance(LONGER_RUNS_OVERRIDES, dict):
        raise SystemExit("LONGER_RUNS_OVERRIDES must map point -> {key: seconds}, "
                         f"not {type(LONGER_RUNS_OVERRIDES).__name__}")
    for point, overrides in LONGER_RUNS_OVERRIDES.items():
        if point not in LONGER_RUNS:
            raise SystemExit(f"LONGER_RUNS_OVERRIDES: point {point!r} is not in "
                             f"LONGER_RUNS; an override only comes with a longer run")
        if not isinstance(overrides, dict) or not overrides:
            raise SystemExit(f"LONGER_RUNS_OVERRIDES[{point}] must be a non-empty "
                             f"{{key: seconds}} dict, not {overrides!r}")
        for key, value in overrides.items():
            if (point, key) not in LONGER_RUNS_OVERRIDE_ALLOWED:
                raise SystemExit(f"LONGER_RUNS_OVERRIDES[{point}]: {key!r} is not "
                                 f"allowed for point_{point}; the allow-list is "
                                 f"{sorted(LONGER_RUNS_OVERRIDE_ALLOWED)}")
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SystemExit(f"LONGER_RUNS_OVERRIDES[{point}][{key!r}] = "
                                 f"{value!r}: expected whole seconds > 0")


def apply_longer_runs(points: dict[int, dict]) -> None:
    """--longer-runs: swap in LONGER_RUNS' n_samples_to_issue, and C17's
    run_timeout_s from LONGER_RUNS_OVERRIDES, after checking them."""
    validate_longer_runs_overrides()
    for point, (new_n, perf_s, client_s) in sorted(LONGER_RUNS.items()):
        info = points[point]
        drop_n = info["n_samples"]
        problems = []
        if new_n <= drop_n:
            problems.append(f"{new_n} is not above NVIDIA's {drop_n}")
        if new_n % DATASET_PROMPTS:
            problems.append(f"{new_n} is not a whole number of "
                            f"{DATASET_PROMPTS}-prompt passes")
        cap_ms = info["max_issue_duration_ms"]
        if cap_ms is not None and perf_s >= LONGER_RUNS_HEADROOM * cap_ms / 1000:
            problems.append(f"the predicted {perf_s} s performance phase reaches "
                            f"{LONGER_RUNS_HEADROOM:.0%} of max_issue_duration_ms "
                            f"{cap_ms}")
        watchdog = info["run_timeout_s"]
        raised = LONGER_RUNS_OVERRIDES.get(point, {}).get(
            "settings.timeouts.run_timeout_s")
        if raised is not None:
            if watchdog is None:
                problems.append("NVIDIA's client.yaml sets no run_timeout_s to raise")
            elif not watchdog < raised <= 2 * watchdog:
                problems.append(f"the run_timeout_s override {raised} is not above "
                                f"NVIDIA's {watchdog:g} and at most twice it")
            # The guard below checks the watchdog this rerun will run with.
            watchdog = raised
        if watchdog is not None and client_s >= LONGER_RUNS_HEADROOM * watchdog:
            problems.append(f"the projected {client_s} s client run reaches "
                            f"{LONGER_RUNS_HEADROOM:.0%} of the whole-run watchdog "
                            f"run_timeout_s {watchdog:g}"
                            + (" (LONGER_RUNS_OVERRIDES)" if raised is not None else ""))
        if problems:
            raise SystemExit(f"point_{point}: --longer-runs refused: "
                             + "; ".join(problems))
        points[point] = dict(info, n_samples=new_n, drop_n_samples=drop_n)
        if raised is not None:
            points[point]["longer_run_timeout_s"] = raised
        print(f"longer run: point_{point} n_samples_to_issue {drop_n:,} -> {new_n:,} "
              f"({drop_n / DATASET_PROMPTS:g} -> {new_n / DATASET_PROMPTS:g} passes); "
              f"predicted performance phase {perf_s:,} s, client run {client_s:,} s"
              + (f" of a {watchdog:,.0f} s watchdog" if watchdog else "")
              + (f" (run_timeout_s raised from NVIDIA's {info['run_timeout_s']:,.0f})"
                 if raised is not None else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slurm-root", type=pathlib.Path, default=DEFAULT_SLURM_ROOT,
                    help="NVIDIA Slurm drop config dir (default: %(default)s)")
    ap.add_argument("--layout", action="store_true",
                    help="print the node/GPU layout tables and exit")
    ap.add_argument("--check", action="store_true",
                    help="diff generated recipes against the Slurm source")
    ap.add_argument("--variants", type=int, nargs="*", default=[],
                    choices=VARIANT_NODES, metavar="N",
                    help="ALSO emit/check reduced-node <N>nodes/point_<C>/ "
                         f"variants (supported: {VARIANT_NODES}). Off by "
                         f"default: {DROP_VERSION} targets the 18-node pool.")
    ap.add_argument("--longer-runs", action="store_true",
                    help="emit/check the RERUN recipes: n_samples_to_issue raised "
                         "on " + ", ".join(f"C{p}" for p in sorted(LONGER_RUNS))
                         + " (LONGER_RUNS), and run_timeout_s raised on C17 "
                         "(LONGER_RUNS_OVERRIDES). Off by default.")
    args = ap.parse_args()

    if not args.slurm_root.is_dir():
        raise SystemExit(f"Slurm drop not found at {args.slurm_root}; "
                         f"pass --slurm-root")
    if args.longer_runs and args.variants:
        raise SystemExit("--longer-runs is for the 18-node submission points only; "
                         "drop --variants")

    validate_server_env_table()
    shared = read_shared(args.slurm_root)
    points: dict[int, dict] = {}
    for p in POINTS:
        info = read_point(args.slurm_root, p, shared)
        assert_table(p, info)
        points[p] = info
    print(f"read {len(points)} points from {args.slurm_root}")
    print("conversion-spec section 1 table: OK (11/11 points match the drop)")
    ported = "; ".join(f"C{p} " + ", ".join(f"{k}={v}" for k, v in ported_env(i).items())
                       for p, i in points.items() if ported_env(i)) or "none"
    print(f"server-env.yaml: OK (11/11 points: every key and value allowlisted in "
          f"SERVER_ENV_KEYS; ported per point: {ported}); "
          f"{shared['path'].name}: SITE_ENV_YAML empty")
    if args.longer_runs:
        apply_longer_runs(points)

    variants = {n: {p: variant_splits(p, points[p], n) for p in POINTS}
                for n in args.variants}
    for n in variants:
        print(f"{n}-node split table: OK ({len(variants[n])}/{len(POINTS)} "
              f"points match EXPECTED_SPLIT)")
    print()

    def print_all_layouts() -> None:
        print_layout(points)
        for n, splits in variants.items():
            print()
            print_variant_layout(points, n, splits)

    if args.layout:
        print_all_layouts()
        return 0

    base = BASE_RECIPE.read_text()

    if args.check:
        print_all_layouts()
        print()
        failures = check(points, variants, base)
        print()
        if failures:
            print(f"CHECK FAILED: {failures} problem(s)")
            return 1
        n_var = sum(len(s) for s in variants.values())
        n_watchdog = sum(longer_watchdog(i) is not None for i in points.values())
        print(f"CHECK PASSED: all 11 recipes match their Slurm sources (server env "
              f"included, key by key), and all "
              f"{n_var} reduced-node variants match their split"
              + (f" ({len(LONGER_RUNS)} recipes with the --longer-runs "
                 f"n_samples_to_issue, {n_watchdog} of them also with its "
                 f"run_timeout_s override)" if args.longer_runs else ""))
        return 0

    # Render and verify EVERYTHING first; write only if all of it passes.
    outputs = []
    for point, info in points.items():
        outputs.append((point, info, out_path(point), render(point, info, base),
                        LAUNCH_TEMPLATE.format(point=point)))
    for nodes, splits in variants.items():
        for point, ranked in splits.items():
            info = points[point]
            outputs.append((point, info, variant_out_path(point, nodes),
                            render_variant(point, info, base, nodes, ranked),
                            render_variant_launch(point, info, nodes, ranked)))
    for point, info, path, text, _ in outputs:
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise SystemExit(f"{path.relative_to(HERE)}: rendered recipe is not "
                             f"valid YAML ({exc}); nothing written") from None
        problems, _ = verify_env(point, info, doc)
        if problems:
            raise SystemExit(f"{path.relative_to(HERE)}: rendered server env does "
                             f"not match the drop; nothing written:\n  "
                             + "\n  ".join(problems))
    for point, info, path, text, launch_text in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        launch = path.parent / "launch_test.sh"
        launch.write_text(launch_text)
        launch.chmod(0o755)
        print(f"wrote {path.relative_to(HERE)}  and  "
              f"{launch.relative_to(HERE)}")
    print()
    print_all_layouts()
    return 0


if __name__ == "__main__":
    sys.exit(main())
