> Site-specific names, paths and addresses are replaced with `<placeholders>`.

# GPT-OSS-120B — MLPerf Endpoints v1.0.3 on GKE (GB300 NVL72 / A4X-max)

This is the GKE sflow port of the NVIDIA Slurm recipe
`code-drop-nvidia/nv-mlpinf-partner-partner-endpoints-v1.0.3/NVIDIA/src/gpt-oss-120b`.
It covers all eleven approved concurrency points:
**C17, C35, C70, C140, C272, C544, C1024, C2048, C3840, C7168, C22528**.

Each point uses 72 GB300 GPUs on **18 nodes** (the Slurm-faithful topology), with
disaggregated TRT-LLM.

The generator is a copy of the v1.0.2 one
([v1.0.2/gpt_oss_120b](../../v1.0.2/gpt_oss_120b/README.md)) with the default drop set to
v1.0.3. In v1.0.3, NVIDIA raised `settings.timeouts.run_timeout_s` in `client.yaml` from
`9000.0` to `16200.0` on **C17, C35, C70, C140**, so C17's `--longer-runs` sample count
(`25584` samples, predicted client run `11,877 s`) now fits natively under NVIDIA's
`16,200.0 s` watchdog (`11,877 / 16,200 = 73.3 % < 85 %`) without needing a GKE
`run_timeout_s` override (`LONGER_RUNS_OVERRIDES` is now empty). All server YAMLs,
`server-env.yaml`, topologies, sample counts and images are byte-identical to v1.0.2.
The GKE layer in
[recipe_base.yaml](recipe_base.yaml) is otherwise unchanged: the `k8s_mpi` operators, the
ComputeDomain, the UCX layer, the atomic hang-detector patch, the request timeout, the
client fixes, `--longer-runs` and the three DaemonSets all behave as in v1.0.2. Follow
[SLURM_TO_GKE.md](../../SLURM_TO_GKE.md) for the full procedure. The
[v1.0.2 README](../../v1.0.2/gpt_oss_120b/README.md),
[v1.0.1 README](../../v1.0.1/gpt_oss_120b/README.md) and the
[v1.0.0 runbook](../../v1.0.0/gpt_oss_120b/README.md) still apply for prerequisites,
validation and traps.

---

## 1. What changed from v1.0.1

All of these come from `diff -r` of the two drops' `gpt-oss-120b/` trees.

| Change in the drop | Points | v1.0.1 → v1.0.2 | In the GKE recipes |
|---|---|---|---|
| `server-env.yaml` gains `TRTLLM_ENABLE_PDL: '0'` | **C2048** | absent → `'0'` (PDL off) | **Ported**, C2048 only: `export` plus `mpirun -x`, in `prefill_server` **and** `decode_server` (§2) |
| `stream_interval` | **C2048** | 100 → **30** (`server-ctx` **and** `server-gen`) | Carried as-is (the server YAMLs are embedded byte-for-byte) |
| Default `MODEL_PATH` | all | `${SCRATCH_DIR}/models/gpt-oss/gpt-oss-120b` → `${SCRATCH_DIR}/models/gpt-oss-120b` | Not ported: GKE reads the node-local SSD. Only the `MODEL_PATH` description changes |
| `_shared/`: `SITE_ENV_YAML` overlay | all | new; default `''`. `run_with_env.py` merges it **after** `server-env.yaml`, and `snapshot_configs` copies it when it is set | Not ported. The generator **refuses a non-empty default**, because its keys would bypass the server-env check |
| README: checkpoint pin | — | none → `openai/gpt-oss-120b@b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, `--exclude 'original/*' --exclude 'metal/*'` | Already what [stage-assets.yaml](stage-assets.yaml) stages; comments updated |

Nothing else changed. Topology, `n_samples_to_issue`, `max_new_tokens`, `client.yaml`,
`endpoint_env.yaml` and the three images are identical to v1.0.1. So are the other ten
points' server YAMLs and `server-env.yaml`. The v1.0.1 topology table
([v1.0.1 README §2](../../v1.0.1/gpt_oss_120b/README.md#2-topology-per-point)) applies
unchanged.

Both server-config changes are at **C2048**. In v1.0.1, attempt 1 at that point died with a
device-side assert in sampling, and its accuracy was low
([v1.0.1 README §6](../../v1.0.1/gpt_oss_120b/README.md#6-results-gke-guide-112)). Whether
v1.0.2 fixes either problem is what the pending C2048 validation run will measure. Nothing
here claims it does.

## 2. The server env: every key is decided (`SERVER_ENV_KEYS`)

The v1.0.1 generator never read `server-env.yaml`. Its model/runtime exports were
hard-coded in `recipe_base.yaml`. With v1.0.2 it would have dropped C2048's
`TRTLLM_ENABLE_PDL` **silently**. The v1.0.2 generator reads every point's
`server-env.yaml`, and the allowlist `SERVER_ENV_KEYS` decides each key:

| Key | Drop value (points) | Decision | In the GKE recipe |
|---|---|---|---|
| `TRTLLM_ENABLE_PDL` | `'0'` (**C2048 only**) | **port** (allowed values `0`, `1`) | C2048 only: `export TRTLLM_ENABLE_PDL=0` + `-x TRTLLM_ENABLE_PDL=0` in prefill **and** decode, plus a `GKE_SERVER_ENV …` echo in the server log. Absent at the other ten points |
| `TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER` | `'1'` (all) | base | `recipe_base.yaml` export + `-x …=1`; asserted equal to the drop |
| `TRTLLM_SERVER_DISABLE_GC` | `'1'` (all) | base | same |
| `TRTLLM_WORKER_DISABLE_GC` | `'1'` (all) | base | same |
| `HF_HOME` | `/tmp/hf_cache_gptoss` (all) | base | export + `-x HF_HOME` |
| `HF_HUB_CACHE` | `/tmp/hf_cache_gptoss/hub` (all) | base | same |
| `TRANSFORMERS_CACHE` | `/tmp/hf_cache_gptoss` (all) | base | same |
| `XDG_CACHE_HOME` | `/tmp/hf_cache_gptoss` (all) | base | same |
| `UCX_CUDA_IPC_ENABLE_MNNVL` | `n` (all) | gke-ucx (**not ported**) | The GKE UCX layer sets `y`. `n` is right on RoCE but catastrophic on GKE ([SLURM_TO_GKE.md](../../SLURM_TO_GKE.md) §5.6, F2) |
| `UCX_RNDV_SCHEME` | `put_zcopy` (all) | gke-ucx (**not ported**) | The GKE UCX layer sets `put_zcopy` |

What the generator enforces, before it writes anything:

- **Fail closed.** These all stop the generator, and every file is rendered and verified
  before any is written: a key not in the table, a value outside the key's `values`, a
  `bool`/`float`/`null` value, a duplicate key, a "base" key missing from a point, a
  "port" key appearing at a point or with a value that `EXPECTED_PORTED_ENV` does not list,
  `TRTLLM_ENV_YAML_CTX`/`_GEN` naming any file other than the point's own
  `server-env.yaml`, or a non-empty `SITE_ENV_YAML` default.
- Errors name the key, never the value of an unknown key, and never quote the file.
  An env file may hold a credential.
- `validate_server_env_table()` checks the table itself. Every allowed value must be a plain
  token (`[A-Za-z0-9_./:-]+`: no `$`, no quoting). A `UCX_*` key can only be `gke-ucx`.
- A ported key goes into **both** server roles as an `export` (for the ranks mpirun starts
  in the pod) **and** an `mpirun -x KEY=VALUE` (MPI forwards only `-x` variables). C2048's
  TP8 decode servers span two nodes, so without the `-x` the second node's ranks would
  never see the key.
- The DSR1 GKE tuning env stays absent at every point: `TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED`,
  `NCCL_GRAPH_MIXING_SUPPORT` and `TLLM_NUMA_AWARE_WORKER_AFFINITY`. So does PDL at every
  point except C2048. Note that C2048's `0` is the **opposite** of DSR1's
  `TRTLLM_ENABLE_PDL=1`. Each recipe's `prefill_server` comment now says this for its own
  point.

`--check` re-parses every recipe and rebuilds the prefill and decode env from the top-level
`export`s and the `mpirun -x` list. For each allowlisted key that the point sets, it
requires exactly one export with the drop's value, exactly one `-x` that forwards that
value, and no other assignment. A "port" key that the point does not set must appear
nowhere in either script. The UCX keys must carry the GKE layer's values.
`TRTLLM_IMAGE`/`CLIENT_IMAGE`/`LCB_IMAGE` must default to the drop's images, and every
script item must be a string. This adds one `[ ok ]` line per recipe: 99 checks, versus
88 in v1.0.1.

## 3. What is here / regenerate

```
gpt_oss_120b/
  README.md  recipe_base.yaml  generate_point_recipes.py
  stage-assets.yaml  mk-report-dir.yaml  prepull-images.yaml
  18nodes/point_<C>/{gpt_oss_120b_gke_disagg_point<C>.yaml, launch_test.sh}   # generated
```

```bash
./generate_point_recipes.py                 # rewrite the eleven 18-node points (v1.0.2 drop)
./generate_point_recipes.py --check         # 99 checks against the v1.0.2 drop
./generate_point_recipes.py --layout        # node/GPU layout table
./generate_point_recipes.py --longer-runs   # the RERUN recipes (C17/35/70/140/544/22528)
./generate_point_recipes.py --variants 16   # also write 16nodes/ variants
```

The generator needs only PyYAML. It runs no subprocess and makes no network access. Run it
with `PYTHONDONTWRITEBYTECODE=1 python3 -B` to leave no `__pycache__`.

`--check --longer-runs` checks recipes that were generated **with** `--longer-runs`. On the
default tree it reports the 25 expected `n_samples_to_issue`/`run_timeout_s` differences,
exactly as in v1.0.1. The `--longer-runs` output was validated in a scratch tree: generate,
then `--check --longer-runs` passes with C17's `run_timeout_s` at 14 400 s and the 85 %
guard. Its v1.0.1 → v1.0.2 diff is line-for-line the same as in default mode.

How the generated recipes differ from v1.0.1:

- **The ten points other than C2048.** Header comments, the `MODEL_PATH` description and
  the `prefill_server` env comment change. In parsed YAML the **only** change is
  `variables.MODEL_PATH.description`. In `launch_test.sh` only the header line changes.
- **C2048.** The same changes, plus `stream_interval: 30` in `SERVER_CTX_YAML` and
  `SERVER_GEN_YAML`. `prefill_server` and `decode_server` each gain
  `export TRTLLM_ENABLE_PDL=0`, its echo line and `-x TRTLLM_ENABLE_PDL=0`.

## 4. Prepare and run

Prerequisites are the same as in
[v1.0.0 README §2](../../v1.0.0/gpt_oss_120b/README.md#2-prerequisites). Assets already
staged for v1.0.1 are valid as-is: same checkpoint revision, same datasets, same images.

```bash
kubectl apply -f stage-assets.yaml && kubectl rollout status ds/stage-gptoss-assets --timeout=60m
kubectl apply -f mk-report-dir.yaml
kubectl apply -f prepull-images.yaml

cd 18nodes/point_2048
SKIP_NODE_CHECK=1 ./launch_test.sh --dry-run      # ✓ Configuration is valid
mkdir -p sflow_output
setsid nohup ./launch_test.sh > sflow_output/launch-$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
```

On a C2048 run, check that both server roles got the ported key:
`grep GKE_SERVER_ENV` in the prefill and decode logs must show `TRTLLM_ENABLE_PDL=0`.

Sweep, from `kubernetes/`:

```bash
G=v1.0.2/gpt_oss_120b/18nodes; mkdir -p $G/sweep
setsid nohup tools/run_sweep.sh $G > $G/sweep/sweep-$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
```

Validate each run with
[SLURM_TO_GKE.md §10](../../SLURM_TO_GKE.md#10-phase-8-validate-every-run).

## 5. Provenance (v1.0.2 inputs)

| Input | Pinned value |
|---|---|
| NVIDIA drop | `nv-mlpinf-partner-partner-endpoints-v1.0.3`, `NVIDIA/src/gpt-oss-120b/configs/GB300-NVL72_GB300-288GB_aarch64x72_TRTLLM/` + `configs/_shared/` |
| NVIDIA's pinned sflow runner | `nv-sflow@b8afb5b86b3f46d9fa771531990fc7afaefc69b9` |
| sflow used here (dry-run) | `0.3.1.dev13+gc98458cc9` = `NVIDIA/nv-sflow@c98458cc9a8fc129a535c28b2f20eef476b3a854` |
| TRT-LLM image | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14`, arm64 manifest `sha256:e71eb8056ac1c717e6b727f9d66ebba9442e95f7ebc3514007b541046e84b088` |
| Client image | `ghcr.io/mlcommons/endpoints:e9f427c` = `mlcommons/endpoints@e9f427ce74fd90e7fafdbc77ee7746cc00245692` |
| LiveCodeBench image | `ghcr.io/mlcommons/endpoints:9d749906-livecodebench` |
| Checkpoint | `openai/gpt-oss-120b@b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, HF layout only (`original/`, `metal/` excluded), the v1.0.2 README's pin. Staged from `<bucket>/models/gpt-oss-120b`: 27 objects, 65 276 859 525 B. Per-file sha256 in `<bucket>/checksums/gpt-oss-120b.SHA256SUMS` |
| Perf dataset | `v4/perf/perf_eval_ref.parquet`: 192 191 924 B, sha256 `be62d3f4a9981e74a3f3c184baeacae7cac7a282c4bb9c01c5542a19f06edf78`, MD5 `e4cd6cef6dd975f3e50c85b3279b358b` (official copy). The staged GCS copy matches it by MD5, size and CRC32C. Its sha256 was not recomputed here |
| Harmony cache | `harmony_cache_gptoss/` from `setup_harmony_cache.sh` (openai-harmony 0.0.8). o200k_base blob `fb374d419588a4632f3f557e76b4b70aebbca790` |
| GPQA cache | `endpoint_accuracy/gpqa/diamond/gpqa_diamond.parquet`, built by the pinned client from gated `Idavidrein/gpqa@83022cefff930aea54f654c0b282e74b9eeda5c6`. 198 rows, sha256 `f9c0206a7446016a32fc4fe9dcddeb3f57823c7b0bb66b756b7dfb33d0e7375a` |

## 6. Status

| Step | State |
|---|---|
| Generator `--check` (v1.0.2 drop) | ✅ CHECK PASSED: 99 checks, 11/11, server env key by key |
| `--longer-runs` → `--check --longer-runs` (scratch tree) | ✅ CHECK PASSED |
| Fail-closed tamper tests (scratch) | ✅ 25/25 refused or flagged |
| `--dry-run` (offline: no network namespace, kubectl/gcloud blocked) | ✅ 11/11 `✓ Configuration is valid` |
| Cluster runs | ⏳ none yet for v1.0.2. C2048 validation is scheduled first |

## 7. Results

None yet for v1.0.2. The v1.0.1 results
([v1.0.1 README §6](../../v1.0.1/gpt_oss_120b/README.md#6-results-gke-guide-112)) were
measured with the v1.0.1 drop. For C2048 they are **not** v1.0.2 results, because PDL and
`stream_interval` differ. For the other ten points the server configs are identical, but
they were still run from the v1.0.1 recipes.
