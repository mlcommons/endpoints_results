# Intel CPU profiles

This guide covers the stock CPU vLLM profiles, model calibration, CPU
placement, and prefill/decode serving. The shared launch and campaign workflow
is in the [root README](../../README.md).

## Profiles

The four profile names remain stable:

| Profile | Runtime and routing |
| --- | --- |
| `qwen-vllm-llm-d` | Stock CPU vLLM with llm-d routing; three TP2 replicas. |
| `qwen-vllm-vllm-router` | Stock CPU vLLM with vLLM Router; three TP2 replicas. |
| `qwen-vllm-llm-d-pd` | CPU vLLM prefill/decode workers with llm-d routing. |
| `qwen-vllm-vllm-router-pd` | CPU vLLM prefill/decode workers with vLLM Router. |

Select one explicitly for a run:

```bash
STACK_PROFILE=qwen-vllm-llm-d bash run_submission.sh
```

Set `VLLM_CPU_IMAGE` in `configs/vllm-cpu.conf`. Preparation reuses the local
image tag and pulls only when absent. Workers use the resolved image ID.

The regular profiles serve `Qwen/Qwen3.6-35B-A3B-FP8`. The two routing
choices are independent of the vLLM worker implementation. The released
`vllm-router==0.1.15` drops assistant `reasoning_content` when it deserializes
and reserializes multi-turn chat history. This changes the prompt received by
vLLM and affects inline accuracy. Upstream
[router PR #194](https://github.com/vllm-project/router/pull/194) fixes the
round trip; check release availability before changing the pin.

## Model calibration

The regular CPU FP8 profiles use
`Qwen/Qwen3.6-35B-A3B-FP8` unchanged by default. They also accept
case-insensitive preparation settings:

```bash
WEIGHT_DTYPE=int8 KV_CACHE_DTYPE=fp8 bash preflight.sh
WEIGHT_DTYPE=int8 KV_CACHE_DTYPE=fp8 bash start_server.sh
```

With `WEIGHT_DTYPE=int8`, preparation downloads the BF16 source
`Qwen/Qwen3.6-35B-A3B` at its approved revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`. Campaigns require that SHA.
Preparation uses `hf download`; calibration can obtain the source through the
prepared runtime image, so it need not already be cached.

`KV_CACHE_DTYPE=fp8` generates ten calibration samples from the pinned performance
dataset and caches them under `HF_HOME/endpoints-agentic/calibration_corpora/`
for reuse on subsequent calibrations.

`WEIGHT_DTYPE` defaults to `fp8`. `int8` directly produces channelwise W8A8
target and `mtp/` draft weights with dynamic tokenwise activations from the
BF16 source. `KV_CACHE_DTYPE` defaults to `auto`; `fp8` calibrates static
tensorwise scales from ten generated samples. Generated checkpoints use
compressed-tensors; FP8 outputs use `FP8_BLOCK`. Ignored weights remain BF16.
This BF16-source INT8 path is supported by the regular CPU profiles.

Checkpoints are stored under `HF_HOME`. Reuse checks the requested weight and
KV quantization in `config.json` through library APIs, plus MTP metadata when
enabled. Generated INT8 checkpoints also require a matching calibration
provenance manifest for reuse. An explicit `MODEL_LOCAL_PATH` keeps the
existing config-based checks and does not require that manifest. vLLM handles
shard and tensor compatibility. Generated outputs retain tensor
verification, a manifest, and `COMPLETE`; incompatible destinations are
never overwritten automatically.

After validation and publication of the INT8 checkpoint, calibration removes
the original BF16 source revision from the Hugging Face cache with
`hf cache rm <source_revision> --yes --cache-dir <HF_HOME>/hub`. The revision
hash selects that cached snapshot; passing a repository ID selects the
whole repository. If calibration fails, the source is retained. If cache
cleanup fails, the published output remains usable and the next preparation
retries cleanup without requantizing. Reuse and server startup do not require
the BF16 source.

Allow about 112 GiB free on the selected Hugging Face cache filesystem for
fresh preparation: roughly 67 GiB for the BF16 source plus the 45 GiB output
reserve. If the complete source is already cached, allow 45 GiB free for the
output.

The runtime leaves vLLM's KV-cache dtype on `auto`. Stock checkpoints without
an FP8 KV scheme use BF16 KV; calibrated checkpoints declare their FP8 scheme
and scales, which vLLM selects automatically.

### MTP

MTP remains opt-in and uses a separate BF16 draft KV-cache group; no MTP KV
calibration is needed. Enable it with `MTP_ENABLED=true`:

```bash
MTP_ENABLED=true bash preflight.sh
MTP_ENABLED=true bash start_server.sh
```

The default `fp8`/`auto` configuration uses the target checkpoint's MTP
weights directly. When calibration and MTP are both requested with
`WEIGHT_DTYPE=int8` or `KV_CACHE_DTYPE=fp8`, the server uses the generated
checkpoint's `mtp/` draft. The launcher enables one-token MTP with BF16 draft
KV and the experimental `disable_eagle_block_drop` setting; validate accuracy
for that configuration.

## Cache sizing and CPU placement

With no explicit override, the CPU launcher requests
`max(384 GiB, ceil(384 GiB * effective target concurrency / 100))`.
The 384 GiB minimum preserves cache capacity for long SWE-bench contexts
at concurrency 100 or below; higher concurrency increases the pool.
The effective target concurrency is
`TARGET_CONCURRENCY` when set, otherwise the benchmark YAML value. It is
configured trajectory concurrency, not observed in-flight HTTP request
concurrency. The regular replica pool divides the requested pool across its
workers, then rounds each per-worker aggregate up to a multiple of its TP
width, so the effective total may exceed the requested pool. vLLM receives
the resulting per-rank value (`aggregate / TP`). For three TP2 replicas,
the minimum is 128 GiB per worker and 64 GiB per rank.

Prefill/decode profiles allocate a separate requested pool to each stage.
With no overrides, their combined effective total is at least 768 GiB.

Set `VLLM_CPU_KVCACHE_SPACE_GIB` for an explicit aggregate-per-worker
override. It takes precedence over adaptive sizing and profile overrides,
must be a positive integer, and must be divisible by every worker's TP width.
Without that global override, a profile may set
`CPU_KV_CACHE_PER_WORKER_GIB` for regular replicas,
`PREFILL_KV_CACHE_PER_WORKER_GIB` for prefill workers, or
`DECODE_KV_CACHE_PER_WORKER_GIB` for decode workers. These are profile-level
aggregate GiB-per-worker values, not per-rank values; an unset role remains
adaptive. Explicit overrides may allocate less than the automatic minimum.

Generic placement discovers online CPU-bearing NUMA nodes, sorts their
numeric IDs, and assigns the first required nodes in worker order. Regular
profiles require `TP_SIZE * REPLICA_COUNT` nodes. Each tensor-parallel rank
uses one placement slot, and startup fails before workers launch if there is
not enough capacity. Extra NUMA nodes remain unused. Worker ports are assigned
sequentially starting at 8870; the router listens on loopback port 30000.

## Prefill/decode profiles

The two prefill/decode profiles have separate topology settings for each
stage. `qwen-vllm-llm-d-pd` ships with three TP1 prefill workers and three TP1
decode workers. `qwen-vllm-vllm-router-pd` ships with two TP2 prefill workers
and one TP2 decode worker. Each worker consumes one NUMA placement slot per
TP rank. The llm-d layout therefore uses six nodes; the vLLM Router layout
uses six nodes as well. Stage replica counts can be changed in the selected
profile when the host has enough capacity.

Select either topology explicitly:

```bash
STACK_PROFILE=qwen-vllm-llm-d-pd bash run_submission.sh
STACK_PROFILE=qwen-vllm-vllm-router-pd bash run_submission.sh
```

Qwen's NIXL Mamba transfer requires the DS convolution-state layout. The
prefill/decode adapter checks this image capability before starting model
workers and rejects images that force the SD layout.

For llm-d prefill/decode, the raw vLLM decode worker is health-checked on
port 8872, while the routed decode endpoint is the sidecar on port 9872.
Clients use the public Envoy endpoint. An EPP or Envoy exit fails the serving
session. The NIXL KV lease is set to 300 seconds because CPU transfer can
exceed vLLM's 30-second default under concurrent load; transfer failures
remain fail-closed.

## Campaign model revisions

The approved repository checkpoints used by these profiles are:

| Model | Approved revision |
| --- | --- |
| `Qwen/Qwen3.6-35B-A3B` | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| `Qwen/Qwen3.6-35B-A3B-FP8` | `95a723d08a9490559dae23d0cff1d9466213d989` |

Campaign points select the approved revision automatically and reject an
explicit `MODEL_REVISION` mismatch. A local checkpoint has no approved
repository SHA.
