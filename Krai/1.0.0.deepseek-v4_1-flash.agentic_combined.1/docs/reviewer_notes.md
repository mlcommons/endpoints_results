# Notes for reviewers

DeepSeek-V4.1-Flash agentic, Standardized / Client on Prem, one HPE Cray XD670 (8x H200-SXM-141GB);
reference client `d3c57f2`, vLLM v0.30.0, checkpoint `dba1be0`.

## Parallelism and maximal engagement (§4.5.3)

- Every point uses exactly the node's 8 GPUs: TP8 x DP1 / TP2 x DP4, with expert parallelism over the same 8 GPUs
  (each GPU holds one eighth of each MoE layer's experts; the server log shows `[EP Rank r/8]`).
- §4.5.3's `A_replica = TP x PP x EP` counts those GPUs twice (TP8 x EP8 = 64; TP2 x EP8 = 16 per replica, 64 with
  DP4) against 8 provisioned. `system_desc.json` therefore states TP/PP/EP/DP in `config_summary` only, and
  engagement needs manual verification. Regardless, every point engages all 8 provisioned accelerators, 
  which meets the rule’s intent.

## Warmup (§6.3)

None. Each point's `point.yaml` `warmup` block declares zero warmup requests and the initialization steps (§6.3.3).

## Speculative decoding (§2.9.4)

DSpark, the native head shipped in `dba1be0`, with the same configuration at every point (`point.yaml`):
k=5, probabilistic drafting, block (lossless) verification, adaptive verification off.

## Tokenization (§2.8)

- The approved revision `dba1be0` ships no `chat_template.jinja`; the next revision, `2cba9e4`, adds only that file.
- The client therefore uses `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` from `2cba9e4`,
  unmodified and vendored with SHA-256 hashes in `src/gentic` (checked at every run, `provenance.txt`). The
  server loads `dba1be0`; on 1,743 real prompts the client's prompt token IDs matched vLLM's encoder exactly.
