# r40 server_configs

- Point: concurrency 40, low concurrency, `--mode both`: performance, inline accuracy and SWE-bench.
- Server: TP8 x DP1, expert parallelism on (EP group = all 8 GPUs), DSpark k=5 (adaptive verification off), 613 trajectories.
- Times: server ready 2026-10-10 14:25:17 UTC; client started 2026-10-10 14:25:22 UTC (the client writes `test_started_at: 0`, so these are the wall-clock start).
- Gates: inline accuracy 53.35% (report.txt), full-run OSL 850.8 (result_summary.json), SWE-bench 99.0%.

| File | Where it comes from | What it is for |
|---|---|---|
| `server_config_resolved.yaml` | `prepare_runs.py`: the server config template with this point's values filled in, as launched | All vLLM flags and env for this point. Defaults not listed are vLLM v0.30.0's. |
| `bench_config_resolved.yaml` | `prepare_runs.py`: the client config template with this point's concurrency and trajectories | Input to `inference-endpoint benchmark from-config`. The client's resolved copy, with its defaults, is `../config.yaml`. |
| `model_server_tp8_dp1_nr1_0.log` | stdout/stderr of the vLLM server container, launch to shutdown | Evidence for the server side; see below. |
| `provenance.txt` | written by `param_search_multireplica_mlperf.sh` once the server is healthy, before the client starts; one line appended after the run | Host and time; container image lines (left empty: `docker image inspect` failed, the image IDs are in `docs/software_disclosure.md`); `nvidia-smi` (GPU model, driver, 700 W power limit, max SM clock); the client tokenizer's `SHA256SUMS` check; the SWE-bench Verified Hugging Face snapshot, before and after the run. |
| `net_baseline_s.txt` | same script: 20 sequential `GET /health` requests through the nginx router, before the client starts | §2.1.1 measured baseline latency: seconds per round trip, connection setup included. |

Useful lines in `model_server_tp8_dp1_nr1_0.log`:

- `non-default args:`: the arguments vLLM parsed, including `revision` and `speculative_config`.
- `snapshots/dba1be0…`: the checkpoint revision loaded.
- `[EP Rank r/8] … Local/global number of experts`: expert placement when expert parallelism is on.
- `Capturing CUDA graphs`: engine start-up, finished before the server reported healthy (time above).
- `Avg prompt tput … Avg generation tput … GPU KV cache used`: 10 s server throughput and KV usage.
- `SpecDecoding metrics`: 10 s speculative-decoding acceptance (DSpark points).

Not in this directory:

- The nginx config is `src/gentic/src/e2e_benchmarking/configs/nginx_dp_router.conf`, rendered with an equal `split_clients` share per DP rank.
- The client's own outputs are one level up: `config.yaml`, `report.txt`, `performance/result_summary.json`, `accuracy/` (mandatory points only) and `events.jsonl` (the run archive only).
- To reproduce this point, see `src/gentic/README.md`: `entry_local.sh --config server_config_resolved.yaml bench_config_resolved.yaml`.
