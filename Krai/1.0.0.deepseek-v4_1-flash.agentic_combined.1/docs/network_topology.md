# Client-to-server network topology (§2.1.1, Client on Prem)

- The reference client and the SUT run on the same node. All containers use host networking.
- Path: client → `http://localhost:8000` (nginx reverse proxy, `proxy_buffering off`) → vLLM on `127.0.0.1:9000`.
  nginx pins each conversation to one data-parallel rank (`X-Session-ID` → `X-data-parallel-rank`); single-rank
  points map every session to rank 0. The SWE-bench service listens on `127.0.0.1:18080`.
- No network hops; one in-host HTTP proxy, the same at every point.
- Measured baseline: before each run, 20 sequential `GET /health` requests through the router
  (`server_configs/net_baseline_s.txt`, seconds per round trip including connection setup): about 1–4 ms,
  median about 1.8 ms.