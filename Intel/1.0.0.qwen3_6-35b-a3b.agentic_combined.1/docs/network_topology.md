# Client-to-server topology and baseline latency

The client and serving stack share one system with two Intel Xeon 6980P
processors. All containers use Docker host networking.

Path: client → `http://localhost:30000` (Envoy) → vLLM on
`127.0.0.1:8870`, `:8871`, or `:8872`. Envoy consults llm-d over local
gRPC on port 9002. The interconnect is IPv4 loopback through the Linux host
network stack, with **zero external network hops** and one local HTTP proxy.
There are three TP2 replicas on NUMA groups `0,1`, `2,3`, and `4,5`.

## Measured baseline

On 2026-10-26 at 23:49:17 UTC (submission host clock), 20 sequential
HTTP/1.1 `GET /health` requests through the benchmark endpoint all returned
HTTP 200 at `127.0.0.1`.

| Routed health-check round-trip latency | Milliseconds |
| --- | ---: |
| Minimum | 2.142 |
| Median | 2.600 |
| Maximum | 5.152 |

Each request used a fresh TCP connection and fixed `X-Session-ID`.
Curl's `time_total` includes connection setup, routing, and the vLLM health
handler, excluding process/container startup. Readiness probes are excluded;
all 20 measured samples, including the first, are retained.

This **idle baseline was measured after submission**, using the restored C8
runtime and no token-generation requests. All seven points share the recorded
endpoint, routing images, Envoy concurrency, and three-TP2 topology; latency
was measured on C8 only.

[Raw samples](network-latency/samples-20261026T234917Z.csv) contain timing
values in **seconds**.
[Metadata](network-latency/metadata-20261026T234917Z.json) records image IDs,
source revision, checkpoint provenance, configuration hashes, and the CSV hash.

## Measurement configuration and reproduction

Copy the submitted source at the revision recorded in the metadata.
Set the copy's `configs/vllm-cpu.conf` `VLLM_CPU_IMAGE` to the recorded C8
runtime image ID. Use the existing INT8/FP8 checkpoint and start the launcher:

```bash
cd /path/to/copied/endpoints_agentic
STACK_PROFILE=qwen-vllm-llm-d \
HF_HOME="$HOME/.cache/huggingface" \
MODEL_LOCAL_PATH=/path/to/calibrated-model \
WEIGHT_DTYPE=int8 KV_CACHE_DTYPE=fp8 MTP_ENABLED=false \
TARGET_CONCURRENCY=8 \
OUTPUT_ROOT=/path/to/new/baseline-output \
bash start_server.sh
```

The capture used 64 GiB KV cache per rank and Envoy concurrency 4.
Generated Envoy/EPP/endpoint configurations matched the original C8 files
byte-for-byte. After routed readiness, run the probe in a second terminal:

```bash
docker run --rm --network host --ipc host --user "$(id -u):$(id -g)" \
  --entrypoint bash \
  sha256:d02c2a5ededb09498e50ba8bbd70a1bb87fcd473fc6d0ed5a95562001abcfc4f \
  -c 'set -euo pipefail
printf "request,http_status,remote_ip,time_connect_s,time_starttransfer_s,time_total_s\n"
for request in $(seq 1 20); do
  printf "%s," "$request"
  curl --noproxy "*" --http1.1 --fail --silent --show-error \
    --connect-timeout 2 --max-time 5 \
    -H "X-Session-ID: issue116-baseline-20261026T234945Z" \
    --output /dev/null \
    --write-out "%{http_code},%{remote_ip},%{time_connect},%{time_starttransfer},%{time_total}\n" \
    http://localhost:30000/health
done' > samples.csv
```

The probe used `endpoints-agentic-client:4235a9c` with curl 8.14.1;
the historical client image was not verified.
Multiply `time_total_s` by 1,000 for milliseconds; summarize all 20 samples
without dropping failures. Stop the launcher with Ctrl-C afterward.
