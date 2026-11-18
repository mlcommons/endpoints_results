# Shared Qwen runtime

Point folders own `run.sh`, `client.sh`, `server.sh` and all three YAML files.
They reference these common service launchers through `QWEN_SHARED_DIR`:

| File | Responsibility |
| --- | --- |
| `dynamo-server.py` | Start four TP1 workers and a session-affine frontend, check readiness, record topology, and stop owned processes. Child output is drained into memory first, so a stalled log file system cannot block a child. A failed log sink (log file or stdout) is reported on stderr, the server keeps running, and the launcher exits 3 when it shuts itself down. |
| `dynamo_cpu_affinity.py` | Assign disjoint GPU-local CPU sets from the allocation's actual NUMA topology. |
| `colocate-client.yaml` | Optional performance-only client placement on the server allocation. |
| `colocated-client.py` | Use the server's reserved client CPUs and adapt the runtime Endpoints worker settings. |
| `coallocate-client.yaml` | Optional two-node server allocation that places the client on its second node. |
| `swe-service.sh` | Bind the service to its allocated client node. |
| `swe-service.py` | Start an installed SWE service, publish its local URL/token, and shut it down. |
| `cuda-compat.sh` | Wrap the server launch: put the image's CUDA forward-compat libcuda first only when the host driver is older. |
| `gpu-mem-clock.sh` | VR-NVL72 DRAM clock lock (`GPU_MEM_CLOCK_LOCK`, 4752 MHz by default; empty skips it): a separate job step locks the memory clock, records it every 30 s and resets it when the job ends. |
| `env/VR-NVL72.env` | Server environment shared by every VR-NVL72 point (`SERVER_ENV_COMMON`); each point's `SERVER_ENV` adds its own lines. |
| `autotune/<SYSTEM>.json` | Pinned FlashInfer autotune file per system; points pass it as `VLLM_FLASHINFER_AUTOTUNE_FILE`. |

The server environment already contains the pinned Dynamo integration and engine
packages. `SWE_SERVICE_PYTHON` selects the installed SWE service environment.
The model [README](../../README.md) lists the cluster prerequisites.

Package installation, Enroot transport, registry authentication and storage are
site-specific; a site prepares those environments and invokes these same runtime launchers.
