> Site-specific names, paths and addresses are replaced with `<placeholders>`.

# Kimi-K3 v1.0.3 SWE-bench on GKE: 1-node pilot and production runs

On Slurm, the drop runs Kimi-K3's SWE-bench accuracy through `swe_service.py` (or `swe-service.py`). The service
starts a sandbox for each agent step and grading step as an `srun` step with Enroot. GKE has
no Slurm. This directory runs the same service on one GKE node and changes only the launcher.
The pilot (below) tests it on 1 node against a stand-in endpoint (or the gold-patch oracle);
[Production](#production-kimi-accboth-runs) runs it on the client node of each `acc`/`both`
run of the v1.0.3 Kimi SWE recipes (`swe_acc/`, `swe_both/`): node 2 for C12..C80 (`FRONTEND_NODE=2`)
and node 4 for C192..C1024 (`FRONTEND_NODE=4`).

Kept as shipped (with two explicit pilot deviations documented below):
- **Deviation 1 (v1.0.1 SWE launcher scripts in the 1-node pilot with v1.0.3 topology/images):** the
  1-node pilot (`swe_pilot.py`) reads `swe-service.py`, `swe-enroot-transport.py`, and
  `swe_cleanup.py` from `drop_v1.0.1_20260330/NVIDIA/src/kimi-k3/scripts` (byte-identical and
  checked by sha256) so it works offline without git-cloning `mlcommons/endpoints` at start;
- **Deviation 2 (`c97ddfcce` SWE service in pilot with `d3c57f2` v1.0.3 client):** the pilot host pod
  runs pinned `mlcommons/endpoints@c97ddfcce` and its `uv.lock` (`swe_source_c97.tgz`, unchanged
  from v1.0.1) inside the v1.0.3 client image `d3c57f2`, while the benchmark client uses the
  v1.0.3 image `d3c57f2` (`ghcr.io/mlcommons/endpoints:d3c57f2-b3296-arm64@sha256:1523051b26ce2eb44726dc94f2fe61938840a8ecbb64c569fe30ca41b9e15a56`);
- the arm64 SWE images and the v1.0.3 client `d3c57f2`;
- the drop's client config, apart from five pilot edits (see `pilot/client-pilot.yaml`, or six
  when `--num-instances N` is passed). The production SWE recipes (`swe_acc/`, `swe_both/`) use
  the v1.0.3 drop's config (`swebench_service_url`, `swebench_service_auth_token`) with no SWE edits.

Replaced: `srun`, `squeue` and `scancel` are local shims, and Enroot 4.2.1 is installed in
the pod with `enroot+caps`, as on Slurm/Pyxis nodes.

## Pods (namespace `kimi-swe`, all on the client node, hostNetwork)

| Pod | Image | Privileges | What it does |
|---|---|---|---|
| `kimi-swe-host` (pilot) / `kimi-swe-host-<alloc>` (`serve`) | client `d3c57f2` | privileged, starts as root | `swe-host-entry.sh`. Installs the pinned Enroot `.deb`s (`enroot`, `enroot+caps`), creates the dedicated service user `kimiswe` (uid 61000) and runs `swe-service.py` as it with the shims first on PATH. Before the service starts, it checks that no socket on the node belongs to that uid yet, blocks the uid's traffic to the GCE and GKE metadata servers on the node and proves the block with a self-test (see [Security notes](#security-notes)). The service listens on a random 127.0.0.1 port. On pod deletion it stops the service the Slurm way (SIGTERM), then checks the teardown, scans for leaked secrets and removes the block. |
| `kimi-swe-model` | TRT-LLM `1.3.0rc14` (or client `d3c57f2` with `--oracle`) | not privileged, no capabilities, 1 GPU (0 GPU with `--oracle`) | Pilot endpoint: GPT-OSS-120B, TP1, at `127.0.0.1:18000` (Kimi-K3 needs 8+ nodes in v1.0.3, so the pilot tests the SWE path on 1 node). With `--oracle` (test only), runs `oracle/oracle_server.py` as `nobody` with a read-only rootfs to verify the grading harness against public SWE-bench gold patches. |
| `kimi-swe-client` | client `d3c57f2` | non-root (`kimiswe`, uid 61000), no capabilities | The drop's benchmark task with `--mode acc`. It waits for the connection file and the endpoint. |
| `kimi-swe-smoke` | client `d3c57f2` | like the host pod | One-shot check, no token needed (see below). |
| `kimi-swe-collect` (pilot) / `kimi-swe-collect-<alloc>` (`serve`) | client `d3c57f2` | non-root (`kimiswe`), read-only mount | Short-lived helper for `collect`. |

**Node storage:** everything lives on the node's local SSD under
`<node-local-ssd>/kimi-swe/`:
- `tmp/` is mounted as the host pod's `/tmp`;
- `runs/<run-id>/` holds the run: `swe_service/`, `report/`, `client/`, `probe/`, the
  `postcheck*.json` files and the connection file;
- `handoff/<run-id>/swe-connection.json` holds the per-run production connection handoff;
- `enroot-cache/` exists only with `--persist-cache`.

## Files

| File | Role |
|---|---|
| [swe-host-entry.sh](swe-host-entry.sh) | Host pod entry (byte-identical to v1.0.1). Checks every pin, then sets up what the drop expects from a Slurm/Pyxis node: a `/tmp` with the standard 1777 mode, Enroot plus `enroot+caps` (it checks the helpers' file caps), and `/etc/enroot/enroot.conf` with only `ENROOT_RESTRICT_DEV yes`, so sandboxes get a minimal `/dev` and no GPUs or disks. That device hook bind-mounts `/dev/log`, and the pod has no syslog daemon, so the script adds an empty, read-only placeholder in the pod's own `/dev`. Creates the service user `kimiswe` (marked section `service user (CS-NET-001)`) and runs the service as it. On SIGTERM it runs the teardown checks. It holds a node lock (`<state>/host-entry.lock`: one host pod per node, smoke or service) and installs, self-tests and removes the metadata block (marked section `metadata block (CS-NET-001)`, see [Security notes](#security-notes)). |
| [srun](srun) | Accepts only the exact command the drop's transport emits: `srun --overlap --jobid=J -N1 -n1 --nodelist=N --chdir=D --job-name=S CMD`. The job id, node and step must match the pod. `D` must be inside the service's `/tmp/kimi-swe-*`. `CMD` must be `bash --noprofile --norc -c …`, `enroot list -f` or `enroot remove -f pyxis_*`. Then it execs `CMD` with stdout and stderr untouched and the exit code passed through. Everything else exits 2, including `--container-*`. |
| [squeue](squeue), [scancel](scancel) | No-ops. There are no Slurm steps; `swe_cleanup.owned_pids` still finds leftover step processes. |
| [swe_probe.py](swe_probe.py) | Root helpers, run with `python3 -I` from `/`: `conn` (readiness and reset checks; accepts only `http://127.0.0.1:<port>`), `snapshot`, `leakscan` and `postcheck-report`. They never follow symlinks or block on FIFOs in directories the service user can write. `smoke` and `prewarm` run as the service user with the service venv, through the drop-patched transport and the shim. With `--metadata-block` (set by the entry script when the block is on), `smoke` also checks from inside a sandbox that TCP connects to `<ip>:80` and `<ip>:988` are refused. |
| [pilot/swe_pilot.py](pilot/swe_pilot.py) | Driver: `--check` (offline verification of pins, ConfigMap sizes, `CLIENT_IMAGE` `d3c57f2`, `SWE_CONTRACT` against `generate_point_recipes.py`, and rendered `swe_<mode>` recipes; zero `kubectl` calls), plus `configmaps`, `secret`, `smoke`, `model`, `host`, `client`, `status`, `down`, `collect`, and `serve` (production). Add `--dry-run` for server-side validation. `smoke` and `host` accept `--no-metadata-block` (fallback, see below); `host` refuses it with `--handoff` or `--scan-reports`, and `serve` has no such option. |
| [run_swe_sweep.sh](run_swe_sweep.sh) | Production wrapper: runs `swe_pilot.py --check` and `generate_point_recipes.py --swe <mode> --check`, starts `serve` around `tools/run_sweep.sh` (or direct point loop when `SKIP_WAIT_IDLE=1`) for Kimi v1.0.3 `swe_acc`/`swe_both` sweeps, then stops `serve` with SIGTERM. |
| [pilot/client-pilot.yaml](pilot/client-pilot.yaml) | The drop's `CLIENT_CONFIG` rendered for `acc` mode. A test checks that only the five marked edits differ (plus `num_instances` when `--num-instances N` is passed). |
| [pilot/model-extra.yaml](pilot/model-extra.yaml) | TRT-LLM options for the pilot endpoint. Not a submission config. |
| [pilot/oracle/](pilot/oracle) | Test-only gold-patch oracle (`oracle_data.py`, `oracle_server.py`) for `model --oracle` grading verification. |
| `artifacts/` | The pinned downloads (git-ignored, see Provenance). |
| [tests/](tests) | Unit tests: `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests` (203 tests, no cluster access). `test_production.py` also runs the v1.0.3 recipes' SWE wait and redaction blocks against temp dirs and a local 127.0.0.1 server. `test_metadata_block.py` runs the entry script's metadata-block section in bash against a simulated iptables (legacy and nft, IPv4 and IPv6, rule counters), curl, getent and `/proc/net` (the socket check), and its `service user` section and `prepare()` against copies of the image's account files with stubbed `useradd`, `groupadd` and `install` (who owns what). |

## Provenance and pins (sha256)

| Artifact | Source | sha256 |
|---|---|---|
| `enroot_4.2.1-1_arm64.deb` | `https://github.com/NVIDIA/enroot/releases/download/v4.2.1/enroot_4.2.1-1_arm64.deb` | `267018b3…be421` |
| `enroot-caps_4.2.1-1_arm64.deb` | `https://github.com/NVIDIA/enroot/releases/download/v4.2.1/enroot%2Bcaps_4.2.1-1_arm64.deb`, renamed (`+` isn't allowed in a ConfigMap key). Its postinst gives `enroot-aufs2ovlfs` and `enroot-mksquashovlfs` the file capabilities that let the non-root service user import images, as on Slurm/Pyxis nodes. The pin is GitHub's published asset digest | `12b811d0…284a1` |
| `swe_source_c97.tgz` | `src/inference_endpoint/evaluation/swebench_service` from the codeload tarball of `mlcommons/endpoints` at `c97ddfcce80392e9f30b86baabaa998355d4327e`, as `swebench_service/` plus a `REVISION` file (the drop checks it) | `674c82a5…b4de9` |
| `swe-service.py`, `swe-enroot-transport.py`, `swe_cleanup.py` | the v1.0.1 drop (`drop_v1.0.1_20260330/NVIDIA/src/kimi-k3/scripts/`) | see `PINS` in the driver |

The full digests are in `PINS` (driver) and `PINNED` (entry script). `configmaps` and `--check`
refuse to run if the two lists differ or any file doesn't match. The pod checks the pins again,
plus `MANIFEST.sha256`, before it runs anything.

## Steps

From `pilot/`:

```bash
./swe_pilot.py --check                          # offline preflight (pins, ConfigMaps, SWE_CONTRACT, recipes)
./swe_pilot.py configmaps                       # after any change to the files above
./swe_pilot.py smoke --node <idle node>         # no token; ends with SMOKE_OK / SMOKE_FAIL
```

When the GitLab `read_registry` token is available, create the netrc with a hidden prompt.
The token never appears on a command line or in shell history (`printf` is a bash builtin).
The file gets two lines: `gitlab.com` issues registry tokens (used by
`check_gitlab_registry.sh`), and Enroot looks up `registry.gitlab.com`:

```bash
install -d -m 700 <workdir>/.config/nv-partner
read -r -p 'GitLab user: ' GL_USER; read -r -s -p 'GitLab token (read_registry only): ' GL_TOKEN; echo
(umask 077; f=<workdir>/.config/nv-partner/gitlab.netrc; rm -f "$f"
 { printf 'machine gitlab.com login %s password %s\n' "$GL_USER" "$GL_TOKEN"
   printf 'machine registry.gitlab.com login %s password %s\n' "$GL_USER" "$GL_TOKEN"; } > "$f")
unset GL_TOKEN
./swe_pilot.py secret                            # Secret kimi-swe-enroot (stdin only)
```

Then run the pilot (with either the real stand-in model or `--oracle` for a fast 5-task
grading check with `--num-instances 5`):

```bash
./swe_pilot.py model  --node N                   # or: ./swe_pilot.py model --oracle --node N
./swe_pilot.py host   --node N --run-id pilot-1  # waits for SWE_READY
./swe_pilot.py client --node N --run-id pilot-1  # add --follow to stream the log
./swe_pilot.py status
./swe_pilot.py down client                       # after the client has finished
./swe_pilot.py down host                         # SIGTERM teardown + checks (up to 15 min); refused while the client pod runs
./swe_pilot.py collect --node N --run-id pilot-1 --dest <dir>
./swe_pilot.py down all                          # pods and the Secret
```

**Pass criteria:**
- `report/swe_bench_results.json` is complete, with 10 instances submitted (or `N` with
  `--num-instances N`);
- the score appears in the accuracy results;
- the host log shows `service user kimiswe (61000:61000): created in this container`, then
  `metadata block: 0 sockets of uid 61000 in the node's network namespace before the service
  starts (...): ok` and `metadata block: active for uid 61000 (...); self-test passed` before
  the service starts;
- the host teardown log ends with `service run finished: ok`. That means the connection file
  was reset to `{}`, the scratch and all `pyxis_*` containers were removed, the leak scan
  found no service or registry token, and the metadata block was still in place at teardown
  and then removed (`metadata block removed: ...`).

The teardown first kills any `kimiswe` process still alive after the drop's own teardown, as
Slurm does at the end of a step, then counts the entries of the run directory that `kimiswe`
does not own (expected: none). Both are warnings, not failures: the leftover count is logged
and saved as `leftover_processes` in `postcheck.json`.

**Optional, needs approval: prewarm.** The agent's first sandbox step has to import its
image within 330 s (`timeout 300 + 30`). If one import takes longer than about 150 s, start
the host with `--persist-cache --prewarm first:10`. That imports the same refs through the
same transport, ahead of time, into a node-wide cache.

**Smoke test (`SWE_POD_MODE=smoke`):**
- egress to PyPI, Hugging Face, GitLab and mirror.gcr.io;
- the drop's full service lifecycle (`uv sync`, start, `/health`, SIGTERM, exit 143, teardown);
- a sandbox round trip on `debian:trixie-slim` through the drop transport and the shim:
  agent-style create and execute, exit code and stdout passthrough, private PID namespace,
  restricted `/dev`, the timeout path, the submission marker, a grader-style ephemeral step
  with file mounts, no leftover containers, rejection of Pyxis flags, and the drop's
  `cleanup_scratch`.
- the metadata block (on by default): its self-test before the service starts, a TCP connect
  from inside a sandbox to `<ip>:80` and `<ip>:988` that must be
  refused, and at teardown the re-check and the removal.

**Fallbacks, only with sign-off:**
- `SWE_RUN_AS=root`, if unprivileged user namespaces fail on COS. It also needs
  `SWE_METADATA_BLOCK=0` (the entry script refuses a block for uid 0: it would cut off
  kubelet and the GKE agents), so it needs its own sign-off for CS-NET-001 too;
- `--no-metadata-block` on `smoke` or `host` (pilot only; sets `SWE_METADATA_BLOCK=0` and
  logs a WARNING): `kimiswe` and its sandboxes can then reach the metadata servers. Production
  refuses it;
- `--keep-default-caps` for the model pod;
- `--smoke-image` with another public arm64 image.

## Production: Kimi acc/both runs

> Needs the production risk sign-off first (Security notes, residual risk 5): the host pod
> runs on the client node of a v1.0.3 Kimi run (node 2 for C12..C80 alongside the frontend;
> node 4 for C192..C1024 alongside the frontend).

In v1.0.3, the perf recipe trees (`8nodes/`, `12nodes/` and `with-pull-secret/{8,12}nodes/`)
stay untouched (`--mode perf` only). `generate_point_recipes.py --swe <acc|both|all>` generates
a parallel SWE recipe tree under `v1.0.3/kimi-k3/[with-pull-secret/]swe_<acc|both>/<N>nodes/point_<C>/`.
Unlike v1.0.1, v1.0.3 has no `etcd` or `nats` tasks (Dynamo uses `DYN_DISCOVERY_BACKEND=kubernetes`
and the TCP request plane), and places the `benchmark` (client) task on `variables.CLIENT_NODE`
(`2` for C12..C80 with `FRONTEND_NODE=2`; `4` for C192..C1024 with `FRONTEND_NODE=4`). On GKE:

1. [run_swe_sweep.sh](run_swe_sweep.sh) runs `pilot/swe_pilot.py --check` and
   `generate_point_recipes.py --swe <mode> --check`, starts `pilot/swe_pilot.py serve` on the
   target `swe_<mode>/<N>nodes` group, then runs `tools/run_sweep.sh` (or a direct point loop
   when `SKIP_WAIT_IDLE=1` for concurrent 8+8 runs on a 16-node pool).
2. For each new run folder `swe_<mode>/<N>nodes/point_<C>/sflow_output/<run id>/` whose
   rendered `client.yaml` has the `swe_bench` dataset (`eval_method: swe_bench_scorer`),
   `serve` reads the client node and sflow allocation `<alloc>` from
   `<run dir>/benchmark/benchmark.k8s.yaml`, verifies that
   `<run dir>/frontend/frontend.k8s.yaml` names the same allocation, checks that no pod of
   another sflow allocation runs on that client node, and starts per-run
   `kimi-swe-host-<alloc>` there with `SWE_HANDOFF=1` and `SWE_SCAN_REPORTS`. The drop's
   `swe-service.py` writes the connection file (url and token, mode 0600, owner `kimiswe`) to
   `<state>/handoff/<run id>/swe-connection.json`.
3. The recipe's `benchmark` task (root, same client node, hostPath artifact `SWE_HANDOFF_DIR`)
   waits for that file and `GET /health` == 200, as `swe-service.py` does before `SWE_READY`.
   The wait starts once the frontend is ready (`depends_on: [frontend]`) and lasts `SWE_WAIT_S`
   (default 2700 s). Then the drop's SWE block runs verbatim and reads the file.
4. After the client, the benchmark applies the drop's teardown redaction (below), then
   mirrors `report/` for sflow's collect.
5. When the benchmark pod (`benchmark-<alloc>`) ends, `serve` deletes `kimi-swe-host-<alloc>`:
   the drop's SIGTERM teardown, then the checks. The leak scan also reads the run's report on
   the node (mounted read-only). `serve` then runs `kimi-swe-collect-<alloc>` to copy the
   service's run directory, without caches, to `<run dir>/swe_service/<run id>/`.

**The redaction port: point reviewers to it.** On Slurm, `swe-service.py` / the client
replaces the service token with `<redacted>` in every `*.yaml` under the
run's output directories when the service stops. On GKE, sflow collects `report/` before the
service pod stops, and that pod cannot write the root-owned report directory. So the
benchmark task applies the same rule (same glob, same replacement) to
`SFLOW_WORKFLOW_OUTPUT_DIR` and `REPORT_DIR`, right after the client and before the mirror,
and stops if any `*.yaml` still holds the token. The pinned client already writes
`<redacted>` for `*_token` fields in `report/config.yaml`. So the rule is expected to change
no file and print `SWE token redacted in 0 file(s): none`. If it ever changes a
file, it prints the file name, never the token. This is the drop's own behavior run from
another task, not a manual edit of results. Both `generate_point_recipes.py --swe all --check`
and `swe_pilot.py --check` assert where it runs.

**Run.** From `examples/self_contained/kubernetes`. Prerequisites: `swe_pilot.py --check` and
`swe_pilot.py configmaps` run with the current files, `swe_pilot.py secret` done, no active
host pod on the target client node, and the usual sweep prerequisites (`../README.md`).

```bash
G=v1.0.3/kimi-k3/swe_acc/8nodes; mkdir -p $G/sweep
MODE=acc GROUP=$G POINTS="24" setsid nohup v1.0.3/kimi-k3/swe/run_swe_sweep.sh \
  > $G/sweep/swe-sweep-$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
```

`serve` logs to `$G/sweep/swe-serve-<UTC>.log` (mode 0600). Each host pod's log goes to
`<workdir>/.cache/kimi-swe-pilot/logs/<run id>-host.log`. To stop everything:
`kill -TERM -- -<process group id>`. `serve` tears the active `kimi-swe-host-<alloc>` pod down
before it exits.

**Failure modes**

| What happens | Effect | What to do |
|---|---|---|
| acc/both launched without `serve` | The benchmark stops `SWE_WAIT_S` after the frontend is ready: `FATAL: no SWE service for this run`. | Use `run_swe_sweep.sh` or the manual fallback. |
| Host pod fails before `SWE_READY` | Same FATAL. `serve` logs `ERROR ... before SWE_READY`, then still tears down and copies. | Read `<run id>-host.log`. Infra failure: one retry. |
| Host pod dies during the run | `serve` logs `the SWE service ended before the benchmark`. The SWE score is not valid. | Infra failure: one retry. |
| Client fails | `set -e` skips the redaction check and the mirror. The report on the node should still hold no token (the client redacts it itself). The host's teardown leak scan checks it anyway; a hit there makes `collect` exit 1, but the service logs are still copied. | As for any failed run. |
| A run folder can't be parsed | Retried for 2 min, then `ERROR: run ...`. That run gets no service. | Check its `benchmark/benchmark.k8s.yaml` and `frontend/frontend.k8s.yaml` manifests; rerun the point. |

**Manual fallback** (one run, without `serve`). Once the run folder `<run id>` exists, read
the client node from its `benchmark/benchmark.k8s.yaml` (`nodeSelector`), then from `pilot/`:

```bash
./swe_pilot.py host --node <client node> --run-id <run id> --allow-busy-node --handoff --scan-reports
# after the run's benchmark pod has finished:
./swe_pilot.py down host
./swe_pilot.py collect --node <client node> --run-id <run id> --dest <run dir>/swe_service
```

If `serve` was killed with `SIGKILL` and left a per-run host pod behind, tear it down with:
`./swe_pilot.py down host --alloc <alloc>`.

**Disk.** Each run keeps its imported images on the client node, in
`<state>/runs/<run id>/swe_service/enroot-cache`: about 2.3 GB per image, so about 460 GB for
200 instances. The node SSD has about 8.6 TB free. Pruning old runs is a separate step.

## Security notes

- **Secrets.**
  - The GitLab token goes from a 0600 netrc to a Kubernetes Secret on stdin. In the pod it
    becomes a 0600 `kimiswe` copy on a memory volume. It is never in argv or env, and never
    printed. Driver output, kubectl errors and pod logs are redacted.
  - The service token is the drop's: random per start, written to a 0600 file and redacted
    from YAML at shutdown. The pinned client also writes `<redacted>` for `*_token` fields in
    `report/config.yaml` and does not print the token.
  - `collect` refuses to copy a run when the pod-side leak scan found a secret in the run
    directory it copies. A hit elsewhere, for example in the node's report after a failed
    client, is never copied: `collect` logs a WARNING and exits 1. It also scans the local
    copy for the registry token.
- **Least privilege.** Only the host pod is privileged. Enroot needs user namespaces, overlay
  and a fresh `/proc` for each sandbox. The service and its sandboxes run as the dedicated
  user `kimiswe` (uid 61000), not root, and sandboxes see a minimal `/dev`. The other pods
  are non-privileged and drop all capabilities. No pod gets a service-account token. Every
  listener is on 127.0.0.1.
- **Root and `kimiswe`.** The root teardown checks read files that `kimiswe` (and so a
  sandbox) can change. They kill leftover `kimiswe` processes first, so nothing can race
  them. They never follow symlinks or block on FIFOs there, and write each report as a new
  root-owned file. The registry token to scan for comes from the root-owned Secret mount.
  Root runs Python with `-I` from `/`. `down host` is refused while the client pod, which
  writes the same run directory, is still running.
- **Production handoff.** The connection file is `kimiswe`'s 0600 file in a fresh root-owned
  0711 per-run directory `<state>/handoff/<run id>/`, so non-root users other than `kimiswe`
  can neither list nor read it. (The parent `<state>/handoff` is 0755 because containerd
  creates it on every task pod's node as the `SWE_HANDOFF_DIR` hostPath before any host pod
  runs.) sflow mounts each `fs://` artifact into every task of the run, and those pods run as
  root, so all of the run's own pods could read it: the same trust as on Slurm, where every
  task runs as the job's user. sflow never collects it (it collects only each pod's output
  directory). The drop's teardown resets it to `{}`, and the root checks verify that. The
  host pod mounts the node's report directory read-only, only for the leak scan.
- **Metadata block (CS-NET-001).** The service and its sandboxes run as `kimiswe` (uid 61000)
  in the node's network namespace. Before the service starts, the entry script adds its own
  filter-table chain `KIMI-SWE-MDBLOCK` and one jump at the top of `OUTPUT` that matches only
  that uid (`-m owner --uid-owner 61000`, with a comment naming the host pod). The chain
  refuses (REJECT) the uid's traffic to the GCE metadata server `<ip>` on every
  port but 53 (it is also the VPC resolver), to GKE's metadata server `<ip>`, and,
  where ip6tables works, to `fd20:ce::254` (again but 53). Without an IPv6 rule, root must not
  reach the metadata server over IPv6, or the pod fails. Root and every other uid keep their
  access, so kubelet and the GKE agents are not affected. The rules go to the node's backend,
  legacy or nft, whichever holds more rules, as kube-proxy's iptables-wrapper decides (a tie
  fails). Every call waits for the node's xtables lock (`-w`; the driver mounts
  `/run/xtables.lock`). No table, chain or rule that is not ours is flushed or deleted; our
  jumps are deleted by their exact spec.
  - Dedicated service user. The rule matches the uid in the whole network namespace of the
    node, not only this pod. The image's own user, `appuser`, is uid 1000, and on these nodes
    GKE's hostNetwork DaemonSets run containers as uid 1000 (10 containers in 7 pods: metrics
    collectors, fluentbit, gke-metrics-agent) and 1001: a block for 1000 would cut them off.
    So the entry script creates `kimiswe` with uid and gid 61000 at start, from one constant in
    the driver (`SERVICE_USER`, `SERVICE_UID`, `SERVICE_GID`, passed as `SWE_SERVICE_*`; the
    script has no default). 61000 is in 60578-61183, which systemd's uid map leaves unused:
    above useradd's automatic range (`UID_MAX` 60000) and below systemd's dynamic users
    (61184-65519), `nonroot` (65532) and `nobody` (65534). Ids outside that band are refused,
    so 1000, 1001, 65532 and 65534 are never used. The script fails closed if the image's
    passwd or group already uses the name, the uid or the gid in any other combination. The
    user has no supplementary groups (`setpriv --clear-groups`), a fresh 0700 home in the
    container, and owns nothing in the image (`appuser` owns the client venv and WORKDIR).
    The pilot client and collect pods run as uid 61000 too; the client image has no passwd
    line for it, so the client pod sets `HOME`, `USER` and `LOGNAME` (no code the client runs
    needs more; checked in the image).
  - Collateral: none by design, and checked at start. Before any rule goes in, the script
    reads the node's socket tables (`/proc/net/tcp`, `udp`, `raw`, their IPv6 tables, and
    `icmp`/`udplite` where present) and stops if any live socket belongs to uid 61000.
    Closing sockets that no process holds any more are listed, not counted. It logs that
    count (0) and, for information, the number of uid 1000 sockets. The operator's pre-check
    before a host pod also stops if any pod on the node uses uid or gid 61000.
  - `--persist-cache` reuses `<state>/enroot-cache` only if `kimiswe` owns every entry. A
    cache from earlier runs as uid 1000 is refused, never chowned: delete it on the node.
  - Self-test before the service starts, fail closed: root gets HTTP 200 from
    `/computeMetadata/v1/instance/id`; `kimiswe` is refused at once (curl exit 7) at
    `<ip>`, at `<ip>:988` and, with the IPv6 rule, at `[fd20:ce::254]`;
    our REJECT counters account for every refused request; `kimiswe` still resolves
    `registry.gitlab.com`. Only HTTP and curl exit codes are logged, never a response body,
    and no token, service-account or attribute path is ever requested. The smoke test also
    connects from inside a sandbox.
  - Teardown, after the other checks: the block must still refuse `kimiswe`; the log shows how
    many packets it refused during the run; then the chain and its jump are removed and the
    removal is confirmed. If the script ends early, an EXIT trap kills this pod's `kimiswe`
    processes and removes the block. A block left by a SIGKILLed pod is removed by the next
    host pod on the node. A node lock (`<state>/host-entry.lock`) keeps a second host pod,
    smoke or service, off the node meanwhile.
  - `SWE_METADATA_BLOCK=0` (driver: `--no-metadata-block`) turns it off with a WARNING: pilot
    fallback only. It then also leaves a stale block of an earlier pod in place.
- **TODO(security), residual risks:**
  1. Sandboxes run model-written commands on the node network (hostNetwork), so they can
     reach node-local ports (CS-NET-002). The metadata block (above) closes the GCE and GKE
     metadata servers to `kimiswe`; with `--no-metadata-block` they stay open. It does not
     cover:
     - a sandbox that becomes root on the node, for example through risk 4;
     - a root-owned service on the node that relays requests to the metadata server (the
       required `Metadata-Flavor: Google` header limits plain request forgery);
     - DNS (53) to `<ip>`, which the VPC resolver needs.
     The rule matches every uid 61000 socket in the node's network namespace, not only this
     pod's. By design nothing else uses that uid, and the check at start proves it for live
     sockets. What remains: a process that starts using uid 61000 on the node while the host
     pod runs (or, after a SIGKILL, until the next host pod removes the rules), for example a
     hostNetwork pod without `runAsUser` whose image user is 61000, would lose metadata access
     too. The pilot client pod (uid 61000, hostNetwork) is blocked as well, by design. A
     minimal node service account would limit the impact of any bypass.
  2. Like on Slurm, a sandbox runs as the same uid as the service. It could read the registry
     token copy and the service token (from `/proc`). Use a read-only registry token and
     revoke it after the run.
  3. The apt dependencies of the two Enroot packages (including `libcap2-bin`) and `iptables`
     (for the metadata block) are verified only by Debian's archive signatures.
  4. `enroot+caps` gives `enroot-aufs2ovlfs` and `enroot-mksquashovlfs` `CAP_SYS_ADMIN`
     file capabilities (plus `CAP_MKNOD` for the first), and `kimiswe` can run them. In this
     privileged pod, a bug in either helper could let a compromised service user become root
     on the node. This is the standard Enroot/Pyxis trust model. Sandboxes can't reach the
     helpers, because they run in the SWE image's rootfs.
  5. Production only: the host pod runs on the client node of the v1.0.3 Kimi run (node 2 for
     C12..C80 alongside the frontend and decode worker; node 4 for C192..C1024 alongside the
     frontend and prefill workers). Its sandboxes (hostNetwork) can reach the run's listeners
     bound to `0.0.0.0` on that node (Dynamo TCP request/event plane ports, frontend HTTP on 8000,
     and worker sshd on 2222). That is the same exposure as Pyxis sandboxes on Slurm. The host
     pod also competes with the co-located tasks for CPU and memory: requests 16 CPU / 64 Gi,
     limits 96 CPU / 384 Gi. Needs the user's sign-off before any Kimi acc/both run; the pilot
     sign-off does not cover it.
