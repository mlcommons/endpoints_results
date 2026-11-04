#!/usr/bin/env python3
"""Kimi-K3 SWE-bench 1-node pilot on GKE: create, watch and remove the pilot's objects.

Everything lives in namespace kimi-swe, on one node of the A4X Max pool.

  configmaps  Namespace and ConfigMaps: the drop's three SWE scripts (byte-identical), our
              host files (entry script, srun/squeue/scancel shims, swe_probe.py, the pinned
              Enroot .deb and service tarball, MANIFEST.sha256), and the pilot client and
              model configs. Every pinned file is checked by sha256 first. Test only: also
              kimi-swe-oracle when oracle/oracle_data.py has written its data (--oracle-dir),
              and --num-instances N puts N in the client config's swe_bench num_instances
              (PILOT 4 says 10; the drop runs 200), e.g. to grade the production task list
              with the oracle.
  secret      Secret kimi-swe-enroot: Enroot .credentials built from your GitLab netrc.
  smoke       One-shot pod, no token needed: tools, egress, the drop's service lifecycle and
              a sandbox round trip on a public arm64 image. Ends with SMOKE_OK or SMOKE_FAIL.
  model       kimi-swe-model: the pilot endpoint, GPT-OSS-120B on 1 GPU at 127.0.0.1:18000.
              --oracle (test only, no GPU): oracle/oracle_server.py at the same address answers
              each task with its public SWE-bench gold patch; a working harness then reports
              every task resolved. A grading-harness check, never a submission.
  host        kimi-swe-host: the drop's swe-service.py behind the shims, until deleted.
              smoke and host pods create the dedicated service user (SERVICE_USER, uid
              SERVICE_UID) and block its access to the GCE metadata server (CS-NET-001, see
              swe-host-entry.sh); --no-metadata-block is a pilot-only fallback that needs
              approval.
  client      kimi-swe-client: `benchmark from-config --mode acc` with client-pilot.yaml, as the
              service user. Give it the same --num-instances as `configmaps`; a different
              value is refused.
  status      Pods and redacted log tails.
  down        Delete pods (and the Secret with `all`). For the host pod, wait for its teardown
              checks and keep its log.
  collect     Copy one run directory from the node, without caches, to a local directory.
  serve       Production: for each acc/both sflow run of the Kimi v1.0.2 recipes, start
              kimi-swe-host on that run's client node (the drop's swe_service task; sflow
              cannot mount the registry Secret), hand the connection file to the recipe's
              benchmark task, and tear the host down and collect its run directory when the
              benchmark ends. v1.0.2 has no etcd: the client node and the run's allocation
              come from the benchmark pod's manifest (node 3 for C8-C80, node 0 for
              C192-C1024), and serve follows only that allocation's benchmark and frontend
              pods. Use it through ../run_swe_sweep.sh.

  --check     Offline self-check, no kubectl at all (not even for the context): the pins, the
              ConfigMaps built from the local files, and the contract with the v1.0.2
              recipes (../generate_point_recipes.py and the recipe files it wrote).

Secrets: the GitLab token goes to the API server on stdin only, never in argv. Nothing here
prints a token: kubectl errors and pod logs pass through a redaction filter first. Local logs
and run records are written with mode 0600 under <workdir>/.cache/kimi-swe-pilot.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import netrc
import os
import re
import signal
import stat
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple

HERE = Path(__file__).resolve().parent                        # .../kimi-k3/swe/pilot
SWE_DIR = HERE.parent                                         # .../kimi-k3/swe
# DEVIATION (v1.0.2): the SWE launcher comes from the v1.0.1 drop. The v1.0.2 drop deleted the
# Kimi SWE scripts (swe-service.py, swe-enroot-transport.py, swe_cleanup.py); the three files
# and their sha256 pins below are the v1.0.1 drop's, byte for byte.
DEFAULT_DROP = (HERE.parents[7] / "code-drop-nvidia" / "nv-mlpinf-partner-partner-endpoints-v1.0.1"
                / "NVIDIA" / "src" / "kimi-k3" / "scripts")
DEFAULT_NETRC = "<workdir>/.config/nv-partner/gitlab.netrc"
LOCAL_DIR = Path.home() / ".cache" / "kimi-swe-pilot"
K8S_ROOT = HERE.parents[3]                                    # .../self_contained/kubernetes
DEFAULT_GROUP = "v1.0.3/kimi-k3/swe_acc/8nodes"               # C24 (first acc run) lives here
# The v1.0.3 recipe generator: `--check` compares the shared contract constants with it.
GENERATOR = SWE_DIR.parent / "generate_point_recipes.py"

NS = "kimi-swe"
FIELD_MANAGER = "kimi-swe-pilot"
PART_OF = {"app.kubernetes.io/part-of": "kimi-swe-pilot"}
DIGEST_KEY = "kimi-swe/content-sha256"

# Client image d3c57f2: the v1.0.3 recipes' CLIENT_IMAGE (drop ghcr.io#mlcommons/endpoints@
# sha256:1523051b141437b25083924d6428082f435713516eee71c54ae207b393900b2d). Used by the host,
# smoke, client, collect and oracle pods. DEVIATION (v1.0.3): the Enroot SWE service stays c97
# (swe_source_c97.tgz + v1.0.1 drop scripts) while the client is d3c57f2.
CLIENT_IMAGE = ("ghcr.io/mlcommons/endpoints@sha256:"
                "1523051b141437b25083924d6428082f435713516eee71c54ae207b393900b2d")
TRTLLM_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14"   # pilot endpoint only (test)
STATE = "<node-local-ssd>/kimi-swe"
MODEL_DIR = "<node-local-ssd>/model-cache/gpt-oss-120b"
HARMONY_DIR = "<node-local-ssd>/mlperf-data/gpt-oss/harmony_cache_gptoss"
MODEL_PORT = 18000
# The dedicated service user (CS-NET-001), the one definition: the host pod's entry script
# creates it from these (SWE_SERVICE_USER/UID/GID) and runs the service and its sandboxes as
# it; the client and collect pods run as it too (app_ids). The metadata block matches this
# uid on the whole node, so nothing else there may use it. Not the image's appuser (1000):
# GKE's own hostNetwork DaemonSets run as 1000 and 1001. 61000 is in 60578-61183, which
# systemd's uid map leaves unused (swe-host-entry.sh: svc_id_ok).
SERVICE_USER = "kimiswe"
SERVICE_UID = SERVICE_GID = 61000
# Production (`serve`). Same paths as ../generate_point_recipes.py (v1.0.2): REPORTS_DIR and
# the swe_<mode> recipes' SWE_HANDOFF_DIR artifact (`--check` and tests/test_production.py
# check that they agree).
# TODO(security): sflow mounts SWE_HANDOFF_DIR into every task pod of an acc/both run, so
# containerd creates HANDOFF_DIR as root:root 0755 on the client node before the host pod
# starts. swe-host-entry.sh stays byte-identical to v1.0.1 (leaving HANDOFF_DIR 0755); each
# run's subdirectory HANDOFF_DIR/<run_id> is created fresh as 0711 (refusing existing paths
# and symlinks) and swe-connection.json is 0600 owned by kimiswe (61000:61000).
HANDOFF_DIR = f"{STATE}/handoff"                    # swe-host-entry.sh: SWE_HANDOFF=1
REPORTS_DIR = "<node-local-ssd>/kimi-reports"
REPORTS_MOUNT = "<node-mount>/kimi-reports"                 # read-only in the host pod
SFLOW_NS = "default"                                # the recipes' K8S_NAMESPACE
# Metadata block (CS-NET-001): the host pod's iptables-legacy calls take the node's xtables
# lock (-w), like kube-proxy, so they never interleave with its table rewrites.
XTABLES_LOCK = "/run/xtables.lock"
NO_MDB_HELP = ("fallback, needs approval (CS-NET-001): no iptables block between the service "
               "user (and its sandboxes) and the GCE metadata server")

POD = {"smoke": "kimi-swe-smoke", "model": "kimi-swe-model", "host": "kimi-swe-host",
       "client": "kimi-swe-client", "collect": "kimi-swe-collect"}
CM_DROP, CM_HOST, CM_CLIENT, CM_MODEL = ("kimi-swe-drop-scripts", "kimi-swe-host",
                                         "kimi-swe-client", "kimi-swe-model")
SECRET = "kimi-swe-enroot"
# Test only (`model --oracle`): oracle/oracle_server.py serves the public SWE-bench gold patches
# from oracle/oracle_data.py's output in place of the model. Nothing else uses these.
CM_ORACLE = "kimi-swe-oracle"
ORACLE_DIR = HERE / "oracle"
ORACLE_DATA = "oracle-data.json.gz"                           # binaryData
ORACLE_TEXT = ("oracle_server.py", "oracle-manifest.json")
ORACLE_MOUNT = "/etc/kimi-swe/oracle"
ORACLE_MODEL_NAME = "swe-oracle-gold-patch"                   # /v1/models id: the run's model name
NOBODY = 65534                                                # the oracle owns no files anywhere
# Test only (--num-instances): ConfigMap kimi-swe-client gets client-pilot.yaml with another
# number on this one line (PILOT 4) and no other change.
CLIENT_CONFIG = "client-pilot.yaml"
NUM_INSTANCES_RE = re.compile(r"^(?P<lead>[ ]+num_instances: )(?P<n>[0-9]+)(?P<tail>[ ]+# PILOT 4)$",
                              re.M)
MAX_INSTANCES = 500                                           # SWE-bench Verified has 500 rows

DROP_FILES = ("swe-service.py", "swe-enroot-transport.py", "swe_cleanup.py")
ARTIFACTS = ("enroot_4.2.1-1_arm64.deb", "enroot-caps_4.2.1-1_arm64.deb", "swe_source_c97.tgz")
HOST_FILES = ("swe-host-entry.sh", "srun", "squeue", "scancel", "swe_probe.py")
# Same pins as the PINNED block in swe-host-entry.sh; `configmaps` checks that they agree.
# enroot-caps_*.deb is NVIDIA's enroot+caps_4.2.1-1_arm64.deb, renamed: '+' is not allowed
# in a ConfigMap key. The digest is the one GitHub publishes for that release asset.
PINS = {
    "swe-service.py": "d6ca1795e4db77f608770498ccb8ccd4f34e7c18e954d1e6939b0a497be3967e",
    "swe-enroot-transport.py": "4a427b88229d1583efa6fbb3072b4bb496cd27708f210b941496d522ec4ed242",
    "swe_cleanup.py": "5491c3e02fad9c55b9abfde98ef80fa5d7aaf723a65cb2f7d773e82b7476d87f",
    "enroot_4.2.1-1_arm64.deb": "267018b335815a6b32a562915de1f3fad579f7c9d6748e98c62e7369b40be421",
    "enroot-caps_4.2.1-1_arm64.deb": "12b811d0c95e879e42e4ac762fa85869a5f3cc9dbb06ab483c01e12e911284a1",
    "swe_source_c97.tgz": "674c82a5f098db8cfe045c8564fe5db953681841f845c4fb7d1a5e3ad22b4de9",
}

RUN_ID_RE = re.compile(r"[A-Za-z0-9]([A-Za-z0-9._-]{0,61}[A-Za-z0-9])?")   # also a label value
NODE_RE = re.compile(r"[a-z0-9]([a-z0-9.-]{0,251}[a-z0-9])?")
LOGIN_RE = re.compile(r"[A-Za-z0-9._-]{1,255}")
TOKEN_RE = re.compile(r"[A-Za-z0-9._-]{16,512}")
PREWARM_RE = re.compile(r"first:[0-9]{1,3}|[A-Za-z0-9_.,-]+")
IMAGE_RE = re.compile(r"[A-Za-z0-9./#:_@-]+")
# sflow allocation ids (the suffix of every pod name of one sflow run, and its label value).
ALLOC_RE = re.compile(r"[a-z0-9]([a-z0-9-]{0,38}[a-z0-9])?")
SFLOW_ALLOC_LABEL = "sflow.ai/allocation"                     # on every pod of an sflow run
ALLOC_LABEL = "kimi-swe/allocation"                           # serve: the run a host pod serves
TOLERATIONS = [
    {"key": "kubernetes.io/arch", "operator": "Equal", "value": "arm64", "effect": "NoSchedule"},
    {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
]
# Left out of `collect`: environments, caches and image layers (large, reproducible).
COLLECT_EXCLUDES = ("--exclude=./swe_service/service-venv", "--exclude=./swe_service/uv-cache",
                    "--exclude=./swe_service/python", "--exclude=enroot-cache",
                    "--exclude=hf-cache", "--exclude=./swe-hf-cache", "--exclude=./client/home")


def log(msg: str) -> None:
    print(f"[swe-pilot {datetime.now(timezone.utc):%H:%M:%SZ}] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[swe-pilot] ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def open_private(path: Path, mode: str = "a"):
    """Open a local file that only the current user can read (dir 0700, file 0600)."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if "a" in mode else os.O_TRUNC)
    return os.fdopen(os.open(path, flags, 0o600), mode)


# ----------------------------------------------------------------------------- redaction
class Redactor:
    """Masks secrets in text before it is printed or written to a local file."""

    _GITLAB = re.compile(r"\bgl[a-z]{2,6}-[A-Za-z0-9_-]{12,}")
    # The service token is secrets.token_urlsafe(48): 64 characters of [A-Za-z0-9_-].
    _URLSAFE64 = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{64}(?![A-Za-z0-9_-])")
    _HEX64 = re.compile(r"[0-9a-f]{64}")                       # sha256 digests stay readable
    _FLAG = re.compile(r"(?i)(\bbearer\s+|--auth-token[=\s]+|\bpassword\s+)[^\s\"',]+")

    def __init__(self) -> None:
        self._literals: set[str] = set()

    def add(self, secret: str | None) -> None:
        if secret and len(secret) >= 8:
            self._literals.add(secret)
            self._literals.add(base64.b64encode(secret.encode()).decode())

    def __call__(self, text: str) -> str:
        for literal in sorted(self._literals, key=len, reverse=True):
            text = text.replace(literal, "<redacted>")
        text = self._GITLAB.sub("<redacted-token>", text)
        text = self._URLSAFE64.sub(
            lambda m: m[0] if self._HEX64.fullmatch(m[0]) else "<redacted-token>", text)
        return self._FLAG.sub(lambda m: m[1] + "<redacted>", text)


# ----------------------------------------------------------------------------- kubectl
class Kube:
    """kubectl with a fixed context. Objects go on stdin; stderr is redacted."""

    def __init__(self, context: str | None, redact: Redactor, dry_run: bool = False) -> None:
        self.redact = redact
        self.dry_run = dry_run
        if not context:
            p = subprocess.run(["kubectl", "config", "current-context"], capture_output=True,
                               text=True, timeout=30, check=False)
            context = p.stdout.strip()
            if p.returncode != 0 or not context:
                die("no current kubectl context; pass --context")
        self.context = context

    def base(self) -> list[str]:
        return ["kubectl", f"--context={self.context}"]

    def run(self, *args: str, stdin: bytes | None = None, check: bool = True,
            timeout: int = 300) -> subprocess.CompletedProcess:
        try:
            p = subprocess.run([*self.base(), *args], input=stdin, capture_output=True,
                               timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            die(f"kubectl {' '.join(args[:3])} timed out after {timeout} s")
        if check and p.returncode != 0:
            err = self.redact(p.stderr.decode(errors="replace")).strip()[-2000:]
            die(f"kubectl {' '.join(args[:3])} failed (exit {p.returncode}): {err}")
        return p

    def _send(self, verb: list[str], obj: dict) -> None:
        extra = ["--dry-run=server"] if self.dry_run else []
        p = self.run(*verb, *extra, "-f", "-", stdin=json.dumps(obj).encode())
        kind, name = obj["kind"], obj["metadata"]["name"]
        log(f"{kind.lower()}/{name}: {self.redact(p.stdout.decode(errors='replace')).strip()}")

    def apply(self, obj: dict) -> None:
        self._send(["apply", "--server-side", "--force-conflicts",
                    f"--field-manager={FIELD_MANAGER}"], obj)

    def create(self, obj: dict) -> None:
        self._send(["create"], obj)

    def get(self, kind: str, name: str, namespaced: bool = True) -> dict | None:
        args = ["get", kind, name, "-o", "json", "--ignore-not-found"]
        if namespaced:
            args += ["-n", NS]
        out = self.run(*args).stdout.strip()
        return json.loads(out) if out else None

    def exists(self, kind: str, name: str) -> bool:
        p = self.run("get", kind, name, "-n", NS, "-o", "name", "--ignore-not-found")
        return bool(p.stdout.strip())

    def logs(self, pod: str, tail: int | None = None) -> str:
        args = ["logs", pod, "-n", NS] + ([f"--tail={tail}"] if tail else [])
        return self.redact(self.run(*args, check=False, timeout=120).stdout.decode(errors="replace"))

    def delete_pod(self, name: str, timeout: int = 180) -> None:
        self.run("delete", "pod", name, "-n", NS, "--ignore-not-found", "--wait=true",
                 f"--timeout={timeout}s", timeout=timeout + 30)


def pod_state(kube: Kube, name: str) -> tuple[str, dict, dict]:
    pod = kube.get("pod", name)
    if not pod:
        return "Absent", {}, {}
    status = pod.get("status", {})
    cs = (status.get("containerStatuses") or [{}])[0]
    return status.get("phase", "Unknown"), cs, pod


def terminated(phase: str, cs: dict) -> bool:
    return phase in ("Absent", "Succeeded", "Failed") or "terminated" in cs.get("state", {})


def wait_started(kube: Kube, name: str, timeout: int = 900) -> None:
    """Wait until the container runs or has exited. Fail fast on pull or config errors."""
    deadline, last = time.monotonic() + timeout, ""
    while time.monotonic() < deadline:
        phase, cs, pod = pod_state(kube, name)
        state = cs.get("state", {})
        if "running" in state or terminated(phase, cs) and phase != "Absent":
            return
        waiting = state.get("waiting", {})
        reason = waiting.get("reason", "")
        if reason in ("ErrImagePull", "ImagePullBackOff", "InvalidImageName",
                      "CreateContainerConfigError", "CreateContainerError"):
            die(f"{name}: {reason}: {kube.redact(waiting.get('message', ''))[:500]}")
        sched = [c.get("message", "") for c in pod.get("status", {}).get("conditions", [])
                 if c.get("reason") == "Unschedulable"]
        msg = f"{name}: {phase} {reason} {sched[0][:300] if sched else ''}".strip()
        if msg != last:
            log(msg)
            last = msg
        time.sleep(5)
    die(f"{name} did not start within {timeout} s")


def wait_ready(kube: Kube, name: str, timeout: int) -> None:
    deadline, last_print = time.monotonic() + timeout, 0.0
    while time.monotonic() < deadline:
        phase, cs, _ = pod_state(kube, name)
        if cs.get("ready"):
            return
        if terminated(phase, cs):
            die(f"{name} exited before it was ready:\n{kube.logs(name, tail=40)}")
        if time.monotonic() - last_print > 60:
            tail = kube.logs(name, tail=1).strip()
            log(f"{name}: {phase}, not ready yet. {tail[-300:]}")
            last_print = time.monotonic()
        time.sleep(10)
    die(f"{name} not ready within {timeout} s")


_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z) ?(.*)$", re.S)


def _ts_key(ts: str) -> str:
    base, _, frac = ts[:-1].partition(".")
    return f"{base}.{frac.ljust(9, '0')}"


def follow_logs(kube: Kube, pod: str, log_path: Path, *,
                echo: Callable[[str], bool] | bool = True,
                stop: Callable[[str], bool] | None = None,
                timeout: float | None = None) -> tuple[list[str], str]:
    """Stream a pod log through the redactor to stdout and a 0600 file.

    Returns (lines, why) with why = stop | exited | timeout. Reconnects if the stream drops
    while the container still runs; --timestamps/--since-time avoid repeated lines.
    """
    lines: list[str] = []
    last_ts = None
    deadline = time.monotonic() + timeout if timeout else None
    with open_private(log_path) as out:
        while True:
            cmd = [*kube.base(), "logs", "-f", "--timestamps", pod, "-n", NS]
            if last_ts:
                cmd.append(f"--since-time={last_ts}")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            timer = None
            if deadline:
                timer = threading.Timer(max(1.0, deadline - time.monotonic()), proc.kill)
                timer.start()
            why = None
            try:
                for raw in proc.stdout:
                    text = raw.decode(errors="replace").rstrip("\n")
                    m = _TS.match(text)
                    if m:
                        if last_ts and _ts_key(m[1]) <= _ts_key(last_ts):
                            continue
                        last_ts, text = m[1], m[2]
                    line = kube.redact(text)
                    out.write(line + "\n")
                    out.flush()
                    lines.append(line)
                    if echo is True or (callable(echo) and echo(line)):
                        print(line, flush=True)
                    if stop and stop(line):
                        why = "stop"
                        break
            finally:
                if timer:
                    timer.cancel()
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
            if why:
                return lines, why
            if deadline and time.monotonic() >= deadline:
                return lines, "timeout"
            phase, cs, _ = pod_state(kube, pod)
            if terminated(phase, cs):
                return lines, "exited"
            time.sleep(2)


# ----------------------------------------------------------------------------- local records
def record_path(run_id: str) -> Path:
    return LOCAL_DIR / "runs" / f"{run_id}.json"


def load_record(run_id: str) -> dict:
    try:
        return json.loads(record_path(run_id).read_text())
    except (OSError, ValueError):
        return {}


def save_record(run_id: str, **values) -> None:
    rec = {**load_record(run_id), **values}
    with open_private(record_path(run_id), "w") as f:
        json.dump(rec, f, indent=2)


def app_ids(args, run_id: str) -> tuple[int, int]:
    """uid, gid for the client and collect pods: --app-uid/--app-gid, else the run record (the
    host log's `service user NAME (UID:GID)` line), else the dedicated service user."""
    rec = load_record(run_id)
    uid = args.app_uid if args.app_uid is not None else int(rec.get("uid", SERVICE_UID))
    gid = args.app_gid if args.app_gid is not None else int(rec.get("gid", SERVICE_GID))
    if uid <= 0 or gid <= 0:
        die("the client and collect pods must run as a non-root user")
    return uid, gid


# ----------------------------------------------------------------------------- ConfigMaps
def entry_pins(text: str) -> dict[str, str]:
    m = re.search(r"declare -A PINNED=\(\n(.*?)\n\)", text, re.S)
    if not m:
        die("swe-host-entry.sh has no PINNED block")
    return dict(re.findall(r"^\s*\[([^\]]+)\]=([0-9a-f]{64})\s*$", m[1], re.M))


def gather_files(drop_dir: Path, artifacts: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for directory, names in ((drop_dir, DROP_FILES), (artifacts, ARTIFACTS), (SWE_DIR, HOST_FILES)):
        for name in names:
            try:
                files[name] = (directory / name).read_bytes()
            except OSError as exc:
                die(f"cannot read {directory / name}: {exc.strerror}")
    if entry_pins(files["swe-host-entry.sh"].decode()) != PINS:
        die("the sha256 pins in swe_pilot.py and swe-host-entry.sh differ")
    for name, want in PINS.items():
        got = sha256(files[name])
        if got != want:
            die(f"sha256 mismatch for {name}: expected {want}, got {got}")
    return files


def manifest(files: dict[str, bytes]) -> str:
    """sha256sum format, checked in the pod with `sha256sum --strict -c`."""
    return "".join(f"{sha256(files[n])}  {n}\n" for n in HOST_FILES + ARTIFACTS)


def content_digest(data: dict[str, str], binary: dict[str, bytes]) -> str:
    items = sorted([(k, sha256(v.encode())) for k, v in data.items()]
                   + [(k, sha256(v)) for k, v in binary.items()])
    return sha256(json.dumps(items).encode())


def configmap(name: str, data: dict[str, str], binary: dict[str, bytes] | None = None) -> dict:
    binary = binary or {}
    obj = {"apiVersion": "v1", "kind": "ConfigMap",
           "metadata": {"name": name, "namespace": NS, "labels": dict(PART_OF),
                        "annotations": {DIGEST_KEY: content_digest(data, binary)}},
           "data": data}
    if binary:
        obj["binaryData"] = {k: base64.b64encode(v).decode() for k, v in binary.items()}
    return obj


def oracle_configmap(directory: Path) -> dict | None:
    """ConfigMap kimi-swe-oracle (test only): oracle_server.py, its data and manifest.

    None if a file is missing or unreadable, or if the data does not match the manifest's
    data_sha256 (oracle/oracle_data.py not run yet). Never fatal: only `model --oracle` needs it.
    """
    try:
        text = {n: (directory / n).read_bytes().decode("utf-8") for n in ORACLE_TEXT}
        data = (directory / ORACLE_DATA).read_bytes()
        listed = json.loads(text["oracle-manifest.json"])
    except (OSError, ValueError):                 # includes UnicodeDecodeError, JSONDecodeError
        return None
    if not isinstance(listed, dict) or listed.get("data_sha256") != sha256(data):
        return None
    return configmap(CM_ORACLE, text, {ORACLE_DATA: data})


def set_num_instances(text: str, n: int) -> str:
    """client-pilot.yaml text with swe_bench num_instances n: only the PILOT 4 line changes.

    Fails closed: n must be 1 to MAX_INSTANCES and the file must have exactly one
    `num_instances: N  # PILOT 4` line. The rewritten line says where its number came from.
    """
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_INSTANCES:
        die(f"--num-instances must be 1 to {MAX_INSTANCES}")
    found = NUM_INSTANCES_RE.findall(text)
    if len(found) != 1:
        die(f"{CLIENT_CONFIG} must have exactly one 'num_instances: N  # PILOT 4' line, "
            f"found {len(found)}")
    old = int(found[0][1])
    if old == n:
        return text
    return NUM_INSTANCES_RE.sub(
        lambda m: f"{m['lead']}{n}{m['tail']}, set by swe_pilot.py --num-instances (file: {old})",
        text)


def client_config(num_instances: int | None = None) -> str:
    """client-pilot.yaml as ConfigMap kimi-swe-client carries it (test only: --num-instances)."""
    text = (HERE / CLIENT_CONFIG).read_text()
    return text if num_instances is None else set_num_instances(text, num_instances)


def desired_configmaps(args) -> list[dict]:
    files = gather_files(Path(args.drop_dir), Path(args.artifacts))
    host = {n: files[n].decode() for n in HOST_FILES}
    host["MANIFEST.sha256"] = manifest(files)
    oracle_dir = getattr(args, "oracle_dir", None)            # None or "": no oracle ConfigMap
    oracle = oracle_configmap(Path(oracle_dir)) if oracle_dir else None
    return [
        configmap(CM_DROP, {n: files[n].decode() for n in DROP_FILES}),
        configmap(CM_HOST, host, {n: files[n] for n in ARTIFACTS}),
        # The key stays client-pilot.yaml whatever --num-instances says: CLIENT_SCRIPT reads it.
        configmap(CM_CLIENT, {CLIENT_CONFIG: client_config(getattr(args, "num_instances", None)),
                              "perf-dataset-unused.jsonl": ""}),
        configmap(CM_MODEL, {"model-extra.yaml": (HERE / "model-extra.yaml").read_text()}),
    ] + ([oracle] if oracle else [])


def namespace_obj() -> dict:
    return {"apiVersion": "v1", "kind": "Namespace",
            "metadata": {"name": NS, "labels": dict(PART_OF)}}


def require_configmaps(kube: Kube, args, names: tuple[str, ...]) -> None:
    """Refuse to start pods on ConfigMaps that differ from the local files."""
    for cm in desired_configmaps(args):
        name = cm["metadata"]["name"]
        if name not in names:
            continue
        live = kube.get("configmap", name)
        want = cm["metadata"]["annotations"][DIGEST_KEY]
        have = ((live or {}).get("metadata", {}).get("annotations") or {}).get(DIGEST_KEY)
        if have != want:
            hint = " (pass the same --num-instances to both, or to neither)" if name == CM_CLIENT else ""
            die(f"ConfigMap {name} is missing or stale; run: swe_pilot.py configmaps{hint}")


def require_oracle(kube: Kube, args) -> None:
    """`model --oracle`: refuse unless ConfigMap kimi-swe-oracle matches the local oracle files."""
    oracle_dir = getattr(args, "oracle_dir", None)
    want = oracle_configmap(Path(oracle_dir)) if oracle_dir else None
    if want is None:
        die(f"no usable oracle data in {oracle_dir or '(empty --oracle-dir)'}: run "
            "oracle/oracle_data.py, then swe_pilot.py configmaps")
    live = kube.get("configmap", CM_ORACLE)
    have = ((live or {}).get("metadata", {}).get("annotations") or {}).get(DIGEST_KEY)
    if have != want["metadata"]["annotations"][DIGEST_KEY]:
        die(f"ConfigMap {CM_ORACLE} is missing or stale; run: swe_pilot.py configmaps")


# ----------------------------------------------------------------------------- pods
def pod_skeleton(name: str, role: str, node: str, *, run_id: str | None = None,
                 grace: int = 30, host_network: bool = True) -> dict:
    labels = {**PART_OF, "kimi-swe/role": role}
    if run_id:
        labels["kimi-swe/run-id"] = run_id
    spec = {"nodeSelector": {"kubernetes.io/hostname": node}, "tolerations": TOLERATIONS,
            "restartPolicy": "Never", "automountServiceAccountToken": False,
            "enableServiceLinks": False, "terminationGracePeriodSeconds": grace,
            "containers": [], "volumes": []}
    if host_network:
        # The service, the model and the client talk over the node's 127.0.0.1.
        spec.update(hostNetwork=True, dnsPolicy="ClusterFirstWithHostNet")
    return {"apiVersion": "v1", "kind": "Pod",
            "metadata": {"name": name, "namespace": NS, "labels": labels}, "spec": spec}


def vol_host(name: str, path: str, kind: str) -> dict:
    return {"name": name, "hostPath": {"path": path, "type": kind}}


def vol_cm(name: str, cm: str) -> dict:
    return {"name": name, "configMap": {"name": cm, "defaultMode": 0o444}}


def vol_mem(name: str, size: str) -> dict:
    return {"name": name, "emptyDir": {"medium": "Memory", "sizeLimit": size}}


def mnt(name: str, path: str, read_only: bool = False) -> dict:
    return {"name": name, "mountPath": path, **({"readOnly": True} if read_only else {})}


def non_root(uid: int, gid: int) -> dict:
    return {"runAsUser": uid, "runAsGroup": gid, "runAsNonRoot": True,
            "allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}}


def host_pod(name: str, role: str, node: str, run_id: str, *, mode: str,
             image_cache: str = "run", prewarm: str | None = None,
             smoke_image: str | None = None, handoff: bool = False,
             reports_dir: str | None = None, metadata_block: bool = True,
             alloc: str | None = None) -> dict:
    """The swe-host pod (service or smoke mode). The only privileged pod of the pilot.

    Production (`serve`, service mode): handoff=True puts the drop's connection file under
    HANDOFF_DIR/<run id>/ for the recipe's benchmark task; reports_dir mounts the node's report
    directory read-only so the teardown leak scan also covers this run's report. alloc (serve
    only) labels the pod with the sflow allocation of the run it serves; nothing else changes.
    metadata_block (default on; CS-NET-001): the entry script blocks the service user's access
    to the GCE metadata server while the service runs. False is a pilot-only fallback.
    """
    service = mode == "service"
    if (handoff or reports_dir) and not service:
        die("handoff and the report scan need service mode")
    if alloc is not None and (not service or not ALLOC_RE.fullmatch(alloc)):
        die(f"invalid allocation id {alloc!r} for {name}")
    pod = pod_skeleton(name, role, node, run_id=run_id, grace=900 if service else 300)
    if alloc is not None:
        pod["metadata"]["labels"][ALLOC_LABEL] = alloc
    env = [{"name": "SWE_POD_MODE", "value": mode}, {"name": "SWE_RUN_ID", "value": run_id},
           {"name": "SWE_IMAGE_CACHE", "value": image_cache},
           {"name": "SWE_METADATA_BLOCK", "value": "1" if metadata_block else "0"},
           # The dedicated service user; the entry script has no defaults for these.
           {"name": "SWE_SERVICE_USER", "value": SERVICE_USER},
           {"name": "SWE_SERVICE_UID", "value": str(SERVICE_UID)},
           {"name": "SWE_SERVICE_GID", "value": str(SERVICE_GID)}]
    if prewarm:
        env.append({"name": "SWE_PREWARM", "value": prewarm})
    if smoke_image:
        env.append({"name": "SWE_SMOKE_IMAGE", "value": smoke_image})
    if handoff:
        env.append({"name": "SWE_HANDOFF", "value": "1"})
    if reports_dir:
        env.append({"name": "SWE_SCAN_REPORTS", "value": REPORTS_MOUNT})
    mounts = [mnt("cm-host", "/opt/kimi-swe/cm/host", True), mnt("cm-drop", "/opt/kimi-swe/cm/drop", True),
              mnt("priv", "/run/kimi-swe"), mnt("state", STATE), mnt("scratch", "/tmp"),
              mnt("dshm", "/dev/shm")]
    vols = [vol_cm("cm-host", CM_HOST), vol_cm("cm-drop", CM_DROP), vol_mem("priv", "16Mi"),
            vol_host("state", STATE, "DirectoryOrCreate"),
            vol_host("scratch", f"{STATE}/tmp", "DirectoryOrCreate"), vol_mem("dshm", "16Gi")]
    if metadata_block:
        # The node's xtables lock, mounted the way kube-proxy mounts it (FileOrCreate).
        mounts.append(mnt("xtables-lock", XTABLES_LOCK))
        vols.append(vol_host("xtables-lock", XTABLES_LOCK, "FileOrCreate"))
    if service:
        mounts.append(mnt("enroot-secret", "/etc/kimi-swe/secret", True))
        vols.append({"name": "enroot-secret", "secret": {
            "secretName": SECRET, "defaultMode": 0o400,
            "items": [{"key": ".credentials", "path": ".credentials"}]}})
    if reports_dir:
        # Created on every node by ../../mk-report-dir.yaml; "Directory" fails loudly if not.
        mounts.append(mnt("reports", REPORTS_MOUNT, True))
        vols.append(vol_host("reports", reports_dir, "Directory"))
    pod["spec"]["containers"] = [{
        "name": "host", "image": CLIENT_IMAGE, "imagePullPolicy": "IfNotPresent",
        "command": ["/bin/bash", "/opt/kimi-swe/cm/host/swe-host-entry.sh"],
        "env": env,
        # Enroot needs user namespaces, overlay and a fresh /proc for the sandboxes. The entry
        # script runs the service as the dedicated service user and gives sandboxes a minimal /dev.
        "securityContext": {"privileged": True, "runAsUser": 0, "runAsGroup": 0},
        "resources": {"requests": {"cpu": "16", "memory": "64Gi"},
                      "limits": {"cpu": "96", "memory": "384Gi"}},
        "volumeMounts": mounts}]
    pod["spec"]["volumes"] = vols
    return pod


def model_pod(node: str, keep_default_caps: bool = False) -> dict:
    pod = pod_skeleton(POD["model"], "model", node, grace=60)
    sc: dict = {"allowPrivilegeEscalation": False}
    if not keep_default_caps:
        sc["capabilities"] = {"drop": ["ALL"]}
    health = f"http://127.0.0.1:{MODEL_PORT}/health"
    probe = ["python3", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request."
             f"urlopen({health!r}, timeout=5).status == 200 else 1)"]
    serve = ["trtllm-serve", MODEL_DIR, "--host", "127.0.0.1", "--port", str(MODEL_PORT),
             "--backend", "pytorch", "--tp_size", "1",
             "--extra_llm_api_options", "/etc/kimi-swe/model/model-extra.yaml"]
    pod["spec"]["containers"] = [{
        "name": "model", "image": TRTLLM_IMAGE, "imagePullPolicy": "IfNotPresent",
        # Start it the way sflow starts the recipes' tasks (`bash -l <entrypoint>`): the image's
        # ENV LD_LIBRARY_PATH lacks the TensorRT libs, which come from its shell init files
        # (BASH_ENV=/etc/bash.bashrc, ENV=/etc/shinit_v2). Exec'ing trtllm-serve directly fails
        # with "libnvonnxparser.so.10: cannot open shared object file". The arguments stay
        # positional ("$0" "$@"), so nothing is interpolated into the shell string.
        "command": ["bash", "-l", "-c", 'exec "$0" "$@"', *serve],
        "env": [{"name": "HF_HUB_OFFLINE", "value": "1"}, {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
                {"name": "TIKTOKEN_RS_CACHE_DIR", "value": HARMONY_DIR},
                {"name": "PYTHONUNBUFFERED", "value": "1"},
                # The image runs as root; Open MPI (worker spawn) refuses root without these,
                # the same as the recipes' `mpirun --allow-run-as-root`.
                {"name": "OMPI_ALLOW_RUN_AS_ROOT", "value": "1"},
                {"name": "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "value": "1"}],
        "securityContext": sc,
        "resources": {"requests": {"cpu": "16", "memory": "128Gi", "nvidia.com/gpu": "1"},
                      "limits": {"cpu": "32", "memory": "256Gi", "nvidia.com/gpu": "1"}},
        "readinessProbe": {"exec": {"command": probe}, "periodSeconds": 10, "timeoutSeconds": 8,
                           "failureThreshold": 3},
        "volumeMounts": [mnt("model", MODEL_DIR, True), mnt("harmony", HARMONY_DIR, True),
                         mnt("model-cfg", "/etc/kimi-swe/model", True), mnt("dshm", "/dev/shm")]}]
    pod["spec"]["volumes"] = [vol_host("model", MODEL_DIR, "Directory"),
                              vol_host("harmony", HARMONY_DIR, "Directory"),
                              vol_cm("model-cfg", CM_MODEL), vol_mem("dshm", "32Gi")]
    return pod


def oracle_pod(node: str) -> dict:
    """Test only (`model --oracle`): kimi-swe-model as the SWE-bench oracle at 127.0.0.1:18000.

    oracle/oracle_server.py (standard library only) from ConfigMap kimi-swe-oracle in the client
    image: no GPU, no host paths, no shell, user nobody, read-only root filesystem. Same pod
    name, port and readiness probe as the model, so host, client, status and down work as is.
    """
    pod = pod_skeleton(POD["model"], "model", node)
    pod["metadata"]["labels"]["kimi-swe/variant"] = "oracle"
    serve = ["python3", "-I", f"{ORACLE_MOUNT}/oracle_server.py",
             "--data", f"{ORACLE_MOUNT}/{ORACLE_DATA}",
             "--manifest", f"{ORACLE_MOUNT}/oracle-manifest.json",
             "--host", "127.0.0.1", "--port", str(MODEL_PORT), "--model-name", ORACLE_MODEL_NAME]
    pod["spec"]["containers"] = [{
        "name": "oracle", "image": CLIENT_IMAGE, "imagePullPolicy": "IfNotPresent",
        "command": serve,
        # -I ignores PYTHON* variables (the server flushes its own log lines); these two cover
        # the readiness probe's python3.
        "env": [{"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                {"name": "PYTHONUNBUFFERED", "value": "1"}],
        "securityContext": {**non_root(NOBODY, NOBODY), "readOnlyRootFilesystem": True,
                            "seccompProfile": {"type": "RuntimeDefault"}},
        "resources": {"requests": {"cpu": "1", "memory": "1Gi"},
                      "limits": {"cpu": "2", "memory": "2Gi"}},
        "readinessProbe": model_pod(node)["spec"]["containers"][0]["readinessProbe"],
        "volumeMounts": [mnt("oracle", ORACLE_MOUNT, True)]}]
    pod["spec"]["volumes"] = [vol_cm("oracle", CM_ORACLE)]
    return pod


# The drop's benchmark task (templates/ifb.yaml), BENCHMARK_MODE=acc, after two pilot waits:
# for the service's connection file and for the pilot endpoint (instead of checks.py ready).
CLIENT_SCRIPT = r"""set -euo pipefail
log() { printf '[swe-client %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
export SWE_CONNECTION="$KIMI_SWE_RUN/swe-connection.json"
mkdir -p -m 0700 "$KIMI_SWE_RUN/client/home"
cd "$KIMI_SWE_RUN/client"
export HOME="$KIMI_SWE_RUN/client/home"
has_url() { python3 -c 'import json,os,sys; sys.exit(0 if json.load(open(os.environ["SWE_CONNECTION"])).get("url") else 1)' 2>/dev/null; }
healthy() { python3 -c 'import sys,urllib.request; sys.exit(0 if urllib.request.urlopen("http://127.0.0.1:18000/health", timeout=5).status == 200 else 1)' 2>/dev/null; }
log "waiting for the SWE service connection file and the pilot endpoint"
for _ in $(seq 360); do has_url && break; sleep 5; done
has_url || { log "no SWE service URL after 30 min"; exit 1; }
for _ in $(seq 360); do healthy && break; sleep 5; done
healthy || { log "pilot endpoint not healthy after 30 min"; exit 1; }
PILOT_MODEL_NAME=$(python3 -c 'import json,urllib.request; print(json.load(urllib.request.urlopen("http://127.0.0.1:18000/v1/models", timeout=10))["data"][0]["id"])')
export PILOT_MODEL_NAME
log "pilot endpoint model: $PILOT_MODEL_NAME"
# ---- from here on: the drop's benchmark task script
unset PYTHONPATH PYTHONHOME UV_PROJECT_ENVIRONMENT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
BENCHMARK_MODE=acc
case "${BENCHMARK_MODE}" in perf|acc|both) ;; *) echo "BENCHMARK_MODE must be perf, acc, or both" >&2; exit 2;; esac
export SWE_SERVICE_URL=http://127.0.0.1:1 SWE_SERVICE_TOKEN=disabled
if [ "${BENCHMARK_MODE}" != perf ]; then
  export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0
  SWE_SERVICE_URL=$(python3 -c 'import json,os; print(json.load(open(os.environ["SWE_CONNECTION"]))["url"])')
  SWE_SERVICE_TOKEN=$(python3 -c 'import json,os; print(json.load(open(os.environ["SWE_CONNECTION"]))["token"])')
  export SWE_SERVICE_URL SWE_SERVICE_TOKEN
fi
exec python3 -m inference_endpoint.main benchmark from-config -c /etc/kimi-swe/client/client-pilot.yaml --mode "${BENCHMARK_MODE}"
"""


def client_pod(node: str, run_id: str, uid: int, gid: int) -> dict:
    run_dir = f"{STATE}/runs/{run_id}"
    pod = pod_skeleton(POD["client"], "client", node, run_id=run_id, grace=120)
    pod["spec"]["containers"] = [{
        "name": "client", "image": CLIENT_IMAGE, "imagePullPolicy": "IfNotPresent",
        "command": ["/bin/bash", "-c", CLIENT_SCRIPT],
        "env": [{"name": "KIMI_SWE_RUN", "value": run_dir},
                {"name": "SFLOW_WORKFLOW_OUTPUT_DIR", "value": run_dir},
                {"name": "MODEL_PATH", "value": MODEL_DIR},
                {"name": "DATASET_PATH", "value": "/etc/kimi-swe/client/perf-dataset-unused.jsonl"},
                # The client image has no passwd line for the service uid: name the user for
                # tools that ask (getpass reads LOGNAME and USER first). HOME: CLIENT_SCRIPT.
                {"name": "USER", "value": SERVICE_USER}, {"name": "LOGNAME", "value": SERVICE_USER}],
        "securityContext": non_root(uid, gid),
        "resources": {"requests": {"cpu": "4", "memory": "16Gi"},
                      "limits": {"cpu": "16", "memory": "64Gi"}},
        "volumeMounts": [mnt("run", run_dir), mnt("model", MODEL_DIR, True),
                         mnt("client-cfg", "/etc/kimi-swe/client", True)]}]
    pod["spec"]["volumes"] = [vol_host("run", run_dir, "Directory"),
                              vol_host("model", MODEL_DIR, "Directory"), vol_cm("client-cfg", CM_CLIENT)]
    return pod


def collect_pod(node: str, run_id: str, uid: int, gid: int, name: str = POD["collect"]) -> dict:
    run_dir = f"{STATE}/runs/{run_id}"
    pod = pod_skeleton(name, "collect", node, run_id=run_id, grace=5, host_network=False)
    pod["spec"]["activeDeadlineSeconds"] = 3900
    pod["spec"]["containers"] = [{
        "name": "collect", "image": CLIENT_IMAGE, "imagePullPolicy": "IfNotPresent",
        "command": ["sleep", "3600"], "securityContext": non_root(uid, gid),
        "resources": {"requests": {"cpu": "1", "memory": "1Gi"}, "limits": {"cpu": "4", "memory": "4Gi"}},
        "volumeMounts": [mnt("run", run_dir, True)]}]
    pod["spec"]["volumes"] = [vol_host("run", run_dir, "Directory")]
    return pod


# ----------------------------------------------------------------------------- checks
def check_node(kube: Kube, node: str, allow_busy: bool, alloc: str | None = None) -> None:
    """The node is arm64, schedulable and Ready, and (unless allow_busy) runs no pods in
    namespace default.

    alloc (serve): the client node runs the served run's own pods, so busy is expected, but a
    pod of ANOTHER sflow run there would share the node's network namespace with the SWE
    sandboxes: refused. Pods of no sflow run (no sflow.ai/allocation label, for example the
    staging DaemonSets) are listed, not refused, as v1.0.1 allowed every pod there.
    """
    if not NODE_RE.fullmatch(node):
        die(f"invalid node name {node!r}")
    if alloc is not None and not ALLOC_RE.fullmatch(alloc):
        die(f"invalid allocation id {alloc!r}")
    obj = kube.get("node", node, namespaced=False)
    if not obj:
        die(f"node {node} not found")
    if obj["metadata"].get("labels", {}).get("kubernetes.io/arch") != "arm64":
        die(f"node {node} is not arm64")
    if obj["spec"].get("unschedulable"):
        die(f"node {node} is cordoned")
    if not any(c.get("type") == "Ready" and c.get("status") == "True"
               for c in obj.get("status", {}).get("conditions", [])):
        die(f"node {node} is not Ready")
    if kube.dry_run:
        return                              # nothing is scheduled on a dry run
    pods = json.loads(kube.run("get", "pods", "-n", "default", "--field-selector",
                               f"spec.nodeName={node}", "-o", "json").stdout)["items"]
    live = [p for p in pods if p.get("status", {}).get("phase") not in ("Succeeded", "Failed")]
    busy = [p["metadata"]["name"] for p in live]
    if alloc is not None:
        def allocation(p: dict):
            return ((p.get("metadata") or {}).get("labels") or {}).get(SFLOW_ALLOC_LABEL)
        other = [p["metadata"]["name"] for p in live if allocation(p) not in (None, alloc)]
        if other:
            die(f"node {node} runs {len(other)} pod(s) of another sflow run ({', '.join(other[:4])}); "
                f"serve attaches only to allocation {alloc}")
        plain = [p["metadata"]["name"] for p in live if allocation(p) is None]
        if plain:
            log(f"node {node} also runs {len(plain)} pod(s) of no sflow run in namespace default "
                f"({', '.join(plain[:4])})")
        return
    if busy and not allow_busy:
        die(f"node {node} runs {len(busy)} pod(s) in namespace default ({', '.join(busy[:4])}). "
            "Pick an idle node, or pass --allow-busy-node.")


def check_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id or ""):
        die("--run-id must be 1-63 characters of [A-Za-z0-9._-], starting and ending alphanumeric")
    return run_id


def down_hint(name: str) -> str:
    """The `down` command that removes pod `name` (a pilot pod or a run's own host/collect pod)."""
    for role, base in POD.items():
        if name == base:
            return f"swe_pilot.py down {role}"
        if name.startswith(base + "-") and role in ("host", "collect"):
            return f"swe_pilot.py down {role} --alloc {name[len(base) + 1:]}"
    return f"kubectl delete pod {name} -n {NS}"


def ensure_absent(kube: Kube, name: str) -> None:
    phase, cs, _ = pod_state(kube, name)
    if phase == "Absent":
        return
    if terminated(phase, cs):
        log(f"removing the finished pod {name} ({phase})")
        kube.delete_pod(name)
        return
    die(f"pod {name} exists ({phase}); remove it first with: {down_hint(name)}")


def require_host(kube: Kube, run_id: str) -> None:
    phase, cs, pod = pod_state(kube, POD["host"])
    if phase != "Running" or "running" not in cs.get("state", {}):
        die(f"{POD['host']} is not running; start it with: swe_pilot.py host")
    if pod["metadata"]["labels"].get("kimi-swe/run-id") != run_id:
        die(f"{POD['host']} serves run {pod['metadata']['labels'].get('kimi-swe/run-id')}, not {run_id}")


# ----------------------------------------------------------------------------- commands
def cmd_configmaps(args, kube: Kube) -> int:
    objs = desired_configmaps(args)
    kube.apply(namespace_obj())
    for obj in objs:
        kube.apply(obj)
    log(f"ConfigMaps in {NS} match the local files; drop scripts and artifacts match their pins")
    if getattr(args, "oracle_dir", None):
        oracle = next((o for o in objs if o["metadata"]["name"] == CM_ORACLE), None)
        count = json.loads(oracle["data"]["oracle-manifest.json"]).get("count") if oracle else None
        log(f"{CM_ORACLE} (test only, for `model --oracle`): "
            + (f"applied, {count} tasks" if oracle else
               f"skipped, no usable data in {args.oracle_dir} (run oracle/oracle_data.py)"))
    n = getattr(args, "num_instances", None)
    if n is not None:
        log(f"{CM_CLIENT} (test only): swe_bench num_instances {n} from --num-instances; "
            f"run `client` with the same --num-instances {n}")
    return 0


def build_credentials(path: Path) -> tuple[str, str, str]:
    """Return (login, token, .credentials text) from a 0600 netrc owned by the caller."""
    try:
        st = path.stat()
    except OSError:
        die(f"{path} not found. Create it with a hidden prompt; see swe/README.md")
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
        die(f"{path} must be a regular file owned by you")
    if st.st_mode & 0o077:
        die(f"{path} must not be readable by others: chmod 600 {path}")
    try:
        parsed = netrc.netrc(str(path))
    except netrc.NetrcParseError as exc:
        die(f"cannot parse {path} (line {exc.lineno}); its content is not shown")
    entry = parsed.authenticators("registry.gitlab.com") or parsed.authenticators("gitlab.com")
    if not entry:
        die(f"{path} has no 'machine registry.gitlab.com' or 'machine gitlab.com' entry")
    login, _, token = entry
    if not login or not LOGIN_RE.fullmatch(login):
        die("the netrc login must be a GitLab user name ([A-Za-z0-9._-])")
    if not token or not TOKEN_RE.fullmatch(token):
        die("the netrc password must be a GitLab token (16+ characters of [A-Za-z0-9._-])")
    # Enroot matches 'machine registry.gitlab.com login USER' exactly; the registry's token
    # service lives on gitlab.com, so that host needs the same entry.
    text = "".join(f"machine {host} login {login} password {token}\n"
                   for host in ("registry.gitlab.com", "gitlab.com"))
    return login, token, text


def cmd_secret(args, kube: Kube) -> int:
    login, token, text = build_credentials(Path(args.netrc).expanduser())
    kube.redact.add(token)
    kube.redact.add(text)
    obj = {"apiVersion": "v1", "kind": "Secret", "type": "Opaque",
           "metadata": {"name": SECRET, "namespace": NS, "labels": dict(PART_OF)},
           "data": {".credentials": base64.b64encode(text.encode()).decode()}}
    if kube.dry_run:
        log(f"dry run: Secret {SECRET} built for login {login} (not sent; token not shown)")
        return 0
    kube.apply(namespace_obj())
    kube.apply(obj)
    log(f"Secret {SECRET}: Enroot .credentials for registry.gitlab.com and gitlab.com, "
        f"login {login} (token not shown). Remove it after the pilot: swe_pilot.py down secret")
    return 0


def _show_dry_run(pod: dict) -> None:
    print(json.dumps(pod, indent=2))


def cmd_smoke(args, kube: Kube) -> int:
    check_node(kube, args.node, args.allow_busy_node)
    require_configmaps(kube, args, (CM_HOST, CM_DROP))
    run_id = check_run_id(args.run_id or f"smoke-{utc_stamp()}")
    if args.smoke_image and not IMAGE_RE.fullmatch(args.smoke_image):
        die("--smoke-image has unexpected characters")
    no_block = getattr(args, "no_metadata_block", False)
    if no_block:
        log("WARNING: --no-metadata-block: the smoke pod runs without the metadata block "
            "(CS-NET-001). Approved fallback only.")
    pod = host_pod(POD["smoke"], "smoke", args.node, run_id, mode="smoke",
                   smoke_image=args.smoke_image, metadata_block=not no_block)
    if kube.dry_run:
        _show_dry_run(pod)
        kube.create(pod)
        return 0
    ensure_absent(kube, POD["smoke"])
    kube.create(pod)
    log_path = LOCAL_DIR / "logs" / f"{run_id}-smoke.log"
    try:
        wait_started(kube, POD["smoke"])
        lines, _ = follow_logs(kube, POD["smoke"], log_path)
        phase, cs, _ = pod_state(kube, POD["smoke"])
        code = cs.get("state", {}).get("terminated", {}).get("exitCode")
    finally:
        if not args.keep:
            kube.delete_pod(POD["smoke"], timeout=400)
    ok = code == 0 and any(line.rstrip().endswith("SMOKE_OK") for line in lines)
    log(f"smoke {'PASSED' if ok else 'FAILED'} (exit {code}). Log: {log_path}. "
        f"Run directory on {args.node}: {STATE}/runs/{run_id}")
    return 0 if ok else 1


def cmd_model(args, kube: Kube) -> int:
    oracle = getattr(args, "oracle", False)
    if oracle and args.keep_default_caps:
        die("--oracle runs without capabilities; --keep-default-caps is for the model only")
    check_node(kube, args.node, args.allow_busy_node)
    if oracle:
        require_oracle(kube, args)
        pod = oracle_pod(args.node)
    else:
        require_configmaps(kube, args, (CM_MODEL,))
        pod = model_pod(args.node, args.keep_default_caps)
    if kube.dry_run:
        _show_dry_run(pod)
        kube.create(pod)
        return 0
    ensure_absent(kube, POD["model"])
    kube.create(pod)
    wait_started(kube, POD["model"])
    wait_ready(kube, POD["model"], args.timeout)
    log(f"{POD['model']} is ready at 127.0.0.1:{MODEL_PORT} on {args.node}"
        + (f" (test-only oracle: public gold patches as model {ORACLE_MODEL_NAME})" if oracle else ""))
    return 0


def cmd_host(args, kube: Kube) -> int:
    no_block = getattr(args, "no_metadata_block", False)
    if no_block and (args.handoff or args.scan_reports):
        die("--no-metadata-block is a pilot fallback; production (--handoff, --scan-reports) "
            "always runs with the metadata block (CS-NET-001)")
    check_node(kube, args.node, args.allow_busy_node)
    require_configmaps(kube, args, (CM_HOST, CM_DROP))
    run_id = check_run_id(args.run_id)
    if args.prewarm and (not PREWARM_RE.fullmatch(args.prewarm) or not args.persist_cache):
        die("--prewarm needs --persist-cache and takes first:N or a comma list of instance ids")
    if no_block:
        log("WARNING: --no-metadata-block: the service user and its sandboxes can reach the GCE "
            "metadata server (CS-NET-001). Approved fallback only.")
    pod = host_pod(POD["host"], "host", args.node, run_id, mode="service",
                   image_cache="persistent" if args.persist_cache else "run", prewarm=args.prewarm,
                   handoff=args.handoff, reports_dir=REPORTS_DIR if args.scan_reports else None,
                   metadata_block=not no_block)
    if kube.dry_run:
        _show_dry_run(pod)
        kube.create(pod)
        return 0
    if not kube.exists("secret", SECRET):
        die(f"Secret {SECRET} is missing; run: swe_pilot.py secret")
    ensure_absent(kube, POD["host"])
    kube.create(pod)
    save_record(run_id, node=args.node, started=utc_stamp())
    wait_started(kube, POD["host"])
    want = ("SWE_READY seen",) + (("PREWARM_DONE", "PREWARM_FAILED") if args.prewarm else ())
    seen: set[str] = set()

    def stop(line: str) -> bool:
        m = re.search(r"service user \S+ \((\d+):(\d+)\)", line)
        if m:
            save_record(run_id, uid=int(m[1]), gid=int(m[2]))
        seen.update(w for w in want if w in line)
        ready = "SWE_READY seen" in seen
        return ready and (not args.prewarm or bool(seen & {"PREWARM_DONE", "PREWARM_FAILED"}))

    _, why = follow_logs(kube, POD["host"], LOCAL_DIR / "logs" / f"{run_id}-host.log",
                         stop=stop, timeout=args.timeout)
    if why != "stop":
        die(f"{POD['host']} {'exited' if why == 'exited' else 'timed out'} before SWE_READY")
    log(f"service ready for run {run_id}. Next: swe_pilot.py client --node {args.node} --run-id {run_id}")
    return 0


def cmd_client(args, kube: Kube) -> int:
    check_node(kube, args.node, args.allow_busy_node)
    require_configmaps(kube, args, (CM_CLIENT,))
    run_id = check_run_id(args.run_id)
    uid, gid = app_ids(args, run_id)
    pod = client_pod(args.node, run_id, uid, gid)
    if kube.dry_run:
        _show_dry_run(pod)
        kube.create(pod)
        return 0
    require_host(kube, run_id)
    if not pod_state(kube, POD["model"])[1].get("ready"):
        log(f"warning: {POD['model']} is not ready; the client waits up to 30 min for it")
    ensure_absent(kube, POD["client"])
    kube.create(pod)
    wait_started(kube, POD["client"])
    log_path = LOCAL_DIR / "logs" / f"{run_id}-client.log"
    if not args.follow:
        log(f"{POD['client']} started. Watch: swe_pilot.py status, or rerun with --follow")
        return 0
    follow_logs(kube, POD["client"], log_path)
    phase, cs, _ = pod_state(kube, POD["client"])
    code = cs.get("state", {}).get("terminated", {}).get("exitCode")
    log(f"{POD['client']} finished with exit {code}. Log: {log_path}")
    return 0 if code == 0 else 1


def run_pod_name(role: str, alloc: str) -> str:
    """serve's pods for one run: kimi-swe-host-<alloc> and kimi-swe-collect-<alloc>. Per run, so
    the services of two concurrent acc/both runs (each on its own client node) never collide."""
    if role not in ("host", "collect") or not ALLOC_RE.fullmatch(alloc or ""):
        die(f"invalid {role} pod for allocation {alloc!r}")
    return f"{POD[role]}-{alloc}"


_OUR_POD = re.compile(r"kimi-swe-[a-z0-9]([a-z0-9-]{0,200}[a-z0-9])?")


def cmd_status(args, kube: Kube) -> int:
    out = kube.run("get", "pods", "-n", NS, "-o", "wide", check=False)
    print(kube.redact((out.stdout or out.stderr).decode(errors="replace")).rstrip())
    names = kube.run("get", "pods", "-n", NS, "-o", "name", check=False).stdout.decode(errors="replace")
    present = {n.removeprefix("pod/") for n in names.split()}
    runs = sorted(n for n in present if _OUR_POD.fullmatch(n) and n not in POD.values()
                  and any(n.startswith(f"{POD[r]}-") for r in ("host", "collect")))
    for name in [n for n in POD.values() if n in present] + runs:
        print(f"--- {name}: last {args.tail} log lines")
        print(kube.logs(name, tail=args.tail).rstrip())
    return 0


_TEARDOWN = re.compile(r"POSTCHECK|leakscan|exited with|run finished|cleanup|FATAL|WARNING"
                       r"|metadata block")


def down_host(kube: Kube, wait_client: bool = True, name: str = POD["host"]) -> bool:
    """Delete a host pod, follow its teardown checks; True if they all passed.

    wait_client: refuse while the pilot client still runs. `serve` passes False: there the
    client is the recipe's benchmark pod, which it has already seen finish. name: the pilot's
    kimi-swe-host (default) or one run's kimi-swe-host-<alloc> (serve).
    """
    phase, _, pod = pod_state(kube, name)
    if phase == "Absent":
        log(f"{name}: not present (no teardown checks to follow)")
        return False
    if wait_client:
        client_phase, client_cs, _ = pod_state(kube, POD["client"])
        if not terminated(client_phase, client_cs):
            # The client pod writes the run directory that the host's teardown checks read as root.
            die(f"{POD['client']} is {client_phase}. Stop it first: swe_pilot.py down client")
    run_id = pod["metadata"]["labels"].get("kimi-swe/run-id", "unknown")
    log_path = LOCAL_DIR / "logs" / f"{run_id}-host.log"
    result: dict = {}
    follower = threading.Thread(
        target=lambda: result.update(lines=follow_logs(
            kube, name, log_path.with_suffix(".teardown.log"),
            echo=lambda line: bool(_TEARDOWN.search(line)))[0]),
        daemon=True)
    follower.start()
    time.sleep(3)
    log(f"deleting {name}: SIGTERM; the drop stops the service, then the checks run "
        "(up to 15 min)")
    kube.delete_pod(name, timeout=1000)
    follower.join(timeout=60)
    lines = result.get("lines", [])
    ok = any("service run finished: ok" in line for line in lines)
    log(f"host teardown {'ok' if ok else 'NOT ok (see log)'}. Log: {log_path.with_suffix('.teardown.log')}")
    return ok


def cmd_down(args, kube: Kube) -> int:
    alloc = getattr(args, "alloc", None)
    if alloc is not None:
        # One run's own pod from `serve` (for example after serve was killed): same teardown.
        if args.target not in ("host", "collect"):
            die("--alloc goes with `down host` or `down collect` (serve's per-run pods)")
        name = run_pod_name(args.target, alloc)
        if args.target == "host":
            down_host(kube, wait_client=False, name=name)
        else:
            kube.delete_pod(name)
            log(f"{name} removed")
        return 0
    order = ["client", "host", "model", "smoke", "collect", "secret"]
    if args.target == "all":
        names = kube.run("get", "pods", "-n", NS, "-l", ALLOC_LABEL, "-o", "name",
                         check=False).stdout.decode(errors="replace").split()
        if names:
            log(f"WARNING: serve's per-run pods stay ({' '.join(names)}): a production run may use "
                "them. Remove one with: swe_pilot.py down host --alloc ALLOCATION")
    for target in (order if args.target == "all" else [args.target]):
        if target == "secret":
            kube.run("delete", "secret", SECRET, "-n", NS, "--ignore-not-found")
            log(f"Secret {SECRET} removed")
        elif target == "host":
            down_host(kube)
        else:
            kube.delete_pod(POD[target], timeout=400 if target == "smoke" else 180)
            log(f"{POD[target]} removed")
    return 0


def _scan_local(root: Path, secret: str) -> list[str]:
    needle, hits = secret.encode(), []
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            try:
                if needle in path.read_bytes():
                    hits.append(str(path))
            except OSError:
                continue
    return hits


def hits_inside(hits: list, root: str) -> list:
    """The leak-scan hits (swe_probe.py leakscan files_with_hits) under `root`. An entry without
    a path counts as inside, so an unexpected report format blocks the copy."""
    prefix = root.rstrip("/") + "/"
    return [h for h in hits if not isinstance(h, dict) or not isinstance(h.get("path"), str)
            or h["path"].startswith(prefix)]


def collect_run(kube: Kube, node: str, run_id: str, uid: int, gid: int, dest_root: Path,
                netrc_path: Path, force: bool = False,
                pod_name: str = POD["collect"]) -> tuple[Path, bool]:
    """Copy STATE/runs/<run_id> from `node`, without caches, to dest_root/<run_id>/ (+ .tgz).

    Refuses (die) without the host's teardown leak report, or if it found a secret in the
    directory being copied, unless `force`. Returns (dest, ok); ok is False if the pod-side scan
    found a secret elsewhere (never copied; e.g. a production run's report on the node) or the
    registry token is found in the local copy. pod_name: the pilot's kimi-swe-collect
    (default) or one run's kimi-swe-collect-<alloc> (serve).
    """
    run_dir = f"{STATE}/runs/{run_id}"
    dest, tgz = dest_root / run_id, dest_root / f"{run_id}.tgz"
    if dest.exists() or tgz.exists():
        die(f"{dest} or {tgz} exists already")
    pod = collect_pod(node, run_id, uid, gid, name=pod_name)
    if kube.dry_run:
        _show_dry_run(pod)
        kube.create(pod)
        return dest, True
    ensure_absent(kube, pod_name)
    kube.create(pod)
    leaks_elsewhere = False
    try:
        wait_started(kube, pod_name, timeout=600)
        exe = ["exec", pod_name, "-n", NS, "--"]
        p = kube.run(*exe, "cat", f"{run_dir}/postcheck-leaks.json", check=False)
        if p.returncode != 0:
            if not force:
                die("no postcheck-leaks.json: tear the host pod down with `swe_pilot.py down host` "
                    "first (or pass --force)")
            log("warning: no postcheck-leaks.json; copying anyway (--force)")
        else:
            report = json.loads(p.stdout)
            hits = report.get("files_with_hits") or []
            inside = hits_inside(hits, run_dir)
            if inside and not force:
                die(f"the pod-side leak scan found secrets in {len(inside)} file(s) of {run_dir}; "
                    "not copying (see the host teardown log)")
            if len(hits) > len(inside):
                leaks_elsewhere = True
                log(f"WARNING: the pod-side leak scan found secrets in {len(hits) - len(inside)} "
                    f"file(s) outside {run_dir}; those are not copied (see the host teardown log)")
            log(f"pod-side leak scan: {json.dumps(report.get('hits'))} over "
                f"{report.get('files_scanned')} files")
        partial = tgz.with_suffix(".partial")
        with open_private(partial, "wb") as f:
            p = subprocess.run([*kube.base(), *exe, "tar", "-C", run_dir, "-czf", "-",
                                *COLLECT_EXCLUDES, "."],
                               stdout=f, stderr=subprocess.PIPE, timeout=3600, check=False)
        if p.returncode not in (0, 1):     # GNU tar: 1 = some file changed while read
            die(f"tar failed (exit {p.returncode}): {kube.redact(p.stderr.decode(errors='replace'))[-1000:]}")
        partial.rename(tgz)
    finally:
        kube.run("delete", "pod", pod_name, "-n", NS, "--ignore-not-found", "--wait=false",
                 check=False)
    dest.mkdir(parents=True, mode=0o700)
    with tarfile.open(tgz) as tar:
        tar.extractall(dest, filter="data")
    log(f"copied {run_dir} from {node} to {dest} ({tgz.stat().st_size / 1e6:.1f} MB archive)")
    if netrc_path.is_file():
        _, token, _ = build_credentials(netrc_path)
        hits = _scan_local(dest, token)
        log("local scan for the registry token: " + (f"FOUND in {hits}" if hits else "not found"))
        if hits:
            return dest, False
    return dest, not leaks_elsewhere


def cmd_collect(args, kube: Kube) -> int:
    run_id = check_run_id(args.run_id)
    if not NODE_RE.fullmatch(args.node):
        die(f"invalid node name {args.node!r}")
    uid, gid = app_ids(args, run_id)
    _, ok = collect_run(kube, args.node, run_id, uid, gid, Path(args.dest).expanduser().resolve(),
                        Path(args.netrc).expanduser(), force=args.force)
    return 0 if ok else 1


# ----------------------------------------------------------------------------- serve
class SflowRun(NamedTuple):
    """One sflow run of a Kimi v1.0.2 SWE recipe, seen from its output folder on this workstation."""
    point: str
    run_dir: Path
    run_id: str      # folder name == basename($SFLOW_WORKFLOW_OUTPUT_DIR) inside the run's pods
    node: str        # the client node: the run's benchmark (client) pod runs there
    alloc: str       # sflow allocation id: the run's pods are named <task>-<alloc>


# The v1.0.2 recipes, as serve depends on them. `--check` compares these with the generator's
# SWE_CONTRACT and with the swe_<mode> recipe files it wrote.
CLIENT_TASK, FRONTEND_TASK = "benchmark", "frontend"   # the run's pods serve reads and follows
SWE_MODES = ("acc", "both")            # swe_<mode>/<N>nodes/point_<C>/: recipe, launch_test.sh, and
                                       # the sflow_output/ folder its runs write
MODE_MARKER = "eval_method: swe_bench_scorer"   # a line of an acc/both run's client.yaml
CONNECTION_FILE = "swe-connection.json"         # swe-host-entry.sh: $STATE/handoff/<run id>/
URL_PREFIX = "http://127.0.0.1:"       # the only service URL the recipe's wait accepts: the host
                                       # pod shares the client node's network (hostNetwork)
SWE_WAIT_S = 2700                      # the recipe waits this long from its benchmark's start:
                                       # host start to SWE_READY must fit (README, Production)
CLIENT_NODE_SPLIT = 80                 # CLIENT_NODE & FRONTEND_NODE: node 2 up to C80, node 4 from C192
PULL_SECRET_SUBDIR = "with-pull-secret"
KNOWN_POINTS = (12, 13, 16, 24, 32, 48, 64, 80, 192, 288, 384, 448, 512, 576, 640, 1024)


def _short(value) -> str:
    return repr(value)[:80]


def task_pod(run_dir: Path, task: str) -> tuple[str, str]:
    """(node, allocation) of the one Pod in the manifest sflow wrote for `task` of this run.

    sflow writes each task's applied manifest (a v1 List) to <run dir>/<task>/<task>.k8s.yaml
    and names the task's pod <task>-<allocation> in the run's namespace, labelled
    sflow.ai/allocation=<allocation>. Anything else raises ValueError: serve never guesses.
    """
    import yaml   # PyYAML, which sflow and the generator need too; only `serve`/`--check` use it
    rel = f"{task}/{task}.k8s.yaml"
    tdir = run_dir / task
    mpath = tdir / f"{task}.k8s.yaml"
    if tdir.is_symlink() or mpath.is_symlink():
        raise ValueError(f"{rel} is a symlink")
    doc = yaml.safe_load(mpath.read_text())
    items = doc.get("items") if isinstance(doc, dict) and doc.get("kind") == "List" else None
    if not isinstance(items, list):
        raise ValueError(f"{rel} is not a List manifest")
    pods = [i for i in items if isinstance(i, dict) and i.get("kind") == "Pod"]
    if len(pods) != 1:
        raise ValueError(f"expected one Pod in {rel}, found {len(pods)}")
    meta, spec = pods[0].get("metadata"), pods[0].get("spec")
    if not isinstance(meta, dict) or not isinstance(spec, dict):
        raise ValueError(f"the Pod in {rel} has no metadata or spec")
    labels, selector = meta.get("labels") or {}, spec.get("nodeSelector") or {}
    if not isinstance(labels, dict) or not isinstance(selector, dict):
        raise ValueError(f"the Pod in {rel} has malformed labels or nodeSelector")
    node, alloc = selector.get("kubernetes.io/hostname"), labels.get(SFLOW_ALLOC_LABEL)
    if not isinstance(node, str) or not NODE_RE.fullmatch(node):
        raise ValueError(f"no valid node in the {task} Pod's nodeSelector: {_short(node)}")
    if (not isinstance(alloc, str) or not ALLOC_RE.fullmatch(alloc)
            or meta.get("name") != f"{task}-{alloc}"):
        raise ValueError(f"unexpected {task} Pod name/allocation: "
                         f"{_short(meta.get('name'))}/{_short(alloc)}")
    if meta.get("namespace") != SFLOW_NS:
        raise ValueError(f"the {task} Pod is in namespace {_short(meta.get('namespace'))}, "
                         f"not {SFLOW_NS} (serve follows pods there only)")
    return node, alloc


def parse_run_dir(run_dir: Path) -> SflowRun:
    """The client node and the allocation of one sflow run, from the manifests sflow wrote for it.

    v1.0.2 has no etcd. The recipes pin the benchmark (the client) to variables.CLIENT_NODE, so
    its Pod's nodeSelector names the node the host must run on: the recipe accepts only a
    service on that node's 127.0.0.1. The frontend manifest must name the same allocation (one
    folder, one run, never a mix of two runs' files). Parallel-tree layout:
    .../swe_<acc|both>/<N>nodes/point_<C>/sflow_output/<run id>/.
    """
    out_dir, point_dir = run_dir.parent, run_dir.parent.parent
    group_dir, mode_dir = point_dir.parent, point_dir.parent.parent
    point = point_dir.name.removeprefix("point_")
    if (out_dir.name != "sflow_output"
            or not point_dir.name.startswith("point_") or not re.fullmatch(r"[0-9]{1,5}", point)
            or not re.fullmatch(r"[0-9]{1,3}nodes", group_dir.name)
            or mode_dir.name not in [f"swe_{m}" for m in SWE_MODES]):
        raise ValueError(
            f"not a swe_<acc|both>/<N>nodes/point_<C>/sflow_output/<run> folder: {run_dir}")
    if not RUN_ID_RE.fullmatch(run_dir.name):
        raise ValueError(f"the run folder name is not a valid run id: {_short(run_dir.name)}")
    node, alloc = task_pod(run_dir, CLIENT_TASK)
    _, frontend_alloc = task_pod(run_dir, FRONTEND_TASK)
    if frontend_alloc != alloc:
        raise ValueError(f"the {CLIENT_TASK} and {FRONTEND_TASK} manifests name different "
                         f"allocations ({alloc}, {frontend_alloc}): not one run")
    return SflowRun(point, run_dir, run_dir.name, node, alloc)


def needs_swe(run_dir: Path) -> bool:
    """acc/both runs list the swe_bench dataset in their rendered client.yaml: the v1.0.1 drop's
    block, which the generator carries into the swe_<mode> recipes (sflow resolves its
    `{% if variables.BENCHMARK_MODE != 'perf' %}` guards). One line is the marker."""
    cpath = run_dir / "client.yaml"
    if cpath.is_symlink():
        raise ValueError("client.yaml is a symlink")
    return any(line.strip() == MODE_MARKER for line in cpath.read_text().splitlines())


def run_ready(run_dir: Path) -> bool:
    """sflow has launched the run's benchmark pod (its manifest exists) and rendered client.yaml."""
    bpath = run_dir / CLIENT_TASK / f"{CLIENT_TASK}.k8s.yaml"
    cpath = run_dir / "client.yaml"
    return (bpath.is_file() and not bpath.is_symlink()
            and cpath.is_file() and not cpath.is_symlink())


UNREADABLE_RETRY_S = 120   # serve: how long a run folder may stay unparseable (being written)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def list_run_dirs(point_dirs: list[Path]) -> set[Path]:
    """The run folders of the watched points in this swe_<mode> node group (no symlinks)."""
    found: set[Path] = set()
    for pdir in point_dirs:
        out = pdir / "sflow_output"
        if out.is_dir() and not out.is_symlink():
            found.update(d for d in out.iterdir() if d.is_dir() and not d.is_symlink())
    return found


def pod_phase(kube: Kube, name: str, ns: str) -> str:
    """The pod's phase, "Absent", or "Unknown" on any kubectl error. Never dies."""
    try:
        p = kube.run("get", "pod", name, "-n", ns, "-o", "jsonpath={.status.phase}",
                     "--ignore-not-found", check=False, timeout=60)
    except SystemExit:
        return "Unknown"
    if p.returncode != 0:
        return "Unknown"
    return p.stdout.decode(errors="replace").strip() or "Absent"


def run_pod_phase(kube: Kube, task: str, run: SflowRun) -> str:
    """The phase of this run's own pod <task>-<alloc> in SFLOW_NS, "Absent", "Unknown" on any
    kubectl error, or "Foreign" if a pod of that name does not carry the run's allocation label
    (then it is not this run's pod). Never dies."""
    try:
        p = kube.run("get", "pod", f"{task}-{run.alloc}", "-n", SFLOW_NS, "-o",
                     'jsonpath={.status.phase}{"|"}{.metadata.labels.sflow\\.ai/allocation}',
                     "--ignore-not-found", check=False, timeout=60)
    except SystemExit:
        return "Unknown"
    if p.returncode != 0:
        return "Unknown"
    out = p.stdout.decode(errors="replace").strip()
    if not out:
        return "Absent"
    phase, sep, label = out.partition("|")
    if not sep or label != run.alloc:
        return "Foreign"
    return phase or "Unknown"


def wait_run_end(kube: Kube, run: SflowRun, host: str, timeout: float, poll: float,
                 stopping: threading.Event) -> str:
    """Wait until this run's benchmark pod has finished, or the run is gone; return why.

    Follows only this run's pods, benchmark-<alloc> and frontend-<alloc> (each must carry the
    run's allocation label), and its host pod. Never dies: an API error only delays the
    decision, so a blip cannot take the SWE service away from a running benchmark.
    """
    t0 = time.monotonic()
    seen_bench = seen_frontend = False
    last = ""
    while True:
        bench = run_pod_phase(kube, CLIENT_TASK, run)
        frontend = run_pod_phase(kube, FRONTEND_TASK, run)
        host_phase = pod_phase(kube, host, NS)
        state = f"benchmark {bench}, frontend {frontend}, {host} {host_phase}"
        if state != last:
            log(f"run {run.run_id}: {state}")
            last = state
        for task, phase in ((CLIENT_TASK, bench), (FRONTEND_TASK, frontend)):
            if phase == "Foreign":
                return (f"pod {task}-{run.alloc} does not carry allocation {run.alloc}: not this "
                        "run's pod; stopping the SWE service")
        if bench in ("Succeeded", "Failed"):
            return f"benchmark pod {bench}"
        seen_bench |= bench not in ("Absent", "Unknown")
        seen_frontend |= frontend not in ("Absent", "Unknown")
        if bench == "Absent" and seen_bench:
            return "benchmark pod removed"
        if bench == "Absent" and frontend == "Absent" and seen_frontend:
            return "the run ended before its benchmark started (frontend pod removed)"
        if not (seen_bench or seen_frontend) and time.monotonic() - t0 > 900:
            return "no frontend or benchmark pod for this run within 15 min"
        if host_phase in ("Succeeded", "Failed", "Absent"):
            return f"{host} is {host_phase}: the SWE service ended before the benchmark"
        if stopping.is_set():
            return "serve was asked to stop"
        if time.monotonic() - t0 > timeout:
            return f"timed out after {int(timeout)} s"
        stopping.wait(poll)


def serve_one(kube: Kube, args, run: SflowRun, stopping: threading.Event) -> bool:
    """One production run: host on the run's client node, SWE_READY, benchmark ends, teardown,
    then collect into <run dir>/swe_service/. True if the teardown checks and the copy both passed.

    The host and collect pods are this run's own (kimi-swe-host-<alloc>, kimi-swe-collect-<alloc>):
    a second serve, for a concurrent acc/both run on its own nodes, never touches them, and this
    one never tears down a pod it did not create.
    """
    host = run_pod_name("host", run.alloc)
    log(f"run {run.run_id} (point {run.point}): client node {run.node}, allocation {run.alloc}, "
        f"host pod {host}")
    bench = run_pod_phase(kube, CLIENT_TASK, run)
    if bench in ("Succeeded", "Failed", "Foreign"):
        log(f"ERROR: run {run.run_id}: its benchmark pod is {bench} already; no SWE service started")
        return False
    if pod_state(kube, host)[0] != "Absent":
        log(f"ERROR: run {run.run_id}: {host} exists already (another serve?); not touching it")
        return False
    check_node(kube, run.node, allow_busy=True, alloc=run.alloc)   # this run's pods only
    pod = host_pod(host, "host", run.node, run.run_id, mode="service",
                   handoff=True, reports_dir=REPORTS_DIR, alloc=run.alloc)
    kube.create(pod)             # dies on AlreadyExists (a concurrent serve won): nothing to undo
    save_record(run.run_id, node=run.node, started=utc_stamp(), sflow_run_dir=str(run.run_dir),
                allocation=run.alloc, host_pod=host)
    try:
        wait_started(kube, host)

        def ready(line: str) -> bool:
            m = re.search(r"service user \S+ \((\d+):(\d+)\)", line)
            if m:
                save_record(run.run_id, uid=int(m[1]), gid=int(m[2]))
            return "SWE_READY seen" in line

        _, why = follow_logs(kube, host, LOCAL_DIR / "logs" / f"{run.run_id}-host.log",
                             echo=lambda line: line.startswith("[swe-host"), stop=ready,
                             timeout=args.ready_timeout)
        if why != "stop":
            log(f"ERROR: {host} {why} before SWE_READY; this run's benchmark will stop at its "
                "SWE wait")
        else:
            log(f"run {run.run_id}: SWE service ready; waiting for the benchmark to finish")
            log(f"run {run.run_id}: {wait_run_end(kube, run, host, args.run_timeout, args.poll, stopping)}")
    finally:
        try:
            teardown_ok = down_host(kube, wait_client=False, name=host)
        except SystemExit:
            log(f"ERROR: could not remove {host}; run: swe_pilot.py down host --alloc {run.alloc}")
            return False
    rec = load_record(run.run_id)
    uid, gid = int(rec.get("uid", SERVICE_UID)), int(rec.get("gid", SERVICE_GID))
    if uid <= 0 or gid <= 0:
        die("the collect pod must run as a non-root user")
    _, copied_ok = collect_run(kube, run.node, run.run_id, uid, gid, run.run_dir / "swe_service",
                               Path(args.netrc).expanduser(),
                               pod_name=run_pod_name("collect", run.alloc))
    return teardown_ok and copied_ok


def cmd_serve(args, kube: Kube) -> int:
    group = Path(args.group).expanduser()
    group = group if group.is_absolute() else K8S_ROOT / group
    if not group.is_dir() or group.is_symlink():
        die(f"no such node group: {group}")
    if (not re.fullmatch(r"[0-9]{1,3}nodes", group.name)
            or group.parent.name not in [f"swe_{m}" for m in SWE_MODES]):
        die(f"{group} is not a SWE node group (expected .../swe_<{'|'.join(SWE_MODES)}>/<N>nodes, "
            f"e.g. {DEFAULT_GROUP})")
    points = args.points or sorted((p.name.removeprefix("point_") for p in group.glob("point_*")),
                                   key=lambda c: int(c) if c.isdigit() else 0)
    point_dirs = []
    for c in points:
        pdir = group / f"point_{c}"
        if not re.fullmatch(r"[0-9]{1,5}", c) or not pdir.is_dir() or pdir.is_symlink():
            die(f"no point folder point_{c} in {group}")
        point_dirs.append(pdir)
    if kube.dry_run:
        die("serve starts pods as runs appear and has no dry run. Validate the production pod "
            "with: swe_pilot.py --dry-run host --node NODE --run-id ID --handoff --scan-reports")
    if not kube.exists("secret", SECRET):
        die(f"Secret {SECRET} is missing; run: swe_pilot.py secret")
    require_configmaps(kube, args, (CM_HOST, CM_DROP))
    seen = list_run_dirs(point_dirs)
    unreadable: dict[Path, float] = {}       # run folder -> first failed parse (monotonic)
    stopping = threading.Event()

    def on_signal(signum, _frame) -> None:
        if not stopping.is_set():
            log(f"serve: signal {signum}: stopping after the current run's teardown, if any")
        stopping.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)   # keep serving if the terminal goes away
    log(f"serve: watching {' '.join(d.name for d in point_dirs)} in {group} "
        f"({len(seen)} existing run(s) ignored). Stop with SIGTERM.")
    served = problems = 0
    while not stopping.is_set():
        for d in sorted(list_run_dirs(point_dirs) - seen, key=_mtime):
            if not run_ready(d):
                continue                       # sflow has not launched the benchmark pod yet
            try:
                run, swe = parse_run_dir(d), needs_swe(d)
            except Exception as exc:           # unreadable manifest: retry, then report
                first = unreadable.setdefault(d, time.monotonic())
                if time.monotonic() - first < UNREADABLE_RETRY_S:
                    continue                   # sflow may still be writing it
                seen.add(d)
                log(f"ERROR: run {d.name}: {exc}")
                problems += 1
                continue
            seen.add(d)
            if not swe:
                log(f"ERROR: run {run.run_id} (point {run.point}): its client.yaml has no "
                    f"`{MODE_MARKER}` line, so it is not an acc/both run; no SWE service")
                problems += 1
                continue
            try:
                ok = serve_one(kube, args, run, stopping)
            except SystemExit:
                ok = False                     # serve_one tears down what it created
            served += 1
            problems += 0 if ok else 1
            log(f"run {run.run_id}: {'done' if ok else 'PROBLEMS (see above)'}")
            if args.once:
                stopping.set()
            break                              # list the folders again after each run
        else:
            stopping.wait(args.poll)
    log(f"serve: stopped. {served} run(s) served, {problems} with problems")
    return 1 if problems else 0


# ----------------------------------------------------------------------------- check
def load_generator(args) -> tuple[Path, object]:
    """Load the v1.0.2 recipe generator without running its main(). Never invokes subprocess."""
    raw = getattr(args, "generator", None) or os.environ.get("KIMI_SWE_GENERATOR") or ""
    if raw:
        path = Path(raw).expanduser().resolve()
    else:
        path = GENERATOR
        if not path.is_file() or "SWE_CONTRACT" not in path.read_text(errors="replace"):
            spec = importlib.util.find_spec("generate_point_recipes")
            if spec and spec.origin:
                path = Path(spec.origin).resolve()
    if not path.is_file() or path.is_symlink():
        die(f"generator not found (or is a symlink): {path}")
    spec = importlib.util.spec_from_file_location("kimi_v102_generator", path)
    if spec is None or spec.loader is None:
        die(f"cannot load generator module from {path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        die(f"loading generator {path} failed: {exc}")
    return path, mod


def check_swe_recipe_text(c: int, mode: str, n_nodes: int, text: str, launch: str,
                          label: str) -> list[str]:
    """Validate one rendered SWE recipe and its launch_test.sh against the serve contract."""
    import yaml
    problems: list[str] = []
    try:
        doc = yaml.safe_load(text)
    except Exception as exc:
        return [f"{label}: invalid YAML: {exc}"]
    if not isinstance(doc, dict):
        return [f"{label}: recipe is not a YAML mapping"]
    vars_ = {k: (v or {}).get("value") for k, v in (doc.get("variables") or {}).items()
             if isinstance(v, dict)}
    want_vars = {
        "CONCURRENCY": c,
        "TOTAL_NODES": n_nodes,
        "CLIENT_NODE": 2 if c <= CLIENT_NODE_SPLIT else 4,
        "FRONTEND_NODE": 2 if c <= CLIENT_NODE_SPLIT else 4,
        "BENCHMARK_MODE": mode,
        "CLIENT_IMAGE": CLIENT_IMAGE,
        "SWE_WAIT_S": SWE_WAIT_S,
        "K8S_NAMESPACE": SFLOW_NS,
    }
    for k, want in want_vars.items():
        if vars_.get(k) != want:
            problems.append(f"{label}: variable {k}={vars_.get(k)!r} != {want!r}")
    arts = {a.get("name"): a for a in (doc.get("artifacts") or []) if isinstance(a, dict)}
    if (arts.get("SWE_HANDOFF_DIR") or {}).get("uri") != f"fs://{HANDOFF_DIR}":
        problems.append(
            f"{label}: SWE_HANDOFF_DIR uri={(arts.get('SWE_HANDOFF_DIR') or {}).get('uri')!r} "
            f"!= 'fs://{HANDOFF_DIR}'")
    if (arts.get("REPORTS_DIR") or {}).get("uri") != f"fs://{REPORTS_DIR}":
        problems.append(
            f"{label}: REPORTS_DIR uri={(arts.get('REPORTS_DIR') or {}).get('uri')!r} "
            f"!= 'fs://{REPORTS_DIR}'")
    cl = (arts.get("CLIENT_CONFIG") or {}).get("content") or ""
    if not any(ln.strip() == MODE_MARKER for ln in cl.splitlines()):
        problems.append(f"{label}: CLIENT_CONFIG lacks `{MODE_MARKER}`")
    if "report_dir: ${REPORT_DIR}" not in cl:
        problems.append(f"{label}: CLIENT_CONFIG lacks `report_dir: ${{REPORT_DIR}}`")
    if ("swebench_service_url: ${SWEBENCH_SERVICE_URL}" not in cl
            or "swebench_service_auth_token: ${SWEBENCH_SERVICE_AUTH_TOKEN}" not in cl):
        problems.append(
            f"{label}: CLIENT_CONFIG lacks swebench_service_url / swebench_service_auth_token")
    tasks = {t.get("name"): t for t in ((doc.get("workflow") or {}).get("tasks") or [])
             if isinstance(t, dict)}
    if FRONTEND_TASK not in tasks:
        problems.append(f"{label}: missing task {FRONTEND_TASK!r}")
    bench = tasks.get(CLIENT_TASK) or {}
    if bench.get("operator") != "client":
        problems.append(f"{label}: {CLIENT_TASK} operator={bench.get('operator')!r} != 'client'")
    if bench.get("depends_on") != [FRONTEND_TASK]:
        problems.append(
            f"{label}: {CLIENT_TASK} depends_on={bench.get('depends_on')!r} != [{FRONTEND_TASK!r}]")
    indices = ((bench.get("resources") or {}).get("nodes") or {}).get("indices")
    if indices != ["${{ variables.CLIENT_NODE }}"]:
        problems.append(f"{label}: {CLIENT_TASK} nodes.indices={indices!r}")
    script_items = [str(x) for x in (bench.get("script") or [])]
    script = "\n".join(script_items)
    if CONNECTION_FILE not in script or URL_PREFIX not in script:
        problems.append(f"{label}: {CLIENT_TASK} script lacks the {CONNECTION_FILE} / {URL_PREFIX} wait")
    if "SWE token redacted in" not in script:
        problems.append(f"{label}: {CLIENT_TASK} script lacks post-run credential redaction")
    if '--mode "${BENCHMARK_MODE}"' not in script:
        problems.append(f"{label}: {CLIENT_TASK} script lacks `--mode \"${{BENCHMARK_MODE}}\"`")
    if any(x.lstrip().startswith("exec ") for x in script_items):
        problems.append(f"{label}: {CLIENT_TASK} script uses `exec` (skips post-run redaction/mirror)")
    rname = f"kimi_k3_gke_disagg_point{c}_swe_{mode}.yaml"
    if f'RECIPE="{rname}"' not in launch:
        problems.append(f"{label}: launch_test.sh RECIPE != {rname!r}")
    if f'REQUIRED_NODES="${{REQUIRED_NODES:-{n_nodes}}}"' not in launch:
        problems.append(f"{label}: launch_test.sh REQUIRED_NODES != {n_nodes}")
    return problems


def run_check(args) -> int:
    """Offline self-check of pins, ConfigMaps, the generator's SWE_CONTRACT, and SWE recipes.

    Never invokes kubectl or touches the cluster.
    """
    problems: list[str] = []
    # 1. Local pins and ConfigMaps (also checks swe-host-entry.sh PINNED == PINS).
    cms = desired_configmaps(args)
    for cm in cms:
        name = cm["metadata"]["name"]
        size = len(json.dumps(cm).encode())
        if size >= 900 * 1024:
            problems.append(f"ConfigMap {name} is {size} B (>= 900 KiB limit)")
    if getattr(args, "oracle_dir", None):
        if not any(cm["metadata"]["name"] == CM_ORACLE for cm in cms):
            problems.append(f"ConfigMap {CM_ORACLE} missing or invalid in {args.oracle_dir}")
    c5, c200 = client_config(5), client_config(200)
    if "num_instances: 5" not in c5 or "num_instances: 200" not in c200:
        problems.append(f"{CLIENT_CONFIG}: client_config(--num-instances) rewrite failed")

    # 2. Driver constants & image pin.
    want_digest = "@sha256:1523051b141437b25083924d6428082f435713516eee71c54ae207b393900b2d"
    if not CLIENT_IMAGE.endswith(want_digest):
        problems.append(f"CLIENT_IMAGE {CLIENT_IMAGE!r} does not end with {want_digest}")
    old_digest = "6cf5" + "16d0"
    for lineno, line in enumerate(Path(__file__).read_text().splitlines(), 1):
        if not line.lstrip().startswith("#") and old_digest in line:
            problems.append(f"swe_pilot.py:{lineno} still references v1.0.1 digest {old_digest}")
    if DEFAULT_GROUP != "v1.0.3/kimi-k3/swe_acc/8nodes":
        problems.append(f"DEFAULT_GROUP={DEFAULT_GROUP!r} != 'v1.0.3/kimi-k3/swe_acc/8nodes'")

    # 3. Generator SWE_CONTRACT cross-check.
    gen_path, gen = load_generator(args)
    contract = getattr(gen, "SWE_CONTRACT", None)
    if not isinstance(contract, dict):
        problems.append(f"{gen_path}: SWE_CONTRACT missing or not a dict")
    else:
        wait_val = (contract.get("wait_s") if "wait_s" in contract
                    else contract.get("wait_timeout_s"))
        checks = [
            ("modes", tuple(contract.get("modes") or ()), SWE_MODES),
            ("handoff_dir", contract.get("handoff_dir"), HANDOFF_DIR),
            ("connection_file", contract.get("connection_file"), CONNECTION_FILE),
            ("wait_s", wait_val, SWE_WAIT_S),
            ("reports_dir", contract.get("reports_dir"), REPORTS_DIR),
            ("mode_marker", contract.get("mode_marker"), MODE_MARKER),
            ("url_prefix", contract.get("url_prefix"), URL_PREFIX),
        ]
        for key, got, want in checks:
            if got != want:
                problems.append(f"SWE_CONTRACT[{key!r}]={got!r} != swe_pilot {want!r}")
        if "client_image" in contract and contract.get("client_image") != CLIENT_IMAGE:
            problems.append(
                f"SWE_CONTRACT['client_image']={contract.get('client_image')!r} != {CLIENT_IMAGE!r}")
        if "pull_secret_subdir" in contract and contract.get("pull_secret_subdir") != PULL_SECRET_SUBDIR:
            problems.append(
                f"SWE_CONTRACT['pull_secret_subdir']={contract.get('pull_secret_subdir')!r} "
                f"!= {PULL_SECRET_SUBDIR!r}")
        if "client_task" in contract and contract.get("client_task") != CLIENT_TASK:
            problems.append(
                f"SWE_CONTRACT['client_task']={contract.get('client_task')!r} != {CLIENT_TASK!r}")
        if "frontend_task" in contract and contract.get("frontend_task") != FRONTEND_TASK:
            problems.append(
                f"SWE_CONTRACT['frontend_task']={contract.get('frontend_task')!r} != {FRONTEND_TASK!r}")
        if "tasks" in contract:
            tasks_val = contract.get("tasks")
            if (tuple(tasks_val) if isinstance(tasks_val, (list, tuple)) else tasks_val) not in (
                    (CLIENT_TASK, FRONTEND_TASK),
                    {"client": CLIENT_TASK, "frontend": FRONTEND_TASK}):
                problems.append(f"SWE_CONTRACT['tasks']={tasks_val!r} != {(CLIENT_TASK, FRONTEND_TASK)!r}")
        if "k8s_namespace" in contract and contract.get("k8s_namespace") != SFLOW_NS:
            problems.append(
                f"SWE_CONTRACT['k8s_namespace']={contract.get('k8s_namespace')!r} != {SFLOW_NS!r}")
        if getattr(gen, "CLIENT_IMAGE", None) != CLIENT_IMAGE:
            problems.append(f"generator CLIENT_IMAGE={getattr(gen, 'CLIENT_IMAGE', None)!r} != {CLIENT_IMAGE!r}")
        if getattr(gen, "REPORTS_DIR", None) != REPORTS_DIR:
            problems.append(f"generator REPORTS_DIR={getattr(gen, 'REPORTS_DIR', None)!r} != {REPORTS_DIR!r}")
        expected = getattr(gen, "EXPECTED", None)
        if not isinstance(expected, dict) or tuple(sorted(expected)) != KNOWN_POINTS:
            problems.append(f"generator EXPECTED keys != {KNOWN_POINTS}")
        else:
            want_client = {c: (2 if c <= CLIENT_NODE_SPLIT else 4) for c in KNOWN_POINTS}
            want_frontend = {c: (2 if c <= CLIENT_NODE_SPLIT else 4) for c in KNOWN_POINTS}
            c_map = (contract.get("client_node") if "client_node" in contract
                     else contract.get("client_node_by_point"))
            f_map = (contract.get("frontend_node") if "frontend_node" in contract
                     else contract.get("frontend_node_by_point"))
            if c_map != want_client:
                problems.append(f"SWE_CONTRACT client_node={c_map!r} != {want_client!r}")
            if f_map != want_frontend:
                problems.append(f"SWE_CONTRACT frontend_node={f_map!r} != {want_frontend!r}")
            for c, tup in expected.items():
                if not isinstance(tup, tuple) or len(tup) < 9:
                    problems.append(f"generator EXPECTED[{c}] malformed: {tup!r}")
                    continue
                if tup[8] != want_client[c] or tup[7] != want_frontend[c]:
                    problems.append(
                        f"generator EXPECTED[{c}] frontend/client=({tup[7]}, {tup[8]}) "
                        f"!= ({want_frontend[c]}, {want_client[c]})")

    # 4. Rendered SWE recipe trees (when --recipes-root is given or swe_<mode>/ exists on disk).
    raw_root = getattr(args, "recipes_root", None) or os.environ.get("KIMI_SWE_RECIPES_ROOT") or ""
    recipes_root = Path(raw_root).expanduser().resolve() if raw_root else SWE_DIR.parent
    checked_recipes = 0
    if raw_root or any((recipes_root / f"swe_{m}").exists() for m in SWE_MODES):
        roots = [recipes_root]
        ps_root = recipes_root / PULL_SECRET_SUBDIR
        if any((ps_root / f"swe_{m}").exists() for m in SWE_MODES):
            roots.append(ps_root)
        expected = getattr(gen, "EXPECTED", None) if isinstance(getattr(gen, "EXPECTED", None), dict) else {}
        for rroot in roots:
            tag = f" [{PULL_SECRET_SUBDIR}]" if rroot != recipes_root else ""
            for mode in SWE_MODES:
                mdir = rroot / f"swe_{mode}"
                if not mdir.is_dir() or mdir.is_symlink():
                    problems.append(f"missing SWE recipe tree {mdir}")
                    continue
                pdirs = sorted(mdir.glob("*nodes/point_*"))
                if not pdirs:
                    problems.append(f"no *nodes/point_* folders in {mdir}")
                    continue
                for pdir in pdirs:
                    m = re.fullmatch(r"point_(\d+)", pdir.name)
                    if not m or int(m.group(1)) not in expected or pdir.is_symlink():
                        problems.append(f"stray or invalid point folder {pdir}")
                        continue
                    c = int(m.group(1))
                    n_nodes = expected[c][6]
                    if pdir.parent.name != f"{n_nodes}nodes":
                        problems.append(f"{pdir}: parent {pdir.parent.name} != {n_nodes}nodes")
                    rfile = pdir / f"kimi_k3_gke_disagg_point{c}_swe_{mode}.yaml"
                    lfile = pdir / "launch_test.sh"
                    if not rfile.is_file() or rfile.is_symlink():
                        problems.append(f"missing recipe {rfile}")
                        continue
                    if not lfile.is_file() or lfile.is_symlink():
                        problems.append(f"missing launch_test.sh in {pdir}")
                        continue
                    if not (lfile.stat().st_mode & 0o111):
                        problems.append(f"{lfile} is not executable")
                    problems.extend(check_swe_recipe_text(
                        c, mode, n_nodes, rfile.read_text(), lfile.read_text(),
                        f"C{c} swe_{mode}{tag}"))
                    checked_recipes += 1
    if problems:
        die("check failed:\n  - " + "\n  - ".join(problems))
    log(f"CHECK OK: {len(cms)} ConfigMaps (<900 KiB), {len(PINS)} sha256 pins, "
        f"CLIENT_IMAGE d3c57f2, SWE_CONTRACT ({gen_path.name}), "
        f"{checked_recipes} rendered SWE recipe(s) checked")
    return 0


# ----------------------------------------------------------------------------- CLI
def num_instances_arg(value: str) -> int:
    """--num-instances: 1 to 3 ASCII digits, 1 to MAX_INSTANCES (nothing else reaches the YAML)."""
    if not re.fullmatch(r"[0-9]{1,3}", value) or not 1 <= int(value) <= MAX_INSTANCES:
        raise argparse.ArgumentTypeError(f"must be a whole number from 1 to {MAX_INSTANCES}")
    return int(value)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="See swe/README.md for the pilot steps.")
    ap.add_argument("--context", help="kubectl context (default: current context)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate objects with the API server (--dry-run=server); change nothing")
    ap.add_argument("--check", action="store_true",
                    help="offline self-check: verify sha256 pins, ConfigMap sizes, CLIENT_IMAGE, "
                         "the generator's SWE_CONTRACT, and any rendered swe_<mode> recipes "
                         "(no kubectl)")
    ap.add_argument("--generator", default=os.environ.get("KIMI_SWE_GENERATOR", ""),
                    help=f"path to generate_point_recipes.py for --check (default: {GENERATOR})")
    ap.add_argument("--recipes-root", default=os.environ.get("KIMI_SWE_RECIPES_ROOT", ""),
                    help=f"root directory containing swe_acc/ and swe_both/ for --check "
                         f"(default: {SWE_DIR.parent})")
    ap.add_argument("--drop-dir", default=os.environ.get("KIMI_SWE_DROP_DIR", str(DEFAULT_DROP)),
                    help="the drop's NVIDIA/src/kimi-k3/scripts directory")
    ap.add_argument("--artifacts", default=str(SWE_DIR / "artifacts"),
                    help="directory with the pinned Enroot .deb and service tarball")
    ap.add_argument("--oracle-dir", default=str(ORACLE_DIR),
                    help=f"test only: oracle_server.py and oracle_data.py's output for ConfigMap "
                         f"{CM_ORACLE} ('' leaves it out of `configmaps`)")
    ap.add_argument("--num-instances", type=num_instances_arg, metavar="N",
                    help=f"test only: swe_bench num_instances in ConfigMap {CM_CLIENT} (default: "
                         f"{CLIENT_CONFIG} as is, 10; the drop runs 200). Give `configmaps` and "
                         "`client` the same N")
    sub = ap.add_subparsers(dest="cmd", required=False)

    def node_cmd(name: str, help_: str) -> argparse.ArgumentParser:
        s = sub.add_parser(name, help=help_)
        s.add_argument("--node", required=True, help="Kubernetes node name")
        s.add_argument("--allow-busy-node", action="store_true",
                       help="allow a node that runs pods in namespace default")
        return s

    sub.add_parser("configmaps", help="create/update the namespace and ConfigMaps")
    s = sub.add_parser("secret", help="create/update the Enroot credentials Secret from a netrc")
    s.add_argument("--netrc", default=DEFAULT_NETRC)
    s = node_cmd("smoke", "one-shot smoke test (no token needed)")
    s.add_argument("--run-id")
    s.add_argument("--smoke-image", help="public arm64 image (Enroot URI without docker://)")
    s.add_argument("--keep", action="store_true", help="keep the finished pod")
    s.add_argument("--no-metadata-block", action="store_true", help=NO_MDB_HELP)
    s = node_cmd("model", "start the pilot endpoint (or, test only, the gold-patch oracle)")
    s.add_argument("--timeout", type=int, default=2400)
    variant = s.add_mutually_exclusive_group()
    variant.add_argument("--keep-default-caps", action="store_true",
                         help="fallback: keep the runtime's default capabilities")
    variant.add_argument("--oracle", action="store_true",
                         help="test only: serve the public SWE-bench gold patches (oracle/) "
                              "instead of the model, to check the grading harness; never submit")
    s = node_cmd("host", "start the SWE service (service mode)")
    s.add_argument("--run-id", required=True)
    s.add_argument("--persist-cache", action="store_true",
                   help="keep imported images in a node-wide cache across runs")
    s.add_argument("--prewarm", help="first:N or comma-separated instance ids (needs --persist-cache)")
    s.add_argument("--timeout", type=int, default=5400, help="seconds to wait for SWE_READY (and prewarm)")
    s.add_argument("--handoff", action="store_true",
                   help="production: write the connection file for the recipe's benchmark task "
                        "(run id = the sflow run folder name; normally done by `serve`)")
    s.add_argument("--scan-reports", action="store_true",
                   help="production: the teardown leak scan also reads this run's report directory")
    s.add_argument("--no-metadata-block", action="store_true",
                   help=NO_MDB_HELP + "; refused with --handoff or --scan-reports")
    for name, help_ in (("client", "start the pilot client"), ("collect", "copy a run directory")):
        s = node_cmd(name, help_) if name == "client" else sub.add_parser(name, help=help_)
        if name == "collect":
            s.add_argument("--node", required=True)
            s.add_argument("--dest", required=True)
            s.add_argument("--force", action="store_true")
            s.add_argument("--netrc", default=DEFAULT_NETRC)
        else:
            s.add_argument("--follow", action="store_true")
        s.add_argument("--run-id", required=True)
        s.add_argument("--app-uid", type=int)
        s.add_argument("--app-gid", type=int)
    s = sub.add_parser("status", help="pods and redacted log tails")
    s.add_argument("--tail", type=int, default=15)
    s = sub.add_parser("down", help="remove pods (all: also the Secret)")
    s.add_argument("target", choices=["client", "host", "model", "smoke", "collect", "secret", "all"])
    s.add_argument("--alloc", help="sflow allocation id for per-run `serve` host/collect pods")
    s = sub.add_parser("serve", help="production: the SWE service for each acc/both Kimi run "
                                     "(use ../run_swe_sweep.sh)")
    s.add_argument("--group", default=DEFAULT_GROUP,
                   help=f"SWE node group folder, absolute or relative to {K8S_ROOT}")
    s.add_argument("--points", nargs="*", help="point folders to watch, e.g. 24 48 (default: all)")
    s.add_argument("--once", action="store_true", help="serve one acc/both run, then exit")
    s.add_argument("--ready-timeout", type=int, default=1800, help="seconds to wait for SWE_READY")
    s.add_argument("--run-timeout", type=int, default=12 * 3600,
                   help="seconds from SWE_READY to the end of the run's benchmark")
    s.add_argument("--poll", type=float, default=15.0, help="seconds between checks")
    s.add_argument("--netrc", default=DEFAULT_NETRC,
                   help="for the local leak scan of each collected run directory")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = parser()
    args = ap.parse_args(argv)
    if args.check:
        if args.cmd is not None:
            die("--check takes no subcommand")
        return run_check(args)
    if not args.cmd:
        ap.error("the following arguments are required: cmd (or pass --check)")
    kube = Kube(args.context, Redactor(), dry_run=args.dry_run)
    log(f"context {kube.context}, namespace {NS}{' (dry run)' if args.dry_run else ''}")
    commands = {"configmaps": cmd_configmaps, "secret": cmd_secret, "smoke": cmd_smoke,
                "model": cmd_model, "host": cmd_host, "client": cmd_client, "status": cmd_status,
                "down": cmd_down, "collect": cmd_collect, "serve": cmd_serve}
    return commands[args.cmd](args, kube)


if __name__ == "__main__":
    sys.exit(main())
