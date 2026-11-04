#!/usr/bin/env python3
"""Node-side fabric validation, clock control, and measured whole-NUMA binding."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


def fabric_identity(text):
    blocks = re.split(r"(?m)^GPU ", text)[1:]
    if len(blocks) != 4:
        raise ValueError("Expected four physical GPUs per node")
    identities = []
    for block in blocks:
        if "\n    Fabric\n" not in block:
            raise ValueError("Missing GPU fabric report")
        fabric = block.split("\n    Fabric\n", 1)[1]
        for key, expected in [
            ("State", "Completed"),
            ("Status", "Success"),
            ("Summary", "Healthy"),
            ("Bandwidth", "Full"),
        ]:
            match = re.search(r"(?m)^\s+" + key + r"\s+:\s*(\S+)", fabric)
            if not match or match[1] != expected:
                raise ValueError(f"Unhealthy or incomplete GPU fabric: {key}")
        identity = []
        for key in ("ClusterUUID", "CliqueId"):
            match = re.search(key + r"\s+:\s*(\S+)", fabric)
            if not match or match[1] == "N/A":
                raise ValueError(f"Missing fabric identity: {key}")
            identity.append(match[1])
        identities.append(identity)
    if any(value != identities[0] for value in identities):
        raise ValueError("Node GPUs have different fabric identities")
    return identities[0]


NGPU = 4


def log(message):
    print(f"[node] {message}", file=sys.stderr, flush=True)


def container_root(action, image, scratch, name):
    """Prepare one disk-backed root per node; Pyxis launches separate namespaces."""
    job = os.environ["SLURM_JOB_ID"]
    if not job.isdigit() or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
        raise ValueError("Invalid job or container name")
    env = dict(os.environ)
    # Pyxis filters these overrides, so query the configuration it actually uses.
    for key in (
        "ENROOT_DATA_PATH",
        "ENROOT_TEMP_PATH",
        "ENROOT_CACHE_PATH",
        "ENROOT_RUNTIME_PATH",
        "ENROOT_LIBRARY_PATH",
        "ENROOT_SYSCONF_PATH",
    ):
        env.pop(key, None)
    info = subprocess.check_output(["enroot", "info"], env=env, text=True, timeout=30)
    matches = re.findall(r"(?m)^\s*ENROOT_DATA_PATH=(.+)$", info)
    if len(matches) != 1 or not Path(matches[0]).is_absolute():
        raise RuntimeError("enroot info must report its configured ENROOT_DATA_PATH")
    configured = Path(matches[0])
    if configured.stat().st_uid != os.getuid():
        raise ValueError("Enroot container directory must belong to the current user")
    cache = Path(scratch).resolve() / f"{os.getuid()}-k3-dynamo-{job}-{name}"
    root = cache / "data" / name
    # Support both Pyxis user scope and job scope without site-specific paths.
    aliases = [configured / f"pyxis_{name}", configured / f"pyxis_{job}_{name}"]
    if action == "cleanup":
        for alias in aliases:
            if alias.is_symlink() and alias.resolve() == root:
                alias.unlink()
        if root.exists():
            # PMIx can leave mode-000 directories; Enroot removes those safely.
            subprocess.run(
                ["enroot", "remove", "--force", name],
                env=dict(env, ENROOT_DATA_PATH=str(cache / "data")),
                check=True,
                timeout=120,
            )
        if cache.exists():
            shutil.rmtree(cache)
        return
    if any(os.path.lexists(alias) for alias in aliases) or cache.exists():
        raise FileExistsError(
            "Job container root already exists; refusing to overwrite"
        )
    (cache / "data").mkdir(parents=True)
    (cache / "tmp").mkdir()
    create_env = dict(
        env,
        ENROOT_DATA_PATH=str(cache / "data"),
        ENROOT_TEMP_PATH=str(cache / "tmp"),
        ENROOT_MAX_PROCESSORS="8",
    )
    source = Path(image)
    if not source.is_file():
        if str(image).startswith(("/", "./", "../")):
            raise FileNotFoundError(image)
        uri = (
            str(image)
            if str(image).startswith(("docker://", "dockerd://"))
            else "docker://" + str(image)
        )
        source = cache / "server.sqsh"
        subprocess.run(
            ["enroot", "import", "--output", str(source), uri],
            env=create_env,
            check=True,
        )
    subprocess.run(
        ["enroot", "create", "--name", name, str(source)], env=create_env, check=True
    )
    for alias in aliases:
        alias.symlink_to(root, target_is_directory=True)
    print("DISK_ROOT_READY", os.environ.get("SLURMD_NODENAME", ""), root, flush=True)


def validate(placement, reports, worker=None, assigned_nodes=None):
    if set(reports) != set(placement["nodes"]):
        raise ValueError("Fabric reports do not match reserved nodes")
    for node, report in reports.items():
        if (
            report.get("healthy") is not True
            or report.get("gpus") != 4
            or report.get("fabric_identity")
            != placement["reports"][node]["fabric_identity"]
        ):
            raise ValueError("Measured fabric changed or unhealthy: " + node)
    used = set()
    for name, spec in placement["workers"].items():
        for slot in spec["slots"]:
            pair = (slot["node"], slot["gpu"])
            if (
                pair in used
                or reports[slot["node"]]["fabric_identity"] != spec["fabric_identity"]
            ):
                raise ValueError("Worker overlaps or crosses measured fabric: " + name)
            used.add(pair)
    if len(used) != placement["active_gpus"] or len(used) > 48:
        raise ValueError("Active GPU budget mismatch")
    if placement["dwdp"]:
        domains = {
            tuple(w["fabric_identity"])
            for name, w in placement["workers"].items()
            if name.startswith("prefill_")
        }
        if len(domains) != 1:
            raise ValueError("DWDP communicator crosses fabrics")
    if worker is not None:
        expected = {s["node"] for s in placement["workers"][worker]["slots"]}
        if (
            len(assigned_nodes) != len(set(assigned_nodes))
            or set(assigned_nodes) != expected
        ):
            raise ValueError(
                "Actual worker node assignment differs from planned placement"
            )


def check_checkpoint(path):
    """Reject incomplete sharded checkpoints before starting model workers."""
    root = Path(path)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        missing = sorted(shard for shard in shards if not (root / shard).is_file())
        if missing:
            raise ValueError(
                f"Incomplete checkpoint {root}: {len(missing)} missing shards, "
                f"including {missing[:3]}"
            )


def check(path, mode, name=None):
    if mode == "preflight":
        for key in ("MODEL_PATH", "DRAFT_PATH"):
            if os.environ.get(key):
                check_checkpoint(os.environ[key])
    placement = json.loads(Path(path).read_text())
    if os.environ.get("NODE_NAMES", "").split(",") != placement["nodes"]:
        raise ValueError(
            "SFlow backend node ordering differs from the measured placement plan"
        )
    root = Path(os.environ["SFLOW_WORKFLOW_OUTPUT_DIR"]) / "fabric"
    reports = {
        n: json.loads((root / (n + ".json")).read_text()) for n in placement["nodes"]
    }
    worker = name if mode == "worker" else None
    assigned = os.environ.get("SFLOW_TASK_ASSIGNED_NODE_NAMES", "").split(",")
    validate(placement, reports, worker, assigned)
    if worker is None:
        (root / "segmented-placement.json").write_text(json.dumps(placement, indent=2))
    elif os.environ.get("SLURM_PROCID", "0") == "0":
        evidence = {
            "worker": worker,
            "assigned_nodes": assigned,
            "planned": placement["workers"][worker],
            "passed": True,
        }
        (root / ("worker-" + worker + "-placement.json")).write_text(
            json.dumps(evidence, indent=2)
        )


def probe(directory):
    directory = Path(directory)
    node = os.environ["SLURMD_NODENAME"]
    assert node and "/" not in node
    directory.mkdir(parents=True, exist_ok=True)
    raw = subprocess.check_output(["nvidia-smi", "-q"], text=True, timeout=30)
    record = {"gpus": 4, "healthy": True, "fabric_identity": fabric_identity(raw)}
    for suffix, content in [("txt", raw), ("json", json.dumps(record))]:
        target = directory / (node + "." + suffix)
        with target.with_suffix("." + suffix + ".tmp").open("w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        target.with_suffix("." + suffix + ".tmp").replace(target)


def clocks(action, directory, memory_clock=4752, restore_range="3186,4752"):
    node = os.environ["SLURMD_NODENAME"]
    os.environ["SLURM_JOB_ID"]  # Refuse use outside an allocation.
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if action in ("pin", "restore"):
        clocks = f"{memory_clock},{memory_clock}" if action == "pin" else restore_range
        deadline = time.monotonic() + 75
        while True:
            result = subprocess.run(
                ["sudo", "-n", "nvidia-smi", "-lmc", clocks],
                capture_output=True,
                text=True,
                timeout=20,
            )
            output = result.stdout + result.stderr
            (root / f"{node}.{action}.txt").write_text(output)
            if (
                result.returncode != 27
                or "not ready yet" not in output
                or time.monotonic() >= deadline
            ):
                result.check_returncode()
                break
            with (root / f"{node}.{action}.retries.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "returncode": result.returncode,
                            "output": output,
                        }
                    )
                    + "\n"
                )
            log(
                "GPU clock controller is not ready after teardown; retrying within 75 seconds"
            )
            time.sleep(min(10, max(0, deadline - time.monotonic())))
        if action == "restore":
            return
    elif action != "monitor":
        raise ValueError(action)
    verification_deadline = time.monotonic() + 30
    while True:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,clocks.mem,clocks.sm,clocks_event_reasons.active",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=30,
        )
        with (root / f"{node}.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.time(), "sample": raw}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        rows = raw.strip().splitlines()
        if len(rows) != 4 or any(
            int(row.split(",")[1].strip()) != memory_clock for row in rows
        ):
            if action == "pin" and time.monotonic() < verification_deadline:
                log("Waiting for all four GPUs to complete the memory-clock transition")
                time.sleep(2)
                continue
            raise RuntimeError(
                f"Memory clocks are not {memory_clock} MHz on all four GPUs"
            )
        if action == "pin":
            return
        time.sleep(30)


def gpu_nodes():
    # Several ranks of one node call this at the same moment; nvidia-smi topo occasionally returned no GPU rows under
    # that concurrency, so retry before falling back to the Vera layout (GPU0/1 -> 0, GPU2/3 -> 1).
    import random
    import time

    last = ""
    for attempt in range(4):
        try:
            r = subprocess.run(
                ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=60
            )
            topo = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout).splitlines()
            nodes = {}
            for line in topo[1:]:
                f = line.split()
                # GPUk <peers...> <CPU affinity> <NUMA affinity> <GPU NUMA ID ("N/A" on Vera)>
                if f and re.fullmatch(r"GPU\d+", f[0]):
                    nums = [x for x in f[1:] if re.fullmatch(r"\d+", x)]
                    if nums:
                        nodes[int(f[0][3:])] = int(nums[-1])
            if len(nodes) >= NGPU:
                return [nodes[g] for g in range(NGPU)]
            last = (
                f"rc={r.returncode} rows={len(nodes)} stderr={r.stderr.strip()[:120]!r}"
            )
        except (OSError, subprocess.SubprocessError) as e:
            last = str(e)
        time.sleep(1 + random.random() * 2)
    log(f"nvidia-smi topo unusable ({last}); falling back to gpu // 2 (Vera layout)")
    return [g // 2 for g in range(NGPU)]


def this_gpu():
    local = int(os.environ.get("SLURM_LOCALID", "0"))
    cvd = [
        x.strip()
        for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if x.strip()
    ]
    if not cvd:
        return local % NGPU
    tok = cvd[local % len(cvd)]
    if tok.isdigit():
        return int(tok)
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    for line in out.splitlines():
        idx, uuid = [x.strip() for x in line.split(",")]
        if uuid == tok or uuid.startswith(tok):
            return int(idx)
    raise SystemExit(f"cannot map CUDA_VISIBLE_DEVICES token {tok!r} to a GPU index")


def numa():
    # These points use the whole CPU NUMA node local to each physical GPU.
    if int(os.environ.get("ENGINE_CORES_PER_RANK", "0") or 0):
        raise ValueError("This recipe supports whole-NUMA binding only")
    if os.environ.get("NUMA_BIND", "1") != "1":
        return
    if shutil.which("numactl") is None:
        raise RuntimeError("NUMA binding requires numactl in the image")
    mem = os.environ.get("NUMA_MEM", "membind")
    if mem not in ("membind", "preferred"):
        raise ValueError("NUMA_MEM must be membind or preferred")
    gpu = this_gpu()
    node = gpu_nodes()[gpu]
    prefix = f"numactl --cpunodebind={node} --{mem}={node}"
    log(f"gpu={gpu} numa={node} -> {prefix}")
    print(prefix)


def install_control_plane(destination):
    import hashlib
    import io
    import tarfile
    import urllib.request

    destination.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(__file__).with_name("control_plane.json").read_text())
    for record in records:
        target = destination / record["name"]
        if (
            target.is_file()
            and hashlib.sha256(target.read_bytes()).hexdigest()
            == record["binary_sha256"]
        ):
            print(f"Verified {target.name}", flush=True)
            continue
        with urllib.request.urlopen(record["source"], timeout=120) as response:
            archive = response.read()
        if hashlib.sha256(archive).hexdigest() != record["archive_sha256"]:
            raise RuntimeError(f"Archive checksum mismatch for {target.name}")
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as stream:
            matches = [
                m
                for m in stream.getmembers()
                if m.isfile() and Path(m.name).name == record["name"]
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Ambiguous binary in {target.name} archive")
            binary = stream.extractfile(matches[0]).read()
        if hashlib.sha256(binary).hexdigest() != record["binary_sha256"]:
            raise RuntimeError(f"Binary checksum mismatch for {target.name}")
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(binary)
        temporary.chmod(0o755)
        temporary.replace(target)
        print(f"Installed {target.name}", flush=True)


def control_plane():
    import socket
    import tempfile
    import urllib.request

    root = Path(os.environ["SFLOW_TASK_OUTPUT_DIR"])
    binary = Path(os.environ["CONTROL_PLANE_DIR"])
    install_control_plane(binary)
    head = os.environ["HEAD_NODE_IP"]
    local = Path(
        tempfile.mkdtemp(prefix="k3-dynamo-control-" + os.environ["SLURM_JOB_ID"] + "-")
    )
    (local / "nats.conf").write_text(
        'port: 4222\nmax_payload: 16777216\njetstream { store_dir: "'
        + str(local / "nats")
        + '" }\n'
    )
    commands = [
        [
            str(binary / "etcd"),
            "--name",
            "k3",
            "--data-dir",
            str(local / "etcd"),
            "--listen-client-urls",
            "http://0.0.0.0:2379",
            "--advertise-client-urls",
            f"http://{head}:2379",
            "--listen-peer-urls",
            "http://127.0.0.1:2380",
            "--initial-advertise-peer-urls",
            "http://127.0.0.1:2380",
            "--initial-cluster",
            "k3=http://127.0.0.1:2380",
        ],
        [str(binary / "nats-server"), "-c", str(local / "nats.conf")],
    ]
    children = []

    def stop(*_):
        raise InterruptedError("Control plane stopping")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for command in commands:
            log = (root / (Path(command[0]).name + ".log")).open("w")
            children.append(
                subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            )
        deadline = time.monotonic() + 90
        while True:
            if any(c.poll() is not None for c in children):
                raise RuntimeError("Control plane process exited")
            try:
                with urllib.request.urlopen(
                    f"http://{head}:2379/health", timeout=2
                ) as r:
                    assert json.load(r)["health"] in (True, "true")
                with socket.create_connection((head, 4222), timeout=2):
                    pass
                break
            except (OSError, AssertionError):
                if time.monotonic() > deadline:
                    raise RuntimeError("Control plane readiness timed out")
                time.sleep(1)
        print("DYNAMO_CONTROL_READY", flush=True)
        while all(c.poll() is None for c in children):
            time.sleep(2)
        raise RuntimeError("Control plane process exited")
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for child in children:
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def wait_ready():
    import urllib.request

    base = "http://" + os.environ["FRONTEND_NODE_IP"] + ":8000"
    root = Path(os.environ["SFLOW_WORKFLOW_OUTPUT_DIR"])
    expected = (int(os.environ["NUM_CTX"]), int(os.environ["NUM_GEN"]))
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        try:

            def get(path):
                with urllib.request.urlopen(base + path, timeout=5) as response:
                    return json.load(response)

            health, models = get("/health"), get("/v1/models")
            (root / "dynamo-health.json").write_text(json.dumps(health, indent=2))
            (root / "dynamo-models.json").write_text(json.dumps(models, indent=2))
            instances = [
                i
                for i in health.get("instances", [])
                if i.get("endpoint") == "generate"
            ]
            actual = (
                sum(i.get("component") == "prefill" for i in instances),
                sum(
                    i.get("component") in ("decode", "backend", "tensorrt_llm")
                    for i in instances
                ),
            )
            if actual == expected and any(
                m.get("id") == "/models/kimi-k3" for m in models.get("data", [])
            ):
                print("DYNAMO_MODEL_READY", actual, flush=True)
                break
            print(
                "Waiting for worker population:",
                actual,
                "expected:",
                expected,
                flush=True,
            )
        except (OSError, ValueError) as exc:
            print("Waiting for Dynamo:", str(exc), flush=True)
        time.sleep(5)
    else:
        raise RuntimeError("Dynamo worker population/model readiness timed out")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("probe").add_argument("directory")
    for action in ("pin", "monitor", "restore"):
        clock = commands.add_parser(action)
        clock.add_argument("directory")
        clock.add_argument("--memory-clock", type=int, required=True)
        clock.add_argument("--restore-range", required=True)
    commands.add_parser("numa")
    commands.add_parser("control-plane")
    commands.add_parser("wait-ready")
    storage = commands.add_parser("container-root")
    storage.add_argument("mode", choices=("prepare", "cleanup"))
    storage.add_argument("image")
    storage.add_argument("scratch")
    storage.add_argument("name")
    validator = commands.add_parser("check")
    validator.add_argument("path")
    validator.add_argument("mode", choices=("preflight", "worker"))
    validator.add_argument("name", nargs="?")
    args = parser.parse_args()
    if args.action == "probe":
        probe(args.directory)
    elif args.action == "check":
        if args.mode == "worker" and args.name is None:
            parser.error("worker check requires a worker name")
        check(args.path, args.mode, args.name)
    elif args.action == "control-plane":
        control_plane()
    elif args.action == "wait-ready":
        wait_ready()
    elif args.action == "numa":
        numa()
    elif args.action == "container-root":
        container_root(args.mode, args.image, args.scratch, args.name)
    else:
        clocks(args.action, args.directory, args.memory_clock, args.restore_range)


if __name__ == "__main__":
    main()
