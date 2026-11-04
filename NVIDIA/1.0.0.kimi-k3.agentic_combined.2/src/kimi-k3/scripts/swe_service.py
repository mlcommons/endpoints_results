"""Launch the pinned SWE-bench service on the client node for a combined perf+SWE run.

Runs as the `swe_service` host task and prepares its pinned source/environment automatically.
Writes {"url", "token"} to $SFLOW_WORKFLOW_OUTPUT_DIR/swe-connection.json (mode 0600),
prints SWE_READY, and keeps the service in the foreground until the workflow stops it.
"""

import json
import fcntl
import hashlib
import io
import zipfile
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import time
import urllib.request

# Step-scoped Slurm variables would pin the service's own srun/Pyxis steps to this task's step.
STEP_VARS = (
    "SLURM_NTASKS",
    "SLURM_NPROCS",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_TASKS_PER_NODE",
    "SLURM_CPUS_PER_TASK",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_TASK_PID",
    "SLURM_CPU_BIND",
    "SLURM_CPU_BIND_LIST",
    "SLURM_CPU_BIND_TYPE",
    "PYTHONPATH",
    "PYTHONHOME",
)


REVISION = "d3c57f2eba127c819e5a8fd83101d9e5f4e15fce"


def bootstrap(root, source, env):
    """Prepare once under a lock; never modify the published benchmark client."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "bootstrap.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        checkout = root / "service"
        if not checkout.exists():
            try:
                subprocess.run(["git", "init", str(checkout)], check=True)
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(checkout),
                        "fetch",
                        "--depth=1",
                        "https://github.com/mlcommons/endpoints.git",
                        REVISION,
                    ],
                    check=True,
                )
                subprocess.run(
                    # Service code needs no LFS benchmark artifacts.
                    [
                        "git", "-C", str(checkout),
                        "-c", "filter.lfs.process=",
                        "-c", "filter.lfs.smudge=",
                        "-c", "filter.lfs.required=false",
                        "checkout", "--detach", "FETCH_HEAD",
                    ],
                    check=True,
                )
            except Exception:
                shutil.rmtree(checkout, ignore_errors=True)
                raise
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != REVISION:
            raise ValueError(
                "SWE service checkout must match the pinned official client"
            )
        uv = root / "uv"
        if not uv.is_file():
            with urllib.request.urlopen(
                "https://pypi.org/pypi/uv/0.8.22/json", timeout=30
            ) as response:
                release = json.load(response)
            wheel = next(
                f
                for f in release["urls"]
                if "manylinux" in f["filename"] and "aarch64" in f["filename"]
            )
            with urllib.request.urlopen(wheel["url"], timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != wheel["digests"]["sha256"]:
                raise ValueError("uv wheel checksum mismatch")
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                binary = next(
                    name for name in archive.namelist()
                    if name.endswith(".data/scripts/uv")
                )
                uv.write_bytes(archive.read(binary))
            uv.chmod(0o755)
        subprocess.run(
            [
                str(uv),
                "sync",
                "--frozen",
                "--no-dev",
                "--python",
                "3.12",
                "--project",
                str(source),
            ],
            env=env,
            check=True,
        )


def main():
    out = Path(os.environ["SFLOW_TASK_OUTPUT_DIR"])
    shared = Path(os.environ["SFLOW_WORKFLOW_OUTPUT_DIR"])
    root = Path(os.environ["SWE_ROOT"])
    source = root / "service/src/inference_endpoint/evaluation/swebench_service"

    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("SLURM_STEP") and k not in STEP_VARS
    }
    env["SLURMD_NODENAME"] = socket.gethostname().split(".")[0]
    scratch = Path(os.environ["CONTAINER_SCRATCH"]) / (
        "k3-swe-" + os.environ["SLURM_JOB_ID"]
    )
    for key, path in {
        "UV_PROJECT_ENVIRONMENT": root / "service-venv",
        "UV_CACHE_DIR": root / "uv-cache",
        "UV_PYTHON_INSTALL_DIR": root / "python",
        "HF_HOME": root / "hf-cache",
        "ENROOT_DATA_PATH": scratch / "data",
        "ENROOT_TEMP_PATH": scratch / "tmp",
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    registry_host = os.environ["SWE_IMAGE_REGISTRY"].split("/")[0].split(":")[0]
    for key in ("NO_PROXY", "no_proxy"):
        env[key] = ",".join(
            filter(None, [env.get(key, ""), registry_host, "localhost", "127.0.0.1"])
        )
    env.update(
        HF_HUB_OFFLINE="0",
        HF_DATASETS_OFFLINE="0",
        TRANSFORMERS_OFFLINE="0",
        PYTHONUNBUFFERED="1",
    )
    bootstrap(root, source, env)
    python = str(root / "service-venv/bin/python")
    env["PATH"] = str(root / "service-venv/bin") + ":" + env["PATH"]

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token = secrets.token_urlsafe(48)
    url = f"http://127.0.0.1:{port}"
    # One argv element: a token starting with "-" would otherwise be parsed as an option.
    command = [
        python,
        "-m",
        "swebench_service",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--auth-token=" + token,
        "--artifact-root",
        str(out / "artifacts"),
        "--runtime",
        "pyxis",
        "--image-registry",
        os.environ["SWE_IMAGE_REGISTRY"],
    ]
    service = subprocess.Popen(
        command,
        cwd=source,
        env=env,
        stdout=(out / "service.log").open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def stop(signum, frame):
        service.terminate()
        try:
            service.wait(timeout=90)
        except subprocess.TimeoutExpired:
            service.kill()
            service.wait()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for _ in range(180):
        if service.poll() is not None:
            raise RuntimeError("SWE service exited; inspect service.log")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            # Connection refusal is expected until the service finishes startup.
            time.sleep(1)
    else:
        raise RuntimeError("SWE service readiness timeout")
    fd = os.open(
        shared / "swe-connection.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
    )
    with os.fdopen(fd, "w") as f:
        json.dump({"url": url, "token": token}, f)
    print("SWE_READY", flush=True)
    raise SystemExit(service.wait())


if __name__ == "__main__":
    main()
