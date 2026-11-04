"""Prepare pinned host MLCflow and elevate only its capture entrypoint."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess


class CollectorError(ValueError):
    pass


def _has_controlling_terminal() -> bool:
    # sudo reads passwords from /dev/tty even with collector stdio redirected.
    try:
        descriptor = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
    except OSError:
        return False
    os.close(descriptor)
    return True


def prepare_host(root: Path, env: dict[str, str], capture_dir: Path) -> dict:
    argv = ["bash", str(root / "prepare_submission_collector.sh")]
    if env.get("SUBMISSION_VALIDATE_ONLY") == "1":
        argv.append("--assert-only")
    with (capture_dir / "collector-preparation.log").open("xb") as log:
        result = subprocess.run(
            argv, env=env, cwd=root, stdout=subprocess.PIPE, stderr=log,
            text=True, timeout=1800, check=False,
        )
    if result.returncode:
        raise CollectorError("Host collector preparation/assertion failed; see collector-preparation.log.")
    try:
        prepared = json.loads(result.stdout)
        if not isinstance(prepared, dict):
            raise ValueError()
        for key in ("venv", "python", "automation"):
            if not isinstance(prepared[key], str) or not Path(prepared[key]).is_absolute():
                raise ValueError()
    except (KeyError, ValueError) as exc:
        raise CollectorError("Host collector preparation returned invalid environment paths.") from exc
    return prepared


def collect(
    *, root: Path, capture_dir: Path, env: dict[str, str],
    pins: tuple[str, str], record: dict,
) -> Path:
    prepared = prepare_host(root, env, capture_dir)
    if (
        prepared.get("mlcflow_version") != pins[0]
        or prepared.get("automation_revision") != pins[1]
    ):
        raise CollectorError("Prepared host collector does not match configs/mlcflow.conf.")
    for name in ("work", "home", "mlc-repos", "collector"):
        (capture_dir / name).mkdir(mode=0o700)
    argv = [
        prepared["python"], "-I", "-B", str(root / "lib/collect_mlcflow.py"),
        *pins, "--capture-dir", str(capture_dir), "--venv", prepared["venv"],
        "--automation", prepared["automation"],
        "--caller-uid", str(os.getuid()), "--caller-gid", str(os.getgid()),
    ]
    elevated = os.geteuid() != 0
    if elevated:
        sudo = ["sudo"] if _has_controlling_terminal() else ["sudo", "-n"]
        argv = [*sudo, "--", *argv]
    result_record = {"argv": argv, "exit_status": None}
    record.update({
        "collector_environment": prepared["venv"],
        "collector_python": prepared["python"],
        "automation_checkout": prepared["automation"],
        "privilege_scope": "collector_only",
        "sudo_used": elevated, "caller_uid": os.getuid(), "caller_gid": os.getgid(),
        "commands": [result_record],
        "os_observation_scope": "submission_host_userspace",
        "kernel_observation_scope": "submission_host_kernel",
    })
    previous = {}

    def cancelled(signum, frame):
        raise KeyboardInterrupt(f"Collector cancelled by signal {signum}")

    process = None
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, cancelled)
        with (capture_dir / "collector.log").open("xb") as log:
            process = subprocess.Popen(
                argv, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
            )
            result_record["pid"] = process.pid
            try:
                process.wait(timeout=600)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                result_record["failure"] = type(exc).__name__
                # sudo forwards this signal to the root entrypoint, which
                # stops its command group and repairs capture ownership.
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired as cleanup_exc:
                    record["cleanup"] = {"completed": False, "pid": process.pid}
                    raise CollectorError(
                        f"Host collector did not finish cancellation cleanup (PID {process.pid})."
                    ) from cleanup_exc
                raise CollectorError("Host collection timed out or was cancelled.") from exc
            finally:
                result_record["exit_status"] = process.returncode
                if process.returncode is not None:
                    record["cleanup"] = {"completed": True, "exit_status": process.returncode}
        if process.returncode:
            raise CollectorError(
                f"Host MLCflow collector failed (exit {process.returncode}); "
                "see collector.log, including any sudo authentication diagnostic."
            )
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        observations = capture_dir / "collector-observations.json"
        if observations.is_file():
            record["native_observations"] = json.loads(observations.read_bytes())
    observations = record.get("native_observations", {})
    if (
        observations.get("status") != "success"
        or observations.get("mlcflow_version") != pins[0]
        or observations.get("automation_revision") != pins[1]
        or observations.get("automation_clean") is not True
    ):
        raise CollectorError("Host collector did not retain verified package/checkout observations.")
    kernel = observations.get("selected_environment", {}).get("MLC_HOST_OS_KERNEL_VERSION")
    uname = observations.get("raw_uname", {}).get("release", {})
    if not kernel or uname.get("exit_status") != 0 or kernel != uname.get("stdout"):
        raise CollectorError("Native host kernel evidence is missing or inconsistent with uname.")
    record.update({
        "mlcflow_version": observations["mlcflow_version"],
        "automation_revision": observations["automation_revision"],
        "automation_tracked_clean": True,
    })
    source = capture_dir / "collector/system_desc.json"
    if not source.is_file() or source.stat().st_uid != os.getuid():
        raise CollectorError("Host collector did not return a caller-owned system_desc.json.")
    return source
