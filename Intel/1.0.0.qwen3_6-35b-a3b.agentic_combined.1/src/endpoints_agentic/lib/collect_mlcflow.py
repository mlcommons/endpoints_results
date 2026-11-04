#!/usr/bin/env python3
"""Run the pinned MLCflow system collector on the submission host."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile


SYS_ROOT = Path("/sys")

REQUIRED_UTILITIES = (
    "git",
    "jq",
    "pstree",
    "systemctl",
    "dmidecode",
    "lsblk",
    "df",
    "uname",
    "lscpu",
    "free",
)
SELECTED_ENV_KEYS = (
    "MLC_HOST_OS_TYPE",
    "MLC_HOST_OS_BITS",
    "MLC_HOST_OS_FLAVOR",
    "MLC_HOST_OS_FLAVOR_LIKE",
    "MLC_HOST_OS_VERSION",
    "MLC_HOST_OS_KERNEL_VERSION",
    "MLC_HOST_OS_GLIBC_VERSION",
    "MLC_HOST_OS_MACHINE",
    "MLC_HOST_PLATFORM_FLAVOR",
)


class CollectorError(RuntimeError):
    """The pinned native MLCflow collection could not be completed safely."""


def validate_sys_readable(sys_root: Path = SYS_ROOT) -> None:
    """Check that MLC can read host DMI and network sysfs without copying it."""
    dmi_dir = sys_root / "class/dmi/id"
    try:
        dmi_names = sorted(path.name for path in dmi_dir.iterdir())
    except OSError as exc:
        raise CollectorError("Host DMI sysfs is not readable.") from exc

    dmi_readable = False
    for name in ("sys_vendor", "product_name", "board_name", "bios_vendor"):
        if name not in dmi_names:
            continue
        try:
            with (dmi_dir / name).open("rb") as source:
                source.read(1)
            dmi_readable = True
            break
        except OSError:
            continue
    if not dmi_readable:
        raise CollectorError("Host DMI sysfs has no readable identity field.")

    network_dir = sys_root / "class/net"
    try:
        interfaces = sorted(network_dir.iterdir())
    except OSError as exc:
        raise CollectorError("Host network sysfs is not readable.") from exc
    if not interfaces:
        raise CollectorError("Host network sysfs contains no interfaces.")

    network_readable = False
    for interface in interfaces:
        try:
            with (interface / "ifindex").open("rb") as source:
                source.read(1)
            network_readable = True
            break
        except OSError:
            continue
    if not network_readable:
        raise CollectorError("Host network sysfs has no readable interface.")


def _validate_capture_directories(capture_root: Path) -> dict[str, Path]:
    paths = {
        name: capture_root / name
        for name in ("work", "home", "mlc-repos", "collector")
    }
    for name, path in paths.items():
        if not path.is_dir():
            raise CollectorError(f"Parent did not pre-create collector directory: {name}.")
        if not os.access(path, os.W_OK | os.X_OK):
            raise CollectorError(f"Capture directory is not writable: {path}.")
    if not capture_root.is_dir() or not os.access(
        capture_root, os.W_OK | os.X_OK
    ):
        raise CollectorError(f"Capture directory is not writable: {capture_root}.")
    if any(paths["mlc-repos"].iterdir()):
        raise CollectorError("The MLC_REPOS registry must be fresh and empty.")
    return paths


def _probe_directory_writable(path: Path) -> None:
    descriptor, probe_name = tempfile.mkstemp(
        prefix=".collector-write-check-", dir=path
    )
    os.close(descriptor)
    Path(probe_name).unlink()


def _execution_environment(
    capture_paths: dict[str, Path], venv_root: Path, automations_root: Path
) -> dict[str, str]:
    env = {"PATH": os.defpath + ":/usr/sbin:/sbin", "LC_ALL": "C.UTF-8"}
    env.update(
        {
            "HOME": str(capture_paths["home"]),
            "MLC_REPOS": str(capture_paths["mlc-repos"]),
            "MLC_SKIP_SUDO": "yes",
            # Skip the upstream package installer, after checking prerequisites.
            # Hardware and OS detection scripts still run unchanged.
            "MLC_SKIP_SYS_UTILS": "yes",
            "MLC_QUIET": "yes",
            "MLC_PYTHON_BIN": str(venv_root / "bin/python"),
            "TMPDIR": str(capture_paths["work"]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": str(automations_root),
        }
    )
    venv_bin = str(venv_root / "bin")
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    return env


def validate_package_version(expected: str, record: dict) -> None:
    try:
        installed = importlib.metadata.version("mlcflow")
    except importlib.metadata.PackageNotFoundError as exc:
        raise CollectorError("The mlcflow package is not installed.") from exc
    record["mlcflow_version"] = installed
    if installed != expected:
        raise CollectorError(
            f"Installed mlcflow version {installed} does not match {expected}."
        )


def _validate_revision_argument(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
        raise CollectorError("REVISION must be a 40-character Git commit.")


def validate_automation_checkout(
    expected_revision: str,
    env: dict[str, str],
    record: dict,
    *,
    automations_root: Path,
    work_dir: Path | None = None,
) -> None:
    _validate_revision_argument(expected_revision)
    if not automations_root.is_dir():
        raise CollectorError("The pinned mlperf-automations checkout is missing.")

    git_env = env.copy()
    git_env["GIT_OPTIONAL_LOCKS"] = "0"
    git_path = str(shutil.which("git", path=env.get("PATH")) or "git")
    command_cwd = work_dir or automations_root.parent
    try:
        head = subprocess.run(
            [
                git_path,
                "-C",
                str(automations_root),
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
            ],
            env=git_env,
            cwd=command_cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        record["automation_revision"] = head.stdout.strip() if head.returncode == 0 else None
        if head.returncode:
            raise CollectorError("Cannot read the automation checkout revision.")
        if record["automation_revision"] != expected_revision:
            raise CollectorError(
                "The automation checkout revision does not match REVISION."
            )

        status = subprocess.run(
            [
                git_path,
                "-C",
                str(automations_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            env=git_env,
            cwd=command_cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CollectorError("Cannot validate the automation checkout.") from exc

    if status.returncode:
        raise CollectorError("Cannot read automation checkout status.")
    record["automation_clean"] = not status.stdout
    if status.stdout:
        raise CollectorError(
            "The automation checkout has tracked or untracked changes."
        )


def validate_utilities_and_entrypoints(
    env: dict[str, str], venv_root: Path, record: dict
) -> dict[str, str]:
    utility_paths: dict[str, str] = {}
    missing = []
    for name in REQUIRED_UTILITIES:
        path = shutil.which(name, path=env.get("PATH"))
        if not path or not os.access(path, os.X_OK):
            missing.append(name)
        else:
            utility_paths[name] = path
    if missing:
        record["missing_utilities"] = missing
        raise CollectorError(
            "Required collector utilities are missing: " + ", ".join(missing)
        )

    entrypoints: dict[str, str] = {}
    for name in ("mlc", "mlcr"):
        path = venv_root / "bin" / name
        if not path.is_file() or not os.access(path, os.X_OK):
            raise CollectorError(f"MLCflow entrypoint {name} is missing.")
        entrypoints[name] = str(path)
    record["missing_utilities"] = []
    return {**utility_paths, **entrypoints}


def _run_mlc_command(
    name: str,
    argv: list[str],
    *,
    env: dict[str, str],
    work_dir: Path,
    output_dir: Path,
    record: dict,
) -> None:
    log_path = output_dir / f"{name}.log"
    command_record = {
        "name": name,
        "argv": argv,
        "exit_status": None,
        "log": str(log_path.relative_to(output_dir.parent)),
    }
    record["commands"].append(command_record)
    process = None
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                argv,
                env=env,
                cwd=work_dir,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            process.wait(timeout=270)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        command_record["failure"] = (
            "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "cancelled"
        )
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
            # The command leader may have exited before a child that ignored
            # SIGTERM. Stop any remaining members of this private group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            command_record["exit_status"] = process.returncode
        raise CollectorError(f"{name} timed out or was cancelled.") from exc
    except OSError as exc:
        command_record["failure"] = type(exc).__name__
        raise CollectorError(f"Could not run {name}.") from exc

    command_record["exit_status"] = process.returncode
    if process.returncode:
        raise CollectorError(f"{name} failed with exit status {process.returncode}.")


def capture_selected_environment(registry_dir: Path) -> tuple[dict, dict]:
    """Read only the selected native OS fields from MLC's cached run state."""
    selected = {key: None for key in SELECTED_ENV_KEYS}
    sources: dict[str, list[str]] = {key: [] for key in SELECTED_ENV_KEYS}
    cache_dir = registry_dir / "local/cache"
    if not cache_dir.is_dir():
        return selected, sources

    for state_path in sorted(cache_dir.rglob("mlc-cached-state.json")):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CollectorError("MLC cached run state is unreadable.") from exc
        if not isinstance(state, dict):
            raise CollectorError("MLC cached run state has an invalid format.")
        new_env = state.get("new_env", {})
        if not isinstance(new_env, dict):
            raise CollectorError("MLC cached environment has an invalid format.")

        relative_source = state_path.relative_to(registry_dir).as_posix()
        for key in SELECTED_ENV_KEYS:
            if key not in new_env or new_env[key] is None:
                continue
            value = new_env[key]
            if not isinstance(value, (str, int, float, bool)):
                raise CollectorError("MLC cached OS environment has an invalid value.")
            value = str(value)
            if selected[key] is not None and selected[key] != value:
                raise CollectorError(
                    f"MLC state has conflicting values for {key}."
                )
            selected[key] = value
            sources[key].append(relative_source)
    return selected, sources


def _validate_system_description(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or path.stat().st_size == 0:
            raise CollectorError("MLCflow did not produce a regular system description.")
        with path.open(encoding="utf-8") as source:
            description = json.load(source)
    except CollectorError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectorError("MLCflow produced an invalid system description.") from exc
    if not isinstance(description, dict):
        raise CollectorError("MLCflow produced an invalid system description.")


def _collect_raw_uname(env: dict[str, str], utility_paths: dict[str, str]) -> dict:
    results = {}
    uname_path = utility_paths.get("uname") or shutil.which(
        "uname", path=env.get("PATH")
    )
    for option, key in (("-r", "release"), ("-a", "all")):
        item = {"argv": [uname_path or "uname", option], "exit_status": None, "stdout": ""}
        try:
            result = subprocess.run(
                item["argv"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )
            item["exit_status"] = result.returncode
            item["stdout"] = result.stdout.rstrip("\n")
            if result.stderr:
                item["stderr"] = result.stderr.rstrip("\n")
        except (OSError, subprocess.TimeoutExpired) as exc:
            item["failure"] = type(exc).__name__
        results[key] = item
    return results


def _write_observations(path: Path, record: dict) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8") as output:
            json.dump(record, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_path, 0o660)
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _restore_capture_owner(root: Path, uid: int, gid: int) -> None:
    """Repair only this fresh capture, without following links outside it."""
    if root.is_symlink() or root.stat().st_uid != uid:
        raise OSError("Capture directory ownership changed")
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in [*directories, *files]:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            os.chown(path, uid, gid, follow_symlinks=False)
            if not stat.S_ISLNK(mode):
                permissions = 0o700 if stat.S_ISDIR(mode) else 0o600 | (mode & 0o100)
                os.chmod(path, permissions)
    os.chown(root, uid, gid, follow_symlinks=False)
    os.chmod(root, 0o700)


def run_collection(
    version: str | None,
    revision: str | None,
    *,
    capture_root: Path,
    venv_root: Path,
    automations_root: Path,
    caller_uid: int,
    caller_gid: int,
    sys_root: Path = SYS_ROOT,
) -> int:
    previous_umask = os.umask(0o077)
    capture_root = Path(capture_root)
    venv_root = Path(venv_root)
    record = {
        "execution": "host",
        "caller_uid": caller_uid,
        "caller_gid": caller_gid,
        "requested_version": version,
        "requested_revision": revision,
        "status": "failure",
        "stage": "arguments",
        "commands": [],
        "preflight": {
            "sys_dmi_readable": None,
            "sys_network_readable": None,
            "automation_clean": None,
        },
        "selected_environment": {key: None for key in SELECTED_ENV_KEYS},
        "selected_environment_sources": {},
    }
    capture_paths: dict[str, Path] = {}
    env = {"PATH": os.defpath + ":/usr/sbin:/sbin", "LC_ALL": "C.UTF-8"}
    utility_paths: dict[str, str] = {}
    failure: Exception | None = None
    capture_safe = False
    environment_captured = False

    def set_stage(stage: str) -> None:
        record["stage"] = stage

    try:
        if os.geteuid() != 0:
            raise CollectorError("Run only this collector entrypoint through sudo; campaign execution stays unprivileged.")
        if caller_uid < 0 or caller_gid < 0 or capture_root.is_symlink():
            raise CollectorError("Invalid capture owner or symlink capture directory.")
        if capture_root.stat().st_uid != caller_uid:
            raise CollectorError("Capture directory does not belong to the caller.")
        capture_safe = True
        if not version or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version):
            raise CollectorError("VERSION is missing or invalid.")
        _validate_revision_argument(revision or "")

        set_stage("capture_directories")
        capture_paths = _validate_capture_directories(capture_root)
        env = _execution_environment(capture_paths, venv_root, automations_root)

        set_stage("sysfs_validation")
        validate_sys_readable(Path(sys_root))
        record["preflight"]["sys_dmi_readable"] = True
        record["preflight"]["sys_network_readable"] = True

        for name, path in capture_paths.items():
            _probe_directory_writable(path)

        set_stage("package_validation")
        validate_package_version(version, record)

        set_stage("checkout_validation")
        validate_automation_checkout(
            revision or "",
            env,
            record,
            automations_root=Path(automations_root),
            work_dir=capture_paths["work"],
        )

        set_stage("utility_validation")
        utility_paths = validate_utilities_and_entrypoints(
            env, venv_root, record
        )

        collector_dir = capture_paths["collector"]
        reserved_outputs = (
            collector_dir / "mlc-add-repo.log",
            collector_dir / "mlcr-collect.log",
            collector_dir / "system_desc.json",
            capture_root / "collector-observations.json",
        )
        if any(path.exists() for path in reserved_outputs):
            raise CollectorError("The collector output path already contains artifacts.")

        set_stage("mlc_add_repo")
        _run_mlc_command(
            "mlc-add-repo",
            [
                utility_paths["mlc"],
                "add",
                "repo",
                str(automations_root),
            ],
            env=env,
            work_dir=capture_paths["work"],
            output_dir=collector_dir,
            record=record,
        )

        set_stage("mlcr_collect")
        _run_mlc_command(
            "mlcr-collect",
            [
                utility_paths["mlcr"],
                "get-mlperf-multi-node-system-info,_endpoints",
                "--system_name=Local host system draft",
                f"--out_dir_path={collector_dir}",
                "--out_file_name=system_desc.json",
            ],
            env=env,
            work_dir=capture_paths["work"],
            output_dir=collector_dir,
            record=record,
        )

        set_stage("environment_capture")
        selected, sources = capture_selected_environment(
            capture_paths["mlc-repos"]
        )
        record["selected_environment"] = selected
        record["selected_environment_sources"] = sources
        environment_captured = True
        if not selected.get("MLC_HOST_OS_KERNEL_VERSION"):
            raise CollectorError(
                "MLC did not expose MLC_HOST_OS_KERNEL_VERSION in its run state."
            )

        set_stage("system_description_validation")
        _validate_system_description(collector_dir / "system_desc.json")
        record["status"] = "success"
        record["stage"] = "complete"
    except (CollectorError, OSError, subprocess.SubprocessError) as exc:
        failure = exc
    except KeyboardInterrupt as exc:
        failure = CollectorError("Collection was interrupted.")
        failure.__cause__ = exc
    except Exception as exc:
        failure = CollectorError(
            f"Unexpected collector error ({type(exc).__name__})."
        )
        failure.__cause__ = exc
    finally:
        if capture_paths and not environment_captured:
            try:
                selected, sources = capture_selected_environment(
                    capture_paths["mlc-repos"]
                )
                record["selected_environment"] = selected
                record["selected_environment_sources"] = sources
            except Exception as exc:
                record["environment_capture_error"] = type(exc).__name__
                if failure is None:
                    failure = CollectorError(
                        "Could not capture the selected MLC OS environment."
                    )

        record["raw_uname"] = _collect_raw_uname(env, utility_paths)
        if failure is not None:
            record["status"] = "failure"
            record["failure"] = {
                "type": type(failure).__name__,
                "message": str(failure),
            }
            print(f"MLCflow collector failed at {record['stage']}: {failure}", file=sys.stderr)

        write_failed = True
        if capture_safe:
            observations_path = capture_root / "collector-observations.json"
            try:
                _write_observations(observations_path, record)
            except OSError as exc:
                print(
                    f"Could not write collector observations ({type(exc).__name__}).",
                    file=sys.stderr,
                )
            else:
                write_failed = False
            try:
                _restore_capture_owner(capture_root, caller_uid, caller_gid)
            except OSError as exc:
                print(f"Capture ownership restoration failed: {exc}", file=sys.stderr)
                write_failed = True
        os.umask(previous_umask)

    return 1 if write_failed or failure is not None else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("revision")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--automation", type=Path, required=True)
    parser.add_argument("--caller-uid", type=int, required=True)
    parser.add_argument("--caller-gid", type=int, required=True)
    args = parser.parse_args(argv)
    for path in (args.capture_dir, args.venv, args.automation):
        if not path.is_absolute():
            parser.error("Collector paths must be absolute.")
    def cancelled(signum, frame):
        raise KeyboardInterrupt(f"Collector cancelled by signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, cancelled)
    return run_collection(
        args.version, args.revision, capture_root=args.capture_dir,
        venv_root=args.venv, automations_root=args.automation,
        caller_uid=args.caller_uid, caller_gid=args.caller_gid,
    )


if __name__ == "__main__":
    raise SystemExit(main())
