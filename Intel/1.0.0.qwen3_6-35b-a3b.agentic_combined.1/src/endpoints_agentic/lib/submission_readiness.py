"""Read-only storage and Docker checks before submission preparation."""
from __future__ import annotations

from pathlib import Path
import subprocess
import tomllib


def storage_readiness(root: Path, env: dict[str, str]) -> dict:
    minimum = env.get("MIN_FREE_GIB", "10")
    if not minimum.isascii() or not minimum.isdecimal():
        raise ValueError("MIN_FREE_GIB must be a non-negative integer")
    result = subprocess.run(
        ["docker", "info", "--format", "{{.DockerRootDir}}"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    docker_root = result.stdout.strip()
    if not docker_root or not Path(docker_root).is_absolute():
        raise ValueError("Docker did not report an absolute storage root")
    containerd_root = "/var/lib/containerd"
    warnings = [
        "Minimum free space is a floor, not a capacity estimate for SWE-bench images and campaign artifacts."
    ]
    containerd_config = Path("/etc/containerd/config.toml")
    try:
        if containerd_config.exists():
            containerd_root = tomllib.loads(containerd_config.read_text()).get("root", containerd_root)
        else:
            warnings.append("Containerd root uses its standard default; verify any custom daemon configuration.")
    except PermissionError:
        warnings.append("Containerd configuration is unreadable; verify its storage root before a live campaign.")
    paths = {
        "output": env.get("OUTPUT_ROOT", str(root / "run_output")),
        "model_cache": env.get("HF_HOME", str(Path.home() / ".cache/huggingface")),
        "docker": docker_root, "containerd": containerd_root,
    }
    filesystems = {}
    import os
    for label, value in paths.items():
        path = Path(value)
        if not path.is_absolute():
            raise ValueError(f"{label} storage path must be absolute")
        probe = path
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        try:
            stat = probe.stat()
            usage = os.statvfs(probe)
        except PermissionError:
            warnings.append(f"Cannot inspect {label} storage; verify capacity before a live campaign.")
            continue
        available = usage.f_bavail * usage.f_frsize
        row = filesystems.setdefault(stat.st_dev, {
            "paths": {}, "available_bytes": available,
        })
        row["paths"][label] = str(path)
        if available < int(minimum) * 1024**3:
            raise ValueError(f"{label} storage has {available / 1024**3:.1f} GiB free; MIN_FREE_GIB={minimum}")
    return {"filesystems": list(filesystems.values()), "warnings": warnings}


def docs_inventory(value: str | None) -> tuple[Path | None, list[str]]:
    if value is None:
        return None, []
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path.is_symlink() or not path.is_dir():
        raise ValueError("SUBMISSION_DOCS_DIR must be an absolute, non-symlink directory")
    files = []
    for entry in sorted(path.rglob("*")):
        if "\\" in entry.name or any(ord(c) < 32 or ord(c) == 127 for c in entry.name):
            raise ValueError(f"Malformed submission documentation path: {entry.name!r}")
        if entry.is_symlink() or not (entry.is_dir() or entry.is_file()):
            raise ValueError(f"Submission documentation must contain only regular files/directories: {entry.name}")
        if entry.is_file():
            files.append(str(entry.relative_to(path)))
    if not files:
        raise ValueError("SUBMISSION_DOCS_DIR must contain specification evidence files")
    return path, files
