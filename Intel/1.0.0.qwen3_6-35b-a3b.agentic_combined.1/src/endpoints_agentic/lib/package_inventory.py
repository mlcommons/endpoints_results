"""Resolve source dependencies for installed hardware families, with or without Git."""
from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path
import re
import subprocess
import sys


class InventoryError(ValueError):
    pass


def available_families(root: Path) -> list[str]:
    return sorted(p.parent.name for p in (root / "platforms").glob("*/package.list")
                  if (p.parent / "platform.conf").is_file())


def select_families(root: Path, selection: str) -> list[str]:
    available = available_families(root)
    if selection == "all":
        families = available
    else:
        families = sorted(set(selection.split(",")))
    if not families or any(not re.fullmatch(r"[a-z][a-z0-9-]*", f) for f in families):
        raise InventoryError("Profiles must be a nonempty comma-separated list of hardware families, or all")
    unknown = set(families) - set(available)
    if unknown:
        raise InventoryError(f"Unknown or omitted hardware families: {', '.join(sorted(unknown))}")
    return families


def family_for_profile(root: Path, profile: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", profile):
        raise InventoryError(f"Invalid stack profile: {profile!r}")
    matches = [family for family in available_families(root)
               if (root / "platforms" / family / "profiles" / f"{profile}.conf").is_file()]
    if len(matches) != 1:
        raise InventoryError(f"Unknown, omitted, or duplicate stack profile: {profile}")
    return matches[0]


def _patterns(root: Path, families: list[str]) -> dict[str, list[str]]:
    manifests = [root / "configs/package-common.list"]
    manifests.extend(root / "platforms" / f / "package.list" for f in families)
    patterns = {}
    for manifest in manifests:
        if not manifest.is_file() or manifest.is_symlink():
            raise InventoryError(f"Missing source manifest: {manifest}")
        entries = []
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("/") or ".." in Path(line).parts:
                raise InventoryError(f"Invalid source pattern in {manifest}: {line}")
            entries.append(line)
        patterns[manifest.relative_to(root).as_posix()] = entries
    return patterns


def _excluded(name: str) -> bool:
    return (
        name == "generate_tarball.sh"
        or name == "configs/submission_metadata.yaml"
        or any(part in {"tests", "cache", "run_output", "output", "__pycache__",
                        ".pytest_cache", ".git", ".codex"} for part in Path(name).parts)
    )


def _matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def resolve_inventory(root: Path, families: list[str], tracked_only: bool = True,
                      exclude: set[str] | None = None) -> list[str]:
    root = root.resolve()
    families = select_families(root, ",".join(families))
    source_patterns = _patterns(root, available_families(root))
    common = source_patterns["configs/package-common.list"]
    selected = common + [
        pattern for family in families
        for pattern in source_patterns[f"platforms/{family}/package.list"]
    ]
    declared = [pattern for manifest in source_patterns.values() for pattern in manifest]
    if tracked_only:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", "."],
            check=True, capture_output=True,
        )
        names = set(result.stdout.decode().rstrip("\0").split("\0")) - {""}
    else:
        names = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    excluded = exclude or set()
    names = {name for name in names if not _excluded(name) and name not in excluded}
    if tracked_only:
        # Retired paths can remain in the index while a refactor is unstaged.
        # A missing declared input still fails below; removed paths outside the
        # manifests are no longer packageable working-tree source.
        declared_names = set()
        undeclared = []
        for name in names:
            if _matches(name, declared):
                declared_names.add(name)
            elif (root / name).exists():
                undeclared.append(name)
        if undeclared:
            raise InventoryError(
                "Undeclared tracked source inputs: " + ", ".join(sorted(undeclared))
            )
        names = declared_names
    inventory = sorted(name for name in names if _matches(name, selected))
    required = {pattern for pattern in selected if not any(c in pattern for c in "*?[")}
    required.add("configs/package-common.list")
    required.update(f"platforms/{family}/package.list" for family in families)
    for name in sorted((set(required) | set(inventory)) - excluded):
        path = root / name
        if name not in names or not path.is_file() or path.is_symlink():
            raise InventoryError(f"Missing or untracked required package input: {name}")
    for family in families:
        profiles = [name for name in inventory if name.startswith(f"platforms/{family}/profiles/")]
        if not profiles:
            raise InventoryError(f"No tracked stack profiles found for family {family}")
    if not inventory:
        raise InventoryError("No source package inputs found")
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profiles", required=True)
    args = parser.parse_args()
    try:
        files = resolve_inventory(args.root, args.profiles.split(","))
    except (InventoryError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    sys.stdout.buffer.write(b"\0".join(name.encode() for name in files) + b"\0")


if __name__ == "__main__":
    main()
