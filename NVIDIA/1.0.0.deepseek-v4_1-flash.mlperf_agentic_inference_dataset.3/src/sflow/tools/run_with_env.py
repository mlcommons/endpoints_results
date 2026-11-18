#!/usr/bin/env python3
"""
Print shell export statements from one or more flat YAML env files.

Usage:
    eval "$(python3 run_with_env.py <env.yaml> [<overlay.yaml> ...] | tee /dev/stderr)"

Files are merged in order, so a later file overrides keys of an earlier one: a point's
server-env.yaml followed by an optional site overlay (NVIDIA/src/sflow/site/*.yaml).
Empty arguments are skipped, so an unset overlay variable expands to nothing harmlessly.

The tee /dev/stderr makes each export line visible in the log,
while eval applies them to the current shell.
"""
import argparse
import os
import shlex
import sys

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("env_yaml", nargs="+",
                   help="Flat YAML files of KEY: value env vars; later files override earlier ones")
    return p.parse_args()


def load_env(paths: list[str]) -> dict:
    """Merge the env files in order; a later file's keys win. Empty paths are skipped."""
    merged: dict = {}
    for path in paths:
        if not path:
            continue
        with open(path) as f:
            merged.update(yaml.safe_load(f) or {})
    return merged


def print_exports(env_vars: dict) -> None:
    """Print 'export KEY=value' lines for each entry in env_vars."""
    for k, v in env_vars.items():
        # Sflow variables are injected into the task process environment.
        expanded = os.path.expandvars(str(v))
        print(f"export {k}={shlex.quote(expanded)}")


def main() -> None:
    args = parse_args()
    print_exports(load_env(args.env_yaml))


if __name__ == "__main__":
    main()
