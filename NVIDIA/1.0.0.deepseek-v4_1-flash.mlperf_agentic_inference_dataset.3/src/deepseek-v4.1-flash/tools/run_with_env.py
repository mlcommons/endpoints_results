"""Render flat server settings or role-specific Dynamo YAML as shell exports."""
import os, shlex, sys
from pathlib import Path
import yaml

role, *paths = sys.argv[1:]
if role not in ("frontend", "worker"):
    raise ValueError("Expected frontend or worker")
settings = {}
for path in paths:
    data = yaml.safe_load(Path(path).read_text())
    if "common" in data:
        data = {**data["common"], **data[role]}
    settings.update(data)
for key, value in settings.items():
    if not key.isidentifier():
        raise ValueError(f"Invalid environment name: {key}")
    # Preserve the source overlay order without adding cwd when PYTHONPATH is unset.
    inherited = os.environ.get("PYTHONPATH", "")
    value = str(value).replace(":${PYTHONPATH}", ":" + inherited if inherited else "")
    value = os.path.expandvars(value)
    if "${" in value:
        raise ValueError(f"Unresolved environment value for {key}: {value}")
    print(f"export {key}={shlex.quote(value)}")
