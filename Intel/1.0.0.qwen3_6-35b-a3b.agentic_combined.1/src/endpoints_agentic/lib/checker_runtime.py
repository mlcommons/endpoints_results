"""Identity and bundled policy of the checker selected for this invocation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import yaml

CHECKER_REPOSITORY = "https://github.com/mlcommons/endpoints-submission-cli"


def installed_checker_ref() -> str:
    """Require an official Git installation, rather than a permanent code pin."""
    try:
        distribution = importlib.metadata.distribution("endpoints-submission-cli")
        source = json.loads(distribution.read_text("direct_url.json") or "{}")
        if not isinstance(source, dict) or not isinstance(source.get("vcs_info"), dict):
            raise ValueError("checker installation lacks Git provenance")
        commit = source["vcs_info"].get("commit_id")
        url = source.get("url")
        url = url.removesuffix(".git") if isinstance(url, str) else None
        if url != CHECKER_REPOSITORY or not isinstance(commit, str) or not re.fullmatch(
            r"[0-9a-f]{40}", commit
        ):
            raise ValueError("checker must be installed from the official Git repository")
        return commit
    except (importlib.metadata.PackageNotFoundError, ValueError, TypeError) as exc:
        raise ValueError(
            f"Cannot identify the selected official checker: {exc}. "
            "Run through prepare_submission.sh."
        ) from exc


def load_checker_policy() -> dict:
    """Explicitly use this checker's registry, independent of environment overrides."""
    from submission_checker.cohorts import Cohort
    from submission_checker.seed_sets import bundled_seed_sets_path, load_seed_sets

    checker_ref = installed_checker_ref()
    path = Path(bundled_seed_sets_path())
    raw = path.read_bytes()
    document = yaml.safe_load(raw)
    cohort = document.get("cohort") if isinstance(document, dict) else None
    target = cohort.get("cohort-id") if isinstance(cohort, dict) else None
    if not isinstance(target, str) or Cohort.parse(target) is None:
        raise ValueError("Selected checker registry must declare a valid cohort.cohort-id")
    sets = load_seed_sets(path)
    eligible = sorted(name for name, value in sets.items() if target in value.cohorts)
    if not eligible:
        raise ValueError(f"Selected checker has no published seed set adoptable for {target}")
    chosen = sets[eligible[0]]
    if any(type(value) is not int for value in chosen.seeds.values()):
        raise ValueError("Selected checker registry must supply integer runtime seeds")
    return {
        "checker_ref": checker_ref,
        "registry_sha256": hashlib.sha256(raw).hexdigest(),
        "registry_bytes": raw,
        "target_cohort": target,
        "seed_set": chosen.id,
        "seeds": chosen.seeds,
    }
