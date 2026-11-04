#!/usr/bin/env python3
"""Sequential local submission campaigns; raw attempts are never rewritten."""
from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

import yaml
from submission_package import PARALLEL_KEYS
from checker_runtime import installed_checker_ref
from submission_metadata import _apply_runtime_metadata, create_draft, load_disclosures
from submission_requirements import (
    APPROVED_REVISIONS, benchmark_datasets, effective_benchmark,
)
from submission_readiness import docs_inventory, storage_readiness
from package_inventory import available_families, family_for_profile, resolve_inventory

ROOT = Path(__file__).resolve().parent.parent
REGULAR_DIMENSIONS = {"tp_size", "replica_count"}
PD_DIMENSIONS = {
    "prefill_tp_size", "prefill_replica_count",
    "decode_tp_size", "decode_replica_count",
}
POINT_DIMENSIONS = REGULAR_DIMENSIONS | PD_DIMENSIONS
SOURCE_DIRS = {"lib", "configs", "platforms", "components", "docker", "docs"}
SOURCE_ROOT_FILES = {
    "calibrate_model.sh", "preflight.sh", "start_server.sh", "run_benchmark.sh",
    "run_submission.sh", "README.md",
    "run_full_submission.sh", "prepare_submission.sh", "resolve_submission_point.sh",
    "prepare_submission_collector.sh", "init_submission_metadata.sh", ".dockerignore",
    "collect_submission_runtime_metadata.sh",
}
SERVING_ENV = (
    "MODEL_REVISION", "MODEL_LOCAL_PATH",
    "KV_CACHE_DTYPE",
    "VLLM_TENSOR_PARALLEL_SIZE", "VLLM_CPU_KVCACHE_SPACE_GIB",
    "VLLM_KV_OFFLOADING_SIZE_GIB", "MTP_ENABLED",
    "VLLM_PD_KV_LEASE_DURATION_SECONDS",
    "LLMD_ENVOY_CONCURRENCY", "LLMD_SESSION_TTL_SECONDS", "LLMD_ROUTED_HEALTH_PATH",
    "LLMD_EPP_GRPC_PORT", "LLMD_EPP_HEALTH_PORT", "LLMD_EPP_METRICS_PORT",
    "LLMD_ENVOY_ADMIN_PORT",
    "SERVER_START_TIMEOUT_SECONDS",
)
BUDGET_ENV = ("PERFORMANCE_TRAJECTORY_COUNT", "MAX_ISSUE_DURATION_MS")


class SubmissionError(ValueError):
    pass


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SubmissionError(f"{path}: expected a JSON object")
    return value


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def positive_int(value: object, name: str) -> int:
    if type(value) is not int or not 1 <= value <= 2147483647:
        raise SubmissionError(f"{name} must be a positive integer <= 2147483647")
    return value


def load_config(path: Path) -> dict:
    from submission_checker.models.regions import compute_regions, classify_concurrency
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise SubmissionError("Submission config must be a mapping")
    extra = set(config) - {"points", "metadata", "disclosures"}
    if extra:
        raise SubmissionError(f"Unknown submission config fields: {sorted(extra, key=str)}")
    points = config.get("points")
    if not isinstance(points, list) or not 7 <= len(points) <= 32:
        raise SubmissionError("points must contain between 7 and 32 point mappings")
    for index, point in enumerate(points):
        label = f"points[{index}]"
        if not isinstance(point, dict):
            raise SubmissionError(
                f"{label}: use a point mapping such as {{concurrency: 16}}, not a scalar"
            )
        extra = set(point) - {"concurrency", "stack_profile"} - POINT_DIMENSIONS
        if extra:
            raise SubmissionError(f"{label}: unknown point fields: {sorted(extra, key=str)}")
        positive_int(point.get("concurrency"), f"{label}.concurrency")
        for name in POINT_DIMENSIONS & point.keys():
            positive_int(point[name], f"{label}.{name}")
        if REGULAR_DIMENSIONS & point.keys() and PD_DIMENSIONS & point.keys():
            raise SubmissionError(f"{label}: cannot mix replica and P/D dimensions")
        if "stack_profile" in point and (
            not isinstance(point["stack_profile"], str)
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", point["stack_profile"])
        ):
            raise SubmissionError(f"{label}.stack_profile must be a valid profile name")
    concurrencies = [point["concurrency"] for point in points]
    if concurrencies != sorted(set(concurrencies)):
        raise SubmissionError("points must be unique and sorted in increasing order")
    if concurrencies[0] not in {8, 16}:
        raise SubmissionError("The lowest concurrency point must be 8 or 16")
    regions = compute_regions(concurrencies[-1], concurrencies[0])
    assignments = {c: classify_concurrency(c, regions) for c in concurrencies}
    required = {"low_latency", "low_concurrency", "med_concurrency", "high_concurrency"}
    missing = required - set(assignments.values())
    if missing:
        raise SubmissionError(f"Missing concurrency-region coverage: {sorted(missing)}")
    disclosures_path = config.get("disclosures", "submission_disclosures.yaml")
    if not isinstance(disclosures_path, str) or not disclosures_path.strip() or Path(disclosures_path).is_absolute():
        raise SubmissionError("disclosures must name a relative submission disclosures YAML")
    disclosure_file = (path.parent / disclosures_path).resolve()
    load_disclosures(disclosure_file)
    return {
        "points": points,
        "regions": assignments, "disclosures_file": str(disclosure_file),
        "config_file": str(path.resolve()),
    }


def collection_identity(collection: dict) -> dict:
    """Bind collector facts without fresh capture paths, PIDs, or log hashes."""
    identity = {
        key: collection[key] for key in (
            "mode", "mlcflow_version", "automation_revision", "automation_tracked_clean",
            "automation_provenance", "privilege_scope", "collector_python",
            "automation_checkout", "os_observation_scope", "kernel_observation_scope",
        ) if key in collection
    }
    native = collection.get("native_observations")
    if isinstance(native, dict):
        identity["host_os_environment"] = native.get("selected_environment", {})
    return identity


def assemble_preflight_metadata(config: dict, env: dict[str, str], root: Path, resolved: list[dict]) -> dict:
    """Capture the same complete metadata used by the initializer, before models."""
    destination, provenance_path, metadata = create_draft(
        config_path=config["config_file"], env=env, root=root, preview=False,
        resolved_points=[{"concurrency": p["concurrency"], **p["serving"]} for p in resolved],
    )
    provenance = read_json(provenance_path)
    collection = provenance.get("collection")
    if not isinstance(collection, dict):
        raise SubmissionError("Metadata provenance lacks a collection record")
    metadata_sources = {
        name: provenance[name] for name in (
            "checker_ref", "registry_sha256", "disclosures_sha256", "runtime_software"
        )
    }
    metadata_sources["collection"] = collection_identity(collection)
    return {
        **config, "metadata": metadata, "metadata_file": str(destination),
        "metadata_provenance": str(provenance_path),
        "metadata_sources": metadata_sources,
        "seed_registry_file": str(provenance_path.parent / "seed_sets.yaml"),
        "runtime_images": {
            str(row["concurrency"]): row["image_id"] for row in provenance["runtime_software"]
        },
    }


def source_inventory(root: Path, metadata_file: str | None = None, disclosures_file: str | None = None,
                     families: list[str] | None = None) -> list[str]:
    generator = root / "generate_tarball.sh"
    selected = families if families is not None else available_families(root)
    if generator.is_file():
        argv = ["bash", str(generator)]
        if selected:
            argv.extend(["--profiles", ",".join(sorted(set(selected)))])
        result = subprocess.run(
            argv,
            env={**os.environ, "PACKAGE_LIST_ONLY": "1", "PACKAGE_ARCHIVE_PATH": str(root.parent / "endpoints_agentic.tar.gz")},
            capture_output=True, check=True,
        )
        paths = set(result.stdout.decode().rstrip("\0").split("\0")) - {""}
    elif selected:
        private = {"configs/submission_metadata.yaml", "configs/submission_disclosures.yaml"}
        for value in (metadata_file, disclosures_file):
            if value:
                try:
                    private.add(str(Path(value).resolve().relative_to(root.resolve())))
                except ValueError:
                    pass
        paths = set(resolve_inventory(root, selected, tracked_only=False, exclude=private))
    else:
        # Packaged sources have no Git inventory.
        paths = {
            str(p.relative_to(root)) for p in root.rglob("*")
            if p.is_file() and not p.is_symlink()
            and (p.relative_to(root).parts[0] in SOURCE_DIRS or str(p.relative_to(root)) in SOURCE_ROOT_FILES)
            and not any(part in {"run_output", "cache", "__pycache__", ".git"} for part in p.relative_to(root).parts)
        }
    return sorted(
        name for name in paths
        if not name.startswith("tests/") and name != "generate_tarball.sh"
        and name != "configs/submission_metadata.yaml"
        and name != "configs/submission_disclosures.yaml"
        and (not metadata_file or (root / name).resolve() != Path(metadata_file).resolve())
        and (not disclosures_file or (root / name).resolve() != Path(disclosures_file).resolve())
    )


def source_identity(root: Path, inventory: list[str]) -> str:
    records = []
    for name in inventory:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise SubmissionError(f"Missing or symlinked source input: {name}")
        records.append([name, path.stat().st_mode & 0o777, file_sha(path)])
    return digest(records)


def point_environment(env: dict[str, str], point: dict) -> dict[str, str]:
    result = {k: v for k, v in env.items() if not k.startswith("SUBMISSION_POINT_")}
    result.pop("ENDPOINT_URL", None)
    if "serving_environment" in point:
        for name, value in point["serving_environment"].items():
            result.pop(name, None)
            if value is not None:
                result[name] = value
    result["TARGET_CONCURRENCY"] = str(point["concurrency"])
    if "stack_profile" in point:
        result["STACK_PROFILE"] = point["stack_profile"]
    for name in POINT_DIMENSIONS & point.keys():
        result["SUBMISSION_POINT_" + name.upper()] = str(point[name])
    return result


def resolve_points(config: dict, env: dict[str, str], root: Path) -> list[dict]:
    points = []
    default_profile = env.get("STACK_PROFILE")
    announced_profiles = set()
    for definition in config["points"]:
        resolver_env = point_environment(env, definition)
        if default_profile and "stack_profile" not in definition:
            resolver_env["STACK_PROFILE"] = default_profile
        result = subprocess.run(
            ["bash", str(root / "resolve_submission_point.sh")],
            env=resolver_env, text=True, stdout=subprocess.PIPE,
            check=True,
        )
        serving = json.loads(result.stdout)
        if not isinstance(serving, dict):
            raise SubmissionError("Point resolver must return a JSON object")
        profile = serving.get("STACK_PROFILE")
        if not isinstance(profile, str) or not profile:
            raise SubmissionError("Point resolver must return a resolved stack profile")
        if "stack_profile" not in definition and not default_profile:
            default_profile = profile
        if profile not in announced_profiles:
            print(f"Resolved stack profile {profile}.", file=sys.stderr)
            announced_profiles.add(profile)
        model = serving.get("MODEL_NAME")
        revision = None
        if not env.get("MODEL_LOCAL_PATH"):
            revision = APPROVED_REVISIONS.get(model)
            if revision is None:
                raise SubmissionError(f"No approved submission checkpoint for {model!r}")
            requested = env.get("MODEL_REVISION")
            if requested is not None and requested != revision:
                raise SubmissionError(f"MODEL_REVISION must be the approved revision {revision} for {model}")
        serving["MODEL_REVISION"] = revision or ""
        points.append({
            **definition, "stack_profile": profile,
            "region": config["regions"][definition["concurrency"]], "serving": serving,
            "parallelism": {k: serving[k] for k in PARALLEL_KEYS if k in serving},
            "serving_environment": {
                **{name: env.get(name) for name in SERVING_ENV},
                "MODEL_REVISION": revision,
            },
        })
    assign_accuracy(points)
    return points


def serving_configuration(point: dict) -> str:
    return digest({
        "profile": point.get("stack_profile") or point.get("serving", {}).get("STACK_PROFILE"),
        "parallelism": point["parallelism"],
        "environment": {
            name: value for name, value in point.get("serving_environment", {}).items()
            if not name.endswith("_PORT") and name != "SERVER_START_TIMEOUT_SECONDS"
        },
    })


def assign_accuracy(points: list[dict]) -> None:
    representatives = {}
    rank = lambda p: (p.get("validation", {}).get("swebench_complete", False), p["concurrency"])
    for point in points:
        point["accuracy_required"] = False
        key = (point["region"], serving_configuration(point))
        previous = representatives.get(key)
        if previous is None or rank(point) > rank(previous):
            representatives[key] = point
    for point in representatives.values():
        point["accuracy_required"] = True


def read_server_manifest(attempt: Path) -> dict:
    """Read the trusted Bash-escaped server manifest."""
    manifests = list(attempt.glob("server_runs/*/resolved-stack.env"))
    if len(manifests) != 1:
        raise SubmissionError("Missing or ambiguous owned server manifest")
    code = (
        "import json,os,re,sys; "
        "names=re.findall(r'^([A-Z][A-Z0-9_]*)=',open(sys.argv[1]).read(),re.M); "
        "print(json.dumps({name:os.environ[name] for name in names}))"
    )
    result = subprocess.run(
        ["bash", "-c", 'set -a; source "$1"; exec "$2" -c "$3" "$1"',
         "submission-manifest", str(manifests[0]), sys.executable, code],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def dataset_entry(report_dir: Path, dataset: str) -> dict:
    accuracy_path = report_dir / "accuracy/accuracy_results.json"
    entries = read_json(accuracy_path).get("accuracy_scores")
    if not isinstance(entries, list):
        raise SubmissionError("Accuracy report must contain accuracy_scores")
    matches = [e for e in entries if isinstance(e, dict) and e.get("dataset_name") == dataset]
    if len(matches) != 1:
        raise SubmissionError(f"Missing measured {dataset} accuracy")
    return matches[0]


def report_artifacts(report_dir: Path, root: Path | None = None) -> dict[str, str]:
    """Fingerprint the small reports consumed by the submission package."""
    hashes = {}
    for name in ("config.yaml", "performance/result_summary.json", "accuracy/accuracy_results.json"):
        path = report_dir / name
        hashes[str(path.relative_to(root or report_dir))] = file_sha(path)
    return hashes


def validate_point(point: dict) -> dict:
    """Check completion and identity for reuse; the checker validates the bundle."""
    report_dir = Path(point["report_dir"])
    attempt = Path(point["attempt_dir"]) if point.get("attempt_dir") else None
    if attempt is not None:
        run_dir = Path(point["run_dir"])
        if not report_dir.resolve().is_relative_to(attempt.resolve()) or not run_dir.resolve().is_relative_to(attempt.resolve()):
            raise SubmissionError("Benchmark artifacts are outside their owned attempt")
        for name in ("config.yaml", "performance/result_summary.json", "accuracy/accuracy_results.json", "events.jsonl"):
            if not (report_dir / name).resolve().is_relative_to(attempt.resolve()):
                raise SubmissionError("Benchmark artifacts are outside their owned attempt")
    summary = read_json(report_dir / "performance/result_summary.json")
    if summary.get("complete") is not True or str(summary.get("state", "")).lower() not in {"completed", "complete"}:
        raise SubmissionError("Performance report is incomplete or interrupted")
    rendered = yaml.safe_load((report_dir / "config.yaml").read_text())
    if rendered["settings"]["load_pattern"].get("target_concurrency") != point["concurrency"]:
        raise SubmissionError("Report concurrency differs from the campaign point")
    events = report_dir / "events.jsonl"
    if not events.is_file() or not events.stat().st_size:
        raise SubmissionError("Missing or empty benchmark events file")
    _, swebench = benchmark_datasets(rendered)
    if dataset_entry(report_dir, "performance").get("complete") is not True:
        raise SubmissionError("Inline accuracy evaluation is incomplete")
    result = {"artifact_hashes": report_artifacts(report_dir, attempt)}
    if point.get("mode") == "Both" or point["accuracy_required"]:
        if dataset_entry(report_dir, swebench["name"]).get("complete") is not True:
            raise SubmissionError("SWE-bench evaluation is incomplete")
        result["swebench_complete"] = True
    return result


def seed_template(root: Path, metadata: dict, dest: Path) -> None:
    """Bind the documented client RNGs without inventing a model sampling seed."""
    data = yaml.safe_load((root / "configs/qwen_agentic_benchmark.yaml").read_text())
    runtime = metadata["point"]["runtime_settings"]["runtime"]
    data["settings"]["runtime"].update(
        scheduler_random_seed=runtime["scheduler_rng_seed"],
        dataloader_random_seed=runtime["sample_index_rng_seed"],
    )
    dest.write_text(yaml.safe_dump(data, sort_keys=False))


def run_command(command: list[str], env: dict[str, str], log: Path | None = None) -> None:
    with contextlib.ExitStack() as stack:
        output = stack.enter_context(log.open("a")) if log else None
        progress = stack.enter_context(log.open(errors="replace")) if log else None
        if progress:
            progress.seek(0, os.SEEK_END)
        pending = ""

        def relay_progress():
            nonlocal pending
            if progress is None:
                return
            lines = (pending + progress.read()).split("\n")
            pending = lines.pop()
            for line in lines:
                if line.startswith(("[corpus]", "[calibration]")) or re.match(
                    r"^\[\d{4}-\d{2}-\d{2}T", line
                ):
                    print(line, flush=True)

        process = subprocess.Popen(
            command, env=env, stdout=output, stderr=subprocess.STDOUT if output else None,
            start_new_session=True,
        )
        try:
            while True:
                try:
                    status = process.wait(timeout=1) if progress else process.wait()
                    relay_progress()
                    break
                except subprocess.TimeoutExpired:
                    relay_progress()
        except BaseException:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
            # Allow ordered launcher cleanup before terminating remaining children.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.05)
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        if status:
            raise SubmissionError(f"{Path(command[1]).name} exited with status {status}; log: {log}")


def resume_validation(stored: dict) -> dict | None:
    if stored.get("status") != "validated":
        return None
    try:
        validation = validate_point(stored)
        if all(stored["validation"]["artifact_hashes"].get(name) == value
               for name, value in validation["artifact_hashes"].items()):
            # Preserve historical hashes as provenance; only package reports are checked.
            validation["artifact_hashes"] = {
                **stored["validation"]["artifact_hashes"],
                **validation["artifact_hashes"],
            }
            return validation
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        pass
    return None


def campaign_families(config: dict, env: dict[str, str], root: Path, *,
                      resolved_points: list[dict]) -> list[str]:
    profiles = {
        point["serving"]["STACK_PROFILE"]
        for point in resolved_points
    }
    if env.get("SUBMISSION_RESUME_DIR"):
        directory = Path(env["SUBMISSION_RESUME_DIR"])
        if not directory.is_absolute() or not directory.is_dir():
            raise SubmissionError("SUBMISSION_RESUME_DIR must be an existing absolute campaign directory")
        previous = read_json(directory / "campaign.json")
        concurrencies = {str(point["concurrency"]) for point in config["points"]}
        profiles.update(
            point["stack_profile"] for key, point in previous.get("points", {}).items()
            if key in concurrencies and isinstance(point.get("stack_profile"), str)
        )
    return sorted({family_for_profile(root, profile) for profile in profiles})


def campaign(config: dict, env: dict[str, str], root: Path = ROOT, *,
             resolved_points: list[dict] | None = None) -> Path:
    from submission_package import build_bundle

    effective_benchmark(root, env)
    if resolved_points is None:
        resolved_points = resolve_points(config, env, root)
    # Bind/export the union of campaign families, never the packager's default.
    # A retained point keeps its recorded profile until it is invalidated below.
    families = campaign_families(
        config, env, root, resolved_points=resolved_points,
    ) if available_families(root) else None
    inventory = source_inventory(root, config.get("metadata_file"), config.get("disclosures_file"), families)
    config_file = Path(config.get("config_file", root / "configs/submission.yaml")).resolve()
    software = [name for name in inventory if (root / name).resolve() != config_file]
    src_id = source_identity(root, software)
    docs_dir, _ = docs_inventory(env.get("SUBMISSION_DOCS_DIR"))
    checker_ref = installed_checker_ref()
    resume = env.get("SUBMISSION_RESUME_DIR")
    if resume:
        directory = Path(resume)
        if not directory.is_absolute() or not directory.is_dir():
            raise SubmissionError("SUBMISSION_RESUME_DIR must be an existing absolute campaign directory")
    else:
        output = Path(env.get("OUTPUT_ROOT", str(root / "run_output")))
        if not output.is_absolute():
            raise SubmissionError("OUTPUT_ROOT must be absolute")
        directory = output / "submission_campaigns" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8])
        directory.mkdir(parents=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SubmissionError("This campaign is already running") from exc
        index_path = directory / "campaign.json"
        runtime_software = {
            str(row["concurrency"]): row
            for row in config.get("metadata_sources", {}).get("runtime_software", [])
        }
        if resume:
            index = read_json(index_path)
            if index.get("version") != 3:
                raise SubmissionError("Resume record lacks serving snapshots; start a new campaign")
            previous_source = index["source_identity"]
            previous_checker = index["checker_ref"]
            if previous_source != src_id or previous_checker != checker_ref:
                history = directory / "resume_history"
                history.mkdir(exist_ok=True)
                snapshot = history / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}.json"
                shutil.copy2(index_path, snapshot)
                index.setdefault("source_history", []).append({
                    "source_identity": previous_source,
                    "checker_ref": previous_checker,
                    "record": str(snapshot.relative_to(directory)),
                })
                index.update(source_identity=src_id, checker_ref=checker_ref)
            for stored in index["points"].values():
                stored.setdefault("source_identity", previous_source)
                stored.setdefault("checker_ref", previous_checker)
        else:
            index = {
                "version": 3, "source_identity": src_id,
                "checker_ref": checker_ref,
                "metadata_sources": config.get("metadata_sources"),
                "status": "preparing", "points": {},
            }
            atomic_json(index_path, index)
            if config.get("metadata_provenance"):
                shutil.copytree(Path(config["metadata_provenance"]).parent, directory / "metadata")
        def verify_metadata_snapshot():
            if not config.get("metadata_sources"):
                return
            snapshot = directory / "metadata"
            snapshot_sources = index["metadata_sources"]
            if file_sha(snapshot / "seed_sets.yaml") != snapshot_sources["registry_sha256"]:
                raise SubmissionError("Captured checker registry changed")
            captured_metadata = yaml.safe_load((snapshot / "metadata.yaml").read_bytes())
            baseline = copy.deepcopy(captured_metadata)
            baseline["system"].update(index.get("system_metadata_additions", {}))
            comparable = copy.deepcopy(config["metadata"])
            comparable["system"] = {
                name: value for name, value in comparable["system"].items()
                if name in baseline["system"]
            }
            # Runtime software may change between attempts; host and checkpoint
            # metadata must still match the original campaign.
            for metadata in (baseline, comparable):
                metadata["system"].pop("serving_framework", None)
                for node in metadata["system"].get("node_types", []):
                    node.pop("inference_backend", None)
                    node.pop("other_software_stack", None)
            if comparable != baseline:
                raise SubmissionError("Captured metadata snapshot changed")
            if json.loads((snapshot / "runtime-software.json").read_bytes()) != snapshot_sources["runtime_software"]:
                raise SubmissionError("Captured container software observations changed")
            if "collection" in snapshot_sources:
                captured_provenance = read_json(snapshot / "provenance.json")
                captured_collection = captured_provenance.get("collection", {})
                if collection_identity(captured_collection) != snapshot_sources["collection"]:
                    raise SubmissionError("Captured collection provenance changed")
            expected_disclosures = snapshot_sources["disclosures_sha256"]
            if file_sha(snapshot / "disclosures.yaml") != expected_disclosures:
                raise SubmissionError("Captured manual submission disclosures changed")
            additions = {
                name: value for name, value in config["metadata"]["system"].items()
                if name not in captured_metadata["system"]
            }
            if additions:
                index["system_metadata_additions"] = additions
        verify_metadata_snapshot()
        seeded_config = directory / "benchmark-template.yaml"
        if not resume:
            seed_template(root, config["metadata"], seeded_config)
        elif not seeded_config.is_file():
            raise SubmissionError("Resume campaign is missing its benchmark template")
        if resume and index.get("benchmark_template_sha256") != file_sha(seeded_config):
            raise SubmissionError("Resume benchmark template changed")
        index["benchmark_template_sha256"] = file_sha(seeded_config)
        atomic_json(index_path, index)
        run_env = dict(env)
        run_env["BENCHMARK_CONFIG_TEMPLATE"] = str(seeded_config)
        run_env["OUTPUT_ROOT"] = str(directory / "preparation")
        run_env.pop("ENDPOINT_URL", None)
        run_env = {k: v for k, v in run_env.items() if not k.startswith("SUBMISSION_POINT_")}
        try:
            retained = {}
            for key, stored in index["points"].items():
                candidate = dict(stored)
                if stored.get("status") == "validated":
                    validation = resume_validation(stored)
                    if validation is None:
                        continue
                    candidate["validation"] = validation
                elif stored.get("status") in {"failed", "interrupted"}:
                    try:
                        candidate["validation"] = validate_point(candidate)
                    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
                        continue
                    candidate["status"] = "validated"
                    candidate.pop("error", None)
                    print(f"Recovered finished concurrency {key}", flush=True)
                else:
                    continue
                candidate["validation_checker_ref"] = checker_ref
                retained[key] = candidate
            pending = [
                copy.deepcopy(point) for point in resolved_points
                if str(point["concurrency"]) not in retained
            ]
            points = sorted([*retained.values(), *pending], key=lambda p: p["concurrency"])
            assign_accuracy(points)
            scheduled = [
                p for p in points if p.get("status") != "validated"
                or (p["accuracy_required"] and not p["validation"].get("swebench_complete"))
            ]
            index["points"].update(retained)
            if scheduled:
                index["status"] = "preparing"
                index.pop("error", None)
            atomic_json(index_path, index)
            print(f"Campaign: {directory}", flush=True)
            # Prepare each configuration at its highest scheduled concurrency.
            preparation = {serving_configuration(point): point for point in scheduled}
            for configuration, representative in preparation.items():
                verify_metadata_snapshot()
                preparation_dir = directory / "preparation" / configuration
                preparation_dir.mkdir(parents=True, exist_ok=True)
                preparation_env = point_environment(run_env, representative)
                preparation_env["OUTPUT_ROOT"] = str(preparation_dir)
                preparation_env["MODE"] = "Both"
                expected_image = config.get("runtime_images", {}).get(str(representative["concurrency"]))
                if expected_image:
                    preparation_env["SUBMISSION_EXPECTED_RUNTIME_IMAGE_ID"] = expected_image
                print(
                    f"Preparing profile={representative['stack_profile']}; "
                    f"log: {preparation_dir / 'preflight.log'}",
                    flush=True,
                )
                run_command(
                    ["bash", str(root / "preflight.sh")], preparation_env,
                    preparation_dir / "preflight.log",
                )
            for point_index, point in enumerate(points):
                verify_metadata_snapshot()
                profile = point["stack_profile"]
                key = str(point["concurrency"])
                if point not in scheduled:
                    index["points"][key] = point
                    print(f"Reusing validated concurrency {key}", flush=True)
                    continue
                point = dict(point)
                attempt = directory / "points" / key / ("attempt-" + uuid.uuid4().hex[:12])
                attempt.mkdir(parents=True)
                point["attempt_dir"] = str(attempt)
                point["status"] = "running"
                point["source_identity"] = source_identity(root, software)
                point["checker_ref"] = checker_ref
                point["mode"] = "Both" if point["accuracy_required"] else "Performance"
                point["budget"] = {name: run_env.get(name) for name in BUDGET_ENV}
                point.pop("validation", None)
                index["points"][key] = point
                index["status"] = "running"
                index.pop("error", None)
                atomic_json(index_path, index)
                snapshot = {
                    "concurrency": point["concurrency"], "stack_profile": profile,
                    "parallelism": point["parallelism"],
                    "serving_environment": point["serving_environment"],
                    "mode": point["mode"], "budget": point["budget"],
                }
                atomic_json(attempt / "submission-settings.json", snapshot)
                point_env = point_environment(run_env, point)
                point_env["OUTPUT_ROOT"] = str(attempt)
                point_env["MODE"] = point["mode"]
                print(
                    f"Running concurrency {key}, profile={profile}, mode={point_env['MODE']}",
                    flush=True,
                )
                point_env["SUBMISSION_SKIP_PREFLIGHT"] = "1"
                expected_image = config.get("runtime_images", {}).get(key)
                if expected_image:
                    point_env["SUBMISSION_EXPECTED_RUNTIME_IMAGE_ID"] = expected_image
                run_command(["bash", str(root / "run_submission.sh")], point_env, attempt / "launcher.log")
                serving = read_server_manifest(attempt)
                if expected_image and serving.get("RUNTIME_IMAGE_ID") != expected_image:
                    raise SubmissionError("Actual server image differs from the metadata software capture")
                if serving.get("MODEL_NAME") != point["serving"]["MODEL_NAME"] or any(
                    (serving.get(key) or None) != point["serving_environment"][key]
                    for key in ("MODEL_REVISION", "MODEL_LOCAL_PATH")
                ):
                    raise SubmissionError("Actual server checkpoint differs from the campaign checkpoint")
                if serving.get("STACK_PROFILE") != profile or any(
                    str(serving.get(k, "replica" if k == "TOPOLOGY_MODE" else "")) != str(v)
                    for k, v in point["parallelism"].items()
                ):
                    raise SubmissionError("Actual server topology differs from the campaign point")
                point["serving"] = serving
                if key in runtime_software:
                    point["runtime_software"] = runtime_software[key]
                snapshot["serving"] = serving
                atomic_json(attempt / "submission-settings.json", snapshot)
                reports = list(attempt.glob("*/reports/performance/result_summary.json"))
                if len(reports) != 1:
                    raise SubmissionError(f"Expected one performance report at concurrency {key}; found {len(reports)}")
                point["report_dir"] = str(reports[0].parent.parent)
                point["run_dir"] = str(reports[0].parent.parent.parent)
                point["validation"] = validate_point(point)
                point["validation_checker_ref"] = checker_ref
                point["status"] = "validated"
                points[point_index] = point
                atomic_json(index_path, index)
            verify_metadata_snapshot()
            index.pop("accuracy_validation", None)
            atomic_json(index_path, index)
            # Export source only for packaging.
            with tempfile.TemporaryDirectory(prefix="submission-source-", dir=directory) as temporary:
                source = Path(temporary) / "endpoints_agentic"
                export_id = source_identity(root, inventory)
                for name in inventory:
                    target = source / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(root / name, target)
                if source_identity(source, inventory) != export_id:
                    raise SubmissionError("Source changed while preparing the package")
                registry_args = (
                    {"seed_registry_path": directory / "metadata/seed_sets.yaml"}
                    if config.get("metadata_sources") else {}
                )
                package_metadata = copy.deepcopy(config["metadata"])
                known_images = {
                    row["image_id"]: row
                    for row in [
                        *(index.get("metadata_sources") or {}).get("runtime_software", []),
                        *runtime_software.values(),
                    ]
                }
                used_software = [
                    row for point in points
                    if (row := point.get("runtime_software") or known_images.get(
                        point["serving"].get("RUNTIME_IMAGE_ID")
                    ))
                ]
                if used_software:
                    disclosures = yaml.safe_load((directory / "metadata/disclosures.yaml").read_bytes())
                    manual_nodes = disclosures.get("system", {}).get("node_types", [])
                    for node_index, node in enumerate(package_metadata["system"]["node_types"]):
                        node.pop("other_software_stack", None)
                        if node_index < len(manual_nodes) and "other_software_stack" in manual_nodes[node_index]:
                            node["other_software_stack"] = manual_nodes[node_index]["other_software_stack"]
                    package_metadata = _apply_runtime_metadata(package_metadata, used_software, disclosures)
                if docs_dir:
                    bundle = build_bundle(directory, package_metadata, points, source, docs_dir=docs_dir, **registry_args)
                else:
                    bundle = build_bundle(directory, package_metadata, points, source, **registry_args)
            index["status"] = "complete"
            index.pop("error", None)
            index["bundle"] = str(bundle)
            atomic_json(index_path, index)
            return bundle
        except BaseException as exc:
            index["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            index["error"] = str(exc)
            for point in index["points"].values():
                if point.get("status") == "running":
                    point["status"] = index["status"]
                    point["error"] = str(exc)
            atomic_json(index_path, index)
            raise


def main() -> int:
    if len(sys.argv) != 1:
        raise SubmissionError("Use environment variables; positional arguments are unsupported")
    if os.environ.get("COLLECT_EMON", "0") not in {"0", "1"}:
        raise SubmissionError("COLLECT_EMON must be 0 or 1")
    config_path = Path(os.environ.get("SUBMISSION_CONFIG", str(ROOT / "configs/submission.yaml"))).resolve()
    config = load_config(config_path)
    for name in ("PERFORMANCE_TRAJECTORY_COUNT", "SWEBENCH_INSTANCE_COUNT"):
        if name in os.environ:
            value = os.environ[name]
            if not re.fullmatch(r"[1-9][0-9]*", value):
                raise SubmissionError(f"{name} must be a positive integer")
            number = positive_int(int(value), name)
            if name == "PERFORMANCE_TRAJECTORY_COUNT" and (number < 613 or number % 613):
                raise SubmissionError("PERFORMANCE_TRAJECTORY_COUNT must be a whole multiple of 613")
    resolved = resolve_points(config, dict(os.environ), ROOT)
    effective_benchmark(ROOT, dict(os.environ))
    docs_inventory(os.environ.get("SUBMISSION_DOCS_DIR"))
    readiness = storage_readiness(ROOT, dict(os.environ))
    config = assemble_preflight_metadata(config, dict(os.environ), ROOT, resolved)
    metadata_point = config["metadata"]["point"]
    baseline_model = yaml.safe_load((ROOT / "configs/qwen_agentic_benchmark.yaml").read_text())["model_params"]["name"]
    if metadata_point["model_name"] != baseline_model:
        raise SubmissionError("point.model_name must match the existing benchmark model")
    if not os.environ.get("SUBMISSION_DOCS_DIR"):
        readiness["warnings"].append(
            "No specification evidence directory supplied; complete public system/power references before submission."
        )
    if os.environ.get("SUBMISSION_VALIDATE_ONLY") == "1":
        print(json.dumps({
            "configuration_valid": True,
            "metadata_file": config["metadata_file"],
            "checker_ref": config["metadata_sources"]["checker_ref"],
            "points": [{"concurrency": p["concurrency"], "serving": p["serving"]} for p in resolved],
            "storage": readiness,
            "target_cohort": config["metadata"]["point"]["target_cohort"],
        }, indent=2))
        return 0
    for warning in readiness["warnings"]:
        print(f"Readiness warning: {warning}", file=sys.stderr)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    bundle = campaign(config, dict(os.environ), resolved_points=resolved)
    print(f"Validated local submission archive: {bundle}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Campaign interrupted; attempts were preserved.", file=sys.stderr)
        sys.exit(130)
    except (SubmissionError, OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Submission failed: {exc}", file=sys.stderr)
        sys.exit(1)
